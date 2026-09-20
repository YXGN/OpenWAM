"""Explicit CUDA-Graph runner for the fixed-shape Wan joint MoT loop."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
from torch import Tensor

_VIDEO_FIELDS = ("hidden_states", "time_mod", "rope_freqs", "context", "context_mask")
_ACTION_FIELDS = ("x_action", "t_mod", "action_freqs", "context", "context_mask")


def _tensor_signature(value: Optional[Tensor]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
    }


def _broadcast_base(value: Tensor) -> Tensor:
    """Return a non-overlapping view with expanded dimensions reduced to one."""
    base = value
    for dim in reversed(range(value.ndim)):
        if value.shape[dim] > 1 and value.stride(dim) == 0:
            base = base.select(dim, 0).unsqueeze(dim)
    return base


def _kv_pairs(value, label: str):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a tuple/list of (key, value) pairs")
    pairs = []
    for layer_id, pair in enumerate(value):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{label}[{layer_id}] must contain exactly (key, value)")
        if not isinstance(pair[0], Tensor) or not isinstance(pair[1], Tensor):
            raise ValueError(f"{label}[{layer_id}] key and value must be tensors")
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


class _StaticTensor:
    """Writable storage preserving a tensor's shape and stride contract."""

    def __init__(self, source: Tensor, *, allow_broadcast: bool):
        self.expanded = any(size > 1 and stride == 0 for size, stride in zip(source.shape, source.stride()))
        if any(stride < 0 for stride in source.stride()):
            raise ValueError("negative-stride tensors are not supported by the MoT CUDA graph")
        if self.expanded and not allow_broadcast:
            raise ValueError("expanded tensor layout is unsupported for this MoT CUDA graph input")

        if self.expanded:
            base_source = _broadcast_base(source)
            self.base = torch.empty_strided(
                base_source.shape,
                base_source.stride(),
                dtype=source.dtype,
                device=source.device,
            )
            self.view = self.base.expand(source.shape)
            self.base.copy_(base_source)
        else:
            self.base = torch.empty_strided(
                source.shape,
                source.stride(),
                dtype=source.dtype,
                device=source.device,
            )
            self.view = self.base
            self.base.copy_(source)

    def copy_from(self, source: Tensor) -> None:
        if self.expanded:
            self.base.copy_(_broadcast_base(source))
        else:
            self.base.copy_(source)


class MoTCudaGraphRunner:
    """Run a fixed-shape Wan joint MoT driver through one explicit CUDA graph.

    The runner owns only graph input/output storage and shallow state wrappers.
    It never changes the driver's per-layer algorithm or the caller's context
    and metadata fields.
    """

    def __init__(self, driver) -> None:
        self.driver = driver
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._buffers: dict[str, _StaticTensor] = {}
        self._output_v: Optional[Tensor] = None
        self._output_a: Optional[Tensor] = None
        self._signature: Optional[dict[str, Any]] = None
        self._capture_count = 0
        self._replay_count = 0
        self._reset_count = 0
        self._last_backend = "eager"
        self._reason = "not_captured"

    def status(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return JSON-serializable graph state for deployment diagnostics."""
        return {
            "enabled": True,
            "captured": self._graph is not None,
            "capture_count": self._capture_count,
            "replay_count": self._replay_count,
            "reset_count": self._reset_count,
            "last_backend": self._last_backend,
            "reason": self._reason,
            "signature": copy.deepcopy(self._signature) if include_signature else None,
        }

    def clear(self, reason: str = "reset") -> None:
        """Drop captured graph and buffers after weights/device/dtype changes."""
        self._graph = None
        self._buffers.clear()
        self._output_v = None
        self._output_a = None
        self._signature = None
        self._reset_count += 1
        self._last_backend = "eager"
        self._reason = reason

    @staticmethod
    def _state_signature(vstate, astate, driver) -> dict[str, Any]:
        payload = astate.payload
        video_kv = _kv_pairs(getattr(vstate, "extras", {}).get("context_kv"), "video.context_kv")
        action_kv = _kv_pairs(getattr(payload, "context_kv", None), "action.context_kv")

        def _kv_signature(pairs):
            if pairs is None:
                return None
            return [{"key": _tensor_signature(key), "value": _tensor_signature(value)} for key, value in pairs]

        return {
            "video": {name: _tensor_signature(getattr(vstate, name)) for name in _VIDEO_FIELDS},
            "action": {name: _tensor_signature(getattr(payload, name)) for name in _ACTION_FIELDS},
            "video_context_kv": _kv_signature(video_kv),
            "action_context_kv": _kv_signature(action_kv),
            "action_state": {
                "action_latents": _tensor_signature(astate.action_latents),
                "timestep": _tensor_signature(astate.timestep),
            },
            "grid": {
                "frames": int(vstate.grid_frames),
                "height": int(vstate.grid_height),
                "width": int(vstate.grid_width),
                "prefix_kv_len": int(vstate.prefix_kv_len),
            },
            "mask_mode": str(driver.attention_mask_mode),
            "video_mask_mode": str(getattr(driver.vb, "video_attention_mask_mode", None)),
            "driver_layers": int(driver.num_layers),
        }

    @staticmethod
    def _context_kv(vstate, astate):
        return (
            _kv_pairs(getattr(vstate, "extras", {}).get("context_kv"), "video.context_kv"),
            _kv_pairs(getattr(astate.payload, "context_kv", None), "action.context_kv"),
        )

    def _iter_inputs(self, vstate, astate):
        payload = astate.payload
        for name in _VIDEO_FIELDS:
            yield f"video.{name}", getattr(vstate, name), name == "context_mask"
        for name in _ACTION_FIELDS:
            yield f"action.{name}", getattr(payload, name), name == "context_mask"
        video_kv, action_kv = self._context_kv(vstate, astate)
        for prefix, pairs in (("video.context_kv", video_kv), ("action.context_kv", action_kv)):
            if pairs is not None:
                for layer_id, (key, value) in enumerate(pairs):
                    yield f"{prefix}.{layer_id}.key", key, False
                    yield f"{prefix}.{layer_id}.value", value, False

    def _validate_layouts(self, vstate, astate) -> Optional[str]:
        for label, value, allow_broadcast in self._iter_inputs(vstate, astate):
            if value is None:
                continue
            if value.device.type != "cuda":
                return f"{label}_not_cuda"
            strides = value.stride()
            if any(stride < 0 for stride in strides):
                return f"{label}_layout_unsupported:negative-stride tensors are unsupported"
            expanded = any(size > 1 and stride == 0 for size, stride in zip(value.shape, strides))
            if expanded and not allow_broadcast:
                return f"{label}_layout_unsupported:expanded tensors are unsupported"
            dense_value = _broadcast_base(value) if expanded else value
            dense_strides = dense_value.stride()
            expected_stride = 1
            for dim in sorted(range(dense_value.ndim), key=dense_strides.__getitem__):
                size = dense_value.shape[dim]
                stride = dense_strides[dim]
                if size > 1:
                    if stride != expected_stride:
                        return f"{label}_layout_unsupported:overlapping or non-dense layout"
                    expected_stride = stride * size
        return None

    def _unsupported_reason(self, vstate, astate, *, use_gradient_checkpointing, use_gradient_checkpointing_offload):
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        if vstate.hidden_states.device.type != "cuda":
            return "non_cuda_state"
        if getattr(self.driver.vb, "training", False) or getattr(self.driver.ab, "training", False):
            return "training_mode"
        if torch.is_grad_enabled():
            return "grad_enabled"
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return "checkpoint_or_offload"
        if getattr(self.driver.ab, "variant", None) != "joint_self_attn":
            return "non_joint_self_attn_action_variant"
        if not self.driver.vb.__class__.__module__.startswith("openwam.model.video_backbone.wan"):
            return "non_wan_video_backbone"
        if vstate.vace_hints is not None:
            return "vace_hints_unsupported"
        if int(vstate.prefix_kv_len) != 0 or vstate.prefix_kv_mask is not None:
            return "prefix_kv_unsupported"
        try:
            self._context_kv(vstate, astate)
        except ValueError as exc:
            return f"context_kv_invalid:{exc}"
        for label, value, _ in self._iter_inputs(vstate, astate):
            if value is not None and value.dtype.is_floating_point and value.dtype != torch.bfloat16:
                return f"{label}_dtype_not_bfloat16"
        return self._validate_layouts(vstate, astate)

    def _make_buffer_states(self, vstate, astate):
        vbuffer = copy.copy(vstate)
        vbuffer.extras = dict(getattr(vstate, "extras", {}))
        for name in _VIDEO_FIELDS:
            value = self._buffers[f"video.{name}"] if f"video.{name}" in self._buffers else None
            setattr(vbuffer, name, None if value is None else value.view)
        vbuffer.vace_hints = None
        vbuffer.prefix_kv_len = 0
        vbuffer.prefix_kv_mask = None
        video_kv = _kv_pairs(vbuffer.extras.get("context_kv"), "video.context_kv")
        if video_kv is not None:
            vbuffer.extras["context_kv"] = tuple(
                (
                    self._buffers[f"video.context_kv.{layer_id}.key"].view,
                    self._buffers[f"video.context_kv.{layer_id}.value"].view,
                )
                for layer_id, _ in enumerate(video_kv)
            )

        abuffer = copy.copy(astate)
        payload = copy.copy(astate.payload)
        for name in _ACTION_FIELDS:
            value = self._buffers[f"action.{name}"] if f"action.{name}" in self._buffers else None
            setattr(payload, name, None if value is None else value.view)
        action_kv = _kv_pairs(getattr(payload, "context_kv", None), "action.context_kv")
        if action_kv is not None:
            payload.context_kv = tuple(
                (
                    self._buffers[f"action.context_kv.{layer_id}.key"].view,
                    self._buffers[f"action.context_kv.{layer_id}.value"].view,
                )
                for layer_id, _ in enumerate(action_kv)
            )
        abuffer.payload = payload
        return vbuffer, abuffer

    def _copy_inputs(self, vstate, astate) -> None:
        for label, source, _ in self._iter_inputs(vstate, astate):
            if source is not None:
                self._buffers[label].copy_from(source)

    def _allocate_buffers(self, vstate, astate) -> None:
        for label, source, allow_broadcast in self._iter_inputs(vstate, astate):
            if source is not None:
                self._buffers[label] = _StaticTensor(source, allow_broadcast=allow_broadcast)

    def _capture(self, vstate, astate) -> None:
        self._allocate_buffers(vstate, astate)
        device = vstate.hidden_states.device
        current_stream = torch.cuda.current_stream(device=device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                warm_v, warm_a = self._make_buffer_states(vstate, astate)
                self.driver.run_joint_loop(
                    warm_v,
                    warm_a,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                )
        current_stream.wait_stream(warmup_stream)

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_v, graph_a = self._make_buffer_states(vstate, astate)
            graph_v, graph_a = self.driver.run_joint_loop(
                graph_v,
                graph_a,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            self._output_v = graph_v.hidden_states
            self._output_a = graph_a.payload.x_action
        current_stream.wait_stream(capture_stream)
        self._graph = graph
        self._capture_count += 1
        self._reason = "captured"

    def run(
        self,
        vstate,
        astate,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        reason = self._unsupported_reason(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if reason is not None:
            self._last_backend = "eager"
            self._reason = reason
            return self.driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        signature = self._state_signature(vstate, astate, self.driver)
        if self._graph is not None and signature != self._signature:
            self._last_backend = "eager"
            self._reason = "static_signature_mismatch"
            return self.driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        if self._graph is None:
            self._signature = signature
            self._capture(vstate, astate)

        self._copy_inputs(vstate, astate)
        self._graph.replay()
        self._replay_count += 1
        self._last_backend = "cuda_graph"
        self._reason = "replayed"
        vstate.hidden_states = self._output_v.clone()
        astate.payload.x_action = self._output_a.clone()
        return vstate, astate
