"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        self._compile_optimization_requested = False
        self._mot_cuda_graph_runner = None
        self._request_context_cache_enabled = False
        self._request_context_cache_status = {
            "enabled": False,
            "last_reason": "disabled",
            "requests_prepared": 0,
            "cache_hits": 0,
        }
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            **self._mot_driver_kwargs,
        )
        if self._mot_cuda_graph_runner is not None:
            self._mot_cuda_graph_runner.clear("mot_driver_rebuilt")
            self._mot_cuda_graph_runner.driver = self._mot_driver
        return self._mot_driver

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        self._compiled_mot_run_joint_loop = None
        self._compile_optimization_requested = compile_enabled(compile_cfg, default=False, strict=True)
        if self._compile_optimization_requested and self._mot_cuda_graph_runner is not None:
            raise ValueError("dual_system_self_attn cannot enable torch.compile and CUDA Graph MoT together")
        super().apply_compile_optimizations(compile_cfg)
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = self_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("dual self-attn MoT compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            logger.warning("dual self-attn MoT torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

    def apply_cuda_graph_optimization(self, enabled: bool = True) -> None:
        """Enable the explicit fixed-shape CUDA Graph MoT runner."""
        self._mot_cuda_graph_runner = None
        if not enabled:
            return
        if self._compile_optimization_requested:
            raise ValueError("dual_system_self_attn cannot enable CUDA Graph MoT and torch.compile together")
        if self.video_backbone is None or self.action_backbone is None:
            raise ValueError("CUDA Graph MoT requires initialized Wan video and action backbones")
        from openwam.model.architectures.dual_system.mot_cuda_graph import MoTCudaGraphRunner

        driver = self._mot_driver or self.build_mot_driver()
        self._mot_cuda_graph_runner = MoTCudaGraphRunner(driver)

    def apply_request_context_cache_optimization(self, enabled: bool = True) -> None:
        """Enable request-local prepared context/KV reuse for supported inference."""
        if bool(enabled) != self._request_context_cache_enabled and self._mot_cuda_graph_runner is not None:
            self._mot_cuda_graph_runner.clear("context_cache_mode_changed")
        self._request_context_cache_enabled = bool(enabled)
        self._request_context_cache_status["enabled"] = self._request_context_cache_enabled
        self._request_context_cache_status["last_reason"] = "enabled" if enabled else "disabled"

    def _request_context_cache_unsupported_reason(
        self,
        pipeline_inputs: dict,
        noisy_actions: Optional[Tensor],
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> Optional[str]:
        if noisy_actions is None or self.action_backbone is None:
            return "action_stream_unavailable"
        if self.training or getattr(self.video_backbone, "training", False) or getattr(self.action_backbone, "training", False):
            return "training_mode"
        if torch.is_grad_enabled():
            return "grad_enabled"
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return "checkpoint_or_offload"
        if getattr(self.action_backbone, "variant", None) != "joint_self_attn":
            return "non_joint_self_attn_action_variant"
        if not self.video_backbone.__class__.__module__.startswith("openwam.model.video_backbone.wan"):
            return "non_wan_video_backbone"
        if bool(getattr(getattr(self.video_backbone, "dit", None), "has_image_input", False)):
            return "image_input_unsupported"
        if float(pipeline_inputs.get("cfg_scale", 1.0)) != 1.0:
            return "cfg_enabled"
        for key in (
            "vace_video",
            "vace_context",
            "vace_reference_image",
            "vace_video_mask",
            "clip_feature",
            "y",
        ):
            if pipeline_inputs.get(key) is not None:
                return f"{key}_unsupported"
        return None

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Create a fresh request-local context cache for eligible generation."""
        if self._request_context_cache_enabled:
            cfg_scale = float(kwargs.get("cfg_scale", 1.0))
            if cfg_scale == 1.0:
                request_cache: dict = {}
                kwargs["_inference_context_cache"] = request_cache
                self._request_context_cache_status["last_reason"] = "pending"
            else:
                self._request_context_cache_status["last_reason"] = "cfg_enabled"
        return super().generate(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        runner = getattr(self, "_mot_cuda_graph_runner", None)
        if runner is not None:
            runner.clear("module_transform")
        return super()._apply(fn, recurse=recurse)

    def set_dtype_device(self, dtype, device):
        runner = getattr(self, "_mot_cuda_graph_runner", None)
        if runner is not None:
            runner.clear("dtype_device_changed")
        super().set_dtype_device(dtype, device)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        runner = getattr(self, "_mot_cuda_graph_runner", None)
        if runner is not None:
            runner.clear("checkpoint_reloaded")
        super().load_checkpoint(path, strict=strict)

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        _inference_context_cache: Optional[dict] = None,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        request_cache = _inference_context_cache if self._request_context_cache_enabled else None
        cache_hit = bool(request_cache and request_cache.get("_prepared"))
        if request_cache is not None and not cache_hit:
            unsupported_reason = self._request_context_cache_unsupported_reason(
                pipeline_inputs,
                noisy_actions,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
            )
            if unsupported_reason is not None:
                request_cache["_unsupported_reason"] = unsupported_reason
                self._request_context_cache_status["last_reason"] = unsupported_reason
                request_cache = None

        if cache_hit:
            pipeline_inputs = dict(pipeline_inputs)
            pipeline_inputs["context"] = request_cache["raw_context"]
            pipeline_inputs["context_mask"] = request_cache["raw_context_mask"]
            if request_cache["raw_seq_lens"] is None:
                pipeline_inputs.pop("seq_lens", None)
            else:
                pipeline_inputs["seq_lens"] = request_cache["raw_seq_lens"]
            prepared_video_context = request_cache["video_context"]
            prepared_video_kv = request_cache["video_kv"]
            prepared_action_context = request_cache["action_context"]
            prepared_action_kv = request_cache["action_kv"]
        else:
            pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
            prepared_video_context = None
            prepared_video_kv = None
            prepared_action_context = None
            prepared_action_kv = None

        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        video_prepare_inputs = dict(pipeline_inputs)
        if cache_hit:
            video_prepare_inputs["prepared_context"] = prepared_video_context
            video_prepare_inputs["context_kv"] = prepared_video_kv
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **video_prepare_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        if cache_hit:
            astate = ab.prepare_state(
                noisy_actions,
                action_timestep,
                prepared_context=prepared_action_context,
                context_kv=prepared_action_kv,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        else:
            astate = ab.prepare_state(
                noisy_actions,
                action_timestep,
                context=action_context,
                context_mask=action_context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        if request_cache is not None and not cache_hit:
            unsupported_reason = None
            if int(getattr(vstate, "prefix_kv_len", 0)) != 0 or getattr(vstate, "prefix_kv_mask", None) is not None:
                unsupported_reason = "prefix_kv_unsupported"
            elif getattr(vstate, "vace_hints", None) is not None:
                unsupported_reason = "vace_hints_unsupported"
            if unsupported_reason is not None:
                request_cache["_unsupported_reason"] = unsupported_reason
                self._request_context_cache_status["last_reason"] = unsupported_reason
            else:
                video_kv = tuple(
                    block.cross_attn.project_context_kv(vstate.context)
                    for block in vstate.extras["dit"].blocks
                )
                action_kv = tuple(
                    block.cross_attn.project_context_kv(astate.payload.context)
                    for block in ab.blocks
                )
                vstate.extras["context_kv"] = video_kv
                astate.payload.context_kv = action_kv
                request_cache.update(
                    {
                        "_prepared": True,
                        "raw_context": pipeline_inputs.get("context"),
                        "raw_context_mask": pipeline_inputs.get("context_mask"),
                        "raw_seq_lens": pipeline_inputs.get("seq_lens"),
                        "video_context": (vstate.context, vstate.context_mask),
                        "video_kv": video_kv,
                        "action_context": (astate.payload.context, astate.payload.context_mask),
                        "action_kv": action_kv,
                    }
                )
                self._request_context_cache_status["requests_prepared"] += 1
                self._request_context_cache_status["last_reason"] = "prepared"
        elif cache_hit:
            self._request_context_cache_status["cache_hits"] += 1
            self._request_context_cache_status["last_reason"] = "cache_hit"
        graph_runner = getattr(self, "_mot_cuda_graph_runner", None)
        if graph_runner is not None:
            vstate, astate = graph_runner.run(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            return vb.finalize(vstate), ab.extract_prediction(astate)

        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        if compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                vstate, astate = compiled_loop(vstate, astate)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("dual self-attn compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate = driver.run_joint_loop(
                    vstate,
                    astate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vb.finalize(vstate), ab.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
