#!/usr/bin/env python3
"""Generate fixed MoonCake open-loop windows from the official checkpoint.

This is a thin probe, not a training or serving process.  It reuses the
service's ``OfficialAdapter`` and the patched reader.  For each requested
window only the reader's first current canvas and state are passed to the
adapter; future video frames are never used as conditions.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parent
OPENWAM_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(OPENWAM_ROOT))

from moon_cake import ACTION_STEPS, EEF_DIM  # noqa: E402
from openloop_validate_moon_cake import reader_window  # noqa: E402
from openwam_front import PROMPT, _official_adapter, load_config  # noqa: E402
from prepare_moon_cake import _reader_from_path  # noqa: E402

FIXED_SEED = 42
TRAINING_LOG_OUTPUT_TEMPLATE = "outputs/${now:%Y-%m-%d_%H-%M-%S}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_ints(value: str, field: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--{field} must be comma-separated integers") from exc
    if not result:
        raise ValueError(f"--{field} must contain at least one integer")
    return result


def requested_windows(episodes: str, starts: str) -> list[tuple[int, int]]:
    episode_ids = _csv_ints(episodes, "episodes")
    local_starts = _csv_ints(starts, "starts")
    if len(episode_ids) == 1 and len(local_starts) > 1:
        episode_ids *= len(local_starts)
    elif len(local_starts) == 1 and len(episode_ids) > 1:
        local_starts *= len(episode_ids)
    if len(episode_ids) != len(local_starts):
        raise ValueError("--episodes and --starts must have equal lengths, or one side length 1")
    return list(zip(episode_ids, local_starts, strict=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--front-config", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--source-patch", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--starts", required=True)
    parser.add_argument("--output", required=True, type=Path, help="output actions NPZ")
    parser.add_argument("--device", required=True)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dit-cache", action="store_true")
    return parser


def _seed_torch() -> None:
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    import torch

    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)


def _artifact(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": known_sha256 if known_sha256 is not None else _sha256(path),
    }


def _trace_architecture_generate(adapter: Any) -> list[dict[str, Any]]:
    """Observe the scalar kwargs handed to the official architecture call."""

    architecture = adapter.policy_server.engine.architecture
    original = architecture.generate
    observed: list[dict[str, Any]] = []

    @functools.wraps(original)
    def traced_generate(*args: Any, **kwargs: Any):
        observed.append(
            {
                "seed": kwargs.get("seed"),
                "num_inference_steps": kwargs.get("num_inference_steps"),
                "num_frames": kwargs.get("num_frames"),
                "action_num_frames": kwargs.get("action_num_frames"),
                "decode_video": kwargs.get("decode_video"),
                "dit_cache_is_none": kwargs.get("dit_cache") is None,
            }
        )
        return original(*args, **kwargs)

    architecture.generate = traced_generate
    return observed


def _metadata_config_snapshot(omega_conf: Any, live_cfg: Any) -> tuple[dict[str, Any], Any, list[dict[str, str]]]:
    """Resolve a metadata copy while preserving Hydra's runtime-only log path.

    The official training config uses Hydra's ``now`` resolver for its log
    directory. That resolver belongs to the training runtime and must not be
    replaced with a timestamp while recording a deployment probe. Every other
    interpolation remains strict: errors propagate from the copied config.
    """

    loaded_config_unresolved = omega_conf.to_container(live_cfg, resolve=False)
    if not isinstance(loaded_config_unresolved, dict):
        raise TypeError("loaded OpenWAM config must be a mapping")
    snapshot = deepcopy(loaded_config_unresolved)
    excluded_non_runtime_fields: list[dict[str, str]] = []
    project = snapshot.get("project")
    if isinstance(project, dict) and project.get("output_dir") == TRAINING_LOG_OUTPUT_TEMPLATE:
        raw_value = project.pop("output_dir")
        excluded_non_runtime_fields.append(
            {
                "path": "project.output_dir",
                "raw_value": str(raw_value),
                "reason": "training_log_template; Hydra now unavailable outside training",
            }
        )
    resolved_config = omega_conf.to_container(omega_conf.create(snapshot), resolve=True)
    return loaded_config_unresolved, resolved_config, excluded_non_runtime_fields


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.denoise_steps <= 0:
        raise ValueError("--denoise-steps must be positive")
    for field in ("ckpt_dir", "front_config", "code_root", "source_patch", "dataset", "stats"):
        path = getattr(args, field)
        if not path.exists():
            raise FileNotFoundError(f"--{field.replace('_', '-')} does not exist: {path}")
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.front_config = args.front_config.resolve()
    args.code_root = args.code_root.resolve()
    args.source_patch = args.source_patch.resolve()
    args.dataset = args.dataset.resolve()
    args.stats = args.stats.resolve()
    args.output = args.output.resolve()
    windows = requested_windows(args.episodes, args.starts)

    _, identity = load_config(args.front_config)
    if identity.action_mode not in ("eef", "delta"):
        raise ValueError(f"unsupported checkpoint action mode: {identity.action_mode}")
    adapter_args = argparse.Namespace(
        openwam_root=str(args.code_root),
        ckpt_dir=str(args.ckpt_dir),
        ckpt_name=args.ckpt_name,
        patch=args.source_patch,
        device=args.device,
        denoise_steps=args.denoise_steps,
        compile=args.compile,
        dit_cache=args.dit_cache,
    )
    _seed_torch()
    adapter = _official_adapter(identity, adapter_args)
    observed_generate_kwargs = _trace_architecture_generate(adapter)
    try:
        from omegaconf import OmegaConf

        loaded_cfg, resolved_cfg, excluded_non_runtime_fields = _metadata_config_snapshot(
            OmegaConf, adapter.policy_server.cfg
        )
    except ImportError as exc:
        raise RuntimeError("OmegaConf is required to record the resolved deploy config") from exc
    reader = _reader_from_path(
        args.dataset,
        code_root=args.code_root,
        stats=args.stats,
        action_mode=identity.action_mode,
        normalized=False,
    )
    by_episode = {int(row["episode_index"]): i for i, row in reader._eps_df.iterrows()}
    actions: list[np.ndarray] = []
    window_manifest: list[dict[str, Any]] = []
    for episode_index, start in windows:
        if episode_index not in by_episode:
            raise KeyError(f"episode_index {episode_index} is not in the reader population")
        item = reader_window(reader, episode_local=by_episode[episode_index], start=start)
        if len(item["video"]) != 9:
            raise ValueError(f"reader returned {len(item['video'])} frames, expected 9")
        current_canvas = item["video"][0].convert("RGB")
        current_canvas_bytes = current_canvas.tobytes()
        raw_state = np.asarray(item["state"], dtype=np.float32).reshape(EEF_DIM)
        adapter.reset()
        output = adapter.generate(image=current_canvas, state=raw_state.tolist(), prompt=PROMPT)
        array = np.asarray(output.actions, dtype=np.float32)
        if array.shape != (ACTION_STEPS, EEF_DIM) or not np.isfinite(array).all():
            raise ValueError(f"adapter returned invalid absolute EEF20 shape/values: {array.shape}")
        actions.append(array)
        window_manifest.append(
            {
                "episode_index": episode_index,
                "start": start,
                "global_index": item["global_index"],
                "conditioning_video_index": 0,
                "conditioning_frame_count_available": len(item["video"]),
                "current_canvas": {
                    "mode": current_canvas.mode,
                    "size": list(current_canvas.size),
                    "rgb_sha256": hashlib.sha256(current_canvas_bytes).hexdigest(),
                },
                "raw_state_float32_sha256": hashlib.sha256(raw_state.tobytes()).hexdigest(),
            }
        )

    actions_array = np.stack(actions, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        actions=actions_array,
        episode_index=np.asarray([item["episode_index"] for item in window_manifest], dtype=np.int64),
        start=np.asarray([item["start"] for item in window_manifest], dtype=np.int64),
    )
    manifest = {
        "predictions": str(args.output),
        "dataset": str(args.dataset),
        "prediction_semantics": "absolute_eef20",
        "action_mode": identity.action_mode,
        "action_shape": list(actions_array.shape),
        "prompt": PROMPT,
        "source": identity.source,
        "checkpoint": {"dir": str(args.ckpt_dir), "name": args.ckpt_name},
        "artifacts": {
            "front_config": _artifact(args.front_config),
            "reader_stats": _artifact(args.stats),
            "model_stats": _artifact(
                args.ckpt_dir / identity.stats,
                known_sha256=identity.stats_sha256,
            ),
            "model_config": _artifact(
                args.ckpt_dir / identity.config,
                known_sha256=identity.config_sha256,
            ),
            # _official_adapter has already verified this digest; reuse it so
            # the probe does not read a 24 GB foundation file a second time.
            "checkpoint": _artifact(args.ckpt_dir / args.ckpt_name, known_sha256=identity.checkpoint_sha256),
        },
        "inference": {
            "declared": {
                "seed": FIXED_SEED,
                "denoise_steps": args.denoise_steps,
                "compile": bool(args.compile),
                "dit_cache": bool(args.dit_cache),
                "device": args.device,
            },
            "observed": {
                "architecture_generate_kwargs": observed_generate_kwargs,
                "loaded_config_unresolved": loaded_cfg,
                "resolved_config": resolved_cfg,
                "excluded_non_runtime_fields": excluded_non_runtime_fields,
            },
        },
        "windows": window_manifest,
        "no_clip_or_filter": True,
        "no_extra_delta_addback": True,
        "conditioning": "reader window first current canvas and state only; future frames excluded",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
