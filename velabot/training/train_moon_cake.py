#!/usr/bin/env python3
"""Thin, explicit MoonCake training launcher for the pinned OpenWAM checkout.

``--dry-run`` only validates and prints the expanded contract. It does not
start training or load model weights. It runs the official Hydra entry point in
configuration-only mode in a short validation process and does not touch the
output directory.
The actual trainer remains the official ``scripts/train.py`` entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from moon_cake import FPS, UNIFY_DIM, VIDEO_STRIDE, WINDOW_FRAMES

VELABOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VELABOT_ROOT))

from source_identity import fingerprint_source  # noqa: E402

PATCH_PATH = Path(__file__).resolve().parents[1] / "patches" / "openwam-moon-cake-training.patch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, object]:
    """Record a file or foundation directory without copying large weights."""

    if path.is_file():
        size = path.stat().st_size
        return {
            "path": str(path.resolve()),
            "kind": "file",
            "bytes": size,
            # Avoid making a dry-run read a multi-GB foundation weight. Small
            # stats/config artifacts remain content-addressed in the manifest.
            "sha256": _sha256(path) if size <= 16 * 1024 * 1024 else None,
            "hash_skipped": "file_over_16MiB" if size > 16 * 1024 * 1024 else None,
        }
    if path.is_dir():
        files = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            files.append({"path": child.relative_to(path).as_posix(), "bytes": child.stat().st_size})
        return {"path": str(path.resolve()), "kind": "directory", "files": files}
    raise FileNotFoundError(path)


def source_identity(code_root: Path) -> dict[str, object]:
    """Record the applied checkout identity without importing model code."""
    tracked = (
        "configs/train.yaml",
        "configs/dataloader/moon_cake.yaml",
        "openwam/dataloader/moon_cake.py",
        "openwam/dataloader/moon_cake_safety.py",
        "openwam/dataloader/registry.py",
        "openwam/deploy/delta_addback.py",
        "openwam/deploy/engine.py",
        "openwam/train/openwam_trainer.py",
        "openwam/train/utils/bounded_optimizer_groups.py",
    )
    return fingerprint_source(code_root, PATCH_PATH, tracked)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--foundation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1")
    parser.add_argument("--code-root", required=True, type=Path, help="pinned OpenWAM checkout")
    parser.add_argument("--action-mode", choices=("eef", "delta"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2000,
        help="checkpoint interval; smoke runs may pass max_steps+1",
    )
    parser.add_argument("--keep-last-k-ckpts", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--zero-stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def expanded_config(args: argparse.Namespace) -> dict[str, object]:
    source = source_identity(args.code_root)
    return {
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "dataset": str(args.dataset.resolve()),
        "stats": str(args.stats.resolve()),
        "foundation": str(args.foundation.resolve()),
        "output": str(args.output.resolve()),
        "gpu": args.gpu,
        "action_mode": args.action_mode,
        "delta_definition": (
            "XYZ relative to window-start state; rot6d/gripper absolute" if args.action_mode == "delta" else None
        ),
        "fps": FPS,
        "window_frames": WINDOW_FRAMES,
        "video_stride": VIDEO_STRIDE,
        "unified_action_dim": UNIFY_DIM,
        "gradient_accumulation_steps": 1,
        "resume_supported": False,
        "mixed_precision": args.mixed_precision,
        "zero_stage": args.zero_stage,
        "save_steps": args.save_steps,
        "save_full_states_for_resume": False,
        "keep_last_k_ckpts": args.keep_last_k_ckpts,
        "debug": False,
        "num_epochs": None,
        "resume_ckpt_path": None,
        "dataset_num_workers": 0,
        "use_gradient_checkpointing": True,
        "initialize_model_on_cpu": True,
        "offload_optimizer_device": "none",
        "wandb": None,
        "learning_rate": 1e-5,
        "action_lr": 5e-5,
        "video_lr": 1e-5,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "source_identity": source,
        "stats_artifact": artifact_identity(args.stats),
        "foundation_artifact": artifact_identity(args.foundation),
    }


def build_overrides(args: argparse.Namespace) -> list[str]:
    """Build only the Hydra overrides shared by dry-run and training."""
    return [
        "dataloader=moon_cake",
        f"dataloader.dataset_dir={args.dataset}",
        f"dataloader.normalization_stats_path={args.stats}",
        f"dataloader.action_mode={args.action_mode}",
        "dataloader.normalize_mode=min-max",
        "dataloader.unify_action=true",
        "dataloader.num_frames=33",
        "dataloader.video_stride=4",
        "dataloader.window_stride=1",
        "dataloader.height=384",
        "dataloader.width=320",
        "dataloader.multiview=true",
        f"training.finetune_ckpt_path={args.foundation}",
        f"training.output_path={args.output}",
        "training.gradient_accumulation_steps=1",
        "training.bound_optimizer_groups=true",
        f"training.save_steps={args.save_steps}",
        "training.save_full_states_for_resume=false",
        f"training.keep_last_k_ckpts={args.keep_last_k_ckpts}",
        f"training.mixed_precision={args.mixed_precision}",
        f"training.zero_stage={args.zero_stage}",
        "training.debug=false",
        "training.num_epochs=null",
        "training.resume_ckpt_path=null",
        "training.dataset_num_workers=0",
        "training.use_gradient_checkpointing=true",
        "training.initialize_model_on_cpu=true",
        "training.offload_optimizer_device=none",
        "project.wandb=null",
        "training.learning_rate=1e-5",
        "training.action_lr=5e-5",
        "training.video_lr=1e-5",
        "training.lr_scheduler=cosine",
        "training.warmup_ratio=0.05",
        f"training.batch_size={args.batch_size}",
        f"project.seed={args.seed}",
        f"training.max_steps={args.max_steps}",
    ]


def build_command(args: argparse.Namespace) -> list[str]:
    script = args.code_root / "scripts" / "train.py"
    gpu_ids = [item.strip() for item in str(args.gpu).split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("--gpu must be a comma-separated list of unique numeric CUDA device ids")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(script),
        *build_overrides(args),
    ]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_steps <= 0:
        raise ValueError("--batch-size and --max-steps must be positive")
    if args.save_steps <= 0 or args.keep_last_k_ckpts <= 0:
        raise ValueError("--save-steps and --keep-last-k-ckpts must be positive")
    for name, path in (
        ("dataset", args.dataset),
        ("stats", args.stats),
        ("foundation", args.foundation),
        ("code-root", args.code_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    # The official train process changes cwd to its checkout for Hydra. Resolve
    # every user path before constructing overrides so relative CLI values keep
    # referring to the caller's workspace.
    args.dataset = args.dataset.resolve()
    args.stats = args.stats.resolve()
    args.foundation = args.foundation.resolve()
    args.output = args.output.resolve()
    args.code_root = args.code_root.resolve()
    if not (args.code_root / "scripts" / "train.py").is_file():
        raise FileNotFoundError(f"official train entry point missing under {args.code_root}")
    config = expanded_config(args)
    command = build_command(args)
    print(json.dumps({"config": config, "command": command}, indent=2, sort_keys=True))
    if args.dry_run:
        # Hydra's ``--cfg job`` mode resolves the real train config and exits
        # before _train is called. This validation process is never launched
        # through torchrun and does not construct a model or touch output.
        hydra_command = [
            sys.executable,
            str(args.code_root / "scripts" / "train.py"),
            *build_overrides(args),
            "--cfg",
            "job",
            "--resolve",
        ]
        result = subprocess.run(hydra_command, cwd=args.code_root, env=os.environ.copy(), check=False)
        if result.returncode:
            return result.returncode
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "status": "started",
        "config": config,
        "command": command,
        "cuda_visible_devices": args.gpu,
        "official_checkout": str(args.code_root.resolve()),
        "python_executable": sys.executable,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # Do not swallow the trainer's exit status or append unparsed arguments.
    returncode = subprocess.run(command, cwd=args.code_root, env=env, check=False).returncode
    manifest["status"] = "completed" if returncode == 0 else "failed"
    manifest["returncode"] = returncode
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
