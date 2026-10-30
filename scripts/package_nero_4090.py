#!/usr/bin/env python3
"""Freeze the current working tree and one self-contained Nero checkpoint.

The output is a standalone rsync source. Large immutable weights are hardlinked
locally, while code/config/tokenizer files are copied. rsync transfers the
checkpoint as a regular file; no server-side symlinks or absolute paths remain.
Never edit the checkpoint inside the bundle (it shares the source inode).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("outputs/nero_atom_4gpu/2026-10-04_15-00-03")
WEIGHT = "checkpoint_step_6248.safetensors"
HANDOFF = Path("migration/nero_4090")
HISTORICAL = Path("outputs/nero_atom_deploy_8017/20261008_101526/smoke")
PACKAGES = ("torch", "torchvision", "numpy", "transformers", "diffusers", "safetensors", "Pillow", "deepspeed")
OPTIONAL_ATTENTION = ("flash_attn", "flash_attn_interface", "sageattention", "xformers")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_file(name, output, destination=None):
    target = output / (destination or name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)


def make_bundle(output):
    output.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".cache", ".pytest_cache", ".ruff_cache")
    for name in ("openwam", "configs", "envs", "tests"):
        shutil.copytree(ROOT / name, output / name, ignore=ignore)
    for name in ("pyproject.toml", "LICENSE", "scripts/setup_openwam_env.sh", "scripts/activate_openwam.sh",
                 "scripts/serve_openwam_atom_8017.sh", "scripts/serve_openwam_atom_4090.sh",
                 "scripts/check_nero_4090.py", "scripts/package_nero_4090.py",
                 "assets/openwam_usage_docs/nero-client-protocol.md"):
        copy_file(name, output)
    shutil.copytree(ROOT / HANDOFF, output / HANDOFF, ignore=ignore)
    copy_file(HANDOFF / "AGENT_HANDOFF.md", output, "AGENTS.md")
    (output / "README.md").write_text(
        "# Nero 4090 inference handoff\n\n"
        "这是服务器当前源码的独立快照，包含未提交修改和 Atom 6248 完整模型。\n\n"
        "先阅读 [4090 部署步骤](migration/nero_4090/README.md) 和 [agent 交接](AGENTS.md)。\n\n"
        "```bash\npython3 scripts/check_nero_4090.py files\n"
        "bash scripts/setup_openwam_env.sh\n"
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_nero_4090.py replay\n"
        "# 只有回放通过后再执行下一条；每个命令分别检查退出码。\n"
        "bash scripts/serve_openwam_atom_4090.sh\n```\n\n"
        "模型、归一化、PNG / EEF20 / Rot6D / 米制夹爪协议与服务器配套；"
        "4090 使用 T5 CPU 和关闭 CUDA Graph 的部署入口。\n"
        "4090 的实际显存、时延与数值回放由本地 agent 验证，结果写入 "
        "`outputs/nero_4090_validation/`。\n",
        encoding="utf-8",
    )
    (output / MODEL).mkdir(parents=True)
    for name in ("config.yaml", "normalization_stats.npy"):
        copy_file(MODEL / name, output)
    shutil.copytree(ROOT / MODEL / "tokenizer", output / MODEL / "tokenizer")
    os.link(ROOT / MODEL / WEIGHT, output / MODEL / WEIGHT)

    # Preserve raw historical artifacts. Today's output mapping is applied by
    # the replay checker, not by rewriting the archived prediction arrays.
    fixture = output / HANDOFF / "recorded_reference"
    fixture.mkdir()
    old = json.loads((ROOT / HISTORICAL / "result.json").read_text())
    for path in sorted((ROOT / HISTORICAL).iterdir()):
        if path.suffix in (".png", ".npz"):
            shutil.copy2(path, fixture / path.name)
    shutil.copy2(ROOT / HISTORICAL / "result.json", fixture / "source_result.json")
    cases = []
    roles = dict(head_camera="head_fpv", left_wrist_camera="left_hand", right_wrist_camera="right_hand")
    for i, window in enumerate(old["windows"]):
        images = {role: f"ep010_t{window['anchor']}_{original}.png" for role, original in roles.items()}
        for role, original in roles.items():
            if sha256(fixture / images[role]) != window["png_sha256"][original]:
                raise ValueError("Recorded PNG does not match the original smoke result")
        cases.append({"name": f"ep010_t{window['anchor']}_request{i}", "prompt": window["prompt"],
                      "state_key": f"state_{i}", "actions_key": f"actions_{i}", "images": images})
    # Generate the existing protocol schema, without importing the robot client.
    from openwam.deploy.nero_chunk_server import CONTRACT, WIRE, validate_schema

    names = ["left_arm", "right_arm"]
    schema = {
        "version": "velabot.io/v2", "action_dim": 20, "include_base": False,
        "layout": [x for name in names for x in [*[f"{name}.arm.{i}" for i in range(9)],
                                                f"{name}.effector.position.opening_width"]],
        "groups": [{"group": name, "arm_format": "rot6d", "arm_dim": 9, "effector_dim": 1,
                    "effector_protocol": "agilex.parallel_gripper.width_m/v1", "effector_names": ["opening_width"],
                    "effector_position_unit": "width_m", "effector_position_limits": [0.0, 0.1],
                    "effector_position_velocity_max": []} for name in names],
    }
    validate_schema(schema)
    provenance = {
        "source": str(HISTORICAL), "recorded_at": old["finished_at"],
        "checkpoint_identity": old["identity"], "denoise_steps": 4, "seed": 42,
        "original_gripper_mapping": "none", "current_gripper_mapping": "clip(raw - 0.002, 0, 0.1)",
        "scope": "Historical A100 two-observation regression (first input repeated), not a new live-server capture.",
    }
    (fixture / "cases.json").write_text(json.dumps({"provenance": provenance, "io_schema": schema, "cases": cases}, indent=2) + "\n")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Bundle contains a symlink: {path}")
        if path.is_file() and path != output / HANDOFF / "manifest.json":
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for field, name in (("checkpoint", WEIGHT), ("config", "config.yaml"), ("stats", "normalization_stats.npy")):
        if files[(MODEL / name).as_posix()]["sha256"] != old["identity"][field + "_sha256"]:
            raise ValueError(f"Current {field} no longer matches the reference")
    extras = [name for name in OPTIONAL_ATTENTION if importlib.util.find_spec(name) is not None]
    if extras:
        raise ValueError(f"Reference environment changed optional attention packages: {extras}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_kind": "working-tree snapshot including uncommitted edits; not a git checkout",
        "checkpoint_dir": str(MODEL), "identity": old["identity"], "wire": WIRE, "contract": CONTRACT,
        "environment": {"python": platform.python_version(),
                        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                        "absent_optional_attention": list(OPTIONAL_ATTENTION)},
        "deployment": {"denoise_steps": 4, "seed": 42, "gripper_close_offset_m": .002, "gripper_threshold_m": None,
                       "text_encoder_cpu": True, "cuda_graph": False, "inference_fast_path": True,
                       "request_context_cache": True, "port": 8017},
        "parity_limits": {"position_max_mm": 2.0, "rotation_max_deg": 1.0, "gripper_max_mm": 1.0},
        "parity_scope": "Conservative migration regression gates, not robot success/safety tolerances or bitwise equivalence.",
        "validation_4090": "pending on target hardware", "files": files,
    }
    (output / HANDOFF / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"bundle": str(output), "files": len(files), "bytes": sum(x["bytes"] for x in files.values()),
                      "checkpoint_hardlinked_on_source": True, "target_gpu_validation": "pending"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing bundles are never overwritten")
    make_bundle(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
