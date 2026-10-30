#!/usr/bin/env python3
"""Verify a frozen Nero migration bundle, then replay recorded inputs offline.

`files` uses only the standard library. `replay` loads the model locally and
calls the same Nero service boundary; it never connects to a robot or 8017.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "migration/nero_4090/manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_files(root, manifest):
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"missing: {name}")
        elif path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            failures.append(f"changed: {name}")
    # An old checkout may contain retired importable modules after rsync.
    for path in (root / "openwam").rglob("*.py"):
        if path.relative_to(root).as_posix() not in manifest["files"]:
            failures.append(f"extra Python source: {path.relative_to(root)}")
    if failures:
        raise ValueError("Bundle verification failed:\n" + "\n".join(failures))
    return {"status": "passed", "checked_files": len(manifest["files"])}


def verify_environment(expected):
    versions = {name: importlib.metadata.version(name) for name in expected["packages"]}
    actual = {"python": platform.python_version(), "packages": versions}
    if actual != {k: expected[k] for k in ("python", "packages")}:
        raise ValueError(f"Environment differs from the reference: {actual}")
    extras = [name for name in expected["absent_optional_attention"] if importlib.util.find_spec(name) is not None]
    if extras:
        raise ValueError(f"Unexpected attention packages may change kernel selection: {extras}")
    return actual


def compare_actions(actual, expected, limits):
    import numpy as np

    from openwam.deploy.nero_chunk_server import finite_eef
    from openwam.evaluation.nero_metrics import rotation_errors

    actual = finite_eef(actual, (32, 20))
    expected = finite_eef(expected, (32, 20))
    delta = actual.astype(np.float64) - expected
    result = {"passed": True, "arms": {}}
    for name, start in (("left", 0), ("right", 10)):
        angles, valid = rotation_errors(actual[:, start + 3:start + 9], expected[:, start + 3:start + 9])
        metrics = {
            "position_max_mm": float(np.linalg.norm(delta[:, start:start + 3], axis=-1).max() * 1000),
            "rotation_max_deg": float(angles.max()),
            "gripper_max_mm": float(np.abs(delta[:, start + 9]).max() * 1000),
        }
        passed = bool(valid.all() and all(metrics[key] <= limit for key, limit in limits.items()))
        result["arms"][name] = {**metrics, "passed": passed}
        result["passed"] &= passed
    return result


def replay(root, manifest, device):
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch

    from openwam.deploy.nero_chunk_server import CONTRACT, WIRE, load_service

    environment = verify_environment(manifest["environment"])
    if WIRE != manifest["wire"] or CONTRACT != manifest["contract"]:
        raise ValueError("Wire contract differs from the source snapshot")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Replay requires a CUDA GPU with BF16 support")
    fixture = root / "migration/nero_4090/recorded_reference"
    cases = json.loads((fixture / "cases.json").read_text())
    arrays = np.load(fixture / "predictions.npz", allow_pickle=False)
    torch.cuda.reset_peak_memory_stats(target)
    started = time.monotonic()
    service = load_service(
        root / manifest["checkpoint_dir"], manifest["identity"]["checkpoint"], str(target),
        config_path=root / "configs/deploy_nero_atom_4090.yaml",
        gripper_close_offset_m=0.002, text_encoder_cpu=True,
    )
    loaded_seconds = time.monotonic() - started
    if service.identity != manifest["identity"]:
        raise ValueError("Loaded checkpoint/config/statistics identity differs from the source server")
    cfg = service.cfg
    if (cfg.inference.denoise_steps != 4 or cfg.inference.denoise_mode != "sync"
            or cfg.inference.cfg_scale != 1 or cfg.optimization.cuda_graph.enabled
            or not cfg.optimization.inference_fast_path.enabled
            or not cfg.optimization.request_context_cache.enabled):
        raise ValueError("Effective inference settings differ from the migration recipe")
    backbone = service.engine.architecture.video_backbone
    if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in backbone.text_encoder.parameters()):
        raise ValueError("T5 is not entirely on CPU in BF16")
    for name, param in service.engine.architecture.named_parameters():
        if not name.startswith("video_backbone.text_encoder.") and param.device.type != "cuda":
            raise ValueError(f"Unexpected CPU model parameter: {name}")
    handshake = service.request(
        "/handshake", {"wire_format": WIRE, "contract": CONTRACT, "io_schema": cases["io_schema"]},
    )
    if handshake["wire_format"] != "openwam/v1" or handshake.get("gripper_close_offset_m") != 0.002:
        raise ValueError("Handshake protocol/gripper settings differ")
    session = handshake["session_id"]
    records, outputs = [], []
    for index, case in enumerate(cases["cases"]):
        state = arrays[case["state_key"]]
        started = time.monotonic()
        reply = service.request("/infer", {
            "wire_format": WIRE, "session_id": session, "request_id": f"migration-replay-{index}",
            "state": state.tolist(), "prompt": case["prompt"],
            "images": {role: base64.b64encode((fixture / file).read_bytes()).decode("ascii")
                       for role, file in case["images"].items()},
        })
        elapsed = time.monotonic() - started
        if reply["identity"] != manifest["identity"] or reply["dt_sec"] != 1 / 60:
            raise ValueError("Inference identity/time contract differs")
        np.testing.assert_array_equal(np.asarray(reply["anchor_state"], dtype=np.float32), state)
        actual = np.asarray(reply["actions"], dtype=np.float32)
        raw_widths = np.asarray(reply["raw_gripper_widths_m"], dtype=np.float32)
        np.testing.assert_array_equal(actual[:, [9, 19]], np.clip(raw_widths - np.float32(.002), 0, .1))
        # The archived reference predates the 2 mm mapping. Preserve it as
        # raw evidence and apply today's EXACT output mapping once here.
        expected = arrays[case["actions_key"]].copy()
        expected[:, [9, 19]] = np.clip(expected[:, [9, 19]] - np.float32(.002), 0, .1)
        comparison = compare_actions(actual, expected, manifest["parity_limits"])
        outputs.append(actual)
        records.append({"case": case["name"], "seconds": elapsed, **comparison})
    repeat_max_abs = float(np.max(np.abs(outputs[0] - outputs[1])))
    passed = all(row["passed"] for row in records) and repeat_max_abs <= 1e-6
    free, total = torch.cuda.mem_get_info(target)
    return {
        "status": "passed" if passed else "failed",
        "scope": "Recorded-input numerical regression; not a closed-loop robot success or generalization test.",
        "reference": cases["provenance"], "identity": service.identity,
        "environment": environment, "gpu": torch.cuda.get_device_name(target),
        "text_encoder_device": "cpu", "cuda_graph": False, "load_seconds": loaded_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(target) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(target) / 2**20,
        "gpu_total_mib": total / 2**20, "gpu_free_mib_after_replay": free / 2**20,
        "cpu_threads": torch.get_num_threads(),
        "parity_limits": manifest["parity_limits"], "repeat_max_abs": repeat_max_abs,
        "cases": records,
    }, np.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("files", "replay"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/nero_4090_validation/report.json")
    args = parser.parse_args()
    report = {"status": "failed", "mode": args.mode}
    try:
        manifest = json.loads(MANIFEST.read_text())
        report["files"] = verify_files(ROOT, manifest)
        if args.mode == "files":
            report["status"] = "passed"
        else:
            result, outputs = replay(ROOT, manifest, args.device)
            report.update(result)
            import numpy as np

            args.report.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.report.with_suffix(".npz"), actions=outputs)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    if args.mode == "replay":
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
