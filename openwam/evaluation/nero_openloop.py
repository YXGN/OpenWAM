"""Nero episode replay: current observations in, full future action chunks out.

No alternative model loader, sampler, normalizer, or legacy robot adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from openwam.dataloader.utils.nero import (
    CAMERAS,
    CONTRACT,
    FK_COLUMNS,
    UNIFY_MAP,
    assemble_eef,
    dataset_fingerprint,
    episode_path,
    read_jsonl,
    relative_actions,
    sha256,
    validate_info,
)
from openwam.dataloader.utils.nero_absolute import CONTRACT as ABSOLUTE_CONTRACT
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import parse_unify_spec
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.deploy.obs_preprocess import ObsPreprocessor


def validate_contract(cfg, stats):
    """Allow only the two audited Nero training representations."""
    dl = cfg.dataloader
    expected = {"nero_multitask": "eef_relative", "nero_multitask_absolute": "eef"}
    if expected.get(dl.type) != dl.action_mode:
        raise ValueError("Expected a Nero reader with its matching EEF action_mode")
    fixed = {
        "num_frames": 33,
        "video_stride": 4,
        "window_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "require_all_cameras": True,
        "unify_action": True,
        "normalize_mode": "min-max",
    }
    for key, value in fixed.items():
        if dl.get(key) != value:
            raise ValueError(f"Nero contract mismatch: {key} must be {value!r}")
    if list(dl.camera_layout) != list(CAMERAS):
        raise ValueError("Nero camera order mismatch")
    for key in ("unify_action_map", "unify_state_map"):
        spec = dl.get(key)
        if spec is None and key == "unify_state_map":
            spec = dl.unify_action_map
        if spec is None or parse_unify_spec(spec).tolist() != UNIFY_MAP:
            raise ValueError(f"Invalid {key}")
    if dl.get("binary_action_dims") or dl.get("base_proprio", "velocity") != "velocity":
        raise ValueError("Nero evaluation does not use legacy command/base compatibility modes")
    if OmegaConf.select(dl, "color_jitter.enabled", default=False):
        raise ValueError("This evaluation contract requires unaugmented Nero training images")
    contract = CONTRACT if dl.action_mode == "eef_relative" else ABSOLUTE_CONTRACT
    if stats.get("metadata", {}).get("contract") != contract:
        raise ValueError("Checkpoint statistics representation contract mismatch")
    if dl.action_mode not in stats or f"{dl.action_mode}_state" not in stats:
        raise ValueError("Separate action and state statistics are required")


def window_starts(length, stride=32, starts=None):
    if stride < 1 or length < 33:
        raise ValueError("Positive stride and at least 33 episode frames are required")
    result = sorted(set(starts)) if starts is not None else sorted(set(range(0, length - 32, stride)) | {length - 33})
    if not result or any(t < 0 or t + 32 >= length for t in result):
        raise ValueError("Every anchor needs 32 real future frames within the episode")
    return result


def load_episode(root, info, episode):
    """Reuse Nero's FK/gripper assembly; never interpret source action16 as EEF."""
    columns = ["observation.state", *FK_COLUMNS, "task_index", "frame_index", "timestamp", "episode_index"]
    table = pq.read_table(episode_path(root, info, episode["episode_index"]), columns=columns).to_pandas()
    if len(table) != episode["length"] or not np.all(table.episode_index == episode["episode_index"]):
        raise ValueError("Episode metadata/parquet mismatch")
    np.testing.assert_array_equal(table.frame_index.to_numpy(), np.arange(len(table)))
    times = table.timestamp.to_numpy(dtype=np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / info["fps"], atol=1e-4):
        raise ValueError("Episode timestamps must be finite and follow the training frame rate")
    eef = assemble_eef(np.stack(table["observation.state"]), *(np.stack(table[c]) for c in FK_COLUMNS))
    return eef, table.task_index.to_numpy(), times


def current_observation(root, info, episode_id, t, state, prompt):
    # Decode ONLY t. Future RGB / cached future video latents never enter inference.
    roles = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    images = {
        role: decode_video_frames(str(episode_path(root, info, episode_id, camera)), [t], 360, 640)[0]
        for role, camera in zip(roles, CAMERAS, strict=True)
    }
    return {"images": images, "state": np.asarray(state, dtype=np.float32).copy(), "prompt": prompt}


def conditions_from_observation(preprocessor, observation, seed):
    obs = preprocessor.preprocess(dict(observation))
    return {
        "first_frame_image": [obs["image"]],
        "proprio": obs["state"],
        "prompt": obs["prompt"],
        "seed": seed,
        "num_frames": 33,
        "video_num_frames": 9,
        "height": 384,
        "width": 320,
    }


class PredictionRecorder:
    """Observe existing normalization once, without changing values or slot masks."""

    def __init__(self, inner):
        self.inner = inner
        self.normalized = None
        self.native = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def unnormalize(self, values):
        result = self.inner.unnormalize(values)
        self.normalized = np.array(values, copy=True)
        self.native = np.array(result, copy=True)
        self.calls += 1
        return result


def predict_chunk(engine, conditions):
    architecture = engine.architecture
    original = architecture.normalizer
    recorder = PredictionRecorder(original)
    architecture.attach_normalizer(recorder)
    try:
        result = engine.generate(conditions)
    finally:
        architecture.attach_normalizer(original)
    if recorder.calls != 1 or recorder.normalized.shape != (32, 80):
        raise ValueError("Expected one native generation of a normalized 32x80 action chunk")
    predicted = np.asarray(result["actions"], dtype=np.float32)
    if predicted.shape != (32, 20) or recorder.native.shape != (32, 20):
        raise ValueError("Expected 32x20 native and deployed EEF predictions")
    if not all(np.isfinite(x).all() for x in (predicted, recorder.native, recorder.normalized)):
        raise ValueError("Non-finite prediction; evaluation must not silently skip failed windows")
    return predicted, recorder.native, recorder.normalized[:, UNIFY_MAP]


def inference_config(steps):
    if steps < 1:
        raise ValueError("denoise_steps must be positive")
    return OmegaConf.create(
        {
            "inference": {"denoise_steps": steps, "denoise_mode": "sync", "cfg_scale": 1.0},
            "optimization": {
                "decode_video": False,
                "compile": {"enabled": False},
                "dit_cache": {"enabled": False},
                "prompt_embed_cache": {"enabled": True, "maxsize": 32},
            },
        }
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(args):
    from openwam.deploy.model_loader import _build_normalizer, load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.evaluation.nero_metrics import DIM_NAMES, evaluate
    from openwam.evaluation.nero_report import write_report

    root, ckpt_dir, output = args.dataset.resolve(), args.ckpt_dir.resolve(), args.output.resolve()
    if len(set(args.episodes)) != len(args.episodes) or args.seed < 0:
        raise ValueError("Episodes must be unique and seed must be nonnegative")
    checkpoint = ckpt_dir / args.ckpt_name
    if Path(args.ckpt_name).name != args.ckpt_name or checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
        raise ValueError("Specify an existing checkpoint filename within --ckpt-dir")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    training_cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    stats = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()
    validate_contract(training_cfg, stats)
    normalizer = _build_normalizer(training_cfg, str(ckpt_dir))
    info = json.loads((root / "meta/info.json").read_text())
    validate_info(root, info)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    fingerprint = dataset_fingerprint(root, info, episodes)
    if fingerprint != stats["metadata"].get("dataset_fingerprint"):
        raise ValueError("Dataset differs from checkpoint statistics population")
    tasks = {int(t["task_index"]): t["task"] for t in read_jsonl(root / "meta/tasks.jsonl")}
    by_id = {int(e["episode_index"]): e for e in episodes}
    plan = []
    mode = training_cfg.dataloader.action_mode
    for eid in args.episodes:
        if eid not in by_id:
            raise ValueError(f"Unknown episode {eid}")
        eef, task_ids, timestamps = load_episode(root, info, by_id[eid])
        for t in window_starts(len(eef), args.stride, args.starts):
            if not np.all(task_ids[t : t + 33] == task_ids[t]):
                raise ValueError("A prediction window crosses a task boundary")
            gt, state = eef[t + 1 : t + 33].copy(), eef[t].copy()
            native_gt = relative_actions(gt, state) if mode == "eef_relative" else gt.copy()
            norm_gt = apply_normalization(native_gt, stats[mode], "min-max")
            # Training and deployment statistics must invert targets at this window.
            unified = np.zeros((32, 80), dtype=np.float32)
            unified[:, UNIFY_MAP] = norm_gt
            np.testing.assert_allclose(normalizer.unnormalize(unified), native_gt, atol=1e-6)
            plan.append(
                {
                    "episode": eid,
                    "anchor": t,
                    "state": state,
                    "gt": gt,
                    "native_gt": native_gt,
                    "normalized_gt": norm_gt,
                    "prompt": tasks[int(task_ids[t])],
                    "task": int(task_ids[t]),
                    "times": timestamps[t + 1 : t + 33],
                }
            )
    if args.max_windows is not None:
        if args.max_windows < 1:
            raise ValueError("max_windows must be positive")
        plan = plan[: args.max_windows]
    if not plan:
        raise ValueError("No evaluation windows selected")
    cfg = merge_deploy_cfg(training_cfg, inference_config(args.denoise_steps))
    preprocessor = ObsPreprocessor.from_cfg(cfg)
    output.mkdir(parents=True)
    (output / "observations").mkdir()
    manifest = {
        "status": "preflight",
        "evaluation": "training_population_openloop_not_task_success",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(ckpt_dir / "config.yaml"),
        "stats_sha256": sha256(ckpt_dir / "normalization_stats.npy"),
        "dataset": str(root),
        "dataset_fingerprint": fingerprint,
        "action_mode": mode,
        "contract": stats["metadata"]["contract"],
        "seed": args.seed,
        "fps": info["fps"],
        "stride": args.stride,
        "denoise_steps": args.denoise_steps,
        "dimensions": DIM_NAMES,
        "metric_weighting": "window-horizon pairs (overlaps counted); episode/task summaries separate",
        "inference": OmegaConf.to_container(inference_config(args.denoise_steps)),
        "windows": [],
    }
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    manifest["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    manifest["source_sha256"] = {
        name: sha256(repo / name)
        for name in (
            "openwam/deploy/engine.py",
            "openwam/deploy/model_loader.py",
            "openwam/deploy/obs_preprocess.py",
            "openwam/model/architectures/base.py",
            "openwam/dataloader/utils/nero.py",
            "openwam/evaluation/nero_openloop.py",
            "openwam/evaluation/nero_metrics.py",
        )
    }
    write_json(output / "manifest.json", manifest)
    engine = None
    if not args.audit_only:
        from openwam.deploy.engine import JointInferenceEngine

        loaded_cfg, architecture = load_from_checkpoint_dir(str(ckpt_dir), device=args.device, ckpt_name=args.ckpt_name)
        engine = JointInferenceEngine(merge_deploy_cfg(loaded_cfg, inference_config(args.denoise_steps)), architecture)
        import torch

        manifest["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(architecture.device),
        }
    manifest["status"] = "running" if engine is not None else "auditing"
    predicted, native_predictions, normalized_predictions, latencies = [], [], [], []
    for i, item in enumerate(plan):
        observation = current_observation(root, info, item["episode"], item["anchor"], item["state"], item["prompt"])
        conditions = conditions_from_observation(preprocessor, observation, args.seed)
        canvas = conditions["first_frame_image"][0]
        image_name = f"observations/{i:05d}.png"
        canvas.save(output / image_name)
        entry = {k: item[k] for k in ("episode", "anchor", "task", "prompt")}
        entry.update(
            {
                "image": image_name,
                "canvas_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
                "state_sha256": hashlib.sha256(item["state"].tobytes()).hexdigest(),
            }
        )
        if engine is not None:
            import torch

            torch.cuda.synchronize(engine.architecture.device)
            begin = time.perf_counter()
            p, native, normalized = predict_chunk(engine, conditions)
            torch.cuda.synchronize(engine.architecture.device)
            latencies.append(time.perf_counter() - begin)
            predicted.append(p)
            native_predictions.append(native)
            normalized_predictions.append(normalized)
            entry["seconds"] = latencies[-1]
            if args.verify_parity and i == 0:
                direct = np.asarray(engine.generate(conditions)["actions"])
                np.testing.assert_array_equal(p, direct)
                manifest["recorder_parity"] = {"exact_equal": True, "max_abs": float(np.max(np.abs(p - direct)))}
            print(
                f"[{i + 1}/{len(plan)}] episode={item['episode']} t={item['anchor']} {latencies[-1]:.2f}s", flush=True
            )
        manifest["windows"].append(entry)
        write_json(output / "manifest.json", manifest)
    arrays = {key: np.stack([x[key] for x in plan]) for key in ("gt", "native_gt", "normalized_gt", "state", "times")}
    arrays.update(
        episode=np.array([x["episode"] for x in plan]),
        anchor=np.array([x["anchor"] for x in plan]),
        task=np.array([x["task"] for x in plan]),
    )
    if engine is not None:
        arrays.update(
            pred=np.stack(predicted),
            native_pred=np.stack(native_predictions),
            normalized_pred=np.stack(normalized_predictions),
            seconds=np.array(latencies),
        )
        metrics = evaluate(arrays)
        manifest["peak_allocated_gib"] = torch.cuda.max_memory_allocated(engine.architecture.device) / 2**30
        write_json(output / "metrics.json", metrics)
        with (output / "predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["episode", "anchor", "horizon", "frame", "timestamp_s", "dimension", "gt", "predict", "error"]
            )
            for w, item in enumerate(plan):
                for h in range(32):
                    for d, name in enumerate(DIM_NAMES):
                        gt, pred = float(arrays["gt"][w, h, d]), float(arrays["pred"][w, h, d])
                        writer.writerow(
                            [
                                item["episode"],
                                item["anchor"],
                                h + 1,
                                item["anchor"] + h + 1,
                                item["times"][h],
                                name,
                                gt,
                                pred,
                                pred - gt,
                            ]
                        )
    np.savez_compressed(output / ("audit.npz" if args.audit_only else "predictions.npz"), **arrays)
    manifest["status"] = "audit_complete" if args.audit_only else "complete"
    if engine is not None:
        write_report(output, arrays, manifest, metrics)
    write_json(output / "manifest.json", manifest)
    print(f"{manifest['status']}: {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--ckpt-name", required=True, help="Explicit saved checkpoint; never follows a moving latest")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--starts", nargs="+", type=int, help="Explicit anchors, applied to every selected episode")
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--max-windows", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--verify-parity",
        action="store_true",
        help="Repeat the first window without the recorder; require exact equality",
    )
    p.add_argument(
        "--audit-only", action="store_true", help="Validate data and render current observations without GPU inference"
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
