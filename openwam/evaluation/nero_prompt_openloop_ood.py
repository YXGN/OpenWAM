"""Isolated population-shift evaluator with audited lossy min-max references.

The frozen main evaluator assumes physical GT is inside training extrema.
This module preserves its production inference path and helpers, and keeps a
versioned copy of run/main solely to replace that preflight assumption. It does
not patch global functions, normalize differently, clip physical GT/state, or
modify the main runner's bytes. The copied flow is an explicit maintenance
tradeoff while existing main-panel jobs/resume manifests remain frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.utils.nero import (
    UNIFY_MAP,
    dataset_fingerprint,
    read_jsonl,
    relative_actions,
    sha256,
    validate_info,
)
from openwam.dataloader.utils.normalization import NORM_EPS, apply_normalization
from openwam.deploy.nero_contract import (
    DEFAULT_INFERENCE_CONFIG,
    conditions_from_observation,
    inference_config,
    validate_contract,
)
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.evaluation import nero_prompt_openloop as frozen
from openwam.evaluation.nero_metrics import DIM_NAMES
from openwam.evaluation.nero_openloop import current_observation, load_episode, window_starts
from openwam.evaluation.nero_prompt_openloop import (
    REPO,
    assert_manifest_compatible,
    atomic_json,
    atomic_npz,
    checkpoint_identity,
    digest,
    future_reference,
    input_identity,
    population_identity,
    predict_joint,
    read_cases,
    save_frames,
    score_case,
    utc_now,
    verify_completed,
    write_report,
)

FROZEN_MAIN_SHA256 = "31f8af37dfa897ccd697f0bdf7a79bae4816cb9e0ba3ce8d35009ac41ffeb0da"
SOURCE_FILES = (*frozen.SOURCE_FILES, "openwam/evaluation/nero_prompt_openloop_ood.py")
REFERENCE_POLICY = (
    "bounded_minmax_clipped_inverse_v1; retain raw physical GT/state; production state clipping unchanged"
)


def clipping_diagnostics(raw, block):
    raw = np.asarray(raw, dtype=np.float32)
    lo, hi = np.asarray(block["min"], dtype=np.float32), np.asarray(block["max"], dtype=np.float32)
    if raw.shape[-1] != 20 or lo.shape != (20,) or hi.shape != (20,) or not np.isfinite(raw).all():
        raise ValueError("Expected finite EEF20 and 20-dimensional training extrema")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError("Invalid training normalization extrema")
    scale = np.maximum(hi - lo, NORM_EPS)
    unbounded = (raw - lo) / scale * 2.0 - 1.0
    clipped = apply_normalization(raw, block, "min-max")
    # The inverse has the epsilon-floored operational range [lo, lo+scale].
    expected_inverse = lo + (clipped + 1.0) * 0.5 * scale
    mask = np.abs(unbounded) > 1.0
    losses = raw - expected_inverse
    flat_raw, flat_mask, flat_losses = raw.reshape(-1, 20), mask.reshape(-1, 20), losses.reshape(-1, 20)
    dimensions = {}
    for index, name in enumerate(DIM_NAMES):
        loss = np.abs(flat_losses[:, index])
        entry = {
            "clipped_count": int(flat_mask[:, index].sum()),
            "below_training_min_count": int((flat_raw[:, index] < lo[index]).sum()),
            "above_training_max_count": int((flat_raw[:, index] > hi[index]).sum()),
            "max_abs_clip_loss_native_units": float(loss.max()),
            "mean_abs_clip_loss_native_units": float(loss.mean()),
        }
        if index % 10 in (0, 1, 2, 9):
            entry["max_abs_clip_loss_mm"] = float(loss.max() * 1000)
        else:
            entry["unit"] = "rot6d component (not angular degrees)"
        dimensions[name] = entry
    report = {
        "total_elements": int(raw.size),
        "clipped_elements": int(mask.sum()),
        "clipped_fraction": float(mask.mean()),
        "clipped_dimensions": [name for i, name in enumerate(DIM_NAMES) if flat_mask[:, i].any()],
        "per_dimension": dimensions,
    }
    return unbounded, clipped, expected_inverse, mask, losses, report


def bounded_reference_audit(normalizer, native_gt, state, stats, mode):
    """Verify production clipping and keep the raw physical references unchanged."""
    original_gt, original_state = np.array(native_gt, copy=True), np.array(state, copy=True)
    au, ac, inverse, amask, aloss, areport = clipping_diagnostics(native_gt, stats[mode])
    su, sc, state_inverse, smask, sloss, sreport = clipping_diagnostics(state, stats[f"{mode}_state"])
    unified = np.zeros((32, 80), dtype=np.float32)
    unified[:, UNIFY_MAP] = ac
    actual_inverse = normalizer.unnormalize(unified)
    np.testing.assert_allclose(actual_inverse, inverse, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_inverse[~amask], original_gt[~amask], atol=1e-6, rtol=1e-6)
    expected_state = np.zeros(80, dtype=np.float32)
    expected_state[UNIFY_MAP] = sc
    np.testing.assert_allclose(normalizer.normalize(state), expected_state, atol=3e-7, rtol=1e-6)
    np.testing.assert_array_equal(native_gt, original_gt)
    np.testing.assert_array_equal(state, original_state)
    arrays = {
        "normalized_gt_unclipped": au,
        "clipped_native_gt": inverse,
        "action_clipping_mask": amask,
        "action_clipping_loss_native_units": aloss,
        "normalized_state_unclipped": su,
        "normalized_state_clipped": sc,
        "state_clipping_mask": smask,
        "state_clipping_loss_native_units": sloss,
        "clipped_state_reference_only": state_inverse,
    }
    report = {
        "policy": REFERENCE_POLICY,
        "action_reference": areport,
        "state_input": sreport,
        "physical_gt_scoring": "Original unmodified gt/native_gt; clipped arrays are diagnostics only",
        "state_conditioning": "Original raw state enters production normalizer; clipping behavior unchanged",
    }
    return ac, arrays, report


def run(args):
    from openwam.deploy.model_loader import _build_normalizer, load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg

    if sha256(Path(frozen.__file__)) != FROZEN_MAIN_SHA256:
        raise ValueError("Frozen main helper source changed; review OOD flow before updating its pinned dependency")
    root, ckpt_dir, output = args.dataset.resolve(), args.ckpt_dir.resolve(), args.output.resolve()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    if args.max_generations is not None and args.max_generations < 1:
        raise ValueError("max-generations must be positive")
    checkpoint = ckpt_dir / args.ckpt_name
    if Path(args.ckpt_name).name != args.ckpt_name or checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
        raise ValueError("Use an explicit existing safetensors filename in --ckpt-dir")
    previous = None
    if output.exists():
        if not args.resume or not (output / "manifest.json").is_file():
            raise FileExistsError(f"Existing output requires --resume and its intact manifest: {output}")
        previous = json.loads((output / "manifest.json").read_text())
    cases, skipped = read_cases(args.cases)
    training_cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    stats = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()
    validate_contract(training_cfg, stats)
    normalizer = _build_normalizer(training_cfg, str(ckpt_dir))
    deploy_cfg = inference_config(args.denoise_steps, config_path=args.config)
    deploy_cfg.optimization.decode_video = True
    cfg = merge_deploy_cfg(training_cfg, deploy_cfg)
    preprocessor = ObsPreprocessor.from_cfg(cfg)
    info = json.loads((root / "meta/info.json").read_text())
    validate_info(root, info)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    population = population_identity(
        stats, dataset_fingerprint(root, info, episodes), args.allow_evaluation_population_shift
    )
    by_id = {int(e["episode_index"]): e for e in episodes}
    tasks = {int(t["task_index"]): t["task"] for t in read_jsonl(root / "meta/tasks.jsonl")}
    identity = {
        "schema_version": 1,
        "reference_normalization_policy": REFERENCE_POLICY,
        "frozen_main_sha256": FROZEN_MAIN_SHA256,
        "checkpoint": checkpoint_identity(checkpoint, previous["identity"]["checkpoint"] if previous else None),
        "config_sha256": sha256(ckpt_dir / "config.yaml"),
        "stats_sha256": sha256(ckpt_dir / "normalization_stats.npy"),
        "dataset": str(root),
        **population,
        "cases_sha256": sha256(args.cases),
        "cases": cases,
        "seeds": args.seeds,
        "inference": OmegaConf.to_container(deploy_cfg, resolve=True),
        "inference_config_sha256": sha256(args.config),
        "action_mode": training_cfg.dataloader.action_mode,
        "contract": stats["metadata"]["contract"],
        "source_sha256": {name: sha256(REPO / name) for name in SOURCE_FILES},
        "tokenizer_sha256": {
            str(p.relative_to(ckpt_dir)): sha256(p) for p in sorted((ckpt_dir / "tokenizer").rglob("*")) if p.is_file()
        },
    }
    if not identity["tokenizer_sha256"]:
        raise ValueError(
            "Checkpoint must include its tokenizer files under tokenizer/; empty provenance is not allowed"
        )
    if previous:
        assert_manifest_compatible(previous, identity)
    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir(exist_ok=True)
    manifest = previous or {
        "identity": identity,
        "identity_sha256": digest(identity),
        "created_at_utc": utc_now(),
        "invocations": [],
        "skipped_rows": skipped,
        "scope": "Open-loop 0.533 second joint predictions; semantic accuracy needs review; no robot success",
    }
    manifest["status"] = "auditing" if args.audit_only else "preflight"
    manifest.pop("error", None)
    manifest.pop("traceback", None)
    manifest["invocations"].append(
        {
            "started_at_utc": utc_now(),
            "argv": sys.argv,
            "device": args.device,
            "audit_only": args.audit_only,
            "max_generations": args.max_generations,
            "retry_failed": args.retry_failed,
        }
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    manifest["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    atomic_json(output / "manifest.json", manifest)
    contexts, loaded = {}, {}
    for case in cases:
        eid, t = case["episode_id"], case["anchor_frame"]
        if eid not in by_id:
            raise ValueError(f"Unknown episode {eid}")
        if eid not in loaded:
            loaded[eid] = load_episode(root, info, by_id[eid])
        eef, task_ids, timestamps = loaded[eid]
        window_starts(len(eef), starts=[t])
        if not np.all(task_ids[t : t + 33] == task_ids[t]):
            raise ValueError(f"Window crosses a task boundary: episode {eid}, t={t}")
        original_prompt = tasks[int(task_ids[t])]
        if case["prompt_class"] == "original" and case["prompt"] != original_prompt:
            raise ValueError(f"Original prompt differs from dataset metadata in {case['group_id']}")
        key = (eid, t)
        if key in contexts:
            continue
        state, gt = eef[t].copy(), eef[t + 1 : t + 33].copy()
        native_gt = relative_actions(gt, state) if identity["action_mode"] == "eef_relative" else gt.copy()
        norm_gt, clipping_arrays, clipping_report = bounded_reference_audit(
            normalizer, native_gt, state, stats, identity["action_mode"]
        )
        observation = current_observation(root, info, eid, t, state, original_prompt)
        conditions = conditions_from_observation(preprocessor, observation, args.seeds[0])
        input_hashes = input_identity(observation, conditions)
        anchor_dir = output / "observations" / f"episode_{eid:06d}_t{t:06d}"
        reference = {
            "state": state,
            "gt": gt,
            "native_gt": native_gt,
            "normalized_gt": norm_gt,
            "times": timestamps[t + 1 : t + 33],
            **clipping_arrays,
        }
        metadata = {
            "episode": eid,
            "anchor": t,
            "task": int(task_ids[t]),
            "original_prompt": original_prompt,
            "normalization_audit": clipping_report,
            **input_hashes,
        }
        if (anchor_dir / "reference.json").exists():
            saved = json.loads((anchor_dir / "reference.json").read_text())
            if saved["metadata"] != metadata:
                raise ValueError(f"Observation changed since preflight: {anchor_dir}")
            for name, expected in saved["artifact_sha256"].items():
                if sha256(anchor_dir / name) != expected:
                    raise ValueError(f"Reference artifact changed: {anchor_dir / name}")
            with np.load(anchor_dir / "reference.npz") as arrays:
                reference["video"] = arrays["video"].copy()
        else:
            anchor_dir.mkdir(exist_ok=True)
            reference["video"] = future_reference(root, info, eid, t, preprocessor)
            np.testing.assert_array_equal(reference["video"][0], np.asarray(conditions["first_frame_image"][0]))
            conditions["first_frame_image"][0].save(anchor_dir / "current.png")
            atomic_npz(anchor_dir / "reference.npz", **reference)
            save_frames(anchor_dir / "reference_frames", reference["video"])
            files = sorted(p for p in anchor_dir.rglob("*") if p.is_file() and p.name != "reference.json")
            atomic_json(
                anchor_dir / "reference.json",
                {"metadata": metadata, "artifact_sha256": {str(p.relative_to(anchor_dir)): sha256(p) for p in files}},
            )
        contexts[key] = {"observation": observation, "reference": reference, "metadata": metadata}
    manifest["inputs"] = [context["metadata"] for context in contexts.values()]
    manifest["planned_generations"] = len(cases) * len(args.seeds)
    if args.audit_only:
        manifest["status"] = "audit_complete"
        manifest["invocations"][-1]["finished_at_utc"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
        write_report(output, manifest, cases)
        print(
            f"audit_complete: {len(cases)} cases, {len(contexts)} observations, {manifest['planned_generations']} planned generations: {output}",
            flush=True,
        )
        return manifest
    engine, performed = None, 0
    manifest["status"] = "running"
    try:
        for case in cases:
            for seed in args.seeds:
                path = output / "cases" / case["case_id"] / f"seed_{seed}"
                result_path = path / "result.json"
                if result_path.exists():
                    if verify_completed(path, manifest["identity_sha256"], case, seed):
                        continue
                    old = json.loads(result_path.read_text())
                    if old["status"] == "failed" and not args.retry_failed:
                        raise RuntimeError(f"Previous failure requires explicit --retry-failed: {path}")
                if args.max_generations is not None and performed >= args.max_generations:
                    break
                path.mkdir(parents=True, exist_ok=True)
                context = contexts[(case["episode_id"], case["anchor_frame"])]
                observation = dict(context["observation"], prompt=case["prompt"])
                conditions = conditions_from_observation(preprocessor, observation, seed)
                if input_identity(observation, conditions) != {
                    k: context["metadata"][k] for k in ("cameras", "canvas_sha256", "state_sha256", "input_sha256")
                }:
                    raise ValueError("Input images or state changed within a prompt pair")
                result = {
                    "status": "running",
                    "identity_sha256": manifest["identity_sha256"],
                    "case_id": case["case_id"],
                    "case": case,
                    "seed": seed,
                    "input_sha256": context["metadata"]["input_sha256"],
                    "started_at_utc": utc_now(),
                }
                atomic_json(result_path, result)
                try:
                    if (path / "generation.json").exists():
                        generated = json.loads((path / "generation.json").read_text())
                        if (
                            generated["identity_sha256"] != manifest["identity_sha256"]
                            or generated["case_id"] != case["case_id"]
                            or generated["seed"] != seed
                            or generated["input_sha256"] != context["metadata"]["input_sha256"]
                            or generated["sha256"] != sha256(path / "prediction.npz")
                        ):
                            raise ValueError("Interrupted generation cache provenance mismatch")
                        with np.load(path / "prediction.npz") as arrays:
                            prediction = {key: arrays[key].copy() for key in arrays.files}
                        result["seconds"] = generated["seconds"]
                        result["recovered_without_resampling"] = True
                    else:
                        if engine is None:
                            import torch

                            from openwam.deploy.engine import JointInferenceEngine

                            loaded_cfg, architecture = load_from_checkpoint_dir(
                                str(ckpt_dir), device=args.device, ckpt_name=args.ckpt_name
                            )
                            architecture.eval()
                            engine = JointInferenceEngine(merge_deploy_cfg(loaded_cfg, deploy_cfg), architecture)
                            manifest["runtime"] = {
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version(),
                                "device": args.device,
                                "gpu": torch.cuda.get_device_name(architecture.device),
                                "model_training": architecture.training,
                            }
                        import torch

                        torch.cuda.synchronize(engine.architecture.device)
                        begin = time.perf_counter()
                        prediction = predict_joint(engine, conditions)
                        torch.cuda.synchronize(engine.architecture.device)
                        result["seconds"] = time.perf_counter() - begin
                        atomic_npz(path / "prediction.npz", **prediction)
                        atomic_json(
                            path / "generation.json",
                            {
                                "identity_sha256": manifest["identity_sha256"],
                                "case_id": case["case_id"],
                                "seed": seed,
                                "input_sha256": context["metadata"]["input_sha256"],
                                "sha256": sha256(path / "prediction.npz"),
                                "seconds": result["seconds"],
                            },
                        )
                    result["metrics"] = score_case(case, prediction, context["reference"])
                    result["reference_path"] = (
                        f"../../../observations/episode_{case['episode_id']:06d}_t{case['anchor_frame']:06d}/reference.npz"
                    )
                    save_frames(path / "video", prediction["video"])
                    files = [
                        path / "prediction.npz",
                        path / "generation.json",
                        path / "video.gif",
                        *sorted((path / "video").glob("*.png")),
                    ]
                    result["artifact_sha256"] = {str(p.relative_to(path)): sha256(p) for p in files}
                    result["status"], result["finished_at_utc"] = "complete", utc_now()
                    atomic_json(result_path, result)
                    performed += 1
                    print(
                        f"complete group={case['group_id']} class={case['prompt_class']} seed={seed} {result['seconds']:.2f}s",
                        flush=True,
                    )
                    atomic_json(output / "manifest.json", manifest)
                except Exception as exc:
                    result.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                        finished_at_utc=utc_now(),
                    )
                    atomic_json(result_path, result)
                    atomic_json(path / f"failure_{time.time_ns()}.json", result)
                    raise
            if args.max_generations is not None and performed >= args.max_generations:
                break
        statuses = Counter()
        for case in cases:
            for seed in args.seeds:
                p = output / "cases" / case["case_id"] / f"seed_{seed}" / "result.json"
                statuses[json.loads(p.read_text())["status"] if p.exists() else "pending"] += 1
        manifest["counts"] = dict(statuses)
        manifest["status"] = "complete" if statuses["complete"] == manifest["planned_generations"] else "partial"
        if engine is not None:
            manifest["peak_allocated_gib"] = torch.cuda.max_memory_allocated(engine.architecture.device) / 2**30
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        manifest["invocations"][-1]["finished_at_utc"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
        write_report(output, manifest, cases)
    print(f"{manifest['status']}: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--denoise-steps", type=int, help="Override shared YAML; current default is 4 synchronous steps"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true", help="CPU-only input/GT/provenance audit, no model load")
    parser.add_argument(
        "--resume", action="store_true", help="Validate immutable manifest and skip verified completed requests"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Explicitly retry a failed request, preserving its failure record"
    )
    parser.add_argument(
        "--max-generations", type=int, help="Stop after this many new results; --resume continues the same frozen plan"
    )
    parser.add_argument(
        "--allow-evaluation-population-shift",
        action="store_true",
        help="Use checkpoint train stats for a separately identified Nero evaluation population",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
