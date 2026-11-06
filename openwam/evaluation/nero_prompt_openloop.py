"""Paired Nero prompt interventions using the production joint inference path.

One immutable CSV and checkpoint per output directory. Only current RGB/state
enter the model; recorded futures are stored separately as scoring references.
No robot connection, training change, alternative sampler, or normalization fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.utils.nero import (
    CAMERAS,
    GRIP_DIMS,
    UNIFY_MAP,
    dataset_fingerprint,
    episode_path,
    read_jsonl,
    relative_actions,
    sha256,
    validate_info,
)
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.deploy.nero_contract import (
    DEFAULT_INFERENCE_CONFIG,
    conditions_from_observation,
    inference_config,
    validate_contract,
)
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.evaluation.nero_metrics import physical_metrics, rotation_errors, summary
from openwam.evaluation.nero_openloop import PredictionRecorder, current_observation, load_episode, window_starts

COLUMNS = (
    "group_id",
    "episode_id",
    "anchor_frame",
    "phase",
    "prompt_class",
    "prompt",
    "change_axis",
    "expected_observable",
    "feasibility",
    "gt_reference",
    "horizon_observable",
    "status",
    "notes",
)
PROMPT_CLASSES = {
    "original",
    "paraphrase_near",
    "paraphrase_far",
    "counterfactual",
    "remaining_task",
    "empty",
    "unavailable",
    "off_topic",
}
GT_CLASSES = {"original", "paraphrase_near", "paraphrase_far", "remaining_task"}
ROLES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "openwam/evaluation/nero_prompt_openloop.py",
    "openwam/evaluation/nero_openloop.py",
    "openwam/evaluation/nero_metrics.py",
    "openwam/deploy/engine.py",
    "openwam/deploy/model_loader.py",
    "openwam/deploy/server.py",
    "openwam/deploy/obs_preprocess.py",
    "openwam/deploy/nero_contract.py",
    "openwam/deploy/denoise_schedule.py",
    "openwam/model/architectures/base.py",
    "openwam/model/video_backbone/wan_backbone.py",
    "openwam/dataloader/utils/nero.py",
    "openwam/model/video_backbone/wan/encode.py",
    "openwam/dataloader/utils/normalization.py",
    "openwam/dataloader/utils/nero_absolute.py",
    "openwam/dataloader/transforms/normalize.py",
    "openwam/dataloader/transforms/multiview.py",
    "openwam/dataloader/utils/video_io.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def read_cases(path):
    """Validate annotation/GT eligibility before any image decoding or GPU load."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Expected exactly the 13 template columns, in order: {COLUMNS}")
        cases, skipped = [], []
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed CSV row at line {line}")
            # Preserve the actual prompt byte-for-byte; normalize only metadata.
            row = {k: v if k == "prompt" else v.strip() for k, v in row.items()}
            if row["status"] == "skip":
                if not row["notes"]:
                    raise ValueError(f"Skipped row {line} needs a reason")
                skipped.append(dict(row, csv_line=line))
                continue
            if row["status"] != "ready":
                raise ValueError(f"Row {line} is not ready/skip; draft rows must be reviewed before running")
            for key in ("group_id", "phase", "change_axis", "expected_observable"):
                if not row[key]:
                    raise ValueError(f"Row {line} requires {key}")
            for key in ("episode_id", "anchor_frame"):
                try:
                    row[key] = int(row[key])
                except ValueError as exc:
                    raise ValueError(f"Row {line}: {key} must be an integer") from exc
                if row[key] < 0:
                    raise ValueError(f"Row {line}: {key} must be nonnegative")
            kind, feasible, gt = row["prompt_class"], row["feasibility"], row["gt_reference"]
            if kind not in PROMPT_CLASSES or feasible not in {"feasible", "infeasible", "not_applicable"}:
                raise ValueError(f"Row {line}: invalid class or unreviewed feasibility")
            if row["horizon_observable"] not in {"yes", "no"} or gt not in {"original", "none"}:
                raise ValueError(f"Row {line}: review horizon observability and GT reference")
            if (kind == "empty" and row["prompt"] != "") or (kind != "empty" and not row["prompt"].strip()):
                raise ValueError(f"Row {line}: only empty cases may have an exactly empty prompt")
            if gt == "original" and (kind not in GT_CLASSES or feasible != "feasible"):
                raise ValueError(f"Row {line}: original GT is not a valid target for this intervention")
            if kind == "original" and gt != "original":
                raise ValueError(f"Row {line}: original control requires its original GT")
            if kind in GT_CLASSES | {"counterfactual"} and feasible != "feasible":
                raise ValueError(f"Row {line}: task cases must be feasible; use unavailable for unsupported tasks")
            if kind in {"empty", "off_topic"} and feasible != "not_applicable":
                raise ValueError(f"Row {line}: empty/off_topic feasibility must be not_applicable")
            if kind == "unavailable" and feasible != "infeasible":
                raise ValueError(f"Row {line}: unavailable must be marked infeasible")
            case_id = digest(row)[:20]
            cases.append(dict(row, case_id=case_id, csv_line=line))
    if not cases:
        raise ValueError("No ready cases")
    groups = {}
    seen = set()
    for case in cases:
        key = (case["group_id"], case["prompt"])
        if key in seen:
            raise ValueError(f"Duplicate prompt within group {case['group_id']}")
        seen.add(key)
        groups.setdefault(case["group_id"], []).append(case)
    for group, members in groups.items():
        if len({(c["episode_id"], c["anchor_frame"]) for c in members}) != 1:
            raise ValueError(f"Group {group} mixes observations")
        if sum(c["prompt_class"] == "original" for c in members) != 1:
            raise ValueError(f"Group {group} requires exactly one ready original control")
    return cases, skipped


def input_identity(observation, conditions):
    if set(conditions) != {
        "first_frame_image",
        "proprio",
        "prompt",
        "seed",
        "num_frames",
        "video_num_frames",
        "height",
        "width",
    }:
        raise ValueError("Unexpected conditioning input; future video/latents must never enter inference")
    if (conditions["num_frames"], conditions["video_num_frames"], conditions["height"], conditions["width"]) != (
        33,
        9,
        384,
        320,
    ):
        raise ValueError("Expected the audited 32-action / 9-video contract")
    state = np.asarray(conditions["proprio"], dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("Expected finite current absolute EEF20")
    np.testing.assert_array_equal(state, observation["state"])
    images = {}
    for role in ROLES:
        im = observation["images"][role]
        images[role] = {"size": list(im.size), "mode": im.mode, "sha256": hashlib.sha256(im.tobytes()).hexdigest()}
    canvas = conditions["first_frame_image"][0]
    result = {
        "cameras": images,
        "canvas_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
    }
    result["input_sha256"] = digest(result)
    return result


def predict_joint(engine, conditions):
    """Record the one production generation; retain pixels from the same draw."""
    architecture = engine.architecture
    original = architecture.normalizer
    recorder = PredictionRecorder(original)
    architecture.attach_normalizer(recorder)
    try:
        result = engine.generate(conditions)
    finally:
        architecture.attach_normalizer(original)
    if recorder.calls != 1 or recorder.normalized.shape != (32, 80):
        raise ValueError("Expected exactly one normalized 32x80 action output")
    pred, native = np.asarray(result["actions"], dtype=np.float32), np.asarray(recorder.native, dtype=np.float32)
    if pred.shape != (32, 20) or native.shape != (32, 20):
        raise ValueError("Expected native/deployed 32x20 action outputs")
    if not all(np.isfinite(x).all() for x in (pred, native, recorder.normalized)):
        raise ValueError("Non-finite action prediction")
    frames = result.get("video")
    if not isinstance(frames, (tuple, list)) or len(frames) != 9:
        raise ValueError("Joint generation must return exactly 9 decoded PIL frames")
    if any(not isinstance(im, Image.Image) or im.size != (320, 384) or im.mode != "RGB" for im in frames):
        raise ValueError("Expected 320x384 RGB PIL video frames; do not silently resize or remap pixels")
    return {
        "pred": pred.copy(),
        "native_pred": native.copy(),
        "normalized_pred": recorder.normalized[:, UNIFY_MAP].copy(),
        "normalized_full": recorder.normalized.copy(),
        "video": np.stack([np.asarray(im) for im in frames]),
    }


def future_reference(root, info, eid, anchor, preprocessor):
    """Scoring-only RGB reference, never supplied to conditions_from_observation."""
    indices = list(range(anchor, anchor + 33, 4))
    views = {
        role: decode_video_frames(str(episode_path(root, info, eid, camera)), indices, 360, 640)
        for role, camera in zip(ROLES, CAMERAS, strict=True)
    }
    return np.stack(
        [
            np.asarray(
                preprocessor.preprocess({"images": {role: views[role][j] for role in ROLES}, "state": np.zeros(20)})[
                    "image"
                ]
            )
            for j in range(9)
        ]
    )


def pixel_metrics(pred, gt):
    # No semantic claim. Conditional t is reported separately from future t+4...t+32.
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2, axis=(1, 2, 3))
    return {
        "first_frame_mse": float(mse[0]),
        "future_mse": float(np.mean(mse[1:])),
        "future_psnr_db": float(10 * np.log10(255**2 / np.mean(mse[1:]))) if np.mean(mse[1:]) > 0 else None,
        "exact_future_match": bool(np.all(mse[1:] == 0)),
        "mse_by_video_frame": mse.tolist(),
    }


def score_case(case, prediction, reference):
    state = reference["state"]
    held = np.broadcast_to(state, (32, 20))[None]
    widths = prediction["native_pred"][:, GRIP_DIMS]
    identity = np.broadcast_to(np.array([1, 0, 0, 0, 1, 0]), (32, 6))
    result = {
        "gt_reference": case["gt_reference"],
        "semantic_branch_eligible": case["prompt_class"] == "counterfactual"
        and case["feasibility"] == "feasible"
        and case["horizon_observable"] == "yes",
        "semantic_judgment": "not_automatically_scored",
        "motion_from_current": physical_metrics(prediction["pred"][None], held),
        "raw_diagnostics": {
            "gripper_out_of_range_fraction": float(np.mean((widths < 0) | (widths > 0.1))),
            "gripper_violation_mm": summary(np.maximum(np.maximum(-widths, widths - 0.1), 0) * 1000),
            "invalid_rotation_count": sum(
                int((~rotation_errors(prediction["native_pred"][:, s + 3 : s + 9], identity)[1]).sum()) for s in (0, 10)
            ),
            "normalized_out_of_range_fraction": float(np.mean(np.abs(prediction["normalized_pred"]) > 1)),
        },
        "action_gt_metrics": None,
        "video_gt_metrics": None,
    }
    if case["gt_reference"] == "original":
        result["action_gt_metrics"] = {
            "prediction": physical_metrics(prediction["pred"][None], reference["gt"][None]),
            "hold_current": physical_metrics(held, reference["gt"][None]),
        }
        result["video_gt_metrics"] = {
            "prediction": pixel_metrics(prediction["video"], reference["video"]),
            "hold_current": pixel_metrics(np.repeat(reference["video"][:1], 9, axis=0), reference["video"]),
            "interpretation": "Pixel reconstruction only; not semantic correctness or task success",
        }
    return result


def save_frames(directory, video):
    directory.mkdir(exist_ok=True)
    frames = [Image.fromarray(x) for x in video]
    for index, im in enumerate(frames):
        im.save(directory / f"{index:02d}.png")
    # GIF is a lossy preview; NPZ/PNG are authoritative. Playback 15 Hz.
    frames[0].save(
        directory.parent / (directory.name + ".gif"), save_all=True, append_images=frames[1:], duration=67, loop=0
    )


def checkpoint_identity(path, previous=None):
    stat = path.stat()
    identity = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    if previous and all(previous.get(k) == v for k, v in identity.items()):
        identity["sha256"] = previous["sha256"]
    else:
        identity["sha256"] = sha256(path)
    return identity


def assert_manifest_compatible(previous, identity):
    if previous["identity"] != identity:
        changed = sorted(
            k for k in set(previous["identity"]) | set(identity) if previous["identity"].get(k) != identity.get(k)
        )
        raise ValueError(
            f"Cannot resume an altered experiment; changed manifest fields: {changed}. Use a new output directory."
        )


def population_identity(stats, eval_fingerprint, allow_shift=False):
    train_fingerprint = stats["metadata"].get("dataset_fingerprint")
    if not train_fingerprint:
        raise ValueError("Training statistics must identify their population")
    shifted = eval_fingerprint != train_fingerprint
    if shifted and not allow_shift:
        raise ValueError(
            "Evaluation population differs from atom training stats; explicitly use --allow-evaluation-population-shift"
        )
    return {
        "train_stats_fingerprint": train_fingerprint,
        "eval_dataset_fingerprint": eval_fingerprint,
        "evaluation_population_shift": shifted,
        "allow_evaluation_population_shift": allow_shift,
        "normalization": "Unchanged checkpoint training statistics; never fitted on evaluation inputs",
    }


def verify_completed(path, identity_hash, case, seed):
    result = json.loads((path / "result.json").read_text())
    if result["status"] != "complete":
        return False
    if result["identity_sha256"] != identity_hash or result["case_id"] != case["case_id"] or result["seed"] != seed:
        raise ValueError(f"Completed case provenance mismatch: {path}")
    for name, expected in result["artifact_sha256"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Completed case artifact changed: {path / name}")
    return True


def write_report(output, manifest, cases):
    def esc(value):
        return html.escape(str(value), quote=True)

    cards, rows = [], []
    for case in cases:
        anchor = f"observations/episode_{case['episode_id']:06d}_t{case['anchor_frame']:06d}"
        cards.append(
            f"<section><h2>{esc(case['group_id'])} / {esc(case['prompt_class'])}</h2><p>{esc(case['prompt']) or '(empty)'}</p><p>{esc(case['expected_observable'])}</p><p>GT: {esc(case['gt_reference'])}; horizon observable: {esc(case['horizon_observable'])}</p><div class='strip'><figure><img src='{anchor}/current.png'><figcaption>Actual current RGB</figcaption></figure><figure><img src='{anchor}/reference_frames.gif'><figcaption>Recorded original future (not counterfactual GT)</figcaption></figure>"
        )
        for seed in manifest["identity"]["seeds"]:
            rel = f"cases/{case['case_id']}/seed_{seed}"
            result_path = output / rel / "result.json"
            rows.append(
                [
                    case["case_id"],
                    case["group_id"],
                    case["episode_id"],
                    case["anchor_frame"],
                    case["prompt_class"],
                    seed,
                    case["prompt"],
                    case["expected_observable"],
                    case["horizon_observable"],
                    "pending",
                    "pending",
                    "pending",
                    "pending",
                    "",
                ]
            )
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            status = result["status"]
            rows[-1][9] = status
            if status == "complete":
                cards.append(
                    f"<figure><img src='{rel}/video.gif'><figcaption>Seed {seed}; <a href='{rel}/result.json'>metrics / provenance</a>; <a href='{rel}/prediction.npz'>arrays</a></figcaption></figure>"
                )
            else:
                cards.append(f"<p>{seed}: {esc(status)} — {esc(result.get('error', ''))}</p>")
        cards.append("</div></section>")
    page = "<!doctype html><meta charset='utf-8'><title>Nero paired prompt audit</title><style>body{font:15px system-ui;max-width:1500px;margin:24px auto;background:#fafafa;color:#222}section{background:white;margin:20px 0;padding:14px;border:1px solid #ddd}.strip{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;width:256px}img{width:256px}figcaption{font-size:12px}p{white-space:pre-wrap}</style>"
    page += (
        f"<h1>Nero paired prompt audit</h1><p>Status: {esc(manifest['status'])}. Checkpoint: {esc(manifest['identity']['checkpoint']['path'])}</p><p>Short-horizon offline predictions; no robot execution or task-success claim. PNG/NPZ are authoritative; GIF previews are quantized. Semantic labels require independent review.</p>"
        + "".join(cards)
    )
    (output / "report.html").write_text(page)
    with (output / "review_template.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_id",
                "group_id",
                "episode_id",
                "anchor_frame",
                "prompt_class",
                "seed",
                "prompt",
                "expected_observable",
                "horizon_observable",
                "run_status",
                "action_judgment",
                "video_judgment",
                "cross_modal_consistency",
                "notes",
            ]
        )
        writer.writerows(rows)


def run(args):
    from openwam.deploy.model_loader import _build_normalizer, load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg

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
        norm_gt = apply_normalization(native_gt, stats[identity["action_mode"]], "min-max")
        unified = np.zeros((32, 80), dtype=np.float32)
        unified[:, UNIFY_MAP] = norm_gt
        np.testing.assert_allclose(normalizer.unnormalize(unified), native_gt, atol=1e-6)
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
        }
        metadata = {
            "episode": eid,
            "anchor": t,
            "task": int(task_ids[t]),
            "original_prompt": original_prompt,
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
