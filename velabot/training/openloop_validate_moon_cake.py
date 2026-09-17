#!/usr/bin/env python3
"""Evaluate fixed-horizon MoonCake open-loop predictions against raw reader rows.

Predictions are supplied as an ``.npz`` with ``actions`` shaped
``(N,32,20)``, ``episode_index`` shaped ``(N,)`` and ``start`` shaped ``(N,)``.
The reader supplies the same decoded nine-frame canvas for each requested
window when a caller uses :func:`reader_window`. Ground truth is the original
row-aligned action ``action[start:start+32]``; no one-frame shift is applied.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from moon_cake import ACTION_STEPS, EEF_DIM, addback_delta_xyz, to_eef20


def _rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all() or x.shape[-1] != 6:
        raise ValueError("rot6d input must be finite with final dimension 6")
    a = x[..., :3]
    b = x[..., 3:6]
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(a_norm <= 1e-10):
        raise ValueError("rot6d first vector is zero")
    a = a / a_norm
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(b_norm <= 1e-10):
        raise ValueError("rot6d vectors are collinear")
    b = b / b_norm
    c = np.cross(a, b)
    return np.stack((a, b, c), axis=-1)


def so3_angle_error(pred_rot6d: np.ndarray, truth_rot6d: np.ndarray) -> np.ndarray:
    pred = _rot6d_to_matrix(pred_rot6d)
    truth = _rot6d_to_matrix(truth_rot6d)
    trace = np.trace(np.swapaxes(pred, -1, -2) @ truth, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "p95": float(np.percentile(arr, 95)), "max": float(np.max(arr))}


def evaluate_predictions(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    """Compute XYZ, SO(3), and physical gripper errors without filtering."""

    pred = np.asarray(predictions, dtype=np.float32)
    truth = np.asarray(ground_truth, dtype=np.float32)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1:] != (ACTION_STEPS, EEF_DIM):
        raise ValueError(f"predictions and ground_truth must both be (N,{ACTION_STEPS},{EEF_DIM})")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("open-loop inputs must be finite; no clip/filter is applied")
    xyz_delta = pred[..., (0, 1, 2, 10, 11, 12)] - truth[..., (0, 1, 2, 10, 11, 12)]
    xyz_error = np.linalg.norm(xyz_delta, axis=-1)
    xyz_per_arm = np.linalg.norm(
        xyz_delta.reshape(pred.shape[0], ACTION_STEPS, 2, 3),
        axis=-1,
    )
    rot_error = so3_angle_error(
        np.concatenate((pred[..., 3:9], pred[..., 13:19]), axis=-1).reshape(-1, 6),
        np.concatenate((truth[..., 3:9], truth[..., 13:19]), axis=-1).reshape(-1, 6),
    ).reshape(pred.shape[0], ACTION_STEPS, 2)
    pred_width = (pred[..., (9, 19)] + 1.0) * 0.05
    truth_width = (truth[..., (9, 19)] + 1.0) * 0.05
    grip_error = np.abs(pred_width - truth_width)
    return {
        "windows": int(pred.shape[0]),
        "horizon": ACTION_STEPS,
        "xyz_m": _summary(xyz_error),
        "xyz_per_arm_m": {"left": _summary(xyz_per_arm[..., 0]), "right": _summary(xyz_per_arm[..., 1])},
        "so3_rad": _summary(rot_error),
        "gripper_width_m": _summary(grip_error),
        "xyz_axis_mean_m": np.mean(np.abs(xyz_delta), axis=(0, 1)).tolist(),
        "offset_curve_xyz_m": np.mean(xyz_error, axis=0).tolist(),
        "clip_or_filter": False,
    }


def reader_window(reader, *, episode_local: int, start: int) -> dict[str, Any]:
    """Read one raw row window and its official reader-decoded canvas."""

    row = reader._eps_df.iloc[int(episode_local)]
    episode_length = int(row["length"])
    # Validate the local boundary before touching the shared parquet table;
    # otherwise a tail request could read rows belonging to the next episode.
    if int(start) < 0 or int(start) + ACTION_STEPS + 1 > episode_length:
        raise ValueError(f"open-loop start {start} is outside episode {row['episode_index']} length {episode_length}")
    table = reader._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
    frame = table.slice(int(reader._ep_data_row_offset[episode_local]) + int(start), ACTION_STEPS + 1).to_pandas()
    if len(frame) != ACTION_STEPS + 1:
        raise ValueError("requested open-loop window crosses the episode tail")
    raw_action = np.stack(frame["action"]).astype(np.float32)
    raw_state = np.stack(frame["observation.state"]).astype(np.float32)
    # Calling the official reader is the source of truth for integer video
    # starts and cumulative per-file episode offsets. This also supplies the
    # exact train-time image canvas to a model caller.
    global_idx = int(reader._cum_n_starts[episode_local]) + int(start)
    sample = reader[global_idx]
    return {
        "episode_index": int(row["episode_index"]),
        "start": int(start),
        "action": to_eef20(raw_action[:ACTION_STEPS]),
        "state": to_eef20(raw_state[0:1])[0],
        "video": sample["video"],
        "global_index": global_idx,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--action-mode", choices=("eef", "delta"), required=True)
    parser.add_argument(
        "--prediction-semantics",
        choices=("absolute_eef20", "xyz_delta_eef20"),
        default="absolute_eef20",
    )
    parser.add_argument("--episodes", help="optional comma-separated episode indices to validate")
    parser.add_argument("--starts", help="optional comma-separated local window starts; pair with --episodes")
    parser.add_argument("--window-indices", help="optional comma-separated NPZ row indices to select")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--denoise-steps", type=int, required=True)
    parser.add_argument("--compile", choices=("on", "off"), required=True)
    return parser


def validate_prediction_semantics(action_mode: str, prediction_semantics: str) -> None:
    if action_mode != "delta" and prediction_semantics == "xyz_delta_eef20":
        raise ValueError("xyz_delta_eef20 predictions require --action-mode delta")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_prediction_semantics(args.action_mode, args.prediction_semantics)
    data = np.load(args.predictions, allow_pickle=False)
    predictions = np.asarray(data["actions"], dtype=np.float32)
    episode_ids = np.asarray(data["episode_index"], dtype=np.int64)
    starts = np.asarray(data["start"], dtype=np.int64)
    indices = None
    if args.window_indices is not None:
        try:
            indices = np.asarray(
                [int(item.strip()) for item in args.window_indices.split(",") if item.strip()],
                dtype=np.int64,
            )
        except ValueError as exc:
            raise ValueError("--window-indices must be comma-separated integers") from exc
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(predictions)):
            raise ValueError("--window-indices contains an out-of-range or empty selection")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("--window-indices must not contain duplicates")
        predictions = predictions[indices]
        episode_ids = episode_ids[indices]
        starts = starts[indices]
    if (args.episodes is None) != (args.starts is None):
        raise ValueError("--episodes and --starts must be supplied together")
    if args.episodes is not None:
        try:
            requested_episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
            requested_starts = [int(value.strip()) for value in args.starts.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("--episodes and --starts must be comma-separated integers") from exc
        if len(requested_episodes) == 1 and len(requested_starts) > 1:
            requested_episodes *= len(requested_starts)
        elif len(requested_starts) == 1 and len(requested_episodes) > 1:
            requested_starts *= len(requested_episodes)
        if not requested_episodes or len(requested_episodes) != len(requested_starts):
            raise ValueError("--episodes and --starts must have equal nonzero lengths (or one side length 1)")
        requested_episode_ids = np.asarray(requested_episodes, dtype=np.int64)
        requested_start_values = np.asarray(requested_starts, dtype=np.int64)
        if not np.array_equal(episode_ids, requested_episode_ids) or not np.array_equal(starts, requested_start_values):
            raise ValueError("requested --episodes/--starts must exactly match NPZ metadata; refusing to re-label GT")
    if len(episode_ids) != len(predictions) or len(starts) != len(predictions):
        raise ValueError("prediction metadata lengths do not match actions")
    # Reader import is deliberately delayed so metric tests stay dependency-light.
    import sys

    sys.path.insert(0, str(args.code_root))
    from prepare_moon_cake import _reader_from_path, source_fingerprint

    reader = _reader_from_path(
        args.dataset,
        code_root=args.code_root,
        stats=args.stats,
        action_mode=args.action_mode,
        normalized=False,
    )
    by_episode = {int(row["episode_index"]): i for i, row in reader._eps_df.iterrows()}
    truth = []
    manifests = []
    reader_items = []
    for episode_id, start in zip(episode_ids, starts, strict=True):
        if int(episode_id) not in by_episode:
            raise KeyError(f"episode_index {int(episode_id)} is not in the reader population")
        item = reader_window(reader, episode_local=by_episode[int(episode_id)], start=int(start))
        reader_items.append(item)
        truth.append(item["action"])
        manifests.append(
            {
                "episode_index": int(episode_id),
                "start": int(start),
                "global_index": item["global_index"],
                "video_frames": len(item["video"]),
            }
        )
    truth_arr = np.stack(truth)
    if args.action_mode == "delta" and args.prediction_semantics == "xyz_delta_eef20":
        # Delta predictions are restored with the exact state from each
        # generating window. No cumsum or rotation reconstruction is allowed.
        pred_abs = np.stack(
            [
                addback_delta_xyz(p, item["state"], action_mode="delta")
                for p, item in zip(predictions, reader_items, strict=True)
            ]
        )
    else:
        pred_abs = predictions
    result = evaluate_predictions(pred_abs, truth_arr)
    source = source_fingerprint(args.code_root)
    result["identity"] = {
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source": source,
        "seed": {"value": args.seed, "kind": "caller_declared"},
        "denoise_steps": {"value": args.denoise_steps, "kind": "caller_declared"},
        "horizon": ACTION_STEPS,
        "chunk": ACTION_STEPS,
        "compile": {"value": args.compile, "kind": "caller_declared"},
        "action_mode": args.action_mode,
        "prediction_semantics": args.prediction_semantics,
        "ground_truth": "reader raw action rows start:start+32; no shift",
        "window_manifest": manifests,
        "window_indices": (
            [int(item) for item in np.asarray(indices).tolist()] if args.window_indices is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
