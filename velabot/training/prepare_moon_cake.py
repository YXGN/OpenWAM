#!/usr/bin/env python3
"""Prepare and preflight MoonCake EEF/XYZ-delta normalization artifacts.

The command is intentionally data-only: it never starts a trainer or loads a
foundation model.  Dataset and output paths are always explicit so a report
cannot silently describe a different private run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

VELABOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VELABOT_ROOT))

from moon_cake import (  # noqa: E402
    ACTION_STEPS,
    EEF_DIM,
    VIDEO_FRAME_INDICES,
    WINDOW_FRAMES,
    to_eef20,
)
from source_identity import fingerprint_source  # noqa: E402

PATCH_PATH = Path(__file__).resolve().parents[1] / "patches" / "openwam-moon-cake-training.patch"
PATCH_SOURCE_FILES = (
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
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def source_fingerprint(code_root: Path) -> dict[str, Any]:
    """Return the exact source format consumed by the serving config."""
    return fingerprint_source(code_root, PATCH_PATH, PATCH_SOURCE_FILES)


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.minimum = np.full(EEF_DIM, np.inf, dtype=np.float64)
        self.maximum = np.full(EEF_DIM, -np.inf, dtype=np.float64)
        self.total = np.zeros(EEF_DIM, dtype=np.float64)
        self.squares = np.zeros(EEF_DIM, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EEF_DIM or not np.isfinite(arr).all():
            raise ValueError(f"stats input must be finite (N,{EEF_DIM}), got {arr.shape}")
        if not len(arr):
            return
        x = arr.astype(np.float64)
        self.count += len(x)
        self.minimum = np.minimum(self.minimum, x.min(axis=0))
        self.maximum = np.maximum(self.maximum, x.max(axis=0))
        self.total += x.sum(axis=0)
        self.squares += np.square(x).sum(axis=0)

    def finish(self, *, population: str) -> dict[str, Any]:
        if not self.count:
            raise ValueError("cannot create stats from an empty population")
        mean = self.total / self.count
        out: dict[str, Any] = {
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(np.maximum(0.0, self.squares / self.count - np.square(mean))).astype(np.float32),
            # Compatibility fields. This profile does not compute quantiles;
            # q01/q99 intentionally alias min/max and are labelled below.
            "q01": self.minimum.astype(np.float32),
            "q99": self.maximum.astype(np.float32),
            "count": self.count,
            "population": population,
            "normalization": "min-max",
            "quantiles_computed": False,
        }
        for dims in ((3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18), (9, 19)):
            for key, value in (("min", -1.0), ("max", 1.0), ("q01", -1.0), ("q99", 1.0), ("mean", 0.0), ("std", 1.0)):
                out[key][list(dims)] = value
        return out


def compute_stats_from_arrays(
    episodes: Iterable[Mapping[str, Any]], *, action_mode: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Compute deterministic EEF and delta stats from raw episode mappings.

    Each mapping must contain ``action`` and ``state`` arrays in the raw 16-D
    schema. The input order is retained in the source identity hash by the
    caller; no episode is silently filtered.
    """

    if action_mode not in ("eef", "delta"):
        raise ValueError("action_mode must be 'eef' or 'delta'")
    eef_acc = _Accumulator()
    delta_acc = _Accumulator()
    report: dict[str, Any] = {
        "episodes": 0,
        "source_frames": 0,
        "windows": 0,
        "alignment_anomalous_episodes": [],
        "trajectory_anomalies_checked": False,
        "gripper_safety_events": [],
        "action_next_state_mismatch_rows": 0,
        "action_state_same_row_mismatch_rows": 0,
    }
    for ep_i, episode in enumerate(episodes):
        if "action" not in episode or "state" not in episode:
            raise ValueError("each episode requires action and state")
        action = np.asarray(episode["action"], dtype=np.float32)
        state = np.asarray(episode["state"], dtype=np.float32)
        if action.shape != state.shape or action.ndim != 2 or action.shape[1] != 16:
            raise ValueError(f"episode {ep_i} action/state must both have shape (T,16)")
        if len(action) < WINDOW_FRAMES:
            raise ValueError(f"episode {ep_i} length {len(action)} is shorter than {WINDOW_FRAMES}")
        action_eef, action_events = to_eef20(action, stream="action", return_events=True)
        state_eef, state_events = to_eef20(state, stream="state", return_events=True)
        eef_acc.add(action_eef)
        eef_acc.add(state_eef)
        episode_index = int(episode.get("episode_index", ep_i))
        for event in (*action_events, *state_events):
            payload = event.as_dict()
            payload["episode_index"] = episode_index
            report["gripper_safety_events"].append(payload)
        next_mismatch = np.any(action[:-1] != state[1:], axis=1)
        same_mismatch = np.any(action != state, axis=1)
        report["action_next_state_mismatch_rows"] += int(next_mismatch.sum())
        report["action_state_same_row_mismatch_rows"] += int(same_mismatch.sum())
        if next_mismatch.any():
            report["alignment_anomalous_episodes"].append(
                {
                    "episode_index": episode_index,
                    "kind": "action_next_state_mismatch",
                    "rows": int(next_mismatch.sum()),
                }
            )
        n_windows = len(action) - WINDOW_FRAMES + 1
        action_windows = np.lib.stride_tricks.sliding_window_view(action_eef, WINDOW_FRAMES, axis=0)
        # sliding_window_view puts the temporal axis last for axis=0.
        action_windows = np.moveaxis(action_windows, -1, 1)[:, :ACTION_STEPS, :]
        state_starts = state_eef[:n_windows]
        delta = action_windows.copy()
        delta[:, :, (0, 1, 2, 10, 11, 12)] -= state_starts[:, None, (0, 1, 2, 10, 11, 12)]
        delta_acc.add(delta.reshape(-1, EEF_DIM))
        report["episodes"] += 1
        report["source_frames"] += len(action)
        report["windows"] += n_windows
    if report["action_next_state_mismatch_rows"]:
        # The reader contract is row aligned; mismatches are recorded and fail
        # preflight rather than being repaired or causing episode deletion.
        report["alignment_valid"] = False
    else:
        report["alignment_valid"] = True
    stats = {
        "eef": eef_acc.finish(population="all_train_action_and_state_rows"),
        "delta": delta_acc.finish(population="all_train_windows_action32_relative_to_window_start_state"),
    }
    delta_xyz = stats["delta"]["max"][[0, 1, 2, 10, 11, 12]] - stats["delta"]["min"][[0, 1, 2, 10, 11, 12]]
    if np.any(delta_xyz <= 1e-6):
        raise ValueError(f"delta XYZ min-max span collapsed: {delta_xyz.tolist()}")
    stats["delta_state"] = stats["eef"]
    report["delta_definition"] = "XYZ action[t] minus state[window_start]; rot6d/gripper absolute"
    report["action_mode"] = action_mode
    return stats, report


def _reader_from_path(dataset: Path, *, code_root: Path, stats: Path, action_mode: str, normalized: bool = False):
    import sys

    sys.path.insert(0, str(code_root))
    from omegaconf import OmegaConf

    from openwam.dataloader.moon_cake import MoonCakeDataset

    cfg = OmegaConf.create(
        {
            "type": "moon_cake",
            "dataset_dir": str(dataset),
            "normalization_stats_path": str(stats),
            "action_mode": action_mode,
            "normalize_mode": "min-max" if normalized else None,
            "unify_action": True,
            "unify_action_map": ["0-9", "34-43"],
            "num_frames": WINDOW_FRAMES,
            "video_stride": 4,
            "window_stride": 1,
            "height": 384,
            "width": 320,
            "multiview": True,
            "enable_action_supervision": True,
        }
    )
    return MoonCakeDataset.from_config(cfg, split="train")


def preflight_reader(reader, *, max_samples: int = 64) -> dict[str, Any]:
    """Decode representative first/last windows using the reader itself.

    This deliberately calls ``reader[idx]`` instead of reproducing camera
    resizing or offsets. The resulting report states that only representative
    windows were decoded; it never claims full video decode coverage.
    """

    if len(reader) <= 0:
        raise ValueError("reader has no legal training windows")
    candidates: set[int] = set()
    n_eps = len(reader._eps_df)
    for ep in np.linspace(0, n_eps - 1, min(n_eps, max_samples // 2 or 1), dtype=int):
        lo, hi = reader._cum_n_starts[ep : ep + 2]
        if hi > lo:
            candidates.update((int(lo), int(hi - 1)))
    sample_indices = sorted(candidates)
    samples: list[dict[str, Any]] = []
    for idx in sample_indices:
        sample = reader[idx]
        videos = sample.get("video", [])
        action = sample["action"]
        proprio = sample["proprio"]
        if len(videos) != len(VIDEO_FRAME_INDICES):
            raise ValueError(f"sample {idx} video count {len(videos)} != {len(VIDEO_FRAME_INDICES)}")
        if any(tuple(image.size) != (320, 384) for image in videos):
            raise ValueError(f"sample {idx} has a non-320x384 video canvas")
        if tuple(action.shape) != (ACTION_STEPS, 80) or tuple(proprio.shape) != (1, 80):
            raise ValueError(f"sample {idx} has action/proprio shapes {tuple(action.shape)}/{tuple(proprio.shape)}")
        if not bool(action.isfinite().all()) or not bool(proprio.isfinite().all()):
            raise ValueError(f"sample {idx} has non-finite normalized tensors")
        samples.append(
            {
                "index": idx,
                "action_shape": list(action.shape),
                "proprio_shape": list(proprio.shape),
                "video_shape": [len(videos), 384, 320],
            }
        )
    video_files: list[dict[str, Any]] = []
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for MoonCake video metadata preflight") from exc
    for camera in reader._video_cameras():
        chunk_col = f"videos/{camera}/chunk_index"
        file_col = f"videos/{camera}/file_index"
        if chunk_col not in reader._eps_df.columns or file_col not in reader._eps_df.columns:
            raise ValueError(f"MoonCake camera {camera!r} has no episode chunk/file columns")
        for (chunk, file), rows in reader._eps_df.groupby([chunk_col, file_col], sort=False):
            path = reader._dataset_dir / reader._video_path_template.format(
                video_key=camera, chunk_index=int(chunk), file_index=int(file)
            )
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                frame_count = int(stream.frames)
                expected = int(rows["length"].sum())
                if frame_count < expected:
                    raise ValueError(f"video {path} has {frame_count} frames, expected at least {expected}")
                video_files.append(
                    {
                        "camera": camera,
                        "path": str(path),
                        "frames": frame_count,
                        "expected_frames": expected,
                        "fps": float(stream.average_rate) if stream.average_rate else None,
                        "codec": stream.codec_context.name,
                    }
                )
    return {
        "decoded_sample_windows": len(samples),
        "decoded_sample_indices": sample_indices,
        "sample_shapes": samples,
        "video_decode_scope": "representative first/last legal window per selected episode; not full-frame decode",
        "reader_windows": len(reader),
        "video_files": video_files,
    }


def iter_reader_episodes(reader) -> Iterable[dict[str, Any]]:
    """Yield raw episode arrays through the official reader's integer offsets."""

    for i, row in reader._eps_df.iterrows():
        table = reader._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        frame = table.slice(int(reader._ep_data_row_offset[i]), int(row["length"])).to_pandas()
        yield {
            "episode_index": int(row["episode_index"]),
            "action": np.stack(frame["action"]).astype(np.float32),
            "state": np.stack(frame["observation.state"]).astype(np.float32),
        }


def _source_identity(
    dataset: Path,
    episodes: Sequence[Mapping[str, Any]],
    derivation: Path | None,
    *,
    universe_episode_count: int | None = None,
) -> dict[str, Any]:
    episode_rows = []
    for i, episode in enumerate(episodes):
        episode_rows.append(
            {
                "position": i,
                "episode_index": int(episode.get("episode_index", i)),
                "frames": len(episode["action"]),
                "action_sha256": _sha256_bytes(np.asarray(episode["action"], dtype=np.float32).tobytes()),
                "state_sha256": _sha256_bytes(np.asarray(episode["state"], dtype=np.float32).tobytes()),
            }
        )
    files = []
    for path in sorted(p for p in dataset.rglob("*") if p.is_file()):
        relative = path.relative_to(dataset).as_posix()
        # The manifest is stable across copies of a dataset: paths are
        # relative, and content hash is reserved for compact metadata. Video
        # bytes are verified by PyAV separately and represented by size here.
        item: dict[str, Any] = {"path": relative, "size": path.stat().st_size}
        if relative.startswith("meta/") and path.stat().st_size <= 16 * 1024 * 1024:
            item["sha256"] = sha256_file(path)
        files.append(item)
    identity = {
        "dataset": str(dataset.resolve()),
        "episode_count": len(episode_rows),
        "universe_episode_count": (
            int(universe_episode_count) if universe_episode_count is not None else len(episode_rows)
        ),
        "episode_rows": episode_rows,
        "episode_list_hash": _json_hash(episode_rows),
        "data_files": files,
        "data_manifest_hash": _json_hash(files),
        "derivation_sha256": sha256_file(derivation) if derivation and derivation.exists() else None,
    }
    identity["source_identity_hash"] = _json_hash(identity)
    return identity


def write_artifacts(
    stats_path: Path,
    report_path: Path,
    stats: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    source: Mapping[str, Any],
    excluded_episodes: Sequence[int],
) -> None:
    if stats_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite {stats_path} or {report_path}")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(stats_path, dict(stats), allow_pickle=True)
    full_report = dict(report)
    full_report["source_identity"] = dict(source_identity)
    full_report["stats_sha256"] = sha256_file(stats_path)
    full_report["source_revision"] = source["revision"]
    full_report["source_repository"] = source["repository"]
    full_report["source"] = dict(source)
    full_report["raw_data_modified"] = False
    full_report["excluded_episodes"] = [int(item) for item in excluded_episodes]
    full_report["selected_episode_count"] = int(source_identity["episode_count"])
    full_report["video_frame_indices"] = VIDEO_FRAME_INDICES.tolist()
    full_report["video_stride"] = 4
    full_report["window_frames"] = WINDOW_FRAMES
    full_report["action_steps"] = ACTION_STEPS
    report_path.write_text(json.dumps(full_report, indent=2, sort_keys=True), encoding="utf-8")


def write_failure_report(
    report_path: Path,
    report: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    source: Mapping[str, Any],
    excluded_episodes: Sequence[int],
) -> None:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload.update(
        source_identity=dict(source_identity),
        source=dict(source),
        source_revision=source["revision"],
        source_repository=source["repository"],
        excluded_episodes=[int(item) for item in excluded_episodes],
        preflight_passed=False,
        stats_written=False,
        raw_data_modified=False,
    )
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="LeRobot v3 MoonCake dataset directory")
    parser.add_argument("--stats-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--action-mode", choices=("eef", "delta"), required=True)
    parser.add_argument("--code-root", type=Path, help="official OpenWAM checkout used for the reader")
    parser.add_argument("--max-episodes", type=int, help="CPU test limit; report says this is a partial preflight")
    parser.add_argument("--allow-existing", action="store_true", help="allow replacing only the two named outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    if args.code_root is None:
        raise ValueError("--code-root is required when reading the official LeRobot dataset")
    reader = _reader_from_path(
        args.dataset,
        code_root=args.code_root,
        stats=args.stats_output,
        action_mode=args.action_mode,
    )
    episodes = list(iter_reader_episodes(reader))
    excluded_episodes = sorted(int(item) for item in reader._load_excluded_episode_indices())
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    stats, report = compute_stats_from_arrays(episodes, action_mode=args.action_mode)
    report["partial_preflight"] = args.max_episodes is not None
    identity = _source_identity(
        args.dataset,
        episodes,
        args.dataset / "meta" / "derivation.json",
        universe_episode_count=getattr(reader, "_n_episodes_before_exclusions", None),
    )
    source = source_fingerprint(args.code_root)
    if args.allow_existing:
        for path in (args.stats_output, args.report_output):
            if path.exists():
                path.unlink()
    if not report["alignment_valid"]:
        write_failure_report(
            args.report_output,
            report,
            source_identity=identity,
            source=source,
            excluded_episodes=excluded_episodes,
        )
        raise RuntimeError(
            "MoonCake preflight found action[t] != observation.state[t+1]; "
            f"see report {args.report_output}. No stats artifact was written."
        )
    write_artifacts(
        args.stats_output,
        args.report_output,
        stats,
        report,
        source_identity=identity,
        source=source,
        excluded_episodes=excluded_episodes,
    )
    # Re-open through the official reader with the newly written artifact to
    # prove the training-time normalized payload and video path are usable.
    normalized_reader = _reader_from_path(
        args.dataset,
        code_root=args.code_root,
        stats=args.stats_output,
        action_mode=args.action_mode,
        normalized=True,
    )
    try:
        reader_report = preflight_reader(normalized_reader)
    except Exception as exc:
        failed = json.loads(args.report_output.read_text(encoding="utf-8"))
        failed["preflight_passed"] = False
        failed["reader_preflight_error"] = f"{type(exc).__name__}: {exc}"
        args.report_output.write_text(json.dumps(failed, indent=2, sort_keys=True), encoding="utf-8")
        raise
    report_path = args.report_output
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    updated["reader_preflight"] = reader_report
    updated["preflight_passed"] = bool(updated.get("alignment_valid", False))
    report_path.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "stats": str(args.stats_output),
                "report": str(args.report_output),
                "windows": report["windows"],
                "action_mode": args.action_mode,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
