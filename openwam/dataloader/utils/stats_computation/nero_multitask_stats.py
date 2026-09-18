"""Audit all Nero episodes and fit relative-action / absolute-state min-max stats."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.nero import (
    ACTION_MODE,
    CAMERAS,
    CONTRACT,
    FK_COLUMNS,
    GRIP_DIMS,
    XYZ_DIMS,
    assemble_eef,
    dataset_fingerprint,
    episode_path,
    read_jsonl,
    validate_info,
)


def table_array(table, name):
    return np.asarray(table[name].combine_chunks().values).reshape(len(table), -1)


class PositionStats:
    def __init__(self):
        self.count = 0
        self.lo, self.hi = np.full(6, np.inf), np.full(6, -np.inf)
        self.total, self.square = np.zeros(6), np.zeros(6)

    def update(self, values):
        x = np.asarray(values, dtype=np.float64)
        self.count += len(x)
        self.lo = np.minimum(self.lo, x.min(axis=0))
        self.hi = np.maximum(self.hi, x.max(axis=0))
        self.total += x.sum(axis=0)
        self.square += (x * x).sum(axis=0)

    def finish(self):
        if not self.count:
            raise ValueError("No full 33-frame windows")
        result = {"min": -np.ones(20), "max": np.ones(20), "mean": np.zeros(20), "std": np.ones(20)}
        result["min"][XYZ_DIMS], result["max"][XYZ_DIMS] = self.lo, self.hi
        result["mean"][XYZ_DIMS] = self.total / self.count
        result["std"][XYZ_DIMS] = np.sqrt(np.maximum(self.square / self.count - (self.total / self.count) ** 2, 0))
        # Physical gripper endpoints, not a per-task empirical range.
        for key, value in (("min", 0), ("max", 0.1), ("mean", 0.05), ("std", 0.05)):
            result[key][GRIP_DIMS] = value
        # Keep even a constant position dimension invertible in both normalizers.
        result["max"] = np.maximum(result["max"], result["min"] + 1e-6)
        return {k: v.astype(np.float32) for k, v in result.items()}


def prepare_dataset(dataset_dir, output_dir):
    import av

    root, output = Path(dataset_dir), Path(output_dir)
    info = json.loads((root / "meta/info.json").read_text())
    validate_info(root, info)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    tasks_list = read_jsonl(root / "meta/tasks.jsonl")
    tasks = {int(t["task_index"]): t["task"] for t in tasks_list}
    if len(tasks) != len(tasks_list) or not all(isinstance(t, str) and t.strip() for t in tasks.values()):
        raise ValueError("Duplicate or blank tasks")
    ids = [int(e["episode_index"]) for e in episodes]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("Empty or duplicate/negative episode indices")
    fingerprint = dataset_fingerprint(root, info, episodes)
    action_stats, state_stats = PositionStats(), PositionStats()
    counts, windows, codecs, rates = Counter(), Counter(), Counter(), Counter()
    gripper_endpoints, frames_by_task = Counter(), Counter()
    frame_offset = 0
    for ep in episodes:
        eid, length = int(ep["episode_index"]), int(ep["length"])
        table = pq.read_table(episode_path(root, info, eid))
        if len(table) != length or length < 33:
            raise ValueError(f"episode {eid}: inconsistent length or shorter than 33 frames")
        for name, expected in (
            ("episode_index", np.full(length, eid)),
            ("frame_index", np.arange(length)),
            ("index", np.arange(frame_offset, frame_offset + length)),
        ):
            if not np.array_equal(np.asarray(table[name]), expected):
                raise ValueError(f"episode {eid}: invalid {name}")
        task_ids = np.unique(np.asarray(table["task_index"]))
        if len(task_ids) != 1 or int(task_ids[0]) not in tasks:
            raise ValueError(f"episode {eid}: task_index must be constant and known")
        task = int(task_ids[0])
        if ep["tasks"] != [tasks[task]]:
            raise ValueError(f"episode {eid}: prompt mismatch")
        state, action = table_array(table, "observation.state"), table_array(table, "action")
        if not np.array_equal(state, action):
            raise ValueError(f"episode {eid}: action must equal current observation.state")
        eef = assemble_eef(state, *(table_array(table, name) for name in FK_COLUMNS))
        ts = np.asarray(table["timestamp"])
        if not np.isfinite(ts).all() or abs(ts[0]) > 1e-6 or np.any(np.abs(np.diff(ts) - 1 / 60) > 1e-4):
            raise ValueError(f"episode {eid}: nonuniform or invalid timestamps")
        n = length - 32
        state_stats.update(eef[:n, XYZ_DIMS])
        # Each full window contributes exactly 32 anchor-relative future labels.
        # Rotation has fixed identity statistics, so need not allocate 26M dR's.
        for horizon in range(1, 33):
            action_stats.update(eef[horizon : horizon + n, XYZ_DIMS] - eef[:n, XYZ_DIMS])
        for camera in CAMERAS:
            with av.open(str(episode_path(root, info, eid, camera))) as container:
                stream = container.streams.video[0]
                if stream.frames != length or (stream.width, stream.height) != (640, 360):
                    raise ValueError(f"episode {eid}: invalid video header for {camera}")
                if stream.average_rate is None or abs(float(stream.average_rate) - 60) > 0.01:
                    raise ValueError(f"episode {eid}: unexpected video rate for {camera}")
                codecs[stream.codec_context.name] += 1
                rates[str(stream.average_rate)] += 1
        counts[task] += 1
        windows[task] += n
        frames_by_task[task] += length
        for arm, col in (("left", 9), ("right", 19)):
            gripper_endpoints[f"{task}/{arm}"] += int(
                np.count_nonzero(np.isclose(eef[:, col], 0, atol=1e-7) | np.isclose(eef[:, col], 0.1, atol=1e-7))
            )
        frame_offset += length
    if (
        len(episodes) != info["total_episodes"]
        or frame_offset != info["total_frames"]
        or len(tasks) != info["total_tasks"]
    ):
        raise ValueError("Dataset totals disagree with metadata")
    if dataset_fingerprint(root, info, episodes) != fingerprint:
        raise ValueError("Dataset changed during preparation")
    metadata = {
        "contract": CONTRACT,
        "dataset_fingerprint": fingerprint,
        "episodes": ids,
        "windows": sum(windows.values()),
        "frames": frame_offset,
        "tasks": tasks,
    }
    payload = {ACTION_MODE: action_stats.finish(), f"{ACTION_MODE}_state": state_stats.finish(), "metadata": metadata}
    report = {
        **metadata,
        "task_episodes": dict(counts),
        "task_windows": dict(windows),
        "task_frames": dict(frames_by_task),
        "video_codecs": dict(codecs),
        "video_rates": dict(rates),
        "gripper_endpoint_fraction": {
            k: v / frames_by_task[int(k.split("/")[0])] for k, v in gripper_endpoints.items()
        },
        "video_validation": "all_headers; frame decode is exercised by reader smoke tests",
        "statistics": "min-max only; XYZ over full windows; rotation identity; gripper calibrated 0..0.1m",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "normalization_stats.npy"
    temp = output / ".normalization_stats.tmp"
    with temp.open("wb") as f:
        np.save(f, payload)
    temp.replace(path)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
