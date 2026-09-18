"""Nero LeRobot v2 reader: anchor-relative EEF poses, absolute grippers/proprio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.utils.nero import (
    ACTION_MODE,
    CAMERAS,
    CONTRACT,
    FK_COLUMNS,
    GRIP_DIMS,
    ROT_DIMS,
    UNIFY_MAP,
    assemble_eef,
    dataset_fingerprint,
    episode_path,
    read_jsonl,
    relative_actions,
    validate_info,
)
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.video_io import decode_video_frames


class NeroMultitaskDataset(LeRobotV3Reader):
    DATASET_NAME = "NeroMultitask"
    ACTION_DIM = 20
    NEEDED_COLS = ("observation.state", *FK_COLUMNS, "task_index")
    HEAD_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA = CAMERAS
    WRIST_DECODE_TOLERATED = ()
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("normalization_stats_path", "action_mode", "require_all_cameras")

    def __init__(
        self, dataset_dir, *, normalization_stats_path, action_mode=ACTION_MODE, require_all_cameras=True, **kwargs
    ):
        if action_mode != ACTION_MODE:
            raise ValueError(f"Nero requires action_mode={ACTION_MODE}")
        if kwargs.get("split", "train") != "train" or kwargs.get("max_hours") is not None:
            raise ValueError("Nero uses all episodes for train; split/subsampling is not configured")
        if not require_all_cameras:
            raise ValueError("Nero requires all three cameras")
        self.action_mode = action_mode
        self._stats_path = Path(normalization_stats_path)
        self._info = json.loads((Path(dataset_dir) / "meta/info.json").read_text())
        validate_info(dataset_dir, self._info)
        self._episodes = read_jsonl(Path(dataset_dir) / "meta/episodes.jsonl")
        super().__init__(dataset_dir, **kwargs)

    def _load_excluded_episode_indices(self):
        if (self._dataset_dir / "meta/excluded_episodes.json").exists():
            raise ValueError("Nero all-data statistics do not allow implicit episode exclusions")
        return set()

    def _build_episode_index(self, info):
        rows = []
        for ep in self._episodes:
            eid = int(ep["episode_index"])
            row = {**ep, "data/chunk_index": eid // info["chunks_size"], "data/file_index": eid, "_data_row_offset": 0}
            for camera in CAMERAS:
                row.update(
                    {
                        f"videos/{camera}/chunk_index": eid // info["chunks_size"],
                        f"videos/{camera}/file_index": eid,
                        f"_video_frame_offset/{camera}": 0,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def _load_prompts(self):
        self._task_idx_to_text = {
            int(t["task_index"]): t["task"] for t in read_jsonl(self._dataset_dir / "meta/tasks.jsonl")
        }

    def _resolve_prompt(self, row, win):
        return self._task_idx_to_text[int(win["task_index"].iloc[0])]

    def _load_stats(self, info):
        payload = np.load(self._stats_path, allow_pickle=True).item()
        metadata = payload.get("metadata", {})
        if metadata.get("contract") != CONTRACT:
            raise ValueError("Incompatible Nero statistics contract; rerun prepare_nero_multitask.py")
        if metadata.get("dataset_fingerprint") != dataset_fingerprint(self._dataset_dir, info, self._episodes):
            raise ValueError("Stale Nero statistics: dataset fingerprint mismatch; rerun preparation")
        blocks = []
        for key in (ACTION_MODE, f"{ACTION_MODE}_state"):
            block = materialize_eef_stats(payload[key], "min-max", dim=20, strict_minmax=True)
            if not all(v.shape == (20,) and np.isfinite(v).all() for v in block.values()):
                raise ValueError("Invalid Nero statistics arrays")
            if np.any(block["max"] <= block["min"]):
                raise ValueError("Nero statistics must have positive ranges")
            for field, value in (("min", -1), ("max", 1)):
                if not np.allclose(block[field][ROT_DIMS], value):
                    raise ValueError("Nero rotation normalization must be identity")
            if not np.allclose(block["min"][GRIP_DIMS], 0) or not np.allclose(block["max"][GRIP_DIMS], 0.1):
                raise ValueError("Nero gripper statistics must use physical 0..0.1m endpoints")
            blocks.append(block)
        self._state_stats = blocks[1]
        self.normalization_stats_path = str(self._stats_path.resolve())
        return blocks[0]

    def _post_init(self, info):
        if (self._num_frames, self._video_stride, self._window_stride) != (33, 4, 1):
            raise ValueError("Nero requires num_frames=33, video_stride=4, window_stride=1")
        if (
            not self._unify
            or self._unify_dst_index.tolist() != UNIFY_MAP
            or self._unify_state_dst_index.tolist() != UNIFY_MAP
        ):
            raise ValueError("Nero requires canonical 80D left/right EEF slot maps")
        if not self._multiview or self._camera_layout != list(CAMERAS) or (self._height, self._width) != (384, 320):
            raise ValueError("Nero requires [head_fpv,left_hand,right_hand] on a 384x320 canvas")
        if self._target_camera is not None or self._head_camera_choices or self._normalize_mode != "min-max":
            raise ValueError("Nero requires fixed cameras and min-max normalization")

    def _train_min_window_len(self):
        return 33

    def _n_supervised_action_steps(self, actual_raw_len):
        return max(0, actual_raw_len - 1)

    def _load_data_table(self, chunk_idx, file_idx):
        return _read_data_table_cached(
            str(episode_path(self._dataset_dir, self._info, file_idx)), tuple(self.NEEDED_COLS)
        )

    @staticmethod
    def _eef(win):
        return assemble_eef(np.stack(win["observation.state"]), *(np.stack(win[name]) for name in FK_COLUMNS))

    def _action_20d(self, win):
        eef = self._eef(win)
        return apply_normalization(relative_actions(eef[1:], eef[0]), self._normalization_stats, "min-max")

    def _proprio_20d(self, win):
        return apply_normalization(self._eef(win.iloc[:1]), self._state_stats, "min-max")

    def _decode_one_camera(self, camera, row, ep_local, offset, real_local_indices, *, kind):
        path = episode_path(self._dataset_dir, self._info, int(row["episode_index"]), camera)
        # Decode at native resolution: the shared canvas applies one BILINEAR
        # resize, exactly as the deployment preprocessor does to raw images.
        return decode_video_frames(str(path), (real_local_indices + offset).tolist(), 360, 640)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        # Strict source: no random replacement or silent black wrist tiles.
        return self._getitem_impl(idx)
