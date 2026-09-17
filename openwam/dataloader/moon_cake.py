"""MoonCake Nero bimanual LeRobot-v3 reader (absolute EEF / XYZ-only delta)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.moon_cake_safety import apply_gripper_safety
from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_file

RAW_DIM = 16
EEF_DIM = 20
WINDOW_FRAMES = 33
ACTION_STEPS = 32
FPS = 60
VIDEO_STRIDE = 4
OPENPI_EEF20_DELTA_MASK = np.zeros(EEF_DIM, dtype=bool)
OPENPI_EEF20_DELTA_MASK[[0, 1, 2, 10, 11, 12]] = True


def to_eef20(raw, *, safety_events=None, stream="unknown"):
    """Convert finite raw ``(T,16)`` xyzw flange rows to canonical EEF20.

    Width conversion intentionally retains the established 0.101 m tolerance
    and ``[-1,1]`` native clipping. ``safety_events`` is an optional mutable
    list used by preflight to expose requested/effective values.
    """

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM or not np.isfinite(raw).all():
        raise ValueError(f"expected finite (T,{RAW_DIM}) flange action/state, got {raw.shape}")
    arms = []
    for start, grip_index in ((0, 9), (8, 19)):
        quat = raw[:, start + 3 : start + 7]
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.max(np.abs(norm - 1.0)) > 1e-4:
            raise ValueError("source quaternion is not unit length")
        effective = apply_gripper_safety(raw[:, start + 7], stream=stream, channel=grip_index, events=safety_events)
        arms.extend((raw[:, start : start + 3], quat_xyzw_to_rot6d(quat / norm), effective[:, None]))
    return np.concatenate(arms, axis=1).astype(np.float32)


def to_eef20_delta(action, state0, *, action_safety_events=None, state_safety_events=None):
    """Use state0 once for XYZ; rot6d and gripper remain absolute."""

    action = np.asarray(action, dtype=np.float32)
    state0 = np.asarray(state0, dtype=np.float32)
    action_eef = (
        to_eef20(action, safety_events=action_safety_events, stream="action")
        if action.shape[-1] == RAW_DIM
        else action
    )
    state_eef = (
        to_eef20(state0.reshape(1, -1), safety_events=state_safety_events, stream="state")[0]
        if state0.size == RAW_DIM
        else state0.reshape(-1)
    )
    if action_eef.ndim != 2 or action_eef.shape[1] != EEF_DIM or action_eef.shape[0] < ACTION_STEPS:
        raise ValueError(f"delta action must be at least ({ACTION_STEPS},{EEF_DIM})")
    if state_eef.shape != (EEF_DIM,):
        raise ValueError(f"delta state0 must have shape ({EEF_DIM},)")
    output = action_eef[:ACTION_STEPS].copy()
    output[:, OPENPI_EEF20_DELTA_MASK] -= state_eef[OPENPI_EEF20_DELTA_MASK]
    return output.astype(np.float32)


class MoonCakeDataset(LeRobotV3Reader):
    DATASET_NAME = "MoonCakeNero"
    ACTION_DIM = EEF_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index", "episode_index", "frame_index")
    HEAD_CAMERA = "observation.images.head_fpv"
    LEFT_WRIST_CAMERA = "observation.images.left_hand"
    RIGHT_WRIST_CAMERA = "observation.images.right_hand"
    WRIST_DECODE_TOLERATED = ()
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("normalization_stats_path", "action_mode")

    def __init__(self, dataset_dir, *, normalization_stats_path, action_mode="eef", **kwargs):
        if action_mode not in ("eef", "delta"):
            raise ValueError("action_mode must be 'eef' or 'delta'")
        if not normalization_stats_path:
            raise ValueError("normalization_stats_path is required")
        self._stats_path = Path(normalization_stats_path)
        self.action_mode = action_mode
        self.safety_events = []
        super().__init__(dataset_dir, **kwargs)

    def _train_min_window_len(self):
        return WINDOW_FRAMES

    def _post_init(self, info):
        derivation_path = self._dataset_dir / "meta" / "derivation.json"
        if not derivation_path.is_file():
            raise FileNotFoundError(f"MoonCake derivation metadata missing: {derivation_path}")
        contract = json.loads(derivation_path.read_text(encoding="utf-8"))
        if info.get("robot_type") != "agilex_nero_dual_arm" or float(info.get("fps")) != FPS:
            raise ValueError("MoonCake requires agilex_nero_dual_arm at 60 Hz")
        if contract.get("quat_order") != "xyzw" or contract.get("pose") != "absolute_body_frame":
            raise ValueError("MoonCake derivation must declare xyzw absolute_body_frame")
        if contract.get("ee_frame") != {"left": "left_gripper_flange", "right": "right_gripper_flange"}:
            raise ValueError("MoonCake derivation must declare both gripper flange frames")
        if self._num_frames != WINDOW_FRAMES or self._video_stride != VIDEO_STRIDE:
            raise ValueError("MoonCake requires num_frames=33 and video_stride=4")
        if self._unify_dst_index.tolist() != list(range(10)) + list(range(34, 44)):
            raise ValueError("MoonCake unified map must be 0-9 and 34-43")

    def _load_stats(self, info):
        if not self._normalize_mode:
            return None
        if self._normalize_mode != "min-max":
            raise ValueError("MoonCake supports normalize_mode=min-max only")
        if not self._stats_path.is_file():
            raise FileNotFoundError(self._stats_path)
        self.normalization_stats_path = str(self._stats_path)
        raw_stats = np.load(self._stats_path, allow_pickle=True).item()
        if not isinstance(raw_stats, dict) or not all(key in raw_stats for key in ("eef", "delta", "delta_state")):
            raise ValueError("MoonCake stats must contain eef, delta, and delta_state blocks")
        for key in ("min", "max", "mean", "std", "q01", "q99"):
            if not np.array_equal(np.asarray(raw_stats["delta_state"][key]), np.asarray(raw_stats["eef"][key])):
                raise ValueError(f"MoonCake stats delta_state.{key} must exactly alias eef.{key}")
        for block_name in ("eef", "delta"):
            block = raw_stats[block_name]
            expected = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}
            for key, value in expected.items():
                for dims, label in (((3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18), "rot6d"), ((9, 19), "gripper")):
                    if not np.allclose(np.asarray(block[key], dtype=np.float32)[list(dims)], value, rtol=0.0, atol=1e-6):
                        raise ValueError(f"MoonCake stats {block_name}.{key} must be identity on {label} dimensions")
        # Proprio is always absolute EEF. Keep its stats separate from the
        # action stats so delta actions cannot accidentally normalize state.
        self._eef_stats = load_stats_file(self._stats_path, action_mode="eef", normalize_mode="min-max", dim=EEF_DIM)
        return load_stats_file(self._stats_path, action_mode=self.action_mode, normalize_mode="min-max", dim=EEF_DIM)

    def _action_20d(self, win):
        action = np.stack(win["action"])
        if self.action_mode == "eef":
            values = to_eef20(action, safety_events=self.safety_events, stream="action")
        else:
            state = to_eef20(np.stack(win["observation.state"]), safety_events=self.safety_events, stream="state")
            values = to_eef20_delta(action, state[0], action_safety_events=self.safety_events)
        return apply_normalization(values, self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win):
        # Deployment and training both consume absolute eef proprio, even for
        # action_mode=delta (delta_state is an artifact alias for eef stats).
        values = to_eef20(np.stack(win["observation.state"]), safety_events=self.safety_events, stream="state")[:1]
        return apply_normalization(values, getattr(self, "_eef_stats", self._normalization_stats), self._normalize_mode)

    def __getitem__(self, idx):
        event_start = len(self.safety_events)
        sample = self._getitem_impl(idx)
        ep = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        episode_index = int(self._eps_df.iloc[ep]["episode_index"])
        for event in self.safety_events[event_start:]:
            event.setdefault("episode_index", episode_index)
        sample["episode_id"] = int(self._eps_df.iloc[ep]["episode_index"])
        return sample
