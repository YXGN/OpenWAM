"""Pure MoonCake action conversion and normalization contract.

This module intentionally has no OpenWAM, torch, or LeRobot import.  It is
used by the preparation/preflight tools and by the focused CPU tests. The
OpenWAM-side patch mirrors these constants and formulas in its official
dependency environment; the patch is fingerprinted separately at launch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

RAW_DIM = 16
EEF_DIM = 20
WINDOW_FRAMES = 33
ACTION_STEPS = 32
FPS = 60
VIDEO_STRIDE = 4
VIDEO_FRAME_INDICES = np.arange(0, WINDOW_FRAMES, VIDEO_STRIDE, dtype=np.int64)
UNIFY_DIM = 80
UNIFY_ACTION_MAP = ("0-9", "34-43")
XYZ_DIMS = np.array((0, 1, 2, 10, 11, 12), dtype=np.int64)
ROT6D_DIMS = np.array((3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18), dtype=np.int64)
GRIPPER_DIMS = np.array((9, 19), dtype=np.int64)
OPENPI_EEF20_DELTA_MASK = np.zeros(EEF_DIM, dtype=bool)
OPENPI_EEF20_DELTA_MASK[XYZ_DIMS] = True


@dataclass(frozen=True)
class SafetyEvent:
    """One explicit, observable gripper range intervention."""

    stream: str
    row: int
    channel: int
    requested: float
    effective: float
    reason: str = "native_gripper_range: opening_width_m outside [0,0.1]"
    selected_intervention: str = "clip_to_-1_1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "row": self.row,
            "channel": self.channel,
            "requested": self.requested,
            "effective": self.effective,
            "reason": self.reason,
            "selected_intervention": self.selected_intervention,
        }


def apply_gripper_safety(
    width: Any, *, stream: str = "unknown", channel: int = -1
) -> tuple[np.ndarray, list[SafetyEvent]]:
    """Apply the retained width admission/native clip as an explicit safety step."""

    values = np.asarray(width, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("gripper width contains non-finite values")
    if np.any(values < -1e-5) or np.any(values > 0.101):
        raise ValueError("gripper opening width outside accepted [0, 0.101] m range")
    requested = 2.0 * values / 0.1 - 1.0
    effective = np.clip(requested, -1.0, 1.0)
    events = [
        SafetyEvent(stream, row, channel, float(before), float(after))
        for row, (before, after) in enumerate(zip(requested, effective, strict=True))
        if before != after
    ]
    return effective.astype(np.float32), events


def _as_raw(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != RAW_DIM:
        raise ValueError(f"expected finite (T,{RAW_DIM}) flange action/state, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("flange action/state contains non-finite values")
    return arr


def to_eef20(raw: Any, *, stream: str = "unknown", return_events: bool = False):
    """Convert raw ``xyz + xyzw + opening_width`` bimanual rows to EEF20.

    The existing MoonCake behavior is retained: opening widths are converted
    with ``2 * width / .1 - 1`` and clipped to ``[-1, 1]``.  Widths through
    0.101 m are accepted; every changed row is returned as a safety event so
    preflight reports requested versus effective values.
    """

    arr = _as_raw(raw)
    arms: list[np.ndarray] = []
    events: list[SafetyEvent] = []
    for start, out_grip in ((0, 9), (8, 19)):
        quat = arr[:, start + 3 : start + 7]
        norms = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.max(np.abs(norms - 1.0)) > 1e-4:
            raise ValueError("source quaternion is not unit length")
        effective, width_events = apply_gripper_safety(arr[:, start + 7], stream=stream, channel=out_grip)
        events.extend(width_events)
        arms.extend((arr[:, start : start + 3], _quat_xyzw_to_rot6d(quat / norms), effective[:, None]))
    result = np.concatenate(arms, axis=1).astype(np.float32)
    return (result, events) if return_events else result


def _quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat.T
    c0 = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=-1)
    c1 = np.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), axis=-1)
    return np.concatenate((c0, c1), axis=-1).astype(np.float32)


def to_eef20_delta(action: Any, state0: Any):
    """Return 32 row XYZ-only deltas and absolute rot6d/gripper values."""

    action_eef = to_eef20(action) if np.asarray(action).shape[-1] == RAW_DIM else np.asarray(action, dtype=np.float32)
    state_arr = np.asarray(state0, dtype=np.float32)
    if state_arr.shape == (RAW_DIM,):
        state_eef = to_eef20(state_arr.reshape(1, RAW_DIM))[0]
    elif state_arr.ndim == 2 and state_arr.shape[1] == RAW_DIM:
        state_eef = to_eef20(state_arr)
    else:
        state_eef = state_arr
    if action_eef.ndim != 2 or action_eef.shape[1] != EEF_DIM:
        raise ValueError(f"expected action shape (T,{EEF_DIM}), got {action_eef.shape}")
    state_eef = np.asarray(state_eef, dtype=np.float32).reshape(-1)
    if state_eef.shape != (EEF_DIM,):
        raise ValueError(f"expected state0 shape ({EEF_DIM},), got {state_eef.shape}")
    if action_eef.shape[0] < ACTION_STEPS:
        raise ValueError(f"expected at least {ACTION_STEPS} action rows, got {action_eef.shape[0]}")
    out = action_eef[:ACTION_STEPS].copy()
    out[:, XYZ_DIMS] -= state_eef[XYZ_DIMS]
    return out.astype(np.float32)


def build_unify_mapping() -> np.ndarray:
    """Return the canonical raw EEF20 -> unified 80-D destination slots."""

    return np.array(list(range(10)) + list(range(34, 44)), dtype=np.int64)


def map_to_unify(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[-1] != EEF_DIM:
        raise ValueError(f"expected final dimension {EEF_DIM}, got {arr.shape}")
    out = np.zeros((*arr.shape[:-1], UNIFY_DIM), dtype=arr.dtype)
    out[..., build_unify_mapping()] = arr
    return out


def _vector(stats: Mapping[str, Any], key: str) -> np.ndarray:
    result = np.asarray(stats[key], dtype=np.float32)
    if result.shape != (EEF_DIM,):
        raise ValueError(f"stats[{key!r}] must have shape ({EEF_DIM},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"stats[{key!r}] contains non-finite values")
    return result


def validate_stats(stats: Mapping[str, Any], *, action_mode: str) -> dict[str, np.ndarray]:
    """Validate strict min-max stats and pin rotation/gripper transforms."""

    if action_mode not in ("eef", "delta"):
        raise ValueError("action_mode must be 'eef' or 'delta'")
    out = {key: _vector(stats, key) for key in ("min", "max", "mean", "std")}
    # q01/q99 are retained for compatibility only. They are aliases, never
    # evidence that quantiles were computed.
    out["q01"] = np.array(stats.get("q01", out["min"]), dtype=np.float32, copy=True)
    out["q99"] = np.array(stats.get("q99", out["max"]), dtype=np.float32, copy=True)
    if out["q01"].shape != (EEF_DIM,) or out["q99"].shape != (EEF_DIM,):
        raise ValueError("stats q01/q99 must have shape (20,)")
    if np.any(out["max"] < out["min"]):
        raise ValueError("stats max must be >= min")
    expected = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}
    for key, neutral in expected.items():
        for dims, label in ((ROT6D_DIMS, "rot6d"), (GRIPPER_DIMS, "gripper")):
            if not np.allclose(out[key][dims], neutral, rtol=0.0, atol=1e-6):
                raise ValueError(f"stats {key} must be identity on {label} dimensions")
    return out


def normalize(values: Any, stats: Mapping[str, Any], *, action_mode: str = "eef") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    s = validate_stats(stats, action_mode=action_mode)
    span = np.maximum(s["max"] - s["min"], 1e-6)
    return np.clip((arr - s["min"]) / span * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def denormalize(values: Any, stats: Mapping[str, Any], *, action_mode: str = "eef") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    s = validate_stats(stats, action_mode=action_mode)
    span = np.maximum(s["max"] - s["min"], 1e-6)
    return ((arr + 1.0) * 0.5 * span + s["min"]).astype(np.float32)


def addback_delta_xyz(actions: Any, raw_state: Any, *, action_mode: str) -> np.ndarray:
    """Add the one generating raw state to XYZ exactly once after denorm."""

    arr = np.asarray(actions, dtype=np.float32)
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if action_mode != "delta":
        return arr
    if arr.ndim != 2 or arr.shape[1] != EEF_DIM or state.shape != (EEF_DIM,):
        raise ValueError("delta addback requires actions (T,20) and raw state (20,)")
    if not np.isfinite(arr).all() or not np.isfinite(state).all():
        raise ValueError("delta addback inputs must be finite")
    out = arr.copy()
    out[:, XYZ_DIMS] += state[XYZ_DIMS]
    return out


__all__ = [
    "ACTION_STEPS",
    "EEF_DIM",
    "FPS",
    "GRIPPER_DIMS",
    "OPENPI_EEF20_DELTA_MASK",
    "RAW_DIM",
    "ROT6D_DIMS",
    "UNIFY_DIM",
    "VIDEO_FRAME_INDICES",
    "VIDEO_STRIDE",
    "WINDOW_FRAMES",
    "XYZ_DIMS",
    "SafetyEvent",
    "addback_delta_xyz",
    "apply_gripper_safety",
    "build_unify_mapping",
    "denormalize",
    "map_to_unify",
    "normalize",
    "to_eef20",
    "to_eef20_delta",
    "validate_stats",
]
