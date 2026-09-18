"""Nero's physical EEF contract, shared by preparation, reading and serving.

Actions are relative to the generating observation, never integrated along a
chunk: dp = p_future - p_anchor (body axes), dR = R_anchor.T @ R_future.
Grippers and proprioception remain absolute. Widths are in metres.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ACTION_MODE = "eef_relative"
CAMERAS = tuple(f"observation.images.{name}" for name in ("head_fpv", "left_hand", "right_hand"))
FK_COLUMNS = tuple(f"observation.fk.fb.{side}.gripper_flange" for side in ("left", "right"))
STATE_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (
        *(f"{side}_arm.joint{i}" for i in range(1, 8)),
        f"{side}_arm.{side}_gripper.opening_width",
    )
)
FK_NAMES = ("x", "y", "z", "rot6d_c0x", "rot6d_c0y", "rot6d_c0z", "rot6d_c1x", "rot6d_c1y", "rot6d_c1z")
XYZ_DIMS = [0, 1, 2, 10, 11, 12]
ROT_DIMS = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
GRIP_DIMS = [9, 19]
UNIFY_MAP = list(range(10)) + list(range(34, 44))
CONTRACT = {
    "version": 1,
    "action_mode": ACTION_MODE,
    "anchor": "window_first_observation",
    "position": "p_future_minus_p_anchor_in_body_axes",
    "rotation": "R_anchor_transpose_matmul_R_future_first_two_columns",
    "proprio": "absolute_body_frame_eef20",
    "gripper": "absolute_opening_width_metres",
    "gripper_bounds_m": [0.0, 0.1],
    "base": "body",
    "tips": ["left_gripper_flange", "right_gripper_flange"],
    "num_frames": 33,
    "action_offset": 1,
    "video_stride": 4,
    "camera_layout": list(CAMERAS),
    "statistics_population": "all_episodes_all_full_windows",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def episode_path(root, info, episode, camera=None):
    chunk = int(episode) // int(info["chunks_size"])
    template = info["video_path"] if camera else info["data_path"]
    return Path(root) / template.format(
        episode_chunk=chunk, chunk_index=chunk, episode_index=int(episode), video_key=camera
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(root, info, episodes):
    """Hash metadata/parquet content and video size/mtime (not video content)."""
    root = Path(root)
    records = []
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "kinematics.urdf"):
        records.append([f"meta/{name}", sha256(root / "meta" / name)])
    for ep in episodes:
        path = episode_path(root, info, ep["episode_index"])
        records.append([str(path.relative_to(root)), sha256(path)])
        for camera in CAMERAS:
            path = episode_path(root, info, ep["episode_index"], camera)
            stat = path.stat()
            records.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def validate_info(root, info):
    if info.get("action_semantics") != "current_state" or info.get("robot_type") != "agilex_nero_dual_arm":
        raise ValueError("Nero requires agilex_nero_dual_arm with action_semantics=current_state")
    if not str(info.get("codebase_version", "")).startswith("v2") or not np.isclose(info["fps"], 60, atol=0.01):
        raise ValueError("Nero requires LeRobot v2 at approximately 60 Hz")
    fk = info.get("fk", {})
    if fk.get("coordinate_frame") != "body" or fk.get("rotation") != "rot6d_first_two_rotation_matrix_columns":
        raise ValueError("Nero FK must use body coordinates and column-major rot6d")
    chains = {c["side"]: (c["base_link"], c["tip_link"]) for c in fk.get("chains", [])}
    if chains != {s: ("body", f"{s}_gripper_flange") for s in ("left", "right")}:
        raise ValueError("Unexpected Nero FK base/tip frames")
    if sha256(Path(root) / "meta/kinematics.urdf") != fk.get("urdf_sha256"):
        raise ValueError("Nero URDF hash mismatch")
    for key, shape in {"observation.state": [16], "action": [16], **dict.fromkeys(FK_COLUMNS, [9])}.items():
        if info["features"].get(key, {}).get("shape") != shape:
            raise ValueError(f"Unexpected feature shape: {key}")
        expected_names = STATE_NAMES if key in ("observation.state", "action") else FK_NAMES
        if info["features"][key].get("names") != list(expected_names):
            raise ValueError(f"Unexpected feature names/order: {key}")
    for camera in CAMERAS:
        if info["features"].get(camera, {}).get("shape") != [360, 640, 3]:
            raise ValueError(f"Missing or incompatible camera: {camera}")


def rotation_matrix(rot6d, *, strict=False):
    """Column convention; project predictions with Gram-Schmidt, reject degeneracy."""
    x = np.asarray(rot6d, dtype=np.float64)
    if x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError("Expected finite rot6d")
    a, b = x[..., :3], x[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rot6d first column")
    if strict and (
        np.any(np.abs(n - 1) > 2e-4)
        or np.any(np.abs(np.linalg.norm(b, axis=-1) - 1) > 2e-4)
        or np.any(np.abs(np.sum(a * b, axis=-1)) > 2e-4)
    ):
        raise ValueError("Source rot6d is not orthonormal")
    a = a / n
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rot6d second column")
    b = b / n
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def matrix_rot6d(matrix):
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1).astype(np.float32)


def assemble_eef(state, left_fk, right_fk):
    state = np.asarray(state, dtype=np.float32)
    left, right = np.asarray(left_fk, dtype=np.float32), np.asarray(right_fk, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 16 or left.shape != (len(state), 9) or right.shape != left.shape:
        raise ValueError("Expected state (T,16) and left/right FK (T,9)")
    if not all(np.isfinite(x).all() for x in (state, left, right)):
        raise ValueError("Non-finite Nero source")
    width = state[:, [7, 15]]
    if np.any(width < -1e-6) or np.any(width > 0.100001):
        raise ValueError("Gripper width outside calibrated [0,0.10] metres")
    width = np.clip(width, 0, 0.1)  # float32 endpoint roundoff only
    rotation_matrix(left[:, 3:], strict=True)
    rotation_matrix(right[:, 3:], strict=True)
    return np.concatenate((left, width[:, :1], right, width[:, 1:]), axis=-1)


def relative_actions(future, anchor):
    future, anchor = np.asarray(future, dtype=np.float32), np.asarray(anchor, dtype=np.float32)
    if future.ndim != 2 or future.shape[1] != 20 or anchor.shape != (20,):
        raise ValueError("Expected future (T,20) and absolute anchor (20,)")
    if not np.isfinite(future).all() or not np.isfinite(anchor).all():
        raise ValueError("Non-finite relative action inputs")
    out = future.copy()
    for start in (0, 10):
        out[:, start : start + 3] -= anchor[start : start + 3]
        r0 = rotation_matrix(anchor[start + 3 : start + 9], strict=True)
        rt = rotation_matrix(future[:, start + 3 : start + 9], strict=True)
        out[:, start + 3 : start + 9] = matrix_rot6d(r0.T @ rt)
    return out


def absolute_actions(relative, anchor):
    """Restore a denormalized native chunk using its original generating state."""
    relative, anchor = np.asarray(relative, dtype=np.float32), np.asarray(anchor, dtype=np.float32).reshape(-1)
    if relative.ndim != 2 or relative.shape[1] != 20 or anchor.shape != (20,):
        raise ValueError("Expected relative actions (T,20) and absolute anchor (20,)")
    if not np.isfinite(relative).all() or not np.isfinite(anchor).all():
        raise ValueError("Non-finite relative action inputs")
    out = relative.copy()
    for start in (0, 10):
        out[:, start : start + 3] += anchor[start : start + 3]
        r0 = rotation_matrix(anchor[start + 3 : start + 9], strict=True)
        dr = rotation_matrix(relative[:, start + 3 : start + 9])
        out[:, start + 3 : start + 9] = matrix_rot6d(r0 @ dr)
    out[:, GRIP_DIMS] = np.clip(out[:, GRIP_DIMS], 0, 0.1)
    return out
