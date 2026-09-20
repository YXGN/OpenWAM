"""Physical EEF errors; never average metres and rotation components together."""

import numpy as np

from openwam.dataloader.utils.nero import GRIP_DIMS, rotation_matrix

DIM_NAMES = [
    f"{side}.{name}"
    for side in ("left", "right")
    for name in ("x_m", "y_m", "z_m", "rot_c0x", "rot_c0y", "rot_c0z", "rot_c1x", "rot_c1y", "rot_c1z", "gripper_m")
]


def rotation_errors(pred, gt):
    """SO(3) error after shared column-wise projection; invalid predictions stay NaN."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    a, b = pred[..., :3], pred[..., 3:]
    n = np.linalg.norm(a, axis=-1)
    unit = a / np.maximum(n[..., None], 1e-30)
    b_orth = b - np.sum(unit * b, axis=-1, keepdims=True) * unit
    valid = np.isfinite(pred).all(-1) & (n >= 1e-8) & (np.linalg.norm(b_orth, axis=-1) >= 1e-8)
    result = np.full(pred.shape[:-1], np.nan)
    if np.any(valid):
        r_pred, r_gt = rotation_matrix(pred[valid]), rotation_matrix(gt[valid], strict=True)
        cos = (np.sum(r_pred * r_gt, axis=(-2, -1)) - 1) / 2
        result[valid] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return result, valid


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    valid = x[np.isfinite(x)]
    return {
        "count": int(x.size),
        "valid_count": int(valid.size),
        "mean": float(np.mean(valid)) if valid.size else None,
        "p95": float(np.percentile(valid, 95)) if valid.size else None,
        "max": float(np.max(valid)) if valid.size else None,
    }


def physical_metrics(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    error = pred.astype(np.float64) - gt
    arms = {}
    for side, start in (("left", 0), ("right", 10)):
        position = np.linalg.norm(error[..., start : start + 3], axis=-1) * 1000
        angles, valid = rotation_errors(pred[..., start + 3 : start + 9], gt[..., start + 3 : start + 9])
        grip = np.abs(error[..., start + 9]) * 1000
        arms[side] = {
            "position_mm": summary(position),
            "rotation_deg": summary(angles),
            "gripper_mm": summary(grip),
            "invalid_rotation_count": int((~valid).sum()),
            "position_mm_by_horizon": np.mean(position, axis=0).tolist(),
            "rotation_deg_by_horizon": [summary(angles[:, h])["mean"] for h in range(32)],
            "gripper_mm_by_horizon": np.mean(grip, axis=0).tolist(),
            "last_step_position_mm": summary(position[:, -1]),
            "last_step_rotation_deg": summary(angles[:, -1]),
        }
    return {
        "per_dimension": {
            name: {"mae": float(np.mean(np.abs(error[..., d]))), "rmse": float(np.sqrt(np.mean(error[..., d] ** 2)))}
            for d, name in enumerate(DIM_NAMES)
        },
        "arms": arms,
    }


def evaluate(arrays):
    pred, gt = arrays["pred"], arrays["gt"]
    if pred.shape != gt.shape or pred.shape[1:] != (32, 20) or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite aligned (windows,32,20) predictions")
    baseline = np.broadcast_to(arrays["state"][:, None], gt.shape)
    result = {"prediction": physical_metrics(pred, gt), "hold_current": physical_metrics(baseline, gt)}
    for group in ("episode", "task"):
        result[f"by_{group}"] = {
            str(int(key)): {
                "prediction": physical_metrics(pred[arrays[group] == key], gt[arrays[group] == key]),
                "hold_current": physical_metrics(baseline[arrays[group] == key], gt[arrays[group] == key]),
            }
            for key in np.unique(arrays[group])
        }
    native = arrays["native_pred"]
    widths = native[..., GRIP_DIMS]
    invalid_rot = sum(
        int((~rotation_errors(native[..., s + 3 : s + 9], arrays["native_gt"][..., s + 3 : s + 9])[1]).sum())
        for s in (0, 10)
    )
    normalized = arrays["normalized_pred"]
    result["raw_diagnostics"] = {
        "gripper_out_of_range_fraction": float(np.mean((widths < 0) | (widths > 0.1))),
        "gripper_violation_mm": summary(np.maximum(np.maximum(-widths, widths - 0.1), 0) * 1000),
        "invalid_rotation_count": invalid_rot,
        "normalized_out_of_range_fraction": float(np.mean(np.abs(normalized) > 1)),
        "normalized_mae_per_dimension": np.mean(np.abs(normalized - arrays["normalized_gt"]), axis=(0, 1)).tolist(),
    }
    seconds = arrays["seconds"]
    result["inference_seconds"] = {"all": summary(seconds), "excluding_first": summary(seconds[1:])}
    return result
