"""Independent absolute-body EEF contract for the Nero absolute experiment."""

import numpy as np

from openwam.dataloader.utils.nero import CONTRACT as RELATIVE_CONTRACT
from openwam.dataloader.utils.nero import GRIP_DIMS, ROT_DIMS
from openwam.dataloader.utils.normalization import materialize_eef_stats

ACTION_MODE = "eef"
CONTRACT = {
    **RELATIVE_CONTRACT,
    "action_mode": ACTION_MODE,
    "anchor": None,
    "position": "absolute_future_position_in_body_axes",
    "rotation": "absolute_future_rotation_first_two_columns",
}


def load_absolute_stats(path):
    """Reject relative or stale-format stats, including mislabeled payloads."""
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("metadata", {}).get("contract") != CONTRACT:
        raise ValueError("Nero absolute EEF requires its own absolute statistics contract")
    blocks = []
    for key in ("eef", "eef_state"):
        if key not in payload:
            raise ValueError(f"Missing Nero absolute statistics: {key}")
        block = materialize_eef_stats(payload[key], "min-max", dim=20, strict_minmax=True)
        if not all(v.shape == (20,) and np.isfinite(v).all() for v in block.values()):
            raise ValueError("Invalid Nero absolute statistics arrays")
        if np.any(block["max"] <= block["min"]):
            raise ValueError("Nero absolute statistics require positive ranges")
        if not np.allclose(block["min"][ROT_DIMS], -1) or not np.allclose(block["max"][ROT_DIMS], 1):
            raise ValueError("Nero absolute rot6D normalization must be identity")
        if not np.allclose(block["min"][GRIP_DIMS], 0) or not np.allclose(block["max"][GRIP_DIMS], 0.1):
            raise ValueError("Nero absolute gripper bounds must be 0..0.1 metres")
        blocks.append(block)
    return payload, blocks[0], blocks[1]
