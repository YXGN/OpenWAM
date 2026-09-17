"""MoonCake delta deployment reconstruction.

Delta actions are ``future_xyz - state0_xyz``.  Reconstruction happens after
the normalizer has gathered and denormalized the model output, and adds the
same generating state exactly once to XYZ slots. Rotation and gripper remain
absolute; no integration, rotation differencing, or output filtering occurs.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.moon_cake import OPENPI_EEF20_DELTA_MASK


def apply_delta_xyz_addback(actions, raw_state, *, action_mode):
    if action_mode != "delta":
        return actions
    result = np.asarray(actions, dtype=np.float32)
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if result.ndim != 2 or result.shape[1] != 20:
        raise ValueError(f"delta addback expects actions (T,20), got {result.shape}")
    if state.shape != (20,):
        raise ValueError(f"delta addback expects absolute EEF state (20,), got {state.shape}")
    if not np.isfinite(result).all() or not np.isfinite(state).all():
        raise ValueError("delta addback inputs must be finite")
    out = result.copy()
    out[:, OPENPI_EEF20_DELTA_MASK] += state[OPENPI_EEF20_DELTA_MASK]
    if not np.isfinite(out).all():
        raise ValueError("delta addback produced non-finite action")
    return out
