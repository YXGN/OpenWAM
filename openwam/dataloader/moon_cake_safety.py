"""Explicit MoonCake gripper safety intervention."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_gripper_safety(width, *, stream: str, channel: int, events: list[dict[str, Any]] | None = None):
    """Preserve the historical width admission and native ``[-1,1]`` clip.

    Widths outside ``[-1e-5, .101]`` are invalid source data and raise. The
    accepted sliver above the physical 0.1 m range is an explicit safety
    intervention whose requested/effective native values are recorded.
    """

    values = np.asarray(width, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("gripper width contains non-finite values")
    if np.any(values < -1e-5) or np.any(values > 0.101):
        raise ValueError("gripper opening width outside accepted [-1e-5, 0.101] m range")
    requested = 2.0 * values / 0.1 - 1.0
    effective = np.clip(requested, -1.0, 1.0)
    if events is not None:
        for row, (before, after) in enumerate(zip(requested, effective)):
            if before != after:
                events.append(
                    {
                        "stream": stream,
                        "row": row,
                        "channel": channel,
                        "requested": float(before),
                        "effective": float(after),
                        "reason": "native_gripper_range: opening_width_m outside [0,0.1]",
                        "selected_intervention": "clip_to_-1_1",
                    }
                )
    return effective.astype(np.float32)
