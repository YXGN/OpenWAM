"""Explicit action safety interventions for the OpenWAM front."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GripperIntervention:
    row: int
    channel: int
    requested: float
    effective: float
    reason: str
    selected_intervention: str

    def wire(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "channel": self.channel,
            "requested": self.requested,
            "effective": self.effective,
            "reason": self.reason,
            "selected_intervention": self.selected_intervention,
        }


class GripperRangeSafety:
    """Clip finite native gripper outputs to the hardware command range."""

    def apply(self, actions: list[list[float]]) -> tuple[list[list[float]], list[GripperIntervention]]:
        effective = [row[:] for row in actions]
        interventions: list[GripperIntervention] = []
        for row_index, row in enumerate(effective):
            for channel in (9, 19):
                requested = row[channel]
                clipped = min(1.0, max(-1.0, requested))
                if clipped != requested:
                    row[channel] = clipped
                    interventions.append(
                        GripperIntervention(
                            row=row_index,
                            channel=channel,
                            requested=requested,
                            effective=clipped,
                            reason="native_gripper_range",
                            selected_intervention="clip_to_-1_1",
                        )
                    )
        return effective, interventions


__all__ = ["GripperIntervention", "GripperRangeSafety"]
