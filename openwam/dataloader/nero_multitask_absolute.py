"""Nero absolute EEF reader; the existing relative reader is unchanged."""

import json
from pathlib import Path

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.nero_multitask import NeroMultitaskDataset
from openwam.dataloader.utils.nero import dataset_fingerprint, read_jsonl, validate_info
from openwam.dataloader.utils.nero_absolute import ACTION_MODE, load_absolute_stats
from openwam.dataloader.utils.normalization import apply_normalization


class NeroMultitaskAbsoluteDataset(NeroMultitaskDataset):
    """Reuse Nero indexing/cameras/proprio, replacing only action semantics/stats."""

    DATASET_NAME = "NeroMultitaskAbsolute"

    def __init__(
        self,
        dataset_dir,
        *,
        normalization_stats_path,
        action_mode=ACTION_MODE,
        require_all_cameras=True,
        **kwargs,
    ):
        if action_mode != ACTION_MODE:
            raise ValueError("Nero absolute reader requires action_mode=eef")
        if kwargs.get("split", "train") != "train" or kwargs.get("max_hours") is not None:
            raise ValueError("Nero absolute uses all episodes for train")
        if not require_all_cameras:
            raise ValueError("Nero absolute requires all three cameras")
        self.action_mode = action_mode
        self._stats_path = Path(normalization_stats_path)
        self._info = json.loads((Path(dataset_dir) / "meta/info.json").read_text())
        validate_info(dataset_dir, self._info)
        self._episodes = read_jsonl(Path(dataset_dir) / "meta/episodes.jsonl")
        # Initialize the common LeRobot machinery directly: the relative reader's
        # constructor deliberately rejects absolute mode. Its pure indexing,
        # camera, EEF assembly and proprio methods are shared unchanged.
        LeRobotV3Reader.__init__(self, dataset_dir, **kwargs)

    def _load_stats(self, info):
        payload, action_stats, self._state_stats = load_absolute_stats(self._stats_path)
        if payload["metadata"].get("dataset_fingerprint") != dataset_fingerprint(
            self._dataset_dir, info, self._episodes
        ):
            raise ValueError("Stale Nero absolute statistics: dataset fingerprint mismatch")
        self.normalization_stats_path = str(self._stats_path.resolve())
        return action_stats

    def _action_20d(self, win):
        # Source action is current_state; supervise t+1..t+32, never t..t+31.
        return apply_normalization(self._eef(win)[1:], self._normalization_stats, "min-max")
