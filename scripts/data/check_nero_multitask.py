#!/usr/bin/env python
"""CPU reader smoke: temporal boundaries, masks, physical reconstruction, camera previews."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.normalize import Normalizer, NormMode, load_mode_stats
from openwam.dataloader.utils.nero import (
    ACTION_MODE,
    FK_COLUMNS,
    UNIFY_MAP,
    absolute_actions,
    assemble_eef,
    episode_path,
)
from openwam.dataloader.utils.stats_computation.nero_multitask_stats import table_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/nero_multitask.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--stats")
    parser.add_argument("--output-dir", default="outputs/nero_multitask_data")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.stats:
        cfg.normalization_stats_path = args.stats
    reader = build_dataset(cfg)
    normalizer = Normalizer(NormMode.MIN_MAX, load_mode_stats(cfg.normalization_stats_path, ACTION_MODE))
    info = json.loads((Path(cfg.dataset_dir) / "meta/info.json").read_text())
    entries, previews = [], []
    for ep_local in np.unique(np.linspace(0, len(reader._eps_df) - 1, min(8, len(reader._eps_df)), dtype=int)):
        row = reader._eps_df.iloc[ep_local]
        eid, length = int(row["episode_index"]), int(row["length"])
        table = pq.read_table(episode_path(cfg.dataset_dir, info, eid))
        raw = assemble_eef(table_array(table, "observation.state"), *(table_array(table, c) for c in FK_COLUMNS))
        for offset in sorted({0, (length - 33) // 2, length - 33}):
            sample = reader[int(reader._cum_n_starts[ep_local]) + offset]
            if sample["action"].shape != (32, 80) or sample["action_mask"].sum().item() != 640:
                raise AssertionError("Action tensor/mask mismatch")
            if not sample["video_mask"].all() or len(sample["video"]) != 9:
                raise AssertionError("Incomplete video window")
            restored = absolute_actions(normalizer.unnormalize(sample["action"].numpy()[:, UNIFY_MAP]), raw[offset])
            error = float(np.max(np.abs(restored - raw[offset + 1 : offset + 33])))
            np.testing.assert_allclose(restored, raw[offset + 1 : offset + 33], atol=2e-6, rtol=0)
            np.testing.assert_allclose(
                sample["proprio"].numpy()[:, UNIFY_MAP],
                normalizer.normalize(raw[offset : offset + 1]),
                atol=2e-6,
                rtol=0,
            )
            entries.append(
                {
                    "episode": eid,
                    "start_frame": offset,
                    "max_reconstruction_error": error,
                    "task_index": int(np.asarray(table["task_index"])[0]),
                }
            )
            tile = Image.new("RGB", (320, 408), "white")
            tile.paste(sample["video"][0], (0, 24))
            ImageDraw.Draw(tile).text((5, 5), f"episode={eid} frame={offset}  top=head / bottom=L,R", fill="black")
            previews.append(tile)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (960, 408 * ((len(previews) + 2) // 3)), "white")
    for i, tile in enumerate(previews):
        canvas.paste(tile, ((i % 3) * 320, (i // 3) * 408))
    canvas.save(output / "camera_preview.jpg")
    result = {
        "windows": len(reader),
        "checked_windows": len(entries),
        "samples": entries,
        "max_reconstruction_error": max(e["max_reconstruction_error"] for e in entries),
    }
    (output / "reader_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
