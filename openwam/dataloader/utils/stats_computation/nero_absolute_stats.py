"""Prepare isolated absolute-action stats using the existing Nero data audit."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.nero import (
    FK_COLUMNS,
    XYZ_DIMS,
    assemble_eef,
    dataset_fingerprint,
    episode_path,
    read_jsonl,
)
from openwam.dataloader.utils.nero_absolute import CONTRACT
from openwam.dataloader.utils.stats_computation.nero_multitask_stats import PositionStats, prepare_dataset, table_array


def prepare_absolute_dataset(dataset_dir, output_dir):
    root, output = Path(dataset_dir), Path(output_dir)
    existing = output / "normalization_stats.npy"
    if (
        existing.exists()
        and np.load(existing, allow_pickle=True).item().get("metadata", {}).get("contract") != CONTRACT
    ):
        raise ValueError("Refusing to overwrite statistics from another action representation")
    # Reuse the complete source/header/time/FK/gripper audit without writing to
    # either the dataset or the relative experiment's statistics directory.
    with tempfile.TemporaryDirectory(prefix="nero_absolute_audit_") as temp:
        report = prepare_dataset(root, temp)
        checked = np.load(Path(temp) / "normalization_stats.npy", allow_pickle=True).item()
        # Existing state stats already describe exactly the absolute anchors.
        state_stats = checked["eef_relative_state"]
    info = json.loads((root / "meta/info.json").read_text())
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    action_stats = PositionStats()
    for ep in episodes:
        table = pq.read_table(episode_path(root, info, ep["episode_index"]), columns=["observation.state", *FK_COLUMNS])
        eef = assemble_eef(table_array(table, "observation.state"), *(table_array(table, key) for key in FK_COLUMNS))
        n = len(eef) - 32
        for horizon in range(1, 33):
            # Match the exact multiplicity of future labels in all full windows.
            action_stats.update(eef[horizon : horizon + n, XYZ_DIMS])
    if dataset_fingerprint(root, info, episodes) != report["dataset_fingerprint"]:
        raise ValueError("Dataset changed during absolute statistics preparation")
    metadata = {**checked["metadata"], "contract": CONTRACT}
    payload = {"eef": action_stats.finish(), "eef_state": state_stats, "metadata": metadata}
    report.update(metadata)
    report["statistics"] = (
        "absolute future XYZ over all full-window labels; absolute anchor state separately; rot6D identity; gripper 0..0.1m"
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".normalization_stats.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, payload)
    temporary.replace(existing)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
