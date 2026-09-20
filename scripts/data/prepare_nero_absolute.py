#!/usr/bin/env python
"""Prepare a separate Nero absolute EEF statistics file; never modify source data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openwam.dataloader.utils.stats_computation.nero_absolute_stats import prepare_absolute_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/nero_absolute_data")
    args = parser.parse_args()
    report = prepare_absolute_dataset(args.dataset_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ("dataset_fingerprint", "frames", "windows", "task_windows")}, indent=2))
