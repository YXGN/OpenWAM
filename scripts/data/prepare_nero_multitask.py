#!/usr/bin/env python
"""Prepare all-data Nero statistics without modifying the source dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openwam.dataloader.utils.stats_computation.nero_multitask_stats import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/nero_multitask_data")
    args = parser.parse_args()
    report = prepare_dataset(args.dataset_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ("dataset_fingerprint", "frames", "windows", "task_windows")}, indent=2))
    print(f"Statistics: {Path(args.output_dir).resolve() / 'normalization_stats.npy'}")


if __name__ == "__main__":
    main()
