"""CPU-only six-rank reader test; no model or training processes are launched."""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.data_loader import prepare_data_loader
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator  # noqa: E402


def main():
    assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES empty"
    torch.set_num_threads(2)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(os.environ["OPENWAM_PROFILE_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train_nero_6gpu")
    dataset = build_dataset(cfg.dataloader, split="train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=list,
        generator=make_dataloader_generator(42, rank=rank),
        worker_init_fn=dataloader_worker_init_fn,
        pin_memory=False,
    )
    loader = prepare_data_loader(
        loader,
        device=torch.device("cpu"),
        num_processes=world,
        process_index=rank,
        put_on_device=False,
        rng_types=[],
    )
    iterator = iter(loader)
    for i in range(16):
        dist.barrier()
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        times = torch.tensor([elapsed])
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        row = {
            "rank": rank,
            "iteration": i,
            "warmup": i < 2,
            "data_wait": elapsed,
            "max_rank_data_wait": times.item(),
            "batch_size": len(batch),
            "simulated_compute_seconds": 3.3,
        }
        with (output / f"rank{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if rank == 0:
            print(json.dumps(row), flush=True)
        del batch
        time.sleep(3.3)
    del iterator, loader
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
