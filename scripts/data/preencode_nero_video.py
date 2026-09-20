#!/usr/bin/env python
"""Build a complete, resumable Nero VAE cache. Launch with torchrun on selected GPUs."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from safetensors import safe_open
from safetensors.torch import save_file

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.nero import sha256
from openwam.dataloader.utils.nero_video_cache import FORMAT, SHAPE, atomic_json, module_digest, source_contract
from openwam.model.video_backbone.wan.encode import preprocess_video
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.train.utils.seeding import seed_everything


def validate_shard(root, sid, count, shard_size):
    record = json.loads((root / f"shard_{sid:05d}.json").read_text())
    path = root / f"shard_{sid:05d}.safetensors"
    if (
        record["file"] != path.name
        or record["start"] != sid * shard_size
        or record["count"] != count
        or path.stat().st_size != record["bytes"]
        or sha256(path) != record["sha256"]
    ):
        raise ValueError(f"Corrupt completed shard {sid}; inspect before rebuilding")
    with safe_open(path, framework="pt", device="cpu") as f:
        view = f.get_slice("latents")
        if view.get_shape() != [count, *SHAPE] or view.get_dtype() != "BF16":
            raise ValueError(f"Invalid shard tensor {sid}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train_nero_absolute_4gpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--reserve-gib", type=float, default=300, help="Leave space for rolling training states")
    parser.add_argument("--max-shards", type=int, help="Bounded validation only; never publishes a complete manifest")
    args = parser.parse_args()
    if args.shard_size <= 0 or args.shard_size % 64:
        raise ValueError("Shard size must be a positive multiple of the audited B64 encoding batch")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("gloo")
    seed_everything(42, rank=rank)
    free, total = torch.cuda.mem_get_info()
    if free < 50 * 2**30:
        raise RuntimeError("VAE-only B64 build requires at least 50 GiB free on every selected GPU")
    torch.cuda.set_per_process_memory_fraction(min(0.95, (free - 3 * 2**30) / total))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name)
    if cfg.dataloader.type != "nero_multitask_absolute" or cfg.dataloader.get("video_latent_cache_dir"):
        raise ValueError("Build cache from the uncached absolute Nero reader")
    dataset = build_dataset(cfg.dataloader)
    source = source_contract(dataset)
    vae = WanVideoVAE38()
    prefix = "video_backbone.vae."
    with safe_open(cfg.training.finetune_ckpt_path, framework="pt", device="cpu") as f:
        weights = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    vae.load_state_dict(weights, strict=True)
    del weights
    vae.to(dtype=torch.bfloat16, device="cuda").eval().requires_grad_(False)
    identity = dict(
        format=FORMAT,
        source=source,
        vae_sha256=module_digest(vae),
        shape=list(SHAPE),
        dtype="bfloat16",
        encode_batch_size=64,
        shard_size=args.shard_size,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Lock on rank 0 prevents a second invocation from writing the same shards.
    import fcntl

    lock = None
    if rank == 0:
        lock = (output / ".build.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "build.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("Existing cache build belongs to different data/weights/preprocessing")
        else:
            atomic_json(path, identity)
    if world > 1:
        dist.barrier()
    if json.loads((output / "build.json").read_text()) != identity:
        raise ValueError("Rank cache identity mismatch")
    shard_count = math.ceil(len(dataset) / args.shard_size)
    if rank == 0:
        existing = sum(p.stat().st_size for p in output.glob("shard_*.safetensors"))
        required = max(0, len(dataset) * math.prod(SHAPE) * 2 - existing) + args.reserve_gib * 2**30
        if shutil.disk_usage(output).free < required:
            raise RuntimeError("Insufficient disk space for cache plus reserved training checkpoints")
    if world > 1:
        dist.barrier()
    limit = min(shard_count, args.max_shards) if args.max_shards is not None else shard_count
    jobs = []
    for sid in range(rank, limit, world):
        count = min(args.shard_size, len(dataset) - sid * args.shard_size)
        if (output / f"shard_{sid:05d}.json").exists():
            validate_shard(output, sid, count, args.shard_size)
        else:
            jobs.append((sid, count))
    indices = [i for sid, count in jobs for i in range(sid * args.shard_size, sid * args.shard_size + count)]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=list,
    )
    completed, done_windows, job_index, pending = 0, 0, 0, []
    started = time.monotonic()
    for batch in loader:
        n = len(batch)
        # Final partial batch is padded only for fixed B64 encoder numerics; padded outputs are discarded.
        batch = batch + [batch[-1]] * (64 - n)
        pixels = torch.cat([preprocess_video(s["video"], dtype=torch.bfloat16, device="cuda") for s in batch])
        with torch.no_grad():
            latents = vae.batch_encode(pixels, device="cuda")[:n].cpu().contiguous()
        if torch.cuda.memory_stats().get("num_alloc_retries", 0):
            raise RuntimeError("VAE allocator retried: stop rather than silently cache a memory-pressure fallback")
        if latents.dtype != torch.bfloat16 or tuple(latents.shape[1:]) != SHAPE or not torch.isfinite(latents).all():
            raise ValueError("Invalid encoded latents")
        pending.append(latents)
        del pixels
        sid, count = jobs[job_index]
        if sum(len(x) for x in pending) == count:
            tensor = torch.cat(pending)
            path = output / f"shard_{sid:05d}.safetensors"
            temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
            save_file({"latents": tensor}, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            record = dict(
                file=path.name, start=sid * args.shard_size, count=count, bytes=path.stat().st_size, sha256=sha256(path)
            )
            atomic_json(output / f"shard_{sid:05d}.json", record)
            pending.clear()
            job_index += 1
            completed += 1
            done_windows += count
            status = dict(
                rank=rank,
                completed_new_shards=completed,
                new_windows=done_windows,
                pending_shards=len(jobs) - job_index,
                last_shard=sid,
                elapsed_seconds=time.monotonic() - started,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                allocator_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0),
            )
            atomic_json(output / f"progress_rank{rank}.json", status)
            print(json.dumps(status), flush=True)
    if pending or job_index != len(jobs):
        raise RuntimeError("Incomplete rank encoding")
    if world > 1:
        dist.barrier()
    if rank == 0:
        if source_contract(dataset) != source:
            raise ValueError("Dataset or preprocessing changed while encoding")
        if args.max_shards is None:
            records = [
                validate_shard(output, sid, min(args.shard_size, len(dataset) - sid * args.shard_size), args.shard_size)
                for sid in range(shard_count)
            ]
            atomic_json(output / "manifest.json", {**identity, "complete": True, "shards": records})
            print(f"Complete cache published: {output / 'manifest.json'}", flush=True)
        else:
            print("Bounded validation finished; complete manifest deliberately not published", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
