"""CPU-only helpers for exact rigid ZeRO-2 repartitioning (no padding)."""

import copy

import torch


def overlaps(part_size, start, stop):
    """Yield source rank, source slice and destination slice for a flat interval."""
    if part_size <= 0 or not 0 <= start < stop:
        raise ValueError("Invalid partition interval")
    for rank in range(start // part_size, (stop - 1) // part_size + 1):
        lo, hi = max(start, rank * part_size), min(stop, (rank + 1) * part_size)
        yield rank, slice(lo - rank * part_size, hi - rank * part_size), slice(lo - start, hi - start)


def reslice(tensors, start, stop):
    part_size = tensors[0].numel()
    if not all(t.ndim == 1 and t.numel() == part_size for t in tensors) or stop > len(tensors) * part_size:
        raise ValueError("Expected equal flat source partitions and an in-bounds interval")
    return torch.cat([tensors[rank][section] for rank, section, _ in overlaps(part_size, start, stop)])


def rescale_post_warmup_scheduler(state, source_world, target_world, source_batch=1, target_batch=1):
    """Preserve LR trajectory in update units, or sample units when batches differ."""

    def scale(value):
        numerator, denominator = value * target_world * source_batch, source_world * target_batch
        if numerator % denominator:
            raise ValueError("Scheduler phase cannot be represented exactly at target world size")
        return numerator // denominator

    if len(state["_milestones"]) != 1 or state["last_epoch"] <= state["_milestones"][0]:
        raise ValueError("Only the post-warmup SequentialLR checkpoint is supported")
    result = copy.deepcopy(state)
    result["last_epoch"] = scale(state["last_epoch"])
    result["_milestones"] = [scale(state["_milestones"][0])]
    linear, cosine = result["_schedulers"]
    linear["total_iters"] = scale(linear["total_iters"])
    linear["start_factor"] = 1 / linear["total_iters"]
    linear["last_epoch"] = linear["total_iters"] - 1
    linear["_step_count"] = linear["total_iters"]
    cosine["T_max"] = scale(cosine["T_max"])
    cosine["last_epoch"] = scale(cosine["last_epoch"])
    cosine["_step_count"] = cosine["last_epoch"] + 2
    return result
