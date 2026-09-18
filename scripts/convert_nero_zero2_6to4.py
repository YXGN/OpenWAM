"""Convert Nero step4000 from six ranks to four, with a validated flat repartition.

Narrow, fail-closed migration: no padding, no round-robin, one AdamW group,
post-warmup scheduler. Original files are never modified.
All three FP32 buffers are checked bitwise after writing each new shard.
"""

import argparse
import copy
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from deepspeed.utils.tensor_fragment import fragment_address

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.train.utils.zero2_reshard import overlaps, rescale_post_warmup_scheduler, reslice  # noqa: E402


def load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, choices=(16, 24, 32), default=32)
    args = parser.parse_args()
    accumulation = 1
    new_batch = 4 * args.micro_batch
    assert 384000 % new_batch == 0
    data_step = 384000 // new_batch
    microstep = 4000 * accumulation
    torch.set_num_threads(8)
    source, target = args.source_run.resolve(), args.target_run.resolve()
    assert not target.exists(), "Use a new target directory"
    old_state = source / "accel_state_step_4000"
    meta = json.loads((old_state / "trainer_state.json").read_text())
    assert meta == {"global_step": 4000, "opt_step": 4000, "epoch": 0}
    tag = (old_state / "latest").read_text().strip()
    old_ds = old_state / tag
    model = load(old_ds / "mp_rank_00_model_states.pt")
    assert model["dp_world_size"] == 6 and model["mp_world_size"] == 1
    assert model["global_steps"] == 4000 and model["global_samples"] == 384000
    assert not model["ds_config"]["zero_optimization"].get("round_robin_gradients", False)
    shapes = model["param_shapes"]
    assert len(shapes) == 1
    total = sum(s.numel() for s in shapes[0].values())
    assert total % 12 == 0
    assert total % 8 == 0
    shards = [load(old_ds / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt") for rank in range(6)]
    for shard in shards:
        opt = shard["optimizer_state_dict"]
        assert opt["partition_count"] == [6] and opt["group_paddings"] == [0]
        assert int(opt["zero_stage"]) == 2
        assert len(opt["base_optimizer_state"]["param_groups"]) == 1
        assert list(opt["base_optimizer_state"]["state"]) == [0]
        assert opt["single_partition_of_fp32_groups"][0].numel() == total // 6
        assert opt["base_optimizer_state"]["state"][0]["step"].item() == 4000
    new_state = target / f"accel_state_step_{microstep}"
    new_ds = new_state / tag
    new_ds.mkdir(parents=True)
    for name in ("config.yaml", "normalization_stats.npy"):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / "tokenizer", target / "tokenizer")
    # Immutable initial weights; retention can unlink this without touching source.
    (target / f"checkpoint_step_{microstep}.safetensors").symlink_to(source / "checkpoint_step_4000.safetensors")
    config = copy.deepcopy(model["ds_config"])
    config["gradient_accumulation_steps"] = accumulation
    config["train_micro_batch_size_per_gpu"] = args.micro_batch
    config["train_batch_size"] = new_batch
    config["zero_optimization"]["offload_optimizer"]["device"] = "none"
    checks = []
    selected = [s["optimizer_state_dict"] for s in shards]
    for rank in range(4):
        start, stop = rank * total // 4, (rank + 1) * total // 4
        # Copy small metadata only; large buffers are reconstructed below.
        state = {
            k: copy.deepcopy(v)
            for k, v in selected[0].items()
            if k not in ("base_optimizer_state", "single_partition_of_fp32_groups", "param_slice_mappings")
        }
        base = selected[0]["base_optimizer_state"]
        state["base_optimizer_state"] = {"param_groups": copy.deepcopy(base["param_groups"]), "state": {0: {}}}
        dest = state["base_optimizer_state"]["state"][0]
        dest["step"] = base["state"][0]["step"].clone()
        for name in ("exp_avg", "exp_avg_sq"):
            dest[name] = reslice([s["base_optimizer_state"]["state"][0][name] for s in selected], start, stop)
        state["single_partition_of_fp32_groups"] = [
            reslice([s["single_partition_of_fp32_groups"][0] for s in selected], start, stop)
        ]
        state["partition_count"] = [4]
        offset, mapping = 0, OrderedDict()
        for name, shape in shapes[0].items():
            end = offset + shape.numel()
            lo, hi = max(start, offset), min(stop, end)
            if lo < hi:
                mapping[name] = fragment_address(start=lo - start, numel=hi - lo)
            offset = end
        state["param_slice_mappings"] = [mapping]
        path = new_ds / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        torch.save({"optimizer_state_dict": state, "ds_config": config, "ds_version": shards[0]["ds_version"]}, path)
        saved = load(path)["optimizer_state_dict"]
        for i, source_slice, section in overlaps(total // 6, start, stop):
            original = selected[i]
            assert torch.equal(
                saved["single_partition_of_fp32_groups"][0][section],
                original["single_partition_of_fp32_groups"][0][source_slice],
            )
            for name in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(
                    saved["base_optimizer_state"]["state"][0][name][section],
                    original["base_optimizer_state"]["state"][0][name][source_slice],
                )
        checks.append({"rank": rank, "all_fp32_buffers_bitwise_equal": True, "numel": total // 4})
        print(json.dumps(checks[-1]), flush=True)
        del state, dest, saved
    model["dp_world_size"] = 4
    model["ds_config"] = config
    torch.save(model, new_ds / "mp_rank_00_model_states.pt")
    scheduler = rescale_post_warmup_scheduler(load(old_state / "scheduler.bin"), 6, 4, 96, new_batch)
    torch.save(scheduler, new_state / "scheduler.bin")
    for rank in range(4):
        rng = torch.load(old_state / f"random_states_{rank}.pkl", map_location="cpu", weights_only=False)
        assert rng["step"] == 4000
        rng["step"] = microstep
        rng["torch_cuda_manual_seed"] = rng["torch_cuda_manual_seed"][:4]
        torch.save(rng, new_state / f"random_states_{rank}.pkl")
    (new_state / "latest").write_text(tag + "\n")
    report = {
        "source": str(source),
        "target": str(target),
        "source_optimizer_step": 4000,
        "target_microstep": microstep,
        "target_microbatch": args.micro_batch,
        "target_accumulation": accumulation,
        "global_batch": new_batch,
        "resume_data_step_offset": microstep - data_step,
        "consumed_windows": 384000,
        "scheduler_units": "preserve LR against consumed samples; new global batch changes updates per epoch",
        "checks": checks,
        "caveat": "Rank/microbatch RNG streams change with topology; not a bitwise continuation of the six-rank run.",
    }
    (target / "migration.json").write_text(json.dumps(report, indent=2))
    # Preserve the component/tokenizer specs which make the checkpoint self-contained.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="train_nero_4gpu")
    cfg.model = OmegaConf.load(source / "config.yaml").model
    cfg.training.resume_ckpt_path = str(target)
    cfg.training.batch_size = args.micro_batch
    cfg.training.resume_data_step_offset = microstep - data_step
    OmegaConf.save(cfg, target / "config.yaml")
    # Completion marker is last, so an incomplete conversion cannot be resumed.
    (new_state / "trainer_state.json").write_text(json.dumps({"global_step": microstep, "opt_step": 4000, "epoch": 0}))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
