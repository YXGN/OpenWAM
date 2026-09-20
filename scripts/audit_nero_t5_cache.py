"""Compare real frozen T5 outputs and full-model loss/gradients without updating weights."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.model.video_backbone.wan.encode import FrozenTextCache, encode_text  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.seeding import seed_everything, seed_process  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-name", default="train_nero_target_8gpu")
    parser.add_argument("--deterministic-backward", action="store_true", help="Diagnostic only; does not change training defaults")
    parser.add_argument("--online-control", action="store_true", help="Compare two online runs to measure ordinary repeatability")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1, "Select one audit GPU"
    seed_everything(42)
    if args.deterministic_backward:
        torch.use_deterministic_algorithms(True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name)
    dataset = build_dataset(cfg.dataloader, split="train")
    representatives = {}
    for i, episode in enumerate(dataset._episodes):
        representatives.setdefault(tuple(episode["tasks"]), int(dataset._cum_n_starts[i]))
    indices = list(representatives.values())
    samples = [dataset[i] for i in indices]
    prompts = [sample["prompt"] for sample in samples]
    assert len(set(prompts)) == 3
    trainer = OpenWAMTrainer(cfg, dataset=dataset)
    arch = trainer.architecture
    backbone = arch.video_backbone
    cache = backbone._training_text_cache
    assert isinstance(cache, FrozenTextCache)
    report = {
        "windows": len(dataset), "indices": indices, "prompts": prompts, "text_checks": [],
        "deterministic_backward": args.deterministic_backward, "online_control": args.online_control,
    }

    def write_report():
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")

    # Cold fill uses the actual production batch size. Also test permutation,
    # duplicate texts and a smaller batch against the ordinary encoder.
    batch64 = [prompts[i % 3] for i in range(64)]
    for batch in (batch64, batch64[::-1], prompts):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            online, online_lengths = encode_text(
                batch, tokenizer=backbone._tokenizer, text_encoder=backbone.text_encoder, device=arch.device
            )
        torch.cuda.synchronize()
        online_s = time.perf_counter() - start
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        start = time.perf_counter()
        actual, lengths = cache.encode(batch, device=arch.device)
        torch.cuda.synchronize()
        cache_s = time.perf_counter() - start
        row = {
            "batch_size": len(batch), "online_seconds": online_s, "cache_seconds": cache_s,
            "exact_equal": torch.equal(online, actual),
            "max_abs_difference": (online.float() - actual.float()).abs().max().item(),
            "seq_lens_equal": torch.equal(online_lengths, lengths),
            "rng_unchanged": torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        }
        report["text_checks"].append(row)
        write_report()
        print(json.dumps(row), flush=True)
        assert row["exact_equal"] and row["seq_lens_equal"] and row["rng_unchanged"], row
        del online, actual

    def forward_backward(use_cache):
        arch.zero_grad(set_to_none=True)
        backbone._training_text_cache = cache if use_cache else None
        seed_process(12345)
        inputs = arch.prepare_inputs(samples)
        input_cpu = {k: v.detach().cpu().clone() for k, v in inputs.items() if torch.is_tensor(v)}
        flags = {k: inputs[k] for k in ("fuse_vae_embedding_in_latents", "num_clean_prefix_frames")}
        result = arch.compute_loss(**inputs, lambda_video=trainer.lambda_video, lambda_action=trainer.lambda_action)
        result["loss"].backward()
        losses = {k: float(v.detach()) for k, v in result.items() if torch.is_tensor(v) and v.numel() == 1}
        return input_cpu, flags, losses

    online_inputs, online_flags, online_losses = forward_backward(False)
    grads = {name: p.grad.detach().clone() for name, p in arch.named_parameters() if p.grad is not None}
    actual_inputs, actual_flags, actual_losses = forward_backward(not args.online_control)
    assert actual_inputs.keys() == online_inputs.keys()
    assert all(torch.equal(online_inputs[k], actual_inputs[k]) for k in online_inputs)
    assert actual_flags == online_flags
    max_diff, diff_squared, ref_squared, unequal, count = 0.0, 0.0, 0.0, 0, 0
    for name, param in arch.named_parameters():
        if param.grad is None:
            assert name not in grads
            continue
        reference = grads.pop(name)
        value = param.grad.detach()
        unequal += int(not torch.equal(reference, value))
        difference = value.float() - reference.float()
        max_diff = max(max_diff, difference.abs().max().item())
        diff_squared += difference.square().sum(dtype=torch.float64).item()
        ref_squared += reference.float().square().sum(dtype=torch.float64).item()
        count += 1
    assert not grads
    report.update({
        "all_prepared_tensors_exact_equal": True, "conditioning_flags_equal": True,
        "online_losses": online_losses, "cached_losses": actual_losses,
        "gradient_tensors": count, "unequal_gradient_tensors": unequal,
        "gradient_max_abs_difference": max_diff,
        "gradient_relative_l2_difference": (diff_squared / max(ref_squared, 1e-30)) ** 0.5,
        "cache_entries": len(cache.entries),
        "cache_bytes": sum(t.numel() * t.element_size() for pair in cache.entries.values() for t in pair),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    write_report()
    print(json.dumps(report, indent=2), flush=True)
    assert online_losses == actual_losses, "Losses differ; inspect before production training"
    if not args.online_control:
        assert unequal == 0, "Gradients differ; inspect before production training"


if __name__ == "__main__":
    main()
