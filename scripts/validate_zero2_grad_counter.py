"""Actual single-rank ZeRO-2 gradient/AdamW/state-resume validation on GPU0."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import deepspeed
import torch
from deepspeed.runtime.zero import stage_1_and_2 as zero
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.train.utils.deepspeed_grad_counter import install_zero2_grad_counter  # noqa: E402
from openwam.train.utils.seeding import seed_everything  # noqa: E402


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32) for _ in range(12)])
        self.unused = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(lambda x, layer=layer: layer(x).tanh() + x, x, use_reentrant=False)
        return x


def flatten(tensors):
    return torch.cat([t.detach().float().cpu().reshape(-1) for t in tensors]) if tensors else torch.empty(0)


def compare(actual, expected):
    assert actual.shape == expected.shape
    d = actual.double() - expected.double()
    return {
        "equal": torch.equal(actual, expected),
        "max_abs": d.abs().max().item(),
        "relative_l2": (d.norm() / expected.double().norm().clamp_min(1e-30)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    root_output = Path(args.output)
    output = root_output / f"rank{os.environ.get('RANK', '0')}"
    output.mkdir(parents=True, exist_ok=True)
    assert not torch.cuda.is_available() if args.cpu else torch.cuda.device_count() == 1
    device = "cpu" if args.cpu else "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    original = zero.DeepSpeedZeroOptimizer.create_gradient_handling_hooks
    report = []

    def run(mode, bf16, accum, resume=None):
        zero.DeepSpeedZeroOptimizer.create_gradient_handling_hooks = original
        if mode != "original":
            install_zero2_grad_counter(audit=mode == "audit")
        seed_everything(123)
        model = Network()
        dtype = torch.bfloat16 if bf16 else torch.float32
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        config = {
            "train_micro_batch_size_per_gpu": 4,
            "train_batch_size": 4 * accum * world_size,
            "gradient_accumulation_steps": accum,
            "gradient_clipping": 1.0,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": bf16},
            "fp16": {"enabled": False},
            "zero_allow_untested_optimizer": True,
            "steps_per_print": 100000,
        }
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        start = 0
        checkpoint_dir = root_output / "resume_checkpoint"
        if resume:
            loaded, _ = engine.load_checkpoint(str(checkpoint_dir), tag="step4")
            assert loaded and engine.global_steps == 4
            start = 4
        snapshots = []
        for i in range(start, 5 * accum):
            torch.manual_seed(1000 + i + 10000 * int(os.environ.get("RANK", "0")))
            x = torch.randn(4, 32, device=device, dtype=dtype)
            target = torch.randn_like(x)
            prediction = engine(x)
            loss = (prediction.float() - target.float()).square().mean()
            engine.backward(loss)
            grads = [g for group in engine.optimizer.averaged_gradients.values() if group for g in group]
            grad = flatten(grads)
            engine.step()
            masters = engine.optimizer.single_partition_of_fp32_groups
            states = engine.optimizer.optimizer.state
            snapshot = {
                "loss": torch.tensor([loss.item()]),
                "gradient": grad,
                "master_weights": flatten(masters),
                "weights": flatten(list(model.parameters())),
            }
            for key in ("exp_avg", "exp_avg_sq"):
                values = [states[p][key] for p in masters if key in states[p]]
                snapshot[key] = flatten(values) if values else torch.empty(0)
            snapshots.append(snapshot)
            if mode == "original" and bf16 and accum == 1 and i == 3:
                engine.save_checkpoint(str(checkpoint_dir), tag="step4")
        counter = getattr(engine.optimizer, "_openwam_grad_counter", None)
        metrics = (
            None
            if counter is None
            else {"calls": counter.calls, "rebuilds": counter.rebuilds, "audit_checks": counter.audit_checks}
        )
        engine.destroy()
        del engine, optimizer, model, states, masters
        gc.collect()
        torch.cuda.empty_cache()
        return snapshots, metrics

    for bf16, accum in ((False, 1), (True, 1), (True, 2)):
        expected, _ = run("original", bf16, accum)
        repeated, _ = run("original", bf16, accum)
        for mode in ("cached", "audit"):
            actual, counts = run(mode, bf16, accum)
            comparisons = []
            for a, e, r in zip(actual, expected, repeated):
                comparisons.append(
                    {
                        key: {"cached_vs_original": compare(a[key], e[key]), "repeat_original": compare(r[key], e[key])}
                        for key in e
                        if e[key].numel()
                    }
                )
            row = {"bf16": bf16, "accumulation": accum, "mode": mode, "counts": counts, "steps": comparisons}
            row["all_bitwise_equal"] = all(
                v["cached_vs_original"]["equal"] for step in comparisons for v in step.values()
            )
            report.append(row)
            (output / "results.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True)
        if bf16 and accum == 1:
            resumed, counts = run("cached", bf16, accum, resume=True)
            row = {
                "resume_from_original": {k: compare(resumed[0][k], expected[4][k]) for k in expected[4]},
                "counts": counts,
            }
            report.append(row)
            (output / "results.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    zero.DeepSpeedZeroOptimizer.create_gradient_handling_hooks = original
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
