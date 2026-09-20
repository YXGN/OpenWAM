"""Opt-in stage timing includes GPU waits once and targets the model device."""

from types import SimpleNamespace

import pytest
import torch

from openwam.model import inference_timing


def test_cuda_stages_include_wait_and_accumulate_without_overlap(monkeypatch):
    clock = [0.0]
    synced = []
    monkeypatch.setattr(inference_timing, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    def synchronize(device):
        synced.append(device)
        clock[0] += 0.002

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    timing = inference_timing.InferenceTiming("cuda:3")
    for name in ("setup", "denoise", "setup"):
        clock[0] += 0.010
        timing.mark(name)
    assert synced == [torch.device("cuda:3")] * 3
    assert timing.milliseconds == pytest.approx({"setup": 24.0, "denoise": 12.0})
    assert sum(timing.milliseconds.values()) == pytest.approx(clock[0] * 1000)


def test_cpu_stage_does_not_touch_cuda(monkeypatch):
    def forbidden(*args):
        raise AssertionError("CPU timing must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    timing = inference_timing.InferenceTiming("cpu")
    timing.mark("setup")
    assert timing.milliseconds["setup"] >= 0
