"""Request-local synchronized wall-clock stages for opt-in inference diagnosis."""

import time

import torch


class InferenceTiming:
    def __init__(self, device):
        self.device = torch.device(device)
        self.milliseconds = {}
        self.text_cache_hit = None
        self._last = time.perf_counter()

    def mark(self, stage):
        # Stage boundaries are outside CUDA graph capture/replay bodies. Target
        # this model's device, not a process-default or another model's GPU.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.milliseconds[stage] = self.milliseconds.get(stage, 0.0) + (now - self._last) * 1000
        self._last = now
