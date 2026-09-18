import copy

import torch
from omegaconf import OmegaConf

from openwam.train.utils.training_utils import build_cosine_scheduler
from openwam.train.utils.zero2_reshard import rescale_post_warmup_scheduler


def test_scheduler_preserves_next_optimizer_update_lrs():
    cfg = OmegaConf.create({"training": {"learning_rate": 1e-4, "warmup_ratio": 0.05, "lr_min_ratio": 0.01}})
    a = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    old = build_cosine_scheduler(a, total_opt_steps=42780, cfg=cfg, num_processes=6)
    for _ in range(4000):
        a.step()
        for _ in range(6):
            old.step()
    state = rescale_post_warmup_scheduler(old.state_dict(), 6, 2)
    b = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    new = build_cosine_scheduler(b, total_opt_steps=42780, cfg=cfg, num_processes=2)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    new.load_state_dict(state)
    for _ in range(1000):
        a.step()
        b.step()
        for _ in range(6):
            old.step()
        for _ in range(2):
            new.step()
        assert abs(a.param_groups[0]["lr"] - b.param_groups[0]["lr"]) < 1e-16
