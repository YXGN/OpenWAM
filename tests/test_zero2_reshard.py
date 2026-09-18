import copy

import pytest
import torch
from omegaconf import OmegaConf

from openwam.train.utils.checkpointing import compute_resume_position
from openwam.train.utils.training_utils import build_cosine_scheduler
from openwam.train.utils.zero2_reshard import overlaps, rescale_post_warmup_scheduler, reslice


@pytest.mark.parametrize("world", [2, 4])
def test_resharding_preserves_all_values_across_old_boundaries(world):
    full = torch.arange(120, dtype=torch.float32)
    source = list(full.chunk(6))
    result = [reslice(source, i * 120 // world, (i + 1) * 120 // world) for i in range(world)]
    assert torch.equal(torch.cat(result), full)
    for i, saved in enumerate(result):
        assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
        for rank, source_slice, target_slice in overlaps(20, i * 120 // world, (i + 1) * 120 // world):
            assert torch.equal(saved[target_slice], source[rank][source_slice])


@pytest.mark.parametrize("new_batch", [96, 128])
def test_four_rank_scheduler_keeps_sample_trajectory(new_batch):
    cfg = OmegaConf.create({"training": {"learning_rate": 1e-4}})
    old_opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    old = build_cosine_scheduler(old_opt, total_opt_steps=42780, cfg=cfg, num_processes=6)
    for _ in range(4000):
        old_opt.step()
        for _ in range(6):
            old.step()
    new_opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    new = build_cosine_scheduler(new_opt, total_opt_steps=42780, cfg=cfg, num_processes=4)
    new_opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
    new.load_state_dict(rescale_post_warmup_scheduler(old.state_dict(), 6, 4, 96, new_batch))
    for _ in range(1000):
        old_opt.step()
        new_opt.step()
        # Four old updates / three new updates both process 384 samples.
        for _ in range(6 * (4 if new_batch == 128 else 1)):
            old.step()
        for _ in range(4 * (3 if new_batch == 128 else 1)):
            new.step()
        assert abs(old_opt.param_groups[0]["lr"] - new_opt.param_groups[0]["lr"]) < 1e-16


def test_batch_change_preserves_consumed_data_and_optimizer_clock():
    assert compute_resume_position(4000, 6417, 1, data_step_offset=1000) == (0, 3000, 4000)
    assert compute_resume_position(10000, 6417, 1, data_step_offset=1000) == (1, 2583, 10000)
    assert compute_resume_position(33085, 6417, 1, data_step_offset=1000) == (5, 0, 33085)
