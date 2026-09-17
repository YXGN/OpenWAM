"""Focused regression tests for static inactive-action indices in generation."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

_ACTION_DIM = 8
_ACTIVE_INDICES = (0, 2, 5)
_INACTIVE_INDICES = (1, 3, 4, 6, 7)
_SCHEDULE = [(1000.0, 1000.0), (750.0, 750.0), (500.0, 500.0), (250.0, 250.0), (0.0, 0.0)]


def _make_probe_arch(dtype):
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def flow_step(model_output, sigma, sigma_next, sample):
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        external_encoder = None

        @staticmethod
        def preprocess_input_for_inference(**_kwargs):
            return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = _ACTION_DIM
        bridge_layers = ()
        uses_proprioception = False

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self._device = torch.device("cpu")
            self._dtype = dtype
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()
            self.seen_action_inputs = []

        def forward(self, noisy_actions, action_timestep, **kwargs):  # noqa: ARG002
            self.seen_action_inputs.append(noisy_actions.detach().clone())
            prediction = torch.empty_like(noisy_actions)
            prediction[..., list(_ACTIVE_INDICES)] = 1.0
            prediction[..., list(_INACTIVE_INDICES)] = 1000.0
            return torch.ones_like(kwargs["latents"]), prediction

    return _Arch()


def _old_bool_index_reference(seed, dtype, active_mask):
    """Run the pre-optimization update using the original bool indexing path."""
    action_latents = torch.randn(
        1,
        len(_SCHEDULE) - 1,
        _ACTION_DIM,
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    inactive_mask = ~active_mask
    inactive_noise = None
    for (_, t_a), (_, t_a_next) in zip(_SCHEDULE, _SCHEDULE[1:]):
        sigma_a = t_a / 1000.0
        sigma_a_next = t_a_next / 1000.0
        prediction = torch.empty_like(action_latents)
        prediction[..., list(_ACTIVE_INDICES)] = 1.0
        prediction[..., list(_INACTIVE_INDICES)] = 1000.0
        if inactive_mask.any() and inactive_noise is None:
            inactive_noise = action_latents[..., inactive_mask].detach().clone() / sigma_a
        action_latents = action_latents + prediction * (sigma_a_next - sigma_a)
        if inactive_mask.any():
            action_latents[..., inactive_mask] = inactive_noise * sigma_a_next
    return action_latents.squeeze(0).float().numpy()


class TestGenerateStaticInactiveIndices(TestCase):
    """Sparse and all-active explicit masks preserve the old trajectory."""

    def test_matches_bool_reference_and_uses_one_static_index(self):
        for dtype in (torch.float32, torch.bfloat16):
            for fast_path_enabled in (True, False):
                for active_indices in (_ACTIVE_INDICES, tuple(range(_ACTION_DIM))):
                    expected_nonzero_calls = 1 if fast_path_enabled and active_indices == _ACTIVE_INDICES else 0
                    with self.subTest(
                        dtype=dtype, fast_path_enabled=fast_path_enabled, active_indices=active_indices
                    ):
                        nonzero_calls = 0
                        original_nonzero = torch.nonzero

                        def _count_nonzero(*args, **kwargs):
                            nonlocal nonzero_calls
                            nonzero_calls += 1
                            return original_nonzero(*args, **kwargs)

                        active_mask = torch.zeros(_ACTION_DIM, dtype=torch.bool)
                        active_mask[list(active_indices)] = True
                        seed = 19
                        with patch.object(torch.compiler, "cudagraph_mark_step_begin", lambda: None), patch.object(
                            torch, "nonzero", side_effect=_count_nonzero
                        ):
                            arch = _make_probe_arch(dtype)
                            arch._inference_fast_path_enabled = fast_path_enabled
                            result = arch.generate(
                                schedule=_SCHEDULE,
                                prompt="",
                                num_frames=5,
                                action_num_frames=5,
                                seed=seed,
                                decode_video=False,
                                active_action_mask=active_mask,
                            )

                        np.testing.assert_array_equal(
                            result["actions"], _old_bool_index_reference(seed, dtype, active_mask)
                        )
                        self.assertEqual(nonzero_calls, expected_nonzero_calls)

                        if active_indices == _ACTIVE_INDICES:
                            eps0 = torch.randn(
                                1,
                                len(_SCHEDULE) - 1,
                                _ACTION_DIM,
                                dtype=dtype,
                                generator=torch.Generator(device="cpu").manual_seed(seed),
                            )
                            for step, seen in enumerate(arch.seen_action_inputs):
                                expected = eps0[..., list(_INACTIVE_INDICES)] * (_SCHEDULE[step][1] / 1000.0)
                                torch.testing.assert_close(
                                    seen[..., list(_INACTIVE_INDICES)], expected, rtol=0, atol=0
                                )
