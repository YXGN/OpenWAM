"""Unit tests for explicit MoT CUDA Graph dispatch and state ownership."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from openwam.model.architectures.dual_system.mot_cuda_graph import MoTCudaGraphRunner


class TestMoTCudaGraphRunner(unittest.TestCase):
    def test_cpu_uses_eager_and_preserves_metadata(self):
        from openwam.model.architectures.base import ActionState
        from openwam.model.video_backbone.base import BlockLoopState

        class _CpuDriver:
            vb = SimpleNamespace(training=False, video_attention_mask_mode="first_frame_causal")
            ab = SimpleNamespace(training=False, variant="joint_self_attn")
            attention_mask_mode = "action_sees_video"
            num_layers = 1

            @staticmethod
            def run_joint_loop(vstate, astate, **_kwargs):
                vstate.hidden_states = vstate.hidden_states + 1
                astate.payload.x_action = astate.payload.x_action + 1
                return vstate, astate

        context = torch.zeros(1, 2, 4)
        vstate = BlockLoopState(
            hidden_states=torch.zeros(1, 3, 4),
            time_mod=torch.zeros(1, 6, 4),
            rope_freqs=torch.zeros(3, 1, 1),
            context=context,
            context_mask=torch.ones(1, 2, dtype=torch.bool),
            grid_frames=3,
            grid_height=1,
            grid_width=1,
        )
        payload = SimpleNamespace(
            x_action=torch.zeros(1, 2, 4),
            t_mod=torch.zeros(1, 6, 4),
            action_freqs=torch.zeros(2, 1, 1),
            context=context.clone(),
            context_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        astate = ActionState(payload=payload)
        driver = _CpuDriver()
        context_before = vstate.context.clone()
        action_context_before = astate.payload.context.clone()
        runner = MoTCudaGraphRunner(driver)

        with torch.no_grad():
            runner.run(vstate, astate)

        status = runner.status()
        self.assertFalse(status["captured"])
        self.assertEqual(status["last_backend"], "eager")
        self.assertIn(status["reason"], {"cuda_unavailable", "non_cuda_state"})
        self.assertTrue(torch.equal(vstate.hidden_states, torch.ones_like(vstate.hidden_states)))
        self.assertTrue(torch.equal(astate.payload.x_action, torch.ones_like(astate.payload.x_action)))
        torch.testing.assert_close(vstate.context, context_before)
        torch.testing.assert_close(astate.payload.context, action_context_before)
        json.dumps(status)

    def test_signature_is_json_safe_and_includes_layout_contract(self):
        from openwam.model.architectures.base import ActionState
        from openwam.model.video_backbone.base import BlockLoopState

        class _Driver:
            attention_mask_mode = "action_sees_video"
            num_layers = 2
            vb = SimpleNamespace(video_attention_mask_mode="first_frame_causal")

        vstate = BlockLoopState(
            hidden_states=torch.zeros(1, 4, 8),
            time_mod=torch.zeros(1, 6, 8),
            rope_freqs=torch.zeros(4, 1, 1),
            context=torch.zeros(1, 3, 8),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
            grid_frames=4,
            grid_height=1,
            grid_width=1,
        )
        payload = SimpleNamespace(
            x_action=torch.zeros(1, 2, 8),
            t_mod=torch.zeros(1, 6, 8),
            action_freqs=torch.zeros(2, 1, 1),
            context=torch.zeros(1, 3, 8),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        astate = ActionState(action_latents=torch.zeros(1, 2, 4), timestep=torch.zeros(1), payload=payload)
        signature = MoTCudaGraphRunner._state_signature(vstate, astate, _Driver())
        self.assertEqual(signature["video"]["hidden_states"]["stride"], [32, 8, 1])
        self.assertEqual(signature["grid"]["frames"], 4)
        self.assertEqual(signature["driver_layers"], 2)
        json.dumps(signature)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for explicit graph capture")
    def test_cuda_capture_replays_and_updates_a_b_a_inputs(self):
        from openwam.model.architectures.base import ActionState
        from openwam.model.video_backbone.base import BlockLoopState

        class _WanLikeVideo:
            __module__ = "openwam.model.video_backbone.wan_backbone"
            training = False
            video_attention_mask_mode = "first_frame_causal"

        class _Action:
            training = False
            variant = "joint_self_attn"

        class _Driver:
            vb = _WanLikeVideo()
            ab = _Action()
            attention_mask_mode = "action_sees_video"
            num_layers = 1

            @staticmethod
            def run_joint_loop(vstate, astate, **_kwargs):
                vstate.hidden_states = vstate.hidden_states + vstate.context[:, :1, :]
                astate.payload.x_action = astate.payload.x_action + astate.payload.context[:, :1, :]
                if vstate.extras.get("context_kv") is not None:
                    vstate.hidden_states = vstate.hidden_states + vstate.extras["context_kv"][0][0][:, :1, :]
                if getattr(astate.payload, "context_kv", None) is not None:
                    astate.payload.x_action = astate.payload.x_action + astate.payload.context_kv[0][1][:, :1, :]
                return vstate, astate

        def _states(context_value, video_tokens=4, kv_value=None):
            context = torch.full((1, 2, 8), context_value, device="cuda", dtype=torch.bfloat16)
            context_mask = torch.ones((1, 2), device="cuda", dtype=torch.bool).unsqueeze(1).expand(1, 3, 2)
            context_kv = None
            if kv_value is not None:
                key = torch.full((1, 2, 8), kv_value, device="cuda", dtype=torch.bfloat16)
                value = torch.full((1, 2, 8), kv_value + 1, device="cuda", dtype=torch.bfloat16)
                context_kv = ((key, value),)
            vstate = BlockLoopState(
                hidden_states=torch.zeros(1, video_tokens, 8, device="cuda", dtype=torch.bfloat16),
                time_mod=torch.zeros(1, 6, 8, device="cuda", dtype=torch.bfloat16),
                rope_freqs=torch.zeros(4, 1, 1, device="cuda", dtype=torch.bfloat16),
                context=context,
                context_mask=context_mask,
                grid_frames=4,
                grid_height=1,
                grid_width=1,
                extras={"context_kv": context_kv},
            )
            payload = SimpleNamespace(
                x_action=torch.zeros(1, 3, 8, device="cuda", dtype=torch.bfloat16),
                t_mod=torch.zeros(1, 6, 8, device="cuda", dtype=torch.bfloat16),
                action_freqs=torch.zeros(3, 1, 1, device="cuda", dtype=torch.bfloat16),
                context=context,
                context_mask=context_mask,
                context_kv=context_kv,
            )
            return vstate, ActionState(payload=payload)

        runner = MoTCudaGraphRunner(_Driver())
        with torch.no_grad():
            v_a, a_a = runner.run(*_states(1.0, kv_value=10.0))
            saved_a = v_a.hidden_states.clone()
            eager_v, eager_a = _states(1.0, kv_value=10.0)
            eager_v, eager_a = runner.driver.run_joint_loop(eager_v, eager_a)
            v_b, _ = runner.run(*_states(1.0, kv_value=20.0))
            v_a2, _ = runner.run(*_states(1.0, kv_value=10.0))
        self.assertTrue(torch.equal(saved_a, eager_v.hidden_states))
        self.assertTrue(torch.equal(a_a.payload.x_action, eager_a.payload.x_action))
        self.assertTrue(torch.equal(saved_a, v_a.hidden_states))
        self.assertFalse(torch.equal(saved_a, v_b.hidden_states))
        self.assertTrue(torch.equal(saved_a, v_a2.hidden_states))
        self.assertEqual(runner.status()["capture_count"], 1)
        self.assertEqual(runner.status()["replay_count"], 3)
        self.assertIsNotNone(a_a.payload.x_action)

        with torch.no_grad():
            mismatch_v, mismatch_a = runner.run(*_states(1.0, video_tokens=5, kv_value=10.0))
        self.assertEqual(runner.status()["capture_count"], 1)
        self.assertEqual(runner.status()["reason"], "static_signature_mismatch")
        self.assertEqual(mismatch_v.hidden_states.shape[1], 5)
        self.assertIsNotNone(mismatch_a.payload.x_action)

        runner.clear("test_reset")
        self.assertFalse(runner.status()["captured"])
        self.assertEqual(runner.status()["reason"], "test_reset")
        with torch.no_grad():
            runner.run(*_states(1.0, kv_value=10.0))
        self.assertEqual(runner.status()["capture_count"], 2)

    def test_capture_error_is_propagated(self):
        runner = MoTCudaGraphRunner(object())
        with (
            patch.object(runner, "_unsupported_reason", return_value=None),
            patch.object(runner, "_state_signature", return_value={}),
            patch.object(runner, "_capture", side_effect=RuntimeError("capture failed")),
            self.assertRaises(RuntimeError),
        ):
            runner.run(None, None)


if __name__ == "__main__":
    unittest.main()
