"""Focused tests for request-local prepared context/KV interfaces."""

from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from openwam.model.action_backbone.separate_action_dit import (
    ActionDiT,
    ActionDiTState,
    BridgeCrossAttention,
)
from openwam.model.architectures.base import ActionState, BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.models.dit import CrossAttention


class TestPreparedContextKV(TestCase):
    def _cross_attention_inputs(self, dtype=torch.float32):
        module = CrossAttention(dim=8, num_heads=2).to(dtype=dtype)
        x = torch.randn(1, 3, 8, dtype=dtype)
        context = torch.randn(1, 4, 8, dtype=dtype)
        return module, x, context

    def test_wan_projected_path_is_exactly_original_path(self):
        module, x, context = self._cross_attention_inputs()
        expected = module(x, context)
        projected_kv = module.project_context_kv(context)
        actual = module(x, context, projected_kv=projected_kv)
        self.assertTrue(torch.equal(expected, actual))

    def test_action_projected_path_is_exactly_original_path(self):
        module = BridgeCrossAttention(hidden_dim=8, num_heads=2, attn_head_dim=4)
        x_action = torch.randn(1, 3, 8)
        context = torch.randn(1, 4, 8)
        expected = module(x_action, context)
        projected_kv = module.project_context_kv(context)
        actual = module(x_action, context, projected_kv=projected_kv)
        self.assertTrue(torch.equal(expected, actual))

    def test_bfloat16_projected_path_when_supported(self):
        module, x, context = self._cross_attention_inputs(torch.bfloat16)
        expected = module(x, context)
        actual = module(x, context, projected_kv=module.project_context_kv(context))
        self.assertTrue(torch.equal(expected, actual))

    def test_action_bfloat16_projected_path(self):
        module = BridgeCrossAttention(hidden_dim=8, num_heads=2, attn_head_dim=4).to(dtype=torch.bfloat16)
        x_action = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        context = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        expected = module(x_action, context)
        actual = module(x_action, context, projected_kv=module.project_context_kv(context))
        self.assertTrue(torch.equal(expected, actual))

    def test_context_a_b_a_is_not_reused(self):
        module, x, context_a = self._cross_attention_inputs()
        context_b = torch.randn_like(context_a)
        with mock.patch.object(module, "project_context_kv", wraps=module.project_context_kv) as project:
            output_a = module(x, context_a)
            output_b = module(x, context_b)
            output_a_again = module(x, context_a)
        self.assertEqual(project.call_count, 3)
        self.assertTrue(torch.equal(output_a, output_a_again))
        self.assertFalse(torch.equal(output_a, output_b))

    def test_provided_kv_skips_kv_projections_but_keeps_q_and_o(self):
        module, x, context = self._cross_attention_inputs()
        projected_kv = module.project_context_kv(context)
        with (
            mock.patch.object(module.k, "forward", wraps=module.k.forward) as k_forward,
            mock.patch.object(module.v, "forward", wraps=module.v.forward) as v_forward,
            mock.patch.object(module.q, "forward", wraps=module.q.forward) as q_forward,
            mock.patch.object(module.o, "forward", wraps=module.o.forward) as o_forward,
        ):
            module(x, context, projected_kv=projected_kv)
        self.assertEqual(k_forward.call_count, 0)
        self.assertEqual(v_forward.call_count, 0)
        self.assertEqual(q_forward.call_count, 1)
        self.assertEqual(o_forward.call_count, 1)

    def test_wan_image_input_rejects_projected_kv(self):
        module = CrossAttention(dim=8, num_heads=2, has_image_input=True)
        x = torch.randn(1, 3, 8)
        y = torch.randn(1, 261, 8)
        projected_kv = module.project_context_kv(y[:, 257:])
        with self.assertRaisesRegex(ValueError, "image input"):
            module(x, y, projected_kv=projected_kv)

    def test_action_state_selects_context_kv_for_layer(self):
        action = ActionDiT(
            action_dim=4,
            dim=8,
            ffn_dim=16,
            num_heads=2,
            num_layers=2,
            video_dim=8,
            bridge_layers=(0, 1),
            variant="joint_self_attn",
            text_dim=8,
            max_action_len=8,
        )
        context = torch.randn(1, 4, 8)
        context_kv = tuple(action.blocks[i].cross_attn.project_context_kv(context) for i in range(2))
        payload = ActionDiTState(
            x_action=torch.randn(1, 3, 8),
            t_mod=torch.zeros(1, 6, 8),
            t_embed=torch.zeros(1, 8),
            action_freqs=action.freqs[:3],
            context=context,
            context_mask=torch.ones(1, 3, 4, dtype=torch.bool),
            context_kv=context_kv,
        )
        astate = SimpleNamespace(payload=payload)
        post_state = (
            payload.x_action,
            torch.zeros(1, 1, 8),
            torch.zeros(1, 1, 8),
            torch.zeros(1, 1, 8),
            torch.zeros(1, 1, 8),
        )
        with mock.patch.object(
            action.blocks[1].cross_attn, "forward", wraps=action.blocks[1].cross_attn.forward
        ) as cross:
            action.post_attn_at_layer_for_compile(1, astate, torch.zeros_like(payload.x_action), post_state)
        self.assertIs(cross.call_args.kwargs["projected_kv"], context_kv[1])

    def test_prepare_state_uses_prepared_context_without_embedding(self):
        action = ActionDiT(
            action_dim=4,
            dim=8,
            ffn_dim=16,
            num_heads=2,
            num_layers=1,
            video_dim=8,
            bridge_layers=(0,),
            variant="joint_self_attn",
            text_dim=8,
            max_action_len=8,
        )
        prepared = torch.randn(1, 4, 8)
        prepared_mask = torch.ones(1, 3, 4, dtype=torch.bool)
        with mock.patch.object(action.text_embedding, "forward", side_effect=AssertionError("raw embedding used")):
            state = action.prepare_state(
                torch.randn(1, 3, 4),
                torch.ones(1),
                prepared_context=(prepared, prepared_mask),
            )
        self.assertIs(state.payload.context, prepared)
        self.assertIs(state.payload.context_mask, prepared_mask)

    def test_prepare_state_without_prepared_context_uses_original_embedding(self):
        action = ActionDiT(
            action_dim=4,
            dim=8,
            ffn_dim=16,
            num_heads=2,
            num_layers=1,
            video_dim=8,
            bridge_layers=(0,),
            variant="joint_self_attn",
            text_dim=8,
            max_action_len=8,
        )
        raw_context = torch.randn(1, 4, 8)
        with mock.patch.object(action.text_embedding, "forward", wraps=action.text_embedding.forward) as embedding:
            state = action.prepare_state(torch.randn(1, 3, 4), torch.ones(1), context=raw_context)
        self.assertEqual(embedding.call_count, 1)
        self.assertIsNot(state.payload.context, raw_context)

    def test_request_cache_is_fresh_and_accepts_ti2v_clean_prefix(self):
        architecture = DualSystemSelfAttnArchitecture.__new__(DualSystemSelfAttnArchitecture)
        architecture._request_context_cache_enabled = True
        architecture._request_context_cache_status = {
            "enabled": True,
            "last_reason": "enabled",
            "requests_prepared": 0,
            "cache_hits": 0,
        }
        with mock.patch.object(BaseWAMArchitecture, "generate", return_value={}) as base_generate:
            architecture.generate(cfg_scale=1.0)
            architecture.generate(cfg_scale=1.0)
        first = base_generate.call_args_list[0].kwargs["_inference_context_cache"]
        second = base_generate.call_args_list[1].kwargs["_inference_context_cache"]
        self.assertIsNot(first, second)

        class _WanLike:
            __module__ = "openwam.model.video_backbone.wan_backbone"
            training = False
            dit = SimpleNamespace(has_image_input=False)

        class _ActionLike:
            variant = "joint_self_attn"
            training = False

        architecture.training = False
        architecture.video_backbone = _WanLike()
        architecture.action_backbone = _ActionLike()
        with torch.no_grad():
            reason = architecture._request_context_cache_unsupported_reason(
                {"first_frame_latents": torch.zeros(1)},
                torch.zeros(1, 2, 4),
                False,
                False,
            )
        self.assertIsNone(reason)

    def test_request_cache_reuses_context_for_ten_ti2v_steps(self):
        class _Cross:
            def __init__(self, offset):
                self.offset = offset
                self.calls = 0

            def project_context_kv(self, context):
                self.calls += 1
                value = context + self.offset
                return value, value

        class _Video:
            __module__ = "openwam.model.video_backbone.wan_backbone"
            training = False
            num_layers = 2

            def __init__(self):
                self.cross = [_Cross(1), _Cross(2)]
                self.dit = SimpleNamespace(
                    has_image_input=False,
                    blocks=[SimpleNamespace(cross_attn=part) for part in self.cross],
                )

            def prepare(self, **kwargs):
                prepared = kwargs.get("prepared_context")
                context, mask = prepared if prepared is not None else (kwargs["context"], kwargs.get("context_mask"))
                return BlockLoopState(
                    hidden_states=torch.zeros(1, 3, 4),
                    time_mod=torch.zeros(1, 6, 4),
                    rope_freqs=torch.zeros(3, 1, 1),
                    context=context,
                    context_mask=mask,
                    extras={"dit": self.dit},
                )

            @staticmethod
            def finalize(state):
                return state.hidden_states

        class _Action:
            variant = "joint_self_attn"
            training = False

            def __init__(self):
                self.cross = [_Cross(3), _Cross(4)]
                self.blocks = [SimpleNamespace(cross_attn=part) for part in self.cross]

            def prepare_state(self, noisy, timestep, **kwargs):
                prepared = kwargs.get("prepared_context")
                context, mask = prepared if prepared is not None else (kwargs["context"], kwargs.get("context_mask"))
                payload = SimpleNamespace(context=context, context_mask=mask, context_kv=None, x_action=noisy)
                return ActionState(action_latents=noisy, timestep=timestep, payload=payload)

            @staticmethod
            def extract_prediction(state):
                return state.payload.x_action

        class _Driver:
            @staticmethod
            def run_joint_loop(vstate, astate, **_kwargs):
                return vstate, astate

        architecture = DualSystemSelfAttnArchitecture.__new__(DualSystemSelfAttnArchitecture)
        architecture.training = False
        architecture.video_backbone = _Video()
        architecture.action_backbone = _Action()
        architecture._mot_driver = _Driver()
        architecture._gradient_checkpointing_layers = None
        architecture._mot_cuda_graph_runner = None
        architecture._compiled_mot_run_joint_loop = None
        architecture._request_context_cache_enabled = True
        architecture._request_context_cache_status = {
            "enabled": True,
            "last_reason": "enabled",
            "requests_prepared": 0,
            "cache_hits": 0,
        }
        cache = {}
        context = torch.zeros(1, 2, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(10):
                architecture.forward(
                    torch.zeros(1, 2, 4),
                    torch.ones(1),
                    context=context,
                    context_mask=mask,
                    first_frame_latents=torch.zeros(1),
                    _inference_context_cache=cache,
                )
        self.assertEqual(architecture._request_context_cache_status["requests_prepared"], 1)
        self.assertEqual(architecture._request_context_cache_status["cache_hits"], 9)
        self.assertEqual([cross.calls for cross in architecture.video_backbone.cross], [1, 1])
        self.assertEqual([cross.calls for cross in architecture.action_backbone.cross], [1, 1])
        self.assertNotIn("vstate", cache)
        self.assertNotIn("astate", cache)


if __name__ == "__main__":
    import unittest

    unittest.main()
