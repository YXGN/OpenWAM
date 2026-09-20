"""Stdlib tests for deploy-time inference fast-path flag routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openwam.deploy.engine import JointInferenceEngine


def _config(*, master=True, graph=True, request_cache=True, compile_enabled=False):
    optimization = SimpleNamespace(
        inference_fast_path=SimpleNamespace(enabled=master),
        cuda_graph=SimpleNamespace(enabled=graph),
        request_context_cache=SimpleNamespace(enabled=request_cache),
        dit_cache=SimpleNamespace(enabled=False),
        compile=SimpleNamespace(enabled=compile_enabled),
        decode_video=False,
    )
    return SimpleNamespace(optimization=optimization)


def _engine(cfg):
    architecture = Mock()
    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.cfg = cfg
    engine.architecture = architecture
    engine._init_optimizations()
    return engine, architecture


class TestInferenceFastPathConfig(unittest.TestCase):
    def test_missing_master_switch_does_not_enable_new_paths(self):
        cfg = _config()
        del cfg.optimization.inference_fast_path
        _engine_instance, architecture = _engine(cfg)
        self.assertFalse(architecture._inference_fast_path_enabled)
        architecture.apply_request_context_cache_optimization.assert_called_once_with(enabled=False)
        architecture.apply_cuda_graph_optimization.assert_called_once_with(enabled=False)

    def test_defaults_enable_supported_new_hooks_and_leave_legacy_cache_off(self):
        engine, architecture = _engine(_config())

        self.assertTrue(architecture._inference_fast_path_enabled)
        architecture.apply_request_context_cache_optimization.assert_called_once_with(enabled=True)
        architecture.apply_cuda_graph_optimization.assert_called_once_with(enabled=True)
        architecture.apply_compile_optimizations.assert_called_once()
        self.assertIsNone(engine._dit_cache)

    def test_master_switch_disables_all_new_hooks(self):
        engine, architecture = _engine(_config(master=False, graph=True, request_cache=True))

        self.assertFalse(architecture._inference_fast_path_enabled)
        architecture.apply_request_context_cache_optimization.assert_called_once_with(enabled=False)
        architecture.apply_cuda_graph_optimization.assert_called_once_with(enabled=False)
        self.assertIsNone(engine._dit_cache)

    def test_compile_and_effective_graph_are_rejected_together(self):
        with self.assertRaisesRegex(ValueError, "compile.*cuda_graph"):
            _engine(_config(compile_enabled=True))

    def test_request_cache_subswitch_is_independent_of_graph(self):
        # The architecture receives the effective request-cache value while
        # CUDA Graph remains enabled by its own sub-switch.
        engine, architecture = _engine(_config(request_cache=False, graph=True))
        architecture.apply_request_context_cache_optimization.assert_called_once_with(enabled=False)
        architecture.apply_cuda_graph_optimization.assert_called_once_with(enabled=True)
        self.assertIsNone(engine._dit_cache)


if __name__ == "__main__":
    unittest.main()
