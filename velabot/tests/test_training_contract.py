# ruff: noqa: E402

from __future__ import annotations

# Test imports below intentionally follow a runtime sys.path insertion.
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TRAINING = Path(__file__).parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from generate_openloop_moon_cake import (
    TRAINING_LOG_OUTPUT_TEMPLATE,
    _metadata_config_snapshot,
    _trace_architecture_generate,
    requested_windows,
)
from moon_cake import (
    addback_delta_xyz,
    denormalize,
    map_to_unify,
    normalize,
    to_eef20,
    to_eef20_delta,
)
from openloop_validate_moon_cake import (
    build_parser as build_openloop_parser,
)
from openloop_validate_moon_cake import (
    evaluate_predictions,
    validate_prediction_semantics,
)
from prepare_moon_cake import compute_stats_from_arrays, preflight_reader
from train_moon_cake import build_command, build_parser, expanded_config


def _raw(n: int = 33) -> np.ndarray:
    row = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1, 0.05, 0.4, 0.5, 0.6, 0, 0, 0, 1, 0.05], dtype=np.float32)
    result = np.repeat(row[None], n, axis=0)
    offsets = np.arange(n, dtype=np.float32) * 0.001
    result[:, 0:3] += offsets[:, None]
    result[:, 8:11] += offsets[:, None]
    return result


def test_eef20_shape_rotation_gripper_and_unify_map():
    converted = to_eef20(_raw())
    assert converted.shape == (33, 20)
    np.testing.assert_allclose(converted[:, [3, 7, 13, 17]], 1.0, atol=1e-6)
    np.testing.assert_allclose(converted[:, [9, 19]], 0.0, atol=1e-6)
    unified = map_to_unify(converted[:1])
    assert unified.shape == (1, 80)
    np.testing.assert_allclose(unified[0, :10], converted[0, :10])
    np.testing.assert_allclose(unified[0, 34:44], converted[0, 10:20])
    np.testing.assert_allclose(unified[0, 10:34], 0.0)


def test_gripper_safety_reports_requested_and_effective_boundary_values():
    raw = _raw(1)
    raw[0, 7] = 0.101
    converted, events = to_eef20(raw, stream="action", return_events=True)
    assert len(events) == 1
    assert events[0].stream == "action"
    assert events[0].channel == 9
    assert events[0].requested > 1.0 and events[0].effective == 1.0
    assert events[0].reason == "native_gripper_range: opening_width_m outside [0,0.1]"
    assert converted[0, 9] == 1.0
    with pytest.raises(ValueError, match="accepted"):
        to_eef20(np.array([[*raw[0, :7], 0.102, *raw[0, 8:]]], dtype=np.float32))


def test_delta_changes_only_xyz_and_addback_is_one_state():
    action = _raw()
    state = _raw(1)[0]
    action_eef = to_eef20(action)
    delta = to_eef20_delta(action, state)
    np.testing.assert_allclose(delta[:, 0:3], action_eef[:32, 0:3] - action_eef[0, 0:3])
    keep = np.ones(20, dtype=bool)
    keep[[0, 1, 2, 10, 11, 12]] = False
    np.testing.assert_allclose(delta[:, keep], action_eef[:32, keep])
    reconstructed = addback_delta_xyz(delta, action_eef[0], action_mode="delta")
    np.testing.assert_allclose(reconstructed, action_eef[:32])


def test_stats_population_aliases_and_normalize_roundtrip():
    action = _raw()
    state = _raw()
    state[1:] = action[:-1]
    episodes = [{"episode_index": 17, "action": action, "state": state}]
    stats, report = compute_stats_from_arrays(episodes, action_mode="delta")
    assert report["windows"] == 1 and report["alignment_valid"]
    assert stats["delta_state"] is stats["eef"]
    assert stats["delta"]["quantiles_computed"] is False
    values = to_eef20(_raw())[:1]
    encoded = normalize(values, stats["eef"])
    decoded = denormalize(encoded, stats["eef"])
    np.testing.assert_allclose(decoded, values, atol=1e-6)


def test_stats_rejects_alignment_mismatch_without_repair():
    action = _raw()
    state = _raw()
    state[1:] = action[:-1]
    state[2, 0] += 0.25
    stats, report = compute_stats_from_arrays([{"action": action, "state": state}], action_mode="eef")
    assert not report["alignment_valid"]
    assert report["action_next_state_mismatch_rows"] == 1
    assert stats["eef"]["count"] == 66


def test_reader_preflight_reports_decoded_windows_and_shapes(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))

    class Tensor(np.ndarray):
        def isfinite(self):
            return np.isfinite(self)

    def tensor(shape):
        return np.zeros(shape, dtype=np.float32).view(Tensor)

    class Image:
        size = (320, 384)

    class Reader:
        def __init__(self):
            self._eps_df = (object(), object())
            self._cum_n_starts = np.array([0, 2, 4])

        def __len__(self):
            return 4

        def __getitem__(self, index):
            assert index in (0, 1, 2, 3)
            return {
                "video": [Image()] * 9,
                "action": tensor((32, 80)),
                "proprio": tensor((1, 80)),
            }

        def _video_cameras(self):
            return []

    report = preflight_reader(Reader(), max_samples=4)
    assert report["decoded_sample_windows"] == 4
    assert report["decoded_sample_indices"] == [0, 1, 2, 3]
    assert all(item["action_shape"] == [32, 80] for item in report["sample_shapes"])
    assert all(item["proprio_shape"] == [1, 80] for item in report["sample_shapes"])
    assert all(item["video_shape"] == [9, 384, 320] for item in report["sample_shapes"])


def test_launcher_requires_explicit_values_and_dry_run_does_not_spawn(tmp_path):
    parser = build_parser()
    code_root = tmp_path / "OpenWAM"
    (code_root / "scripts").mkdir(parents=True)
    (code_root / "scripts" / "train.py").write_text("# fixture\n")
    stats_path = tmp_path / "stats.npy"
    foundation = tmp_path / "foundation"
    stats_path.write_bytes(b"stats")
    foundation.write_bytes(b"foundation")
    args = parser.parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--stats",
            str(tmp_path / "stats.npy"),
            "--foundation",
            str(foundation),
            "--output",
            str(tmp_path / "out"),
            "--gpu",
            "4,6",
            "--code-root",
            str(code_root),
            "--action-mode",
            "delta",
            "--batch-size",
            "1",
            "--max-steps",
            "2",
            "--dry-run",
        ]
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        expanded_config(args)
    command = build_command(args)
    assert "dataloader.action_mode=delta" in command
    assert args.save_steps == 2000 and args.keep_last_k_ckpts == 1
    assert "training.save_steps=2000" in command
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", str(tmp_path)])


def test_openloop_is_row_aligned_and_rejects_invalid_rot6d():
    values = np.zeros((1, 32, 20), dtype=np.float32)
    values[..., [3, 7, 13, 17]] = 1.0
    report = evaluate_predictions(values, values)
    assert report["xyz_m"]["max"] == 0.0 and report["clip_or_filter"] is False
    invalid = values.copy()
    invalid[..., 3] = 0.0
    with pytest.raises(ValueError, match="rot6d"):
        evaluate_predictions(invalid, values)


def test_openloop_delta_prediction_semantics_cannot_be_used_for_eef_mode():
    parser = build_openloop_parser()
    args = parser.parse_args(
        [
            "--predictions",
            "predictions.npz",
            "--output",
            "report.json",
            "--dataset",
            "dataset",
            "--code-root",
            "code",
            "--stats",
            "stats.npy",
            "--action-mode",
            "eef",
            "--prediction-semantics",
            "xyz_delta_eef20",
            "--seed",
            "42",
            "--denoise-steps",
            "10",
            "--compile",
            "off",
        ]
    )
    assert args.action_mode == "eef"
    with pytest.raises(ValueError, match="require --action-mode delta"):
        validate_prediction_semantics(args.action_mode, args.prediction_semantics)


def test_openloop_window_selection_expands_one_episode_without_relabeling():
    assert requested_windows("78", "0,128,672") == [(78, 0), (78, 128), (78, 672)]
    with pytest.raises(ValueError, match="equal lengths"):
        requested_windows("78,79", "0,128,672")


def test_openloop_trace_preserves_architecture_call_and_records_scalar_kwargs():
    class Architecture:
        def generate(self, **kwargs):
            return kwargs

    class Engine:
        architecture = Architecture()

    class PolicyServer:
        engine = Engine()

    class Adapter:
        policy_server = PolicyServer()

    observed = _trace_architecture_generate(Adapter())
    result = Adapter.policy_server.engine.architecture.generate(
        seed=42,
        num_inference_steps=10,
        num_frames=49,
        action_num_frames=49,
        decode_video=False,
        dit_cache=None,
    )
    assert result["seed"] == 42
    assert observed == [
        {
            "seed": 42,
            "num_inference_steps": 10,
            "num_frames": 49,
            "action_num_frames": 49,
            "decode_video": False,
            "dit_cache_is_none": True,
        }
    ]


def test_generator_metadata_snapshot_is_strict_except_training_log_now():
    omega_conf = pytest.importorskip("omegaconf")
    OmegaConf = omega_conf.OmegaConf
    live_cfg = OmegaConf.create(
        {
            "project": {"output_dir": TRAINING_LOG_OUTPUT_TEMPLATE, "seed": 42},
            "training": {"batch_size": 1, "label": "${project.seed}"},
        }
    )
    unresolved, resolved, excluded = _metadata_config_snapshot(OmegaConf, live_cfg)
    assert unresolved["project"]["output_dir"] == TRAINING_LOG_OUTPUT_TEMPLATE
    assert "output_dir" not in resolved["project"]
    assert resolved["training"]["label"] == 42
    assert excluded == [
        {
            "path": "project.output_dir",
            "raw_value": TRAINING_LOG_OUTPUT_TEMPLATE,
            "reason": "training_log_template; Hydra now unavailable outside training",
        }
    ]
    assert OmegaConf.to_container(live_cfg, resolve=False)["project"]["output_dir"] == TRAINING_LOG_OUTPUT_TEMPLATE

    with pytest.raises(Exception, match="Unsupported interpolation type"):
        _metadata_config_snapshot(
            OmegaConf,
            OmegaConf.create({"project": {"output_dir": TRAINING_LOG_OUTPUT_TEMPLATE, "bad": "${unknown:foo}"}}),
        )
    with pytest.raises(Exception, match="Interpolation key"):
        _metadata_config_snapshot(
            OmegaConf,
            OmegaConf.create(
                {"project": {"output_dir": TRAINING_LOG_OUTPUT_TEMPLATE}, "runtime": "${project.output_dir}"}
            ),
        )
