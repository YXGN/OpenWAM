"""Exercise the real Nero reader / deploy seams, not a second inference implementation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.nero import UNIFY_MAP, absolute_actions
from openwam.deploy.model_loader import _build_normalizer
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.evaluation.nero_metrics import evaluate, physical_metrics, rotation_errors
from openwam.evaluation.nero_openloop import (
    PredictionRecorder,
    conditions_from_observation,
    current_observation,
    inference_config,
    load_episode,
    predict_chunk,
    validate_contract,
    window_starts,
)
from openwam.evaluation.nero_report import write_report
from tests.dataloader import test_nero_multitask as relative_tests

bucket = relative_tests.bucket


def test_episode_reading_matches_training_pixels_actions_state_and_prompt(bucket):
    root, dl, eefs, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    stats = np.load(dl.normalization_stats_path, allow_pickle=True).item()
    validate_contract(cfg, stats)
    normalizer = _build_normalizer(cfg, str(root / "prepared"))
    info = json.loads((root / "meta/info.json").read_text())
    reader = build_dataset(dl)
    prep = ObsPreprocessor.from_cfg(cfg)
    for eid, index, t, length, ep in ((0, 7, 7, 40, 0), (1001, 10, 2, 35, 1), (2, 11, 0, 33, 2)):
        eef, task, times = load_episode(root, info, {"episode_index": eid, "length": length})
        sample = reader[index]
        observation = current_observation(root, info, eid, t, eef[t], sample["prompt"])
        conditions = conditions_from_observation(prep, observation, 42)
        assert set(conditions) == {
            "first_frame_image",
            "proprio",
            "prompt",
            "seed",
            "num_frames",
            "video_num_frames",
            "height",
            "width",
        }
        assert conditions["prompt"] == f"task {ep}\nsecond instruction"
        np.testing.assert_array_equal(np.asarray(conditions["first_frame_image"][0]), np.asarray(sample["video"][0]))
        np.testing.assert_allclose(normalizer.normalize(conditions["proprio"]), sample["proprio"][0], atol=3e-7)
        gt = absolute_actions(normalizer.unnormalize(sample["action"]), eef[t])
        np.testing.assert_allclose(gt, eef[t + 1 : t + 33], atol=5e-7)
        np.testing.assert_array_equal(eef, eefs[ep])
        assert times[t + 1] > times[t] and task[t] == ep


def test_current_observation_never_decodes_future(monkeypatch):
    from PIL import Image

    calls = []

    def decode(path, indices, height, width):
        calls.append(indices)
        return [Image.new("RGB", (width, height))]

    monkeypatch.setattr("openwam.evaluation.nero_openloop.decode_video_frames", decode)
    info = {"chunks_size": 1000, "video_path": "{episode_index}/{video_key}.mp4"}
    current_observation(Path("/unused"), info, 7, 19, np.zeros(20), "task")
    assert calls == [[19], [19], [19]]


@pytest.mark.parametrize("length,expected", [(33, [0]), (35, [0, 2]), (65, [0, 32]), (66, [0, 32, 33])])
def test_full_windows_and_tail(length, expected):
    assert window_starts(length) == expected


@pytest.mark.parametrize("length,stride,starts", [(32, 32, None), (40, 0, None), (40, 32, [8]), (40, 32, [-1])])
def test_invalid_windows_rejected(length, stride, starts):
    with pytest.raises(ValueError):
        window_starts(length, stride, starts)


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_mode", "delta"),
        ("camera_layout", ["wrong"]),
        ("unify_action_map", ["0-19"]),
        ("binary_action_dims", [9, 19]),
        ("video_stride", 1),
        ("base_proprio", "global_pose"),
    ],
)
def test_reject_semantic_drift(bucket, key, value):
    _, dl, _, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    cfg.dataloader[key] = value
    stats = np.load(dl.normalization_stats_path, allow_pickle=True).item()
    with pytest.raises(ValueError):
        validate_contract(cfg, stats)


def test_recorder_preserves_engine_behavior_and_restores_on_failure(bucket):
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    _, dl, eefs, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    cfg = merge_deploy_cfg(cfg, inference_config(10))
    inner = _build_normalizer(cfg, str(Path(dl.normalization_stats_path).parent))
    recorder = PredictionRecorder(inner)
    np.testing.assert_array_equal(recorder._dst_index, UNIFY_MAP)
    np.testing.assert_array_equal(recorder.normalize(eefs[0][0]), inner.normalize(eefs[0][0]))

    class Architecture:
        normalizer = inner
        action_backbone = SimpleNamespace(shift_action=5.0)
        video_backbone = SimpleNamespace(shift_video=5.0)
        action_scheduler = FlowMatchScheduler("Wan")
        video_scheduler = FlowMatchScheduler("Wan")

        def apply_compile_optimizations(self, cfg):
            assert cfg.enabled is False

        def attach_normalizer(self, n):
            self.normalizer = n

        def normalize_deploy_proprio(self, state):
            return torch.from_numpy(self.normalizer.normalize(state))

        def generate(self, **kwargs):
            values = np.zeros((32, 80), dtype=np.float32)
            # Identity relative rotations and deliberately out-of-range gripper.
            values[:, UNIFY_MAP] = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 2], 2)
            return {"actions": self.normalizer.unnormalize(values)}

    arch = Architecture()
    engine = JointInferenceEngine(cfg, arch)
    conditions = {"proprio": eefs[0][0], "prompt": "test"}
    direct = engine.generate(conditions)["actions"]
    wrapped, raw, normalized = predict_chunk(engine, conditions)
    np.testing.assert_array_equal(wrapped, direct)
    assert raw[0, 9] > 0.1 and wrapped[0, 9] == pytest.approx(0.1)
    assert normalized[0, 9] == 2 and arch.normalizer is inner

    def fail(_):
        raise RuntimeError("probe failure")

    engine.generate = fail
    with pytest.raises(RuntimeError, match="probe failure"):
        predict_chunk(engine, conditions)
    assert arch.normalizer is inner


def test_absolute_loader_contract_and_separate_state_stats(bucket, tmp_path):
    from openwam.dataloader.utils.nero_absolute import CONTRACT

    _, dl, _, _ = bucket
    source = np.load(dl.normalization_stats_path, allow_pickle=True).item()
    payload = {
        "metadata": {"contract": CONTRACT},
        "eef": source["eef_relative"],
        "eef_state": source["eef_relative_state"],
    }
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    cfg.dataloader.type = "nero_multitask_absolute"
    cfg.dataloader.action_mode = "eef"
    np.save(tmp_path / "normalization_stats.npy", payload)
    validate_contract(cfg, payload)
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is not None
    payload["metadata"]["contract"] = source["metadata"]["contract"]
    np.save(tmp_path / "normalization_stats.npy", payload)
    with pytest.raises(ValueError, match="contract"):
        _build_normalizer(cfg, str(tmp_path))


def test_physical_metrics_known_errors_and_invalid_rotation(bucket):
    _, _, eefs, _ = bucket
    gt = eefs[0][1:33][None]
    pred = gt.copy()
    pred[..., 0] += 0.003
    pred[..., 1] += 0.004
    pred[..., 9] += 0.002
    m = physical_metrics(pred, gt)
    assert m["arms"]["left"]["position_mm"]["mean"] == pytest.approx(5, abs=1e-4)
    assert m["arms"]["left"]["gripper_mm"]["mean"] == pytest.approx(2, abs=1e-4)
    angles, valid = rotation_errors(np.zeros((1, 6)), gt[0, :1, 3:9])
    assert not valid[0] and np.isnan(angles[0])
    p = np.array([[0, 1, 0, -1, 0, 0]])
    angle, valid = rotation_errors(p, np.array([[1, 0, 0, 0, 1, 0]]))
    assert valid[0] and angle[0] == pytest.approx(90)


def test_report_and_metrics_preserve_all_dimensions_and_escape_prompt(bucket, tmp_path):
    _, _, eefs, _ = bucket
    gt, state = eefs[0][1:33][None], eefs[0][0:1]
    arrays = dict(
        gt=gt,
        pred=gt,
        state=state,
        native_gt=gt,
        native_pred=gt,
        normalized_pred=gt,
        normalized_gt=gt,
        episode=np.array([0]),
        anchor=np.array([0]),
        task=np.array([0]),
        times=np.arange(1, 33)[None] / 60,
        seconds=np.array([1.0]),
    )
    metrics = evaluate(arrays)
    assert len(metrics["prediction"]["per_dimension"]) == 20
    assert metrics["prediction"]["arms"]["left"]["position_mm"]["mean"] == 0
    assert metrics["inference_seconds"]["excluding_first"]["mean"] is None
    manifest = {"windows": [{"prompt": "</script><script>alert(1)</script>"}]}
    write_report(tmp_path, arrays, manifest, metrics)
    html = (tmp_path / "report.html").read_text()
    assert "</script><script>alert(1)" not in html
    assert "left.rot_c0x" in html and "right.gripper_m" in html
