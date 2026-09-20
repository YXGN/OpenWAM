"""Absolute EEF labels, normalization, cameras and separation from relative runs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.normalize import Normalizer, NormMode, load_mode_stats
from openwam.dataloader.utils.nero import GRIP_DIMS, ROT_DIMS, UNIFY_MAP, XYZ_DIMS, sha256
from openwam.dataloader.utils.stats_computation.nero_absolute_stats import prepare_absolute_dataset
from tests.dataloader import test_nero_multitask as relative_tests

REPO = relative_tests.REPO
bucket = relative_tests.bucket


@pytest.fixture(scope="module")
def absolute_bucket(bucket):
    root, relative_cfg, eefs, _ = bucket
    original_hash = sha256(relative_cfg.normalization_stats_path)
    output = root / "absolute"
    report = prepare_absolute_dataset(root, output)
    assert sha256(relative_cfg.normalization_stats_path) == original_hash
    cfg = OmegaConf.create(OmegaConf.to_container(relative_cfg))
    cfg.type = "nero_multitask_absolute"
    cfg.action_mode = "eef"
    cfg.normalization_stats_path = str(output / "normalization_stats.npy")
    return cfg, eefs, report


def test_absolute_future_labels_state_and_unchanged_observations(absolute_bucket, bucket):
    dl, eefs, report = absolute_bucket
    reader, relative = build_dataset(dl), build_dataset(bucket[1])
    normalizer = Normalizer(NormMode.MIN_MAX, load_mode_stats(dl.normalization_stats_path, "eef"))
    assert len(reader) == len(relative) == report["windows"] == 12
    for index, ep, offset in ((0, 0, 0), (7, 0, 7), (8, 1, 0), (10, 1, 2), (11, 2, 0)):
        sample, old = reader[index], relative[index]
        assert sample["action"].shape == (32, 80)
        assert sample["proprio"].shape == (1, 80)
        assert sample["action_mask"].sum() == 32 * 20
        assert sample["proprio_mask"].sum() == 20
        assert sample["action_mask"][0].nonzero().flatten().tolist() == UNIFY_MAP
        assert sample["prompt"] == old["prompt"] == f"task {ep}\nsecond instruction"
        np.testing.assert_allclose(
            normalizer.unnormalize(sample["action"].numpy()[:, UNIFY_MAP]),
            eefs[ep][offset + 1 : offset + 33],
            atol=5e-7,
        )
        np.testing.assert_allclose(
            sample["proprio"].numpy()[:, UNIFY_MAP], normalizer.normalize(eefs[ep][offset : offset + 1]), atol=3e-7
        )
        for key in ("proprio", "proprio_mask", "action_mask", "video_mask"):
            np.testing.assert_array_equal(sample[key], old[key])
        assert len(sample["video"]) == len(old["video"]) == 9
        for image, old_image in zip(sample["video"], old["video"]):
            np.testing.assert_array_equal(np.asarray(image), np.asarray(old_image))


def test_absolute_stats_exact_population_and_physical_bounds(absolute_bucket):
    dl, eefs, _ = absolute_bucket
    payload = np.load(dl.normalization_stats_path, allow_pickle=True).item()
    targets = np.concatenate([eef[t + 1 : t + 33] for eef in eefs for t in range(len(eef) - 32)])
    anchors = np.concatenate([eef[:-32] for eef in eefs])
    for key, values in (("eef", targets), ("eef_state", anchors)):
        block = payload[key]
        xyz = values[:, XYZ_DIMS].astype(np.float64)
        for field, expected in (
            ("min", xyz.min(0)),
            ("max", np.maximum(xyz.max(0), xyz.min(0) + 1e-6)),
            ("mean", xyz.mean(0)),
            ("std", xyz.std(0)),
        ):
            np.testing.assert_allclose(block[field][XYZ_DIMS], expected, atol=1e-7)
        np.testing.assert_array_equal(block["min"][ROT_DIMS], -1)
        np.testing.assert_array_equal(block["max"][ROT_DIMS], 1)
        np.testing.assert_allclose(block["min"][GRIP_DIMS], 0)
        np.testing.assert_allclose(block["max"][GRIP_DIMS], 0.1)
    # Episode tails appear in future labels but cannot serve as full-window anchors.
    assert payload["eef"]["max"][0] > payload["eef_state"]["max"][0]


def test_cross_representation_stats_rejected(absolute_bucket, bucket):
    absolute_cfg, _, _ = absolute_bucket
    relative_cfg = bucket[1]
    for source, other in ((absolute_cfg, relative_cfg), (relative_cfg, absolute_cfg)):
        cfg = OmegaConf.create(OmegaConf.to_container(source))
        cfg.normalization_stats_path = other.normalization_stats_path
        with pytest.raises(ValueError, match="contract"):
            build_dataset(cfg)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_absolute_dataset(bucket[0], Path(relative_cfg.normalization_stats_path).parent)


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_mode", "eef_relative"),
        ("camera_layout", ["observation.images.head_fpv"]),
        ("unify_action_map", ["0-19"]),
        ("normalize_mode", "mean-std"),
    ],
)
def test_absolute_reader_rejects_incompatible_config(absolute_bucket, key, value):
    cfg = OmegaConf.create(OmegaConf.to_container(absolute_bucket[0]))
    cfg[key] = value
    with pytest.raises(ValueError):
        build_dataset(cfg)


def test_absolute_deploy_does_not_add_anchor(absolute_bucket, bucket, monkeypatch):
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import _build_normalizer, repr_contract_from_cfg

    dl, eefs, _ = absolute_bucket
    cfg = OmegaConf.create(
        {
            "dataloader": OmegaConf.to_container(dl),
            "model": {"architecture": {"action_dim": 80}},
            "inference": {"denoise_steps": 1, "num_frames": 33, "video_num_frames": 9},
        }
    )
    normalizer = _build_normalizer(cfg, str(Path(dl.normalization_stats_path).parent))
    anchor = eefs[0][3].copy()
    sample = build_dataset(dl)[3]

    def generate(**kwargs):
        np.testing.assert_allclose(kwargs["proprio"], sample["proprio"].numpy(), atol=3e-7)
        anchor[0] += 10
        return {"actions": normalizer.unnormalize(sample["action"].numpy()), "video": None}

    architecture = SimpleNamespace(
        normalize_deploy_proprio=normalizer.normalize, generate=generate, video_scheduler=None, action_scheduler=None
    )
    monkeypatch.setattr("openwam.deploy.engine.make_schedule", lambda *args, **kwargs: [])
    engine = JointInferenceEngine(cfg, architecture=architecture)
    np.testing.assert_allclose(engine.generate({"proprio": anchor[None]})["actions"], eefs[0][4:36], atol=5e-7)
    assert repr_contract_from_cfg(cfg)["model_representation"] == "eef"
    assert repr_contract_from_cfg(cfg)["output_frame"] == "absolute_body_gripper_flange"
    with pytest.raises(ValueError, match="contract"):
        _build_normalizer(cfg, str(Path(bucket[1].normalization_stats_path).parent))


def test_absolute_recipe_is_fresh_and_independent():
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        absolute = compose(config_name="train_nero_absolute_4gpu")
        relative = compose(config_name="train_nero_target_8gpu")
    assert absolute.dataloader.type == "nero_multitask_absolute"
    assert absolute.dataloader.action_mode == "eef"
    assert absolute.training.resume_ckpt_path is None
    assert not absolute.training.get("resume_data_step_offset", 0)
    assert absolute.training.batch_size == relative.training.batch_size == 64
    assert absolute.training.gradient_accumulation_steps == relative.training.gradient_accumulation_steps == 1
    assert absolute.training.output_path != relative.training.output_path
    assert absolute.dataloader.normalization_stats_path != relative.dataloader.normalization_stats_path
    assert relative.dataloader.action_mode == "eef_relative"
