"""Lossy reference normalization must not alter physical GT or production state."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.dataloader.utils.nero import UNIFY_MAP, relative_actions, sha256
from openwam.dataloader.utils.normalization import NORM_EPS
from openwam.deploy.model_loader import _build_normalizer
from openwam.evaluation import nero_prompt_openloop as frozen
from openwam.evaluation.nero_prompt_openloop_ood import (
    FROZEN_MAIN_SHA256,
    SOURCE_FILES,
    bounded_reference_audit,
    clipping_diagnostics,
    predict_joint,
    score_case,
)
from tests.dataloader import test_nero_multitask as relative_tests

bucket = relative_tests.bucket


def setup_normalizer(bucket):
    root, dl, eefs, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    stats = np.load(dl.normalization_stats_path, allow_pickle=True).item()
    normalizer = _build_normalizer(cfg, str(root / "prepared"))
    state = eefs[0][0].copy()
    native_gt = relative_actions(eefs[0][1:33], state)
    return normalizer, stats, state, native_gt


def test_in_range_reference_matches_old_roundtrip(bucket):
    normalizer, stats, state, native_gt = setup_normalizer(bucket)
    normalized, arrays, report = bounded_reference_audit(normalizer, native_gt, state, stats, "eef_relative")
    unified = np.zeros((32, 80), dtype=np.float32)
    unified[:, UNIFY_MAP] = normalized
    np.testing.assert_allclose(normalizer.unnormalize(unified), native_gt, atol=1e-6)
    np.testing.assert_allclose(arrays["clipped_native_gt"], native_gt, atol=1e-6)
    assert report["physical_gt_scoring"].startswith("Original unmodified")


def test_out_of_range_expected_clipping_keeps_physical_gt_and_state(bucket):
    normalizer, stats, state, native_gt = setup_normalizer(bucket)
    native_gt[:, 0] = stats["eef_relative"]["max"][0] + 0.01
    native_gt[:, 10] = stats["eef_relative"]["min"][10] - 0.02
    state[0] = stats["eef_relative_state"]["max"][0] + 0.05
    original_gt, original_state = native_gt.copy(), state.copy()
    normalized, arrays, report = bounded_reference_audit(normalizer, native_gt, state, stats, "eef_relative")
    np.testing.assert_array_equal(native_gt, original_gt)
    np.testing.assert_array_equal(state, original_state)
    assert np.all(normalized[:, 0] == 1) and np.all(normalized[:, 10] == -1)
    assert np.all(arrays["normalized_gt_unclipped"][:, 0] > 1)
    assert report["action_reference"]["per_dimension"]["left.x_m"]["clipped_count"] == 32
    assert report["action_reference"]["per_dimension"]["left.x_m"]["max_abs_clip_loss_mm"] == pytest.approx(
        10, abs=0.001
    )
    assert report["action_reference"]["per_dimension"]["right.x_m"]["max_abs_clip_loss_mm"] == pytest.approx(
        20, abs=0.001
    )
    assert report["state_input"]["per_dimension"]["left.x_m"]["max_abs_clip_loss_mm"] == pytest.approx(50, abs=0.001)
    np.testing.assert_array_equal(normalizer.normalize(state)[UNIFY_MAP], arrays["normalized_state_clipped"])
    # A prediction at the range boundary still incurs its true 10/20 mm errors;
    # it must not receive zero error by replacing the reference with clipped GT.
    video = np.zeros((9, 384, 320, 3), dtype=np.uint8)
    prediction = {
        "pred": arrays["clipped_native_gt"],
        "native_pred": arrays["clipped_native_gt"],
        "normalized_pred": normalized,
        "video": video,
    }
    case = {
        "prompt_class": "original",
        "feasibility": "feasible",
        "horizon_observable": "yes",
        "gt_reference": "original",
    }
    metrics = score_case(case, prediction, {"state": state, "gt": native_gt, "video": video})
    assert metrics["action_gt_metrics"]["prediction"]["arms"]["left"]["position_mm"]["mean"] == pytest.approx(
        10, abs=0.001
    )
    assert metrics["action_gt_metrics"]["prediction"]["arms"]["right"]["position_mm"]["mean"] == pytest.approx(
        20, abs=0.001
    )


def test_wrong_inverse_or_state_normalization_still_fails(bucket):
    normalizer, stats, state, gt = setup_normalizer(bucket)
    wrong_inverse = SimpleNamespace(
        unnormalize=lambda x: normalizer.unnormalize(x) + 0.002, normalize=normalizer.normalize
    )
    with pytest.raises(AssertionError):
        bounded_reference_audit(wrong_inverse, gt, state, stats, "eef_relative")
    wrong_state = SimpleNamespace(
        unnormalize=normalizer.unnormalize, normalize=lambda x: normalizer.normalize(x) + 0.02
    )
    with pytest.raises(AssertionError):
        bounded_reference_audit(wrong_state, gt, state, stats, "eef_relative")


def test_epsilon_floored_constant_dimension_uses_operational_inverse():
    stats = {"min": np.zeros(20, dtype=np.float32), "max": np.zeros(20, dtype=np.float32)}
    values = np.zeros(20, dtype=np.float32)
    values[0], values[1] = NORM_EPS * 0.5, NORM_EPS * 2
    _, clipped, inverse, mask, loss, report = clipping_diagnostics(values, stats)
    assert not mask[0] and mask[1]
    assert clipped[0] == pytest.approx(0) and clipped[1] == 1
    assert inverse[0] == pytest.approx(NORM_EPS * 0.5) and inverse[1] == pytest.approx(NORM_EPS)
    assert loss[1] == pytest.approx(NORM_EPS)
    assert report["per_dimension"]["left.x_m"]["above_training_max_count"] == 1


def test_frozen_inference_helpers_and_main_bytes_remain_unchanged():
    assert predict_joint is frozen.predict_joint
    assert score_case is frozen.score_case
    assert sha256(Path(frozen.__file__)) == FROZEN_MAIN_SHA256
    assert "openwam/evaluation/nero_prompt_openloop_ood.py" in SOURCE_FILES
    assert "openwam/evaluation/nero_prompt_openloop.py" in SOURCE_FILES
