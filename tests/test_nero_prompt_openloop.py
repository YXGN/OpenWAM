"""Evaluation integrity: paired inputs, genuine joint outputs, and legal GT masks."""

import csv
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.utils.nero import UNIFY_MAP, sha256
from openwam.evaluation.nero_prompt_openloop import (
    COLUMNS,
    assert_manifest_compatible,
    atomic_json,
    input_identity,
    population_identity,
    predict_joint,
    read_cases,
    run,
    score_case,
    verify_completed,
)
from tests.dataloader import test_nero_multitask as relative_tests

bucket = relative_tests.bucket


def row(**changes):
    result = dict(
        zip(
            COLUMNS,
            [
                "group_a",
                "0",
                "0",
                "before_pick",
                "original",
                "task 0\nsecond instruction",
                "none",
                "Right hand approaches the visible target",
                "feasible",
                "original",
                "yes",
                "ready",
                "visual check recorded",
            ],
            strict=True,
        )
    )
    result.update(changes)
    return result


def write_cases(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "change,match",
    [
        ({"status": "draft"}, "not ready"),
        ({"feasibility": "pending"}, "unreviewed"),
        ({"horizon_observable": "pending"}, "review horizon"),
        ({"prompt_class": "counterfactual"}, "not a valid target"),
        ({"prompt": ""}, "only empty"),
        (
            {"prompt_class": "empty", "prompt": " ", "gt_reference": "none", "feasibility": "not_applicable"},
            "exactly empty",
        ),
    ],
)
def test_reject_unreviewed_or_invalid_semantic_gt(tmp_path, change, match):
    path = write_cases(tmp_path / "cases.csv", [row(**change)])
    with pytest.raises(ValueError, match=match):
        read_cases(path)


def test_group_requires_same_observation_and_original_control(tmp_path):
    path = tmp_path / "cases.csv"
    alternate = row(
        prompt_class="counterfactual", prompt="Other feasible task", gt_reference="none", change_axis="hand"
    )
    write_cases(path, [row(), dict(alternate, anchor_frame="1")])
    with pytest.raises(ValueError, match="mixes observations"):
        read_cases(path)
    write_cases(path, [alternate])
    with pytest.raises(ValueError, match="original control"):
        read_cases(path)
    write_cases(path, [row(), alternate, row(status="skip", notes="Not visible")])
    cases, skipped = read_cases(path)
    assert len(cases) == 2 and len(skipped) == 1
    assert cases[0]["prompt"] == "task 0\nsecond instruction"
    assert cases[0]["case_id"] != cases[1]["case_id"]


def test_conditions_reject_future_inputs_and_state_mismatch():
    observation = {
        "images": {
            role: Image.new("RGB", (640, 360)) for role in ("head_camera", "left_wrist_camera", "right_wrist_camera")
        },
        "state": np.zeros(20, dtype=np.float32),
    }
    conditions = {
        "first_frame_image": [Image.new("RGB", (320, 384))],
        "proprio": observation["state"].copy(),
        "prompt": "task",
        "seed": 42,
        "num_frames": 33,
        "video_num_frames": 9,
        "height": 384,
        "width": 320,
    }
    original = input_identity(observation, conditions)
    assert original == input_identity(observation, dict(conditions, prompt="different", seed=43))
    with pytest.raises(ValueError, match="future video/latents"):
        input_identity(observation, dict(conditions, input_video_latents=np.zeros(1)))
    with pytest.raises(AssertionError):
        input_identity(observation, dict(conditions, proprio=np.ones(20)))


def make_prediction():
    state = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05], 2).astype(np.float32)
    pred = np.repeat(state[None], 32, axis=0)
    normalized = np.zeros((32, 80), dtype=np.float32)
    normalized[:, UNIFY_MAP] = pred
    return {
        "pred": pred.copy(),
        "native_pred": pred.copy(),
        "normalized_pred": pred.copy(),
        "normalized_full": normalized,
        "video": np.zeros((9, 384, 320, 3), dtype=np.uint8),
    }, state


def test_joint_recorder_one_draw_and_restore_on_failure():
    prediction, _ = make_prediction()

    class Inner:
        def unnormalize(self, values):
            return values[:, UNIFY_MAP]

    class Architecture:
        normalizer = Inner()

        def attach_normalizer(self, normalizer):
            self.normalizer = normalizer

    arch, calls = Architecture(), []
    original = arch.normalizer

    def generate(conditions):
        calls.append(conditions)
        return {
            "actions": arch.normalizer.unnormalize(prediction["normalized_full"]),
            "video": [Image.fromarray(x) for x in prediction["video"]],
        }

    engine = SimpleNamespace(architecture=arch, generate=generate)
    actual = predict_joint(engine, {"prompt": "task"})
    assert len(calls) == 1 and arch.normalizer is original
    np.testing.assert_array_equal(actual["video"], prediction["video"])
    np.testing.assert_array_equal(actual["native_pred"], prediction["native_pred"])

    def fail(_):
        raise RuntimeError("intentional failure")

    engine.generate = fail
    with pytest.raises(RuntimeError, match="intentional failure"):
        predict_joint(engine, {})
    assert arch.normalizer is original


def test_counterfactual_never_scores_original_gt_and_na_is_predeclared():
    prediction, state = make_prediction()
    # Missing gt/video keys prove that counterfactual scoring cannot consult them.
    case = row(prompt_class="counterfactual", gt_reference="none")
    metrics = score_case(case, prediction, {"state": state})
    assert metrics["semantic_branch_eligible"] is True
    assert metrics["action_gt_metrics"] is None and metrics["video_gt_metrics"] is None
    assert metrics["semantic_judgment"] == "not_automatically_scored"
    assert (
        score_case(dict(case, horizon_observable="no"), prediction, {"state": state})["semantic_branch_eligible"]
        is False
    )
    reference = {"state": state, "gt": prediction["pred"], "video": prediction["video"]}
    scored = score_case(row(), prediction, reference)
    assert scored["action_gt_metrics"]["prediction"]["arms"]["left"]["position_mm"]["mean"] == 0
    assert scored["video_gt_metrics"]["prediction"]["exact_future_match"] is True


def test_population_shift_preserves_training_stats_identity():
    stats = {"metadata": {"dataset_fingerprint": "training"}}
    with pytest.raises(ValueError, match="explicitly use"):
        population_identity(stats, "evaluation")
    result = population_identity(stats, "evaluation", allow_shift=True)
    assert result["train_stats_fingerprint"] == "training"
    assert result["eval_dataset_fingerprint"] == "evaluation"
    assert result["evaluation_population_shift"] is True
    assert stats == {"metadata": {"dataset_fingerprint": "training"}}


def test_resume_refuses_changed_plan_or_corrupted_completed_output(tmp_path):
    assert_manifest_compatible({"identity": {"seeds": [42]}}, {"seeds": [42]})
    with pytest.raises(ValueError, match="altered experiment"):
        assert_manifest_compatible({"identity": {"seeds": [42]}}, {"seeds": [43]})
    artifact = tmp_path / "prediction.npz"
    artifact.write_bytes(b"original")
    atomic_json(
        tmp_path / "result.json",
        {
            "status": "complete",
            "identity_sha256": "run",
            "case_id": "case",
            "seed": 42,
            "artifact_sha256": {"prediction.npz": sha256(artifact)},
        },
    )
    assert verify_completed(tmp_path, "run", {"case_id": "case"}, 42)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_completed(tmp_path, "run", {"case_id": "case"}, 42)


def test_audit_only_real_reader_preprocessor_references_and_resume(bucket, tmp_path, monkeypatch):
    root, dl, _, _ = bucket
    prepared = root / "prepared"
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    OmegaConf.save(cfg, prepared / "config.yaml")
    (prepared / "fixture.safetensors").write_bytes(b"never loaded in CPU audit")
    (prepared / "tokenizer").mkdir(exist_ok=True)
    (prepared / "tokenizer/test.json").write_text("{}")
    cases_path = write_cases(
        tmp_path / "cases.csv",
        [row(), row(prompt_class="empty", prompt="", gt_reference="none", feasibility="not_applicable")],
    )
    monkeypatch.setattr(
        "openwam.deploy.model_loader.load_from_checkpoint_dir", lambda *a, **k: pytest.fail("audit-only loaded a model")
    )
    args = Namespace(
        dataset=root,
        ckpt_dir=prepared,
        ckpt_name="fixture.safetensors",
        output=tmp_path / "audit",
        cases=cases_path,
        seeds=[42, 43],
        config=Path(__file__).resolve().parents[1] / "configs/deploy_nero.yaml",
        denoise_steps=None,
        device="cuda:0",
        audit_only=True,
        resume=False,
        retry_failed=False,
        max_generations=None,
        allow_evaluation_population_shift=False,
    )
    manifest = run(args)
    assert manifest["status"] == "audit_complete" and manifest["planned_generations"] == 4
    assert len(manifest["inputs"]) == 1
    reference = np.load(args.output / "observations/episode_000000_t000000/reference.npz")
    assert reference["video"].shape == (9, 384, 320, 3)
    assert reference["gt"].shape == (32, 20)
    first_hash = manifest["identity_sha256"]
    args.resume = True
    assert run(args)["identity_sha256"] == first_hash
    args.seeds = [44]
    with pytest.raises(ValueError, match="altered experiment"):
        run(args)
