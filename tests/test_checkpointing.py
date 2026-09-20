"""Tests for checkpoint discovery / resume-position / retention helpers.

Pure filesystem + integer logic — no GPU, no real Accelerator (a tiny fake covers
the save/load_full_state metadata round-trip and the atomic completion marker).
"""

from pathlib import Path

import pytest

from openwam.train.utils.checkpointing import (
    archive_weights,
    compute_resume_position,
    finalize_keep_weights_only,
    find_latest_accel_state,
    find_latest_weights,
    load_full_state,
    manage_checkpoints,
    save_full_state,
    verify_resume_normalization_stats,
)


def _make_accel_state(root: Path, step: int, *, marker: bool) -> Path:
    d = root / f"accel_state_step_{step}"
    d.mkdir()
    if marker:
        (d / "trainer_state.json").write_text("{}")
    return d


def test_weight_archive_is_loadable_and_survives_rolling_pruning(tmp_path):
    import numpy as np
    from safetensors.numpy import load_file, save_file

    run = tmp_path / "runs" / "2026-09-18_12-00-00"
    run.mkdir(parents=True)
    weight = run / "checkpoint_step_4000.safetensors"
    tensor = np.arange(8, dtype=np.float32)
    save_file({"weight": tensor}, str(weight))
    original = weight.read_bytes()
    (run / "config.yaml").write_text("model: {}\n")
    np.save(run / "normalization_stats.npy", {"mean": tensor})
    (run / "tokenizer").mkdir()
    (run / "tokenizer" / "tokenizer.json").write_text("{}")
    _make_accel_state(run, 4000, marker=True)

    archive = Path(archive_weights(str(run), str(tmp_path / "archives"), 4000))
    assert {p.name for p in archive.iterdir()} == {
        weight.name, "config.yaml", "normalization_stats.npy", "tokenizer"
    }
    assert (archive / weight.name).read_bytes() == original
    assert (archive / "config.yaml").read_bytes() == (run / "config.yaml").read_bytes()
    assert (archive / "normalization_stats.npy").read_bytes() == (run / "normalization_stats.npy").read_bytes()
    assert (archive / "tokenizer" / "tokenizer.json").read_text() == "{}"
    (run / "checkpoint_step_6000.safetensors").write_bytes(original)
    _make_accel_state(run, 6000, marker=True)
    manage_checkpoints(str(run), keep_last_k=1)
    assert not weight.exists()
    assert not (run / "accel_state_step_4000").exists()
    assert (run / "accel_state_step_6000").exists()
    np.testing.assert_array_equal(load_file(find_latest_weights(str(archive)))["weight"], tensor)


def test_weight_archive_copy_failure_does_not_publish_or_damage_source(tmp_path, monkeypatch):
    import shutil

    run = tmp_path / "run"
    run.mkdir()
    weight = run / "checkpoint_step_4000.safetensors"
    weight.write_bytes(b"original")

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated full disk")

    monkeypatch.setattr(shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="simulated full disk"):
        archive_weights(str(run), str(tmp_path / "archives"), 4000)
    assert weight.read_bytes() == b"original"
    assert list((tmp_path / "archives" / "run").iterdir()) == []


def test_weight_archive_rejects_overwrite_and_nested_destination(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    existing = tmp_path / "archives" / "run" / "step_4000"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        archive_weights(str(run), str(tmp_path / "archives"), 4000)
    assert (existing / "sentinel").read_text() == "preserve"
    with pytest.raises(ValueError, match="outside"):
        archive_weights(str(run), str(run / "archives"), 4000)


# --- find_latest_weights ---


def test_find_latest_weights_picks_highest_step(tmp_path):
    for s in (100, 200, 300):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
    assert find_latest_weights(str(tmp_path)).endswith("checkpoint_step_300.safetensors")


def test_find_latest_weights_skips_malformed(tmp_path):
    (tmp_path / "checkpoint_step_50.safetensors").write_text("x")
    (tmp_path / "checkpoint_step_garbage.safetensors").write_text("x")
    assert find_latest_weights(str(tmp_path)).endswith("checkpoint_step_50.safetensors")


def test_find_latest_weights_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_weights(str(tmp_path))


# --- find_latest_accel_state (marker = trainer_state.json) ---


def test_find_latest_accel_state_requires_marker(tmp_path):
    _make_accel_state(tmp_path, 100, marker=True)
    _make_accel_state(tmp_path, 200, marker=False)  # half-written: no completion marker
    # 200 lacks the marker -> skipped; 100 is the latest usable.
    assert find_latest_accel_state(str(tmp_path)).endswith("accel_state_step_100")


def test_find_latest_accel_state_picks_highest_usable(tmp_path):
    _make_accel_state(tmp_path, 100, marker=True)
    _make_accel_state(tmp_path, 200, marker=True)
    assert find_latest_accel_state(str(tmp_path)).endswith("accel_state_step_200")


def test_find_latest_accel_state_none_when_empty(tmp_path):
    assert find_latest_accel_state(str(tmp_path)) is None


# --- compute_resume_position (grad_accum alignment / off-by fix) ---


def test_resume_position_grad_accum_1_is_identity():
    # grad_accum=1: aligned == global_step always (zero regression vs. pre-fix behaviour).
    assert compute_resume_position(25, 10, 1) == (2, 5, 25)


def test_resume_position_floors_skip_and_pulls_back_global_step():
    # gs=10, bpe=100, grad_accum=4: skip 10 -> 8, aligned 10 -> 8 (no re-train, step matched).
    assert compute_resume_position(10, 100, 4) == (0, 8, 8)


def test_resume_position_alignment_across_epoch():
    # gs=16, bpe=10, grad_accum=4: start=1, skip 6 -> 4, aligned = 1*10 + 4 = 14.
    assert compute_resume_position(16, 10, 4) == (1, 4, 14)


def test_resume_position_already_aligned_unchanged():
    # gs=12, bpe=100, grad_accum=4: skip 12 already a multiple of 4 -> unchanged.
    assert compute_resume_position(12, 100, 4) == (0, 12, 12)


# --- finalize_keep_weights_only ---


def test_finalize_keeps_configured_recent_weights_and_removes_all_state(tmp_path):
    for s in (100, 200, 300, 400):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
        _make_accel_state(tmp_path, s, marker=True)
    finalize_keep_weights_only(str(tmp_path), keep_last_k=3)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [
        "checkpoint_step_200.safetensors",
        "checkpoint_step_300.safetensors",
        "checkpoint_step_400.safetensors",
    ]


def test_finalize_default_keeps_only_final_weight(tmp_path):
    for s in (100, 200, 300):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
    finalize_keep_weights_only(str(tmp_path))
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint_step_300.safetensors"]


# --- manage_checkpoints: weights + accel_state pruned in lockstep ---


def test_manage_checkpoints_prunes_accel_state_in_lockstep(tmp_path):
    for s in (100, 200, 300):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
        _make_accel_state(tmp_path, s, marker=True)
    manage_checkpoints(str(tmp_path), keep_last_k=2)
    weights = sorted(p.name for p in tmp_path.glob("checkpoint_step_*"))
    states = sorted(p.name for p in tmp_path.glob("accel_state_step_*"))
    assert weights == ["checkpoint_step_200.safetensors", "checkpoint_step_300.safetensors"]
    assert states == ["accel_state_step_200", "accel_state_step_300"]


# --- save/load_full_state metadata round-trip + atomic completion marker ---


class _FakeAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def save_state(self, state_dir):
        import os

        os.makedirs(state_dir, exist_ok=True)  # stand-in for the sharded state write

    def load_state(self, state_dir):
        pass


def test_full_state_meta_round_trip(tmp_path):
    acc = _FakeAccelerator()
    save_full_state(acc, str(tmp_path), global_step=42, opt_step=10, epoch=3)
    state_dir = tmp_path / "accel_state_step_42"
    assert (state_dir / "trainer_state.json").is_file()  # atomic completion marker present
    assert load_full_state(acc, str(state_dir)) == {"global_step": 42, "opt_step": 10, "epoch": 3}


# --- verify_resume_normalization_stats ---


def test_verify_resume_normalization_stats_noop_without_dataset_path(tmp_path):
    class _Dataset:
        normalization_stats_path = None

    verify_resume_normalization_stats(str(tmp_path), _Dataset())


def test_verify_resume_normalization_stats_missing_checkpoint_artifact(tmp_path):
    import numpy as np

    src = tmp_path / "dataset_stats.npy"
    np.save(src, {"eef": {"min": np.zeros(2)}})
    (tmp_path / "run").mkdir()

    class _Dataset:
        normalization_stats_path = str(src)

    with pytest.raises(FileNotFoundError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(str(tmp_path / "run"), _Dataset())


def test_verify_resume_normalization_stats_mismatch(tmp_path):
    import numpy as np

    src = tmp_path / "dataset_stats.npy"
    dst_dir = tmp_path / "run"
    dst_dir.mkdir()
    np.save(src, {"eef": {"min": np.zeros(2, dtype=np.float32)}})
    np.save(dst_dir / "normalization_stats.npy", {"eef": {"min": np.ones(2, dtype=np.float32)}})

    class _Dataset:
        normalization_stats_path = str(src)

    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(str(dst_dir), _Dataset())


def test_verify_resume_normalization_stats_match(tmp_path):
    import numpy as np

    payload = {"eef": {"min": np.arange(3, dtype=np.float32), "max": np.arange(3, dtype=np.float32) + 1}}
    src = tmp_path / "dataset_stats.npy"
    dst_dir = tmp_path / "run"
    dst_dir.mkdir()
    np.save(src, payload)
    np.save(dst_dir / "normalization_stats.npy", payload)

    class _Dataset:
        normalization_stats_path = str(src)

    verify_resume_normalization_stats(str(dst_dir), _Dataset())


def test_verify_resume_normalization_stats_rejects_allclose_only_match(tmp_path):
    import numpy as np

    src = tmp_path / "dataset_stats.npy"
    dst_dir = tmp_path / "run"
    dst_dir.mkdir()
    np.save(src, {"eef": {"min": np.ones(2, dtype=np.float32)}})
    # np.allclose accepts this delta at values near 1, but strict resume must not.
    np.save(
        dst_dir / "normalization_stats.npy",
        {"eef": {"min": np.ones(2, dtype=np.float32) + np.float32(1e-6)}},
    )

    class _Dataset:
        normalization_stats_path = str(src)

    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(str(dst_dir), _Dataset())


def test_verify_resume_normalization_stats_accepts_legacy_reduced_checkpoint(tmp_path):
    import numpy as np

    transform = {
        "mean": np.zeros(3, dtype=np.float32),
        "std": np.ones(3, dtype=np.float32),
        "min": -np.ones(3, dtype=np.float32),
        "max": np.ones(3, dtype=np.float32),
        "q01": -np.ones(3, dtype=np.float32),
        "q99": np.ones(3, dtype=np.float32),
    }
    full = {
        "eef": {
            **transform,
            "q50": [0.0, 0.0, 0.0],
            "action_rows": 100,
            "state_rows": 100,
            "gripper_convention": "minus1_closed_plus1_open",
            "representation": "absolute_eef10",
        }
    }
    src = tmp_path / "dataset_stats.npy"
    dst_dir = tmp_path / "run"
    dst_dir.mkdir()
    np.save(src, full)
    np.save(dst_dir / "normalization_stats.npy", {"eef": transform})

    class _Dataset:
        normalization_stats_path = str(src)

    verify_resume_normalization_stats(str(dst_dir), _Dataset())


def test_verify_resume_normalization_stats_rejects_shared_semantic_mismatch(tmp_path):
    import numpy as np

    transform = {
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
        "min": -np.ones(2, dtype=np.float32),
        "max": np.ones(2, dtype=np.float32),
        "q01": -np.ones(2, dtype=np.float32),
        "q99": np.ones(2, dtype=np.float32),
    }
    src = tmp_path / "dataset_stats.npy"
    dst_dir = tmp_path / "run"
    dst_dir.mkdir()
    np.save(src, {"eef": {**transform, "gripper_convention": "minus1_closed_plus1_open"}})
    np.save(
        dst_dir / "normalization_stats.npy",
        {"eef": {**transform, "gripper_convention": "plus1_closed_minus1_open"}},
    )

    class _Dataset:
        normalization_stats_path = str(src)

    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(str(dst_dir), _Dataset())


def test_setup_output_dir_verifies_stats_before_reusing_resume_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    import openwam.train.openwam_trainer as trainer_module

    run_dir = tmp_path / "run"
    state_dir = run_dir / "accel_state_step_12"
    state_dir.mkdir(parents=True)
    (state_dir / "trainer_state.json").write_text("{}")

    calls = []
    monkeypatch.setattr(
        trainer_module,
        "verify_resume_normalization_stats",
        lambda output_dir, dataset: calls.append((output_dir, dataset)),
    )
    trainer = object.__new__(trainer_module.OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({"training": {"output_path": str(tmp_path / "unused")}})
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.dataset = object()

    output_path, resume_state_dir = trainer.setup_output_dir(debug=False, resume_path=str(run_dir))

    assert output_path == str(run_dir)
    assert resume_state_dir == str(state_dir)
    assert calls == [(str(run_dir), trainer.dataset)]
