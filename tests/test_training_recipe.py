"""Native training preparation, using real parquet/video and no model execution."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module("openwam_training_recipe", ROOT / "scripts/prepare_training.py")
# Reuse the native physical/time/camera fixture rather than inventing weaker data.
bucket = load_module("nero_recipe_fixture", ROOT / "tests/dataloader/test_nero_multitask.py").bucket


@pytest.fixture
def document(bucket, tmp_path):
    checkpoint = tmp_path / "foundation"
    checkpoint.mkdir()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        model = OmegaConf.to_container(compose(config_name="train").model, resolve=True)
    # Deliberately differ from defaults: preparation must preserve these values.
    model["architecture"]["video_attention_mask_mode"] = "bidirectional"
    model["architecture"]["bridge_interval"] = 2
    model["video_backbone"]["components"] = [
        {"attr": name, "model_class": "openwam.model.video_backbone.wan.models." + class_name, "extra_kwargs": {}}
        for name, class_name in (
            ("dit", "dit.WanModel"),
            ("text_encoder", "text_encoder.WanTextEncoder"),
            ("vae", "vae.WanVideoVAE38"),
        )
    ]
    model["video_backbone"]["tokenizer"] = {
        "subdir": "tokenizer",
        "class": "openwam.model.video_backbone.wan.models.text_encoder.HuggingfaceTokenizer",
    }
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "tokenizer/tokenizer_config.json").write_text("{}")
    (checkpoint / "config.yaml").write_text(json.dumps({"model": model}))
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"not-loaded-in-tests")
    return {
        "name": "nero_recipe_test",
        "dataset_root": str(bucket[0]),
        "foundation_checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "run with spaces"),
        "num_devices": 2,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 2,
    }


def test_metadata_inspect_is_stdlib_only(bucket):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/prepare_training.py"),
            "inspect",
            "--dataset-root",
            str(bucket[0]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert (info["episodes"], info["frames"], info["full_windows_33"]) == (3, 108, 12)
    assert info["required_mapping"]["cameras"] == prep.CAMERAS


def test_native_epoch_padding_and_tail_accumulation(document):
    recipe = prep.validate_recipe(document)
    plan = prep.training_plan(recipe)
    assert recipe["epochs"] == 1 and recipe["horizon"] == 32
    assert recipe["action_mode"] == "relative"
    assert plan["windows"] == 12
    assert plan["microsteps_per_epoch"] == 3
    assert plan["optimizer_steps_per_epoch"] == 2
    assert plan["final_accumulation_microsteps"] == 1
    assert plan["effective_batch_size"] == 8
    recipe["per_device_batch_size"] = 5
    recipe["epochs"] = 3
    plan = prep.training_plan(recipe)
    assert plan["microsteps_per_epoch"] == 2
    assert plan["padded_window_repeats_per_epoch"] == 8
    assert plan["total_microsteps"] == 6
    assert plan["total_optimizer_steps"] == 3


def test_plan_matches_native_accelerate_batch_shards(document):
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, SequentialSampler

    recipe = prep.validate_recipe({**document, "per_device_batch_size": 5})
    plan = prep.training_plan(recipe)
    batches = BatchSampler(SequentialSampler(range(plan["windows"])), batch_size=5, drop_last=False)
    shards = [BatchSamplerShard(batches, num_processes=2, process_index=rank) for rank in range(2)]
    assert [len(shard) for shard in shards] == [plan["microsteps_per_epoch"]] * 2
    seen = [index for shard in shards for batch in shard for index in batch]
    assert len(seen) - len(set(seen)) == plan["padded_window_repeats_per_epoch"]
    assert set(seen) == set(range(plan["windows"]))


@pytest.mark.parametrize(
    "mode,reader",
    [("relative", "nero_multitask"), ("absolute", "nero_multitask_absolute")],
)
def test_generated_hydra_resolves_native_defaults(document, mode, reader):
    recipe = prep.validate_recipe({**document, "action_mode": mode})
    result = prep.prepare_recipe(recipe)
    output = Path(recipe["output_dir"])
    # Exercise the exact native CLI search path; --cfg resolves config without training.
    config_only = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--config-dir",
            str(output),
            "--config-name",
            recipe["name"],
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = OmegaConf.to_container(OmegaConf.create(config_only.stdout), resolve=True)
    assert resolved["dataloader"]["type"] == reader
    assert resolved["dataloader"]["dataset_dir"] == recipe["dataset_root"]
    assert resolved["training"]["num_epochs"] == 1
    assert resolved["training"]["max_steps"] is None
    assert resolved["training"]["gradient_accumulation_steps"] == 2
    assert resolved["training"]["save_steps"] == 3
    assert resolved["training"]["speed_preset"] is None
    assert resolved["model"]["architecture"]["action_dim"] == 80
    assert resolved["model"] == prep.foundation_model(recipe)
    assert resolved["model"]["architecture"]["video_attention_mask_mode"] == "bidirectional"
    assert resolved["model"]["architecture"]["bridge_interval"] == 2
    assert prep.verify_prepared(result["recipe"])["verified"]
    text = Path(result["train_script"]).read_text()
    assert "expected = 2" in text and "--nproc_per_node=2" in text
    assert "scripts/prepare_training.py stats" in text
    assert text.index("prepare_training.py verify") < text.index("import torch")
    assert "CUDA_VISIBLE_DEVICES=" not in text  # Do not silently choose GPUs.
    assert "uv " not in text
    subprocess.run(["bash", "-n", result["train_script"]], check=True)
    override = subprocess.run(["bash", result["train_script"], "dataloader=moon_cake"], capture_output=True, text=True)
    assert override.returncode == 2 and "accepts no overrides" in override.stderr
    with pytest.raises(ValueError, match="never overwritten"):
        prep.prepare_recipe(recipe)


@pytest.mark.parametrize("mode,key", [("relative", "eef_relative"), ("absolute", "eef")])
def test_real_native_stats_and_loader_share_representation(document, bucket, mode, key):
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.utils.nero import relative_actions
    from openwam.dataloader.utils.normalization import apply_normalization

    recipe = prep.validate_recipe({**document, "action_mode": mode})
    result = prep.prepare_stats(recipe)
    assert result["cached"] is False and result["windows"] == 12
    assert prep.prepare_stats(recipe)["cached"] is True
    stats_path = Path(recipe["output_dir"]) / "stats/normalization_stats.npy"
    stats = np.load(stats_path, allow_pickle=True).item()
    assert stats["metadata"]["contract"]["action_offset"] == 1
    assert stats["metadata"]["contract"]["gripper"] == "absolute_opening_width_metres"
    # Resolve inheritance through Hydra, exactly as a generated training config does.
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"dataloader={'nero_multitask_absolute' if mode == 'absolute' else 'nero_multitask'}"],
        )
    dl = cfg.dataloader
    dl.dataset_dir = recipe["dataset_root"]
    dl.normalization_stats_path = str(stats_path)
    dataset = build_dataset(dl)
    assert len(dataset) == 12
    sample = dataset[0]
    eef = bucket[2][0]
    physical = eef[1:33] if mode == "absolute" else relative_actions(eef[1:33], eef[0])
    expected = apply_normalization(physical, stats[key], "min-max")
    actual = sample["action"].numpy()
    np.testing.assert_allclose(actual[:, :10], expected[:, :10], atol=3e-6)
    np.testing.assert_allclose(actual[:, 34:44], expected[:, 10:], atol=3e-6)
    # Rotation-relative and absolute runs have distinct stats contracts, never reused.
    changed = {
        **recipe,
        "action_mode": "absolute" if mode == "relative" else "relative",
    }
    with pytest.raises(ValueError, match="different data/representation"):
        prep.prepare_stats(changed)
    stats_path.write_bytes(stats_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="statistics changed"):
        prep.prepare_stats(recipe)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "moon_cake"),
        ("representation", "quaternion_xyzw"),
        ("action_mode", "delta"),
        ("state_key", "state"),
        ("action_key", "actions"),
        ("cameras", {"head": "cam0"}),
        ("fk_keys", {"left": "left_pose", "right": "right_pose"}),
        ("horizon", 50),
        ("num_devices", 0),
        ("per_device_batch_size", True),
        ("gradient_accumulation_steps", 1.5),
        ("name", "train"),
        ("epochs", 0),
        ("typo", 1),
    ],
)
def test_rejects_unsupported_or_ambiguous_recipe(document, field, value):
    with pytest.raises(ValueError):
        prep.validate_recipe({**document, field: value})


def test_checkpoint_and_venv_identity(document, tmp_path):
    with pytest.raises(ValueError, match="companion config.yaml"):
        prep.validate_recipe({**document, "foundation_checkpoint": str(tmp_path / "missing")})
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    recipe = prep.validate_recipe({**document, "python_executable": str(executable)})
    assert recipe["python_executable"] == str(executable)


def test_speed_preset_is_explicit_not_inherited(document, monkeypatch):
    monkeypatch.setenv("OPENWAM_PRESET", "/ambient/should-not-be-used.env")
    recipe = prep.validate_recipe(document)
    assert prep.native_config(recipe)["training"]["speed_preset"] is None
    recipe["speed_preset"] = "fast"
    assert prep.native_config(recipe)["training"]["speed_preset"] == str(ROOT / "presets/fast.env")
    with pytest.raises(ValueError, match="null or fast"):
        prep.validate_recipe({**document, "speed_preset": "other"})


def test_environment_expansion_and_native_audit_rejection(document, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = tmp_path / "data-copy"
    shutil.copytree(document["dataset_root"], data)
    monkeypatch.setenv("RECIPE_DATASET", str(data))
    recipe = prep.validate_recipe({**document, "dataset_root": "${RECIPE_DATASET}"})
    assert recipe["dataset_root"] == str(data)
    parquet = next(data.glob("data/**/*.parquet"))
    table = pq.read_table(parquet)
    columns = table.to_pydict()
    columns["action"][0][0] += 1.0
    pq.write_table(pa.table(columns), parquet)
    with pytest.raises(ValueError, match="current_state|observation.state|action"):
        prep.prepare_stats(recipe)
    monkeypatch.delenv("RECIPE_DATASET")
    with pytest.raises(ValueError, match="unresolved environment"):
        prep.validate_recipe({**document, "dataset_root": "${RECIPE_DATASET}"})


def test_cache_relative_is_rejected_and_absolute_requires_complete_manifest(document, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="only by the absolute"):
        prep.validate_recipe({**document, "video_latent_cache_dir": str(cache)})
    with pytest.raises(ValueError, match="complete native cache"):
        prep.validate_recipe(
            {
                **document,
                "action_mode": "absolute",
                "video_latent_cache_dir": str(cache),
            }
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture.framework", "tri_system"),
        ("architecture.action_dim", 20),
        ("architecture.variant", "unknown"),
        ("video_backbone.name", "cosmos3_edge"),
        ("video_backbone.components", []),
        ("video_backbone.tokenizer", None),
        ("freeze", []),
    ],
)
def test_foundation_contract_rejects_unsupported_reconstruction(document, field, value):
    config_path = Path(document["foundation_checkpoint"]) / "config.yaml"
    config = OmegaConf.load(config_path)
    OmegaConf.update(config, "model." + field, value, merge=False)
    config_path.write_text(OmegaConf.to_yaml(config))
    with pytest.raises(ValueError, match="requires|require|Foundation"):
        prep.validate_recipe(document)


@pytest.mark.parametrize("target", ["recipe", "yaml", "plan", "script", "weights", "foundation_config", "tokenizer"])
def test_prepared_binding_rejects_changes_before_launch(document, target):
    recipe = prep.validate_recipe(document)
    prepared = prep.prepare_recipe(recipe)
    output = Path(recipe["output_dir"])
    checkpoint = Path(recipe["foundation_checkpoint"])
    paths = {
        "recipe": output / "recipe.json",
        "yaml": output / (recipe["name"] + ".yaml"),
        "plan": output / "training_plan.json",
        "script": output / "train.sh",
        "weights": checkpoint,
        "foundation_config": checkpoint.parent / "config.yaml",
        "tokenizer": checkpoint.parent / "tokenizer/tokenizer_config.json",
    }
    changed = paths[target]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding changed"):
        prep.verify_prepared(prepared["recipe"])


def test_foundation_directory_is_pinned_to_specific_weights(document):
    directory = Path(document["foundation_checkpoint"])
    recipe = prep.validate_recipe(document)
    assert recipe["foundation_checkpoint"] == str(directory / "checkpoint_step_1.safetensors")
    prepared = prep.prepare_recipe(recipe)
    (directory / "checkpoint_step_99.safetensors").write_bytes(b"new-unselected-weights")
    assert prep.verify_prepared(prepared["recipe"])["verified"]
    generated = json.loads((Path(recipe["output_dir"]) / (recipe["name"] + ".yaml")).read_text())
    assert generated["training"]["finetune_ckpt_path"] == recipe["foundation_checkpoint"]
