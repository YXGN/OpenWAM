#!/usr/bin/env python3
"""Inspect Nero data and prepare a native OpenWAM training recipe without training.

The supported profile is deliberately narrow: LeRobot v2 Nero FK, 33-frame
windows, 32 future actions, three fixed cameras. MoonCake v3 ``delta`` is a
different representation and is not an alias for this profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = "nero_fk_v2"
CAMERAS = {
    "head": "observation.images.head_fpv",
    "left_wrist": "observation.images.left_hand",
    "right_wrist": "observation.images.right_hand",
}
FK = {
    "left": "observation.fk.fb.left.gripper_flange",
    "right": "observation.fk.fb.right.gripper_flange",
}
FIELDS = {
    "name",
    "profile",
    "dataset_root",
    "cameras",
    "state_key",
    "action_key",
    "fk_keys",
    "representation",
    "action_mode",
    "horizon",
    "epochs",
    "per_device_batch_size",
    "num_devices",
    "gradient_accumulation_steps",
    "foundation_checkpoint",
    "output_dir",
    "python_executable",
    "video_latent_cache_dir",
    "seed",
    "dataset_num_workers",
    "speed_preset",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def path_value(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "${" in expanded:
        raise ValueError(f"{name} contains an unresolved environment variable")
    path = Path(expanded)
    path = (base / path if not path.is_absolute() else path).absolute()
    # Keep the venv executable path; resolving its symlink selects system Python.
    return str(path if name == "python_executable" else path.resolve())


def inspect_dataset(dataset_root):
    """Metadata inspection only: no video decoding, statistics, or model loading."""
    root = Path(dataset_root).expanduser().resolve()
    info = read_json(root / "meta/info.json")
    episodes = jsonl(root / "meta/episodes.jsonl")
    lengths = [positive_int(ep["length"], "episode length") for ep in episodes]
    return {
        "dataset_root": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "action_semantics": info.get("action_semantics"),
        "fps": info.get("fps"),
        "features": info.get("features", {}),
        "episodes": len(episodes),
        "frames": sum(lengths),
        "full_windows_33": sum(max(0, length - 32) for length in lengths),
        "episodes_shorter_than_33": [ep["episode_index"] for ep in episodes if ep["length"] < 33],
        "supported_profile": PROFILE,
        "required_mapping": {
            "state_key": "observation.state",
            "action_key": "action",
            "fk_keys": FK,
            "cameras": CAMERAS,
        },
        "validation": "metadata inspection only; prepare checks profile; stats performs the native full dataset audit",
    }


def validate_recipe(document, *, base_dir=None):
    if not isinstance(document, dict):
        raise ValueError("Recipe must be a JSON object")
    unknown = set(document) - FIELDS
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    recipe = dict(document)
    required = {
        "name",
        "dataset_root",
        "foundation_checkpoint",
        "output_dir",
        "num_devices",
        "per_device_batch_size",
    }
    missing = required - set(recipe)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if not isinstance(recipe["name"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", recipe["name"]):
        raise ValueError("name must contain only letters, digits, underscore and hyphen")
    if (ROOT / "configs" / (recipe["name"] + ".yaml")).exists():
        raise ValueError("name must not shadow an existing native Hydra configuration")
    defaults = {
        "profile": PROFILE,
        "state_key": "observation.state",
        "action_key": "action",
        "fk_keys": FK,
        "cameras": CAMERAS,
        "representation": "rot6d",
        "action_mode": "relative",
        "horizon": 32,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "python_executable": sys.executable,
        "video_latent_cache_dir": None,
        "seed": 42,
        "dataset_num_workers": 0,
        "speed_preset": None,
    }
    for key, value in defaults.items():
        recipe.setdefault(key, value)
    for key in (
        "profile",
        "state_key",
        "action_key",
        "fk_keys",
        "cameras",
        "representation",
        "horizon",
    ):
        if recipe[key] != defaults[key]:
            raise ValueError(f"{PROFILE} requires {key}={defaults[key]!r}; arbitrary mappings are not supported")
    if recipe["action_mode"] not in ("relative", "absolute"):
        raise ValueError("action_mode must be relative or absolute; legacy MoonCake delta is XYZ-only")
    if recipe["speed_preset"] not in (None, "fast"):
        raise ValueError("speed_preset must be null or fast")
    for key in (
        "epochs",
        "per_device_batch_size",
        "num_devices",
        "gradient_accumulation_steps",
    ):
        positive_int(recipe[key], key)
    for key in ("seed", "dataset_num_workers"):
        if type(recipe[key]) is not int or recipe[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    base = Path(base_dir or Path.cwd()).absolute()
    for key in (
        "dataset_root",
        "foundation_checkpoint",
        "output_dir",
        "python_executable",
    ):
        recipe[key] = path_value(recipe[key], base, key)
    if not Path(recipe["python_executable"]).is_file() or not os.access(recipe["python_executable"], os.X_OK):
        raise ValueError("python_executable must be an existing executable Python path")
    checkpoint = Path(recipe["foundation_checkpoint"])
    if (
        not checkpoint.exists()
        or not (checkpoint.parent if checkpoint.is_file() else checkpoint).joinpath("config.yaml").is_file()
    ):
        raise ValueError("foundation_checkpoint must exist with its native companion config.yaml")
    if checkpoint.is_dir():
        candidates = [
            (int(match.group(1)), path)
            for path in checkpoint.glob("checkpoint_step_*.safetensors")
            if (match := re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", path.name))
        ]
        if not candidates:
            raise ValueError("foundation_checkpoint directory contains no native checkpoint_step_*.safetensors")
        checkpoint = max(candidates, key=lambda item: item[0])[1]
    if checkpoint.suffix != ".safetensors":
        raise ValueError("foundation_checkpoint file must be a native .safetensors checkpoint")
    # Resolve a directory once, so a newly written later checkpoint cannot change this run.
    recipe["foundation_checkpoint"] = str(checkpoint)
    foundation_model(recipe)
    if recipe["video_latent_cache_dir"] is not None:
        if recipe["action_mode"] != "absolute":
            raise ValueError("Native VAE cache is supported only by the absolute Nero reader")
        recipe["video_latent_cache_dir"] = path_value(recipe["video_latent_cache_dir"], base, "video_latent_cache_dir")
        manifest = read_json(Path(recipe["video_latent_cache_dir"]) / "manifest.json")
        if manifest.get("complete") is not True:
            raise ValueError("video_latent_cache_dir must contain a complete native cache")
    root = Path(recipe["dataset_root"])
    output = Path(recipe["output_dir"])
    if output == root or root in output.parents:
        raise ValueError("output_dir must be outside the source dataset")
    from openwam.dataloader.utils.nero import validate_info

    info = read_json(root / "meta/info.json")
    validate_info(root, info)
    inspected = inspect_dataset(root)
    episodes = jsonl(root / "meta/episodes.jsonl")
    ids = [ep["episode_index"] for ep in episodes]
    if len(set(ids)) != len(ids) or not episodes or inspected["episodes_shorter_than_33"]:
        raise ValueError("Nero requires unique episodes with at least 33 frames each")
    if inspected["frames"] != info.get("total_frames") or len(episodes) != info.get("total_episodes"):
        raise ValueError("Episode metadata totals disagree with info.json")
    if (root / "meta/excluded_episodes.json").exists():
        raise ValueError("Nero all-data statistics do not permit implicit episode exclusions")
    return recipe


def foundation_model(recipe):
    """Validate reconstruction metadata only; do not instantiate modules or load tensors."""
    from omegaconf import OmegaConf

    directory = Path(recipe["foundation_checkpoint"]).parent
    model = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True).get("model")
    if not isinstance(model, dict):
        raise ValueError("Foundation config.yaml requires a complete model section")
    architecture = model.get("architecture", {})
    backbone = model.get("video_backbone", {})
    required = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 80,
        "state_dim": 80,
        "use_proprioception": True,
    }
    for key, expected in required.items():
        if architecture.get(key) != expected:
            raise ValueError(f"Current Nero recipe requires foundation model.architecture.{key}={expected!r}")
    if backbone.get("name") != "wan22_ti2v_5b" or backbone.get("from_scratch", False) is not False:
        raise ValueError("Current Nero recipe requires the native wan22_ti2v_5b foundation backbone")
    if not isinstance(model.get("action_backbone"), dict):
        raise ValueError("Foundation model requires action_backbone configuration")
    components = backbone.get("components")
    if not isinstance(components, list) or not all(isinstance(entry, dict) for entry in components):
        raise ValueError("Foundation requires native component reconstruction specs")
    attrs = [entry.get("attr") for entry in components]
    if len(set(attrs)) != len(attrs) or not {"dit", "text_encoder", "vae"}.issubset(attrs):
        raise ValueError("Foundation requires unique dit/text_encoder/vae components")
    if any(
        not isinstance(entry.get("model_class"), str) or not isinstance(entry.get("extra_kwargs", {}), dict)
        for entry in components
    ):
        raise ValueError("Foundation component specs require model_class and constructor kwargs")
    frozen = model.get("freeze", [])
    if not {"video_backbone.text_encoder", "video_backbone.vae"}.issubset(frozen):
        raise ValueError("Recipe text cache and native video preprocessing require frozen T5 and VAE")
    tokenizer = backbone.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("subdir"), str) or not tokenizer["subdir"]:
        raise ValueError("Foundation requires a checkpoint-local tokenizer spec")
    tokenizer_dir = (directory / tokenizer["subdir"]).resolve()
    if (
        not tokenizer_dir.is_relative_to(directory.resolve())
        or not tokenizer_dir.is_dir()
        or not any(path.is_file() for path in tokenizer_dir.rglob("*"))
    ):
        raise ValueError("Foundation checkpoint-local tokenizer files are missing")
    return model


def load_recipe(path):
    path = Path(path).absolute()
    return validate_recipe(read_json(path), base_dir=path.parent)


def training_plan(recipe):
    data = inspect_dataset(recipe["dataset_root"])
    windows = data["full_windows_33"]
    per_device, devices, gas = (
        recipe[k] for k in ("per_device_batch_size", "num_devices", "gradient_accumulation_steps")
    )
    # Native DataLoader(drop_last=False), Accelerate split_batches=False/even_batches=True.
    microsteps = math.ceil(windows / (per_device * devices))
    updates = math.ceil(microsteps / gas)
    return {
        "profile": PROFILE,
        "action_mode": recipe["action_mode"],
        "action_horizon": 32,
        "raw_window_frames": 33,
        "action_offset": 1,
        "video_stride": 4,
        "fps": data["fps"],
        "frames": data["frames"],
        "episodes": data["episodes"],
        "windows": windows,
        "global_microbatch_size": per_device * devices,
        "effective_batch_size": per_device * devices * gas,
        "microsteps_per_epoch": microsteps,
        "optimizer_steps_per_epoch": updates,
        "total_microsteps": microsteps * recipe["epochs"],
        "total_optimizer_steps": updates * recipe["epochs"],
        "padded_window_repeats_per_epoch": microsteps * per_device * devices - windows,
        "final_accumulation_microsteps": (microsteps - 1) % gas + 1,
        "epoch_semantics": "one native shuffled pass over full windows; Accelerate repeats initial windows to equalize ranks/batches; final accumulation group is flushed",
        "step_units": "native max_steps/save_steps/checkpoint_step count microsteps, not optimizer updates",
        "statistics": "native all-episode full-window future labels; no terminal clamping",
        "mapping": {key: recipe[key] for key in ("state_key", "action_key", "fk_keys", "cameras")},
        "action_source": "future FK poses and observation.state gripper widths; raw action is audited equal to current observation.state",
        "relative_semantics": "p_future-p_anchor in body axes; R_anchor.T @ R_future; absolute gripper width metres",
        "checkpoint_output": str(Path(recipe["output_dir"]) / "checkpoints" / "<native timestamp run directory>"),
        "resource_note": "User-selected batch/device counts; no automatic VRAM capacity guarantee",
    }


def native_config(recipe):
    absolute = recipe["action_mode"] == "absolute"
    plan = training_plan(recipe)
    dl = {
        "dataset_dir": recipe["dataset_root"],
        "normalization_stats_path": str(Path(recipe["output_dir"]) / "stats/normalization_stats.npy"),
    }
    if absolute:
        dl["video_latent_cache_dir"] = recipe["video_latent_cache_dir"]
    return {
        "defaults": [
            "/train",
            {"override /model": None},
            {"override /dataloader": "nero_multitask_absolute" if absolute else "nero_multitask"},
            "_self_",
        ],
        "model": foundation_model(recipe),
        "dataloader": dl,
        "training": {
            "speed_preset": str(ROOT / "presets/fast.env") if recipe["speed_preset"] == "fast" else None,
            "num_epochs": recipe["epochs"],
            "max_steps": None,
            "batch_size": recipe["per_device_batch_size"],
            "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
            "finetune_ckpt_path": recipe["foundation_checkpoint"],
            "resume_ckpt_path": None,
            "output_path": str(Path(recipe["output_dir"]) / "checkpoints"),
            "save_steps": plan["microsteps_per_epoch"],
            "save_full_states_for_resume": False,
            "initialize_model_on_cpu": True,
            "cache_frozen_text_embeddings": True,
            "dataset_num_workers": recipe["dataset_num_workers"],
        },
        "project": {"seed": recipe["seed"], "wandb": {"run_name": recipe["name"]}},
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_stats(recipe):
    """Delegate all data audit/statistics mathematics to existing native preparation."""
    from openwam.dataloader.utils.nero import CONTRACT, dataset_fingerprint
    from openwam.dataloader.utils.nero_absolute import CONTRACT as ABSOLUTE_CONTRACT

    root, output = Path(recipe["dataset_root"]), Path(recipe["output_dir"]) / "stats"
    identity = {
        "contract": ABSOLUTE_CONTRACT if recipe["action_mode"] == "absolute" else CONTRACT,
        "dataset_fingerprint": dataset_fingerprint(
            root,
            read_json(root / "meta/info.json"),
            jsonl(root / "meta/episodes.jsonl"),
        ),
        "preparation_sources": {
            name: sha256(ROOT / "openwam/dataloader/utils" / name)
            for name in (
                "nero.py",
                "nero_absolute.py",
                "stats_computation/nero_multitask_stats.py",
                "stats_computation/nero_absolute_stats.py",
            )
        },
    }
    stats, receipt = output / "normalization_stats.npy", output / "preparation.json"
    if stats.exists() and receipt.exists():
        previous = read_json(receipt)
        if previous.get("identity") != identity or previous.get("sha256") != sha256(stats):
            raise ValueError(
                "Prepared statistics changed or belong to different data/representation; use a new output_dir"
            )
        return {"cached": True, **previous}
    if stats.exists() or receipt.exists():
        raise ValueError("Incomplete or untracked statistics; inspect the output before preparing a new run")
    if recipe["action_mode"] == "absolute":
        from openwam.dataloader.utils.stats_computation.nero_absolute_stats import (
            prepare_absolute_dataset,
        )

        report = prepare_absolute_dataset(root, output)
    else:
        from openwam.dataloader.utils.stats_computation.nero_multitask_stats import (
            prepare_dataset,
        )

        report = prepare_dataset(root, output)
    result = {
        "identity": identity,
        "sha256": sha256(stats),
        "windows": report["windows"],
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return {"cached": False, **result}


BOUND_SOURCES = (
    "scripts/prepare_training.py",
    "scripts/train.py",
    "configs/train.yaml",
    "configs/dataloader/nero_multitask.yaml",
    "configs/dataloader/nero_multitask_absolute.yaml",
    "openwam/train/openwam_trainer.py",
    "openwam/train/utils/ckpt_model_loader.py",
    "openwam/dataloader/nero_multitask.py",
    "openwam/dataloader/nero_multitask_absolute.py",
    "openwam/dataloader/utils/normalization.py",
    "openwam/dataloader/utils/video_io.py",
    "openwam/dataloader/bases/lerobot_v3_reader.py",
    "openwam/train/utils/grad_accumulation.py",
    "openwam/dataloader/utils/nero.py",
    "openwam/dataloader/utils/nero_absolute.py",
    "openwam/dataloader/utils/stats_computation/nero_multitask_stats.py",
    "openwam/dataloader/utils/stats_computation/nero_absolute_stats.py",
)


def prepared_identity(recipe, output):
    checkpoint = Path(recipe["foundation_checkpoint"])
    tokenizer = checkpoint.parent / foundation_model(recipe)["video_backbone"]["tokenizer"]["subdir"]
    files = [
        output / "recipe.json",
        output / "training_plan.json",
        output / (recipe["name"] + ".yaml"),
        output / "train.sh",
        checkpoint,
        checkpoint.parent / "config.yaml",
        *sorted(path for path in tokenizer.rglob("*") if path.is_file()),
    ]
    if recipe["speed_preset"] == "fast":
        files.append(ROOT / "presets/fast.env")
    files.extend(ROOT / name for name in BOUND_SOURCES)
    return {str(path.resolve()): sha256(path) for path in files}


def verify_prepared(recipe_path):
    """Fail before CUDA/training when reviewed generated files or their inputs changed."""
    recipe_path = Path(recipe_path).resolve()
    output = recipe_path.parent
    receipt = read_json(output / "prepared_run.json")
    for name, expected in receipt["files_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Prepared run binding changed: {path}; prepare a new output_dir")
    recipe = load_recipe(recipe_path)
    if output != Path(recipe["output_dir"]):
        raise ValueError("Prepared recipe moved away from its bound output_dir")
    # Also compare semantics; a receipt cannot make a mismatched YAML valid.
    if read_json(output / (recipe["name"] + ".yaml")) != native_config(recipe):
        raise ValueError("Generated YAML does not match the prepared recipe")
    if read_json(output / "training_plan.json") != training_plan(recipe):
        raise ValueError("Training plan no longer matches recipe/dataset window counts")
    return {"verified": True, "recipe": str(recipe_path)}


def prepare_recipe(recipe, *, model_root=ROOT):
    recipe = validate_recipe(recipe)
    output = Path(recipe["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be empty; existing runs are never overwritten")
    plan = training_plan(recipe)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("recipe.json", recipe),
        ("training_plan.json", plan),
        (recipe["name"] + ".yaml", native_config(recipe)),
    ):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    quote = shlex.quote
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'if (($#)); then echo "train.sh accepts no overrides; prepare a new recipe for changed settings" >&2; exit 2; fi',
            f"cd {quote(str(model_root))}",
            f"recipe_python={quote(recipe['python_executable'])}",
            "export PYTHONNOUSERSITE=1",
            'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"',
            'export WANDB_MODE="${WANDB_MODE:-offline}"',
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"',
            f'"$recipe_python" scripts/prepare_training.py verify --recipe {quote(str(output / "recipe.json"))}',
            "\"$recipe_python\" - <<'PY'",
            "import torch",
            f"expected = {recipe['num_devices']}",
            "if not torch.cuda.is_available() or torch.cuda.device_count() != expected:",
            '    raise SystemExit(f"Select exactly {expected} visible CUDA devices using CUDA_VISIBLE_DEVICES")',
            "PY",
            f'"$recipe_python" scripts/prepare_training.py stats --recipe {quote(str(output / "recipe.json"))}',
            f'exec "$recipe_python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={recipe["num_devices"]} '
            f"scripts/train.py --config-dir {quote(str(output))} --config-name {quote(recipe['name'])}",
            "",
        ]
    )
    path = output / "train.sh"
    path.write_text(script)
    path.chmod(0o755)
    (output / "prepared_run.json").write_text(
        json.dumps({"files_sha256": prepared_identity(recipe, output)}, indent=2) + "\n"
    )
    return {
        "recipe": str(output / "recipe.json"),
        "train_script": str(path),
        "plan": plan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect").add_argument("--dataset-root", required=True)
    for name in ("prepare", "stats", "verify"):
        commands.add_parser(name).add_argument("--recipe", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_dataset(args.dataset_root)
    elif args.command == "verify":
        result = verify_prepared(args.recipe)
    else:
        recipe = load_recipe(args.recipe)
        result = prepare_recipe(recipe) if args.command == "prepare" else prepare_stats(recipe)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
