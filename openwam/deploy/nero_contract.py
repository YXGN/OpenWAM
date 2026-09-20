"""Shared Nero checkpoint contract and official-engine observation conditions."""

from pathlib import Path

from omegaconf import OmegaConf

from openwam.dataloader.utils.nero import CAMERAS, CONTRACT, UNIFY_MAP
from openwam.dataloader.utils.nero_absolute import CONTRACT as ABSOLUTE_CONTRACT
from openwam.dataloader.utils.unify_action import parse_unify_spec

DEFAULT_INFERENCE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "deploy_nero.yaml"


def validate_contract(cfg, stats):
    """Allow only the two audited Nero training representations."""
    dl = cfg.dataloader
    expected = {"nero_multitask": "eef_relative", "nero_multitask_absolute": "eef"}
    if expected.get(dl.type) != dl.action_mode:
        raise ValueError("Expected a Nero reader with its matching EEF action_mode")
    fixed = {
        "num_frames": 33,
        "video_stride": 4,
        "window_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "require_all_cameras": True,
        "unify_action": True,
        "normalize_mode": "min-max",
    }
    for key, value in fixed.items():
        if dl.get(key) != value:
            raise ValueError(f"Nero contract mismatch: {key} must be {value!r}")
    if list(dl.camera_layout) != list(CAMERAS):
        raise ValueError("Nero camera order mismatch")
    for key in ("unify_action_map", "unify_state_map"):
        spec = dl.get(key)
        if spec is None and key == "unify_state_map":
            spec = dl.unify_action_map
        if spec is None or parse_unify_spec(spec).tolist() != UNIFY_MAP:
            raise ValueError(f"Invalid {key}")
    if dl.get("binary_action_dims") or dl.get("base_proprio", "velocity") != "velocity":
        raise ValueError("Nero evaluation does not use legacy command/base compatibility modes")
    if OmegaConf.select(dl, "color_jitter.enabled", default=False):
        raise ValueError("This evaluation contract requires unaugmented Nero training images")
    contract = CONTRACT if dl.action_mode == "eef_relative" else ABSOLUTE_CONTRACT
    if stats.get("metadata", {}).get("contract") != contract:
        raise ValueError("Checkpoint statistics representation contract mismatch")
    if dl.action_mode not in stats or f"{dl.action_mode}_state" not in stats:
        raise ValueError("Separate action and state statistics are required")


def conditions_from_observation(preprocessor, observation, seed):
    obs = preprocessor.preprocess(dict(observation))
    return {
        "first_frame_image": [obs["image"]],
        "proprio": obs["state"],
        "prompt": obs["prompt"],
        "seed": seed,
        "num_frames": 33,
        "video_num_frames": 9,
        "height": 384,
        "width": 320,
    }


def inference_config(steps=None, *, fast_path=None, config_path=DEFAULT_INFERENCE_CONFIG):
    """Read only the four Nero tuning knobs; keep model/robot semantics fixed."""
    values = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    fields = {"denoise_steps", "inference_fast_path", "cuda_graph", "request_context_cache"}
    if not isinstance(values, dict) or set(values) != fields:
        raise ValueError(f"Nero inference YAML must contain exactly: {', '.join(sorted(fields))}")
    if type(values["denoise_steps"]) is not int or values["denoise_steps"] < 1:
        raise ValueError("denoise_steps must be a positive integer")
    for name in fields - {"denoise_steps"}:
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be a YAML boolean (true/false)")
    if steps is not None:
        if type(steps) is not int or steps < 1:
            raise ValueError("denoise_steps must be a positive integer")
        values["denoise_steps"] = steps
    if fast_path is not None:
        if type(fast_path) is not bool:
            raise ValueError("fast_path must be a boolean")
        values["inference_fast_path"] = fast_path
    return OmegaConf.create(
        {
            "inference": {"denoise_steps": values["denoise_steps"], "denoise_mode": "sync", "cfg_scale": 1.0},
            "optimization": {
                "inference_fast_path": {"enabled": values["inference_fast_path"]},
                "cuda_graph": {"enabled": values["cuda_graph"]},
                "request_context_cache": {"enabled": values["request_context_cache"]},
                "decode_video": False,
                "compile": {"enabled": False},
                "dit_cache": {"enabled": False},
                "prompt_embed_cache": {"enabled": True, "maxsize": 32},
            },
        }
    )
