"""Small HTTP front for the pinned OpenWAM MoonCake model.

The process is deliberately independent from the robot packages.  It accepts
the JSON dialect emitted by ``OpenWAMCodec`` and owns OpenWAM image
preprocessing.  The model adapter is the only boundary that knows about the
official checkout; tests can replace it with a CPU fixture.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from safety import GripperRangeSafety
from source_identity import normalize_source, verify_integrated

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised by the independent env
    Image = None  # type: ignore[assignment,misc]


WIRE_FORMAT = "openwam_native"
ACTION_SEMANTICS = "absolute_flange_rot6d_20d"
ACTION_DIM = 20
ACTION_HORIZON = 32
DT_SEC = 1.0 / 60.0
CLIENT_IMAGE_ROLES = ("head_fpv", "left_hand", "right_hand")
SERVER_IMAGE_ROLES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
IMAGE_ROLE_MAP = {
    "head_fpv": "head_camera",
    "left_hand": "left_wrist_camera",
    "right_hand": "right_wrist_camera",
}
SLOT_SIZES = {
    "head_camera": (320, 256),
    "left_wrist_camera": (160, 128),
    "right_wrist_camera": (160, 128),
}
PROMPT = (
    "Pick up the box with the left hand and place it in front of the robot. "
    "Pick up a mooncake from the conveyor belt with the right hand and place it into the box. "
    "Push the lid down with the right hand to close the box. "
    "Then press the box and place the closed box aside properly with the left hand."
)
SOURCE_REVISION = "c171854a18a2acbeaadf46a746c4118c0e0c00f7"
ACTION_MODES = ("eef", "delta")
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")


def _schema_group(name: str) -> dict[str, Any]:
    return {
        "group": name,
        "arm_format": "rot6d",
        "arm_dim": 9,
        "effector_protocol": "agilex.parallel_gripper.width_m/v1",
        "effector_dim": 1,
        "effector_names": ["opening_width"],
        "effector_position_unit": "width_m",
        "effector_position_limits": [0.0, 0.1],
        "effector_position_velocity_max": [],
    }


OPENWAM_IO_SCHEMA: dict[str, Any] = {
    "version": "velabot.io/v2",
    "action_dim": ACTION_DIM,
    "include_base": False,
    "layout": [
        *[f"left_arm.arm.{i}" for i in range(9)],
        "left_arm.effector.position.opening_width",
        *[f"right_arm.arm.{i}" for i in range(9)],
        "right_arm.effector.position.opening_width",
    ],
    "groups": [_schema_group("left_arm"), _schema_group("right_arm")],
}

# IOSchema does not carry pose frames or delta reconstruction policy.  Keep
# those model obligations explicit and independently checked in this service.
_MODEL_CONTRACT_BASE: dict[str, Any] = {
    "action_semantics": ACTION_SEMANTICS,
    "coordinate_frame": "body",
    "pose_frame": "gripper_flange",
    "position_unit": "m",
    "rotation_representation": "rot6d_first_two_columns",
    "gripper_unit": "width_m",
    "native_gripper_unit": "normalized_width",
    "native_gripper_range": [-1.0, 1.0],
    "gripper_affine_scale": 0.05,
    "gripper_affine_offset": 0.05,
    "policy_hz": 60,
}


def model_contract_for(action_mode: str) -> dict[str, Any]:
    """Return the fixed serving contract for one checkpoint action mode."""
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")
    contract = dict(_MODEL_CONTRACT_BASE)
    contract["action_mode"] = action_mode
    contract["delta_xyz_addback"] = "none_for_absolute_eef" if action_mode == "eef" else "engine_once_xyz_state0"
    return contract


MODEL_CONTRACTS = {mode: model_contract_for(mode) for mode in ACTION_MODES}
MODEL_CONTRACT = MODEL_CONTRACTS["eef"]


class ProtocolError(ValueError):
    """A client request cannot be interpreted under the fixed dialect."""


class ModelError(RuntimeError):
    """The loaded model violated its serving boundary or failed to generate."""


class BusyError(ProtocolError):
    """A second request arrived while the single model is generating."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _finite_vector(value: Any, width: int, field: str) -> list[float]:
    _require(isinstance(value, list), f"{field} must be a JSON list")
    _require(len(value) == width, f"{field} must have width {width}; got {len(value)}")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)),
            f"{field}[{index}] must be a finite number",
        )
        result.append(float(item))
    return result


def _finite_actions(value: Any) -> list[list[float]]:
    _require(isinstance(value, list), "model actions must be a JSON-like list")
    _require(len(value) == ACTION_HORIZON, f"model actions must have shape ({ACTION_HORIZON}, {ACTION_DIM})")
    return [_finite_vector(row, ACTION_DIM, f"actions[{index}]") for index, row in enumerate(value)]


def _decode_jpeg(value: Any, role: str) -> Any:
    _require(Image is not None, "Pillow is required by the OpenWAM service environment")
    _require(isinstance(value, str) and value != "", f"image {role!r} must be non-empty base64 JPEG text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ProtocolError(f"image {role!r} is not valid base64: {exc}") from exc
    _require(raw, f"image {role!r} must not be empty")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ProtocolError(f"image {role!r} is not a decodable JPEG: {exc}") from exc
    _require(image.format == "JPEG", f"image {role!r} must be JPEG bytes, got {image.format!r}")
    return image.convert("RGB")


def make_multiview_canvas(images: Mapping[str, Any]) -> Any:
    """Resize each view once with LANCZOS and compose the 320x384 canvas."""
    _require(Image is not None, "Pillow is required by the OpenWAM service environment")
    slots: dict[str, Any] = {}
    for role in SERVER_IMAGE_ROLES:
        image = images[role]
        target = SLOT_SIZES[role]
        slots[role] = image.resize(target, Image.Resampling.LANCZOS) if image.size != target else image.copy()
    canvas = Image.new("RGB", (320, 384))
    canvas.paste(slots["head_camera"], (0, 0))
    canvas.paste(slots["left_wrist_camera"], (0, 256))
    canvas.paste(slots["right_wrist_camera"], (160, 256))
    return canvas


@dataclass(frozen=True)
class ModelIdentity:
    checkpoint: str
    config: str
    stats: str
    source: dict[str, Any]
    model_contract: dict[str, Any]
    action_mode: str = "eef"
    checkpoint_sha256: str = ""
    config_sha256: str = ""
    stats_sha256: str = ""

    def wire(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "config": self.config,
            "stats": self.stats,
            "source": dict(self.source),
            "model_contract": dict(self.model_contract),
            "action_mode": self.action_mode,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "stats_sha256": self.stats_sha256,
        }


@dataclass(frozen=True)
class ModelOutput:
    actions: list[list[float]]
    step: int = 0
    latency_ms: float = 0.0


class ModelAdapter(Protocol):
    identity: ModelIdentity

    def reset(self) -> None: ...

    def generate(self, *, image: Any, state: list[float], prompt: str) -> ModelOutput: ...


class OpenWAMFront:
    """Strict protocol front over a single loaded model adapter."""

    def __init__(self, model: ModelAdapter, *, prompt: str = PROMPT, safety: GripperRangeSafety | None = None) -> None:
        _require(prompt == PROMPT, "this MoonCake service requires the exact training prompt")
        _require(model.identity.action_mode in ACTION_MODES, f"action_mode must be one of {ACTION_MODES}")
        _require(
            model.identity.model_contract == model_contract_for(model.identity.action_mode),
            "model identity contract does not match its action_mode",
        )
        self.model = model
        self.safety = safety or GripperRangeSafety()
        self.prompt = prompt
        self._handshaken = False
        self._lock = threading.Lock()
        self._requests = 0

    def _common(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "openwam",
            "wire_format": WIRE_FORMAT,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_semantics": ACTION_SEMANTICS,
            "dt_sec": DT_SEC,
            "supports_rtc": False,
            "return_hidden": False,
            "image_roles": list(CLIENT_IMAGE_ROLES),
            "prompt": self.prompt,
            "io_schema": json.loads(json.dumps(OPENWAM_IO_SCHEMA)),
            **self.model.identity.wire(),
        }

    def health(self) -> dict[str, Any]:
        return {**self._common(), "session_active": self._handshaken}

    def handshake(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise BusyError("an inference or handshake is already in flight")
        try:
            self._validate_handshake(request)
            self.model.reset()
            self._handshaken = True
            self._requests = 0
            return {"type": "handshake_ack", **self._common()}
        finally:
            self._lock.release()

    def _validate_handshake(self, request: Mapping[str, Any]) -> None:
        _require(request.get("wire_format") == WIRE_FORMAT, f"wire_format must be {WIRE_FORMAT!r}")
        _require(request.get("action_dim") == ACTION_DIM, f"action_dim must be {ACTION_DIM}")
        _require(request.get("action_horizon") == ACTION_HORIZON, f"action_horizon must be {ACTION_HORIZON}")
        _require(request.get("action_semantics") == ACTION_SEMANTICS, f"action_semantics must be {ACTION_SEMANTICS!r}")
        dt = request.get("dt_sec")
        _require(
            isinstance(dt, (int, float))
            and not isinstance(dt, bool)
            and math.isclose(float(dt), DT_SEC, abs_tol=1e-12),
            f"dt_sec must be {DT_SEC}",
        )
        _require(request.get("supports_rtc") is False, "OpenWAM does not support RTC")
        _require(request.get("return_hidden") is False, "OpenWAM does not support hidden features")
        _require(
            tuple(request.get("image_roles", ())) == CLIENT_IMAGE_ROLES, f"image_roles must be {CLIENT_IMAGE_ROLES}"
        )
        advertised = request.get("io_schema")
        _require(isinstance(advertised, Mapping), "handshake requires a complete velabot.io/v2 io_schema")
        _require(
            dict(advertised) == OPENWAM_IO_SCHEMA, "handshake io_schema does not match the configured OpenWAM layout"
        )

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise BusyError("an inference or handshake is already in flight")
        try:
            _require(self._handshaken, "OpenWAM inference requires a successful /handshake first")
            _require(request.get("wire_format") == WIRE_FORMAT, f"wire_format must be {WIRE_FORMAT!r}")
            _require(request.get("action_dim") == ACTION_DIM, f"action_dim must be {ACTION_DIM}")
            _require(
                request.get("action_semantics") == ACTION_SEMANTICS, f"action_semantics must be {ACTION_SEMANTICS!r}"
            )
            _require(request.get("rtc") in (None, False, {}), "OpenWAM does not support RTC")
            _require(not request.get("return_hidden"), "OpenWAM does not support hidden features")
            _require(request.get("prompt") == self.prompt, "prompt must match the exact training prompt")
            state = _finite_vector(request.get("state"), ACTION_DIM, "state")
            images = request.get("images")
            _require(isinstance(images, Mapping), "inference requires an images object")
            _require(set(images) == set(SERVER_IMAGE_ROLES), f"images must contain exactly {SERVER_IMAGE_ROLES}")
            native_images = {role: _decode_jpeg(images[role], role) for role in SERVER_IMAGE_ROLES}
            canvas = make_multiview_canvas(native_images)
            try:
                result = self.model.generate(image=canvas, state=state, prompt=self.prompt)
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError(f"OpenWAM model generation failed: {exc}") from exc
            try:
                actions = _finite_actions(result.actions)
                _require(
                    isinstance(result.step, int) and not isinstance(result.step, bool) and result.step >= 0,
                    "model step must be a non-negative integer",
                )
                _require(
                    isinstance(result.latency_ms, (int, float))
                    and not isinstance(result.latency_ms, bool)
                    and math.isfinite(float(result.latency_ms))
                    and result.latency_ms >= 0.0,
                    "model latency_ms must be a finite non-negative number",
                )
            except ProtocolError as exc:
                raise ModelError(str(exc)) from exc
            actions, interventions = self.safety.apply(actions)
            self._requests += 1
            return {
                "type": "action_sequence",
                "ok": True,
                "actions": actions,
                "action_dim": ACTION_DIM,
                "action_horizon": ACTION_HORIZON,
                "wire_format": WIRE_FORMAT,
                "action_semantics": ACTION_SEMANTICS,
                "dt_sec": DT_SEC,
                "step": result.step or self._requests,
                "latency_ms": float(result.latency_ms),
                "safety": {
                    "interventions": [item.wire() for item in interventions],
                    "clipped_count": len(interventions),
                },
                "gripper_clips": len(interventions),
                **self.model.identity.wire(),
            }
        finally:
            self._lock.release()


def load_config(path: str | Path) -> tuple[dict[str, Any], ModelIdentity]:
    """Load and validate the service-owned identity/config boundary."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load service config {config_path}: {exc}") from exc
    _require(isinstance(raw, dict), "service config must be a JSON object")
    _require(raw.get("io_schema") == OPENWAM_IO_SCHEMA, "service config io_schema is not the fixed OpenWAM schema")
    contract = raw.get("model_contract")
    action_mode = raw.get("action_mode")
    _require(action_mode in ACTION_MODES, f"service action_mode must be one of {ACTION_MODES}")
    _require(
        contract == model_contract_for(action_mode),
        "service model_contract must match body-frame gripper-flange OpenWAM action semantics",
    )
    for key in ("checkpoint", "config", "stats"):
        _require(isinstance(raw.get(key), str) and raw[key].strip(), f"service config requires identity field {key!r}")
    _require(raw["config"] == "config.yaml", "service identity config must be the official loader's config.yaml")
    _require(
        raw["stats"] == "normalization_stats.npy",
        "service identity stats must be the official loader's normalization_stats.npy",
    )
    source = raw.get("source")
    _require(isinstance(source, dict), "service config requires source identity")
    source = _source_identity(source)
    for key in ("checkpoint_sha256", "config_sha256", "stats_sha256"):
        digest = raw.get(key)
        _require(
            isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower()),
            f"service config requires a lowercase SHA-256 identity field {key!r}",
        )
    identity = ModelIdentity(
        checkpoint=raw["checkpoint"],
        config=raw["config"],
        stats=raw["stats"],
        source=dict(source),
        model_contract=dict(contract),
        action_mode=action_mode,
        checkpoint_sha256=raw["checkpoint_sha256"],
        config_sha256=raw["config_sha256"],
        stats_sha256=raw["stats_sha256"],
    )
    return raw, identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelError(f"cannot read identity artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the service source fingerprint."""
    try:
        return normalize_source(dict(source))
    except ValueError as exc:
        raise ModelError(str(exc)) from exc


def _changed_source_paths(source_root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ModelError(f"cannot inspect OpenWAM checkout changes: {exc}") from exc
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            paths.update(part.strip() for part in path.split(" -> "))
        else:
            paths.add(path)
    return paths


def _verify_artifacts(
    identity: ModelIdentity,
    checkpoint_dir: Path,
    checkpoint_name: str,
    source_root: Path,
    patch_path: Path | None = None,
) -> None:
    """Verify the bytes and checkout actually handed to the official loader."""
    checkpoint = checkpoint_dir / checkpoint_name
    config = checkpoint_dir / identity.config
    stats = checkpoint_dir / identity.stats
    for path in (checkpoint, config, stats):
        if not path.is_file():
            raise ModelError(f"required model identity artifact is missing: {path}")
    for path, expected in (
        (checkpoint, identity.checkpoint_sha256),
        (config, identity.config_sha256),
        (stats, identity.stats_sha256),
    ):
        actual = _sha256_file(path)
        if actual != expected:
            raise ModelError(f"identity digest mismatch for {path.name}: config={expected}, actual={actual}")
    source = _source_identity(identity.source)
    if source.get("base_revision"):
        expected_patch = source_root / "velabot" / "patches" / "openwam-moon-cake-training.patch"
        if patch_path is None:
            patch_path = expected_patch
        elif patch_path.resolve() != expected_patch.resolve():
            raise ModelError(f"integrated source requires the registered patch path: {expected_patch}")
        try:
            verify_integrated(source_root, source)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelError(f"integrated OpenWAM source identity verification failed: {exc}") from exc
        return
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ModelError(f"cannot read OpenWAM checkout revision: {exc}") from exc
    if revision != SOURCE_REVISION:
        raise ModelError(f"OpenWAM checkout revision is {revision!r}; expected pinned {SOURCE_REVISION!r}")
    files = source["files_sha256"]
    patched = bool(source["patch_sha256"] or files)
    if bool(source["patch_sha256"]) != bool(files):
        raise ModelError("source.patch_sha256 and source.files_sha256 must be provided together")
    if identity.action_mode == "delta" and not patched:
        raise ModelError("delta source must declare patch_sha256 and files_sha256")
    if patched:
        if patch_path is None:
            raise ModelError("patched OpenWAM serving requires the applied training patch path")
        try:
            patch_path.resolve().relative_to(source_root.resolve())
        except ValueError:
            pass
        else:
            raise ModelError("training patch must be outside the OpenWAM checkout")
        if _sha256_file(patch_path) != source["patch_sha256"]:
            raise ModelError(f"training patch SHA-256 does not match source.patch_sha256: {patch_path}")
        try:
            subprocess.run(
                ["git", "-C", str(source_root), "apply", "--reverse", "--check", str(patch_path)],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ModelError("declared training patch is not applied to the OpenWAM checkout") from exc
    for relative, expected in files.items():
        path = source_root / relative
        try:
            path.resolve().relative_to(source_root.resolve())
        except ValueError as exc:
            raise ModelError(f"source file escapes the OpenWAM checkout: {relative}") from exc
        if not path.is_file():
            raise ModelError(f"registered OpenWAM source file is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ModelError(f"source file SHA-256 mismatch for {relative}: expected {expected}, actual {actual}")
    changed = _changed_source_paths(source_root)
    unregistered = sorted(changed - set(files))
    if unregistered:
        raise ModelError(f"OpenWAM checkout has unregistered changes: {', '.join(unregistered)}")
    if not patched and changed:
        raise ModelError("eef OpenWAM checkout must be clean")


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _verify_stats_file(path: Path, action_mode: str) -> None:
    """Check the mode-specific stats before the official loader uses them."""
    try:
        raw = np.load(path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, AttributeError) as exc:
        raise ModelError(f"cannot load MoonCake normalization stats {path}: {exc}") from exc

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ModelError(message)

    require(isinstance(raw, Mapping), "MoonCake normalization stats must be a mapping")
    require(action_mode in ACTION_MODES, f"unknown action mode {action_mode!r}")

    def check_block(block_name: str) -> Mapping[str, Any]:
        block = raw.get(block_name)
        require(isinstance(block, Mapping), f"MoonCake stats require {block_name!r} block")
        for key in STAT_KEYS:
            try:
                values = np.asarray(block.get(key), dtype=float)
                finite = np.isfinite(values).all()
            except (TypeError, ValueError) as exc:
                raise ModelError(f"MoonCake stats {block_name}.{key} must be numeric") from exc
            require(values.shape == (20,), f"MoonCake stats {block_name}.{key} must have shape (20,)")
            require(finite, f"MoonCake stats {block_name}.{key} contains non-finite values")
        return block

    eef = check_block("eef")
    state = raw.get("eef_state")
    if state is not None:
        state = check_block("eef_state")
        for key in STAT_KEYS:
            if not np.array_equal(np.asarray(state[key]), np.asarray(eef[key])):
                raise ModelError(f"MoonCake stats eef_state.{key} must exactly equal eef.{key}")
    if action_mode == "delta":
        check_block("delta")
        delta_state = check_block("delta_state")
        for key in STAT_KEYS:
            if not np.array_equal(np.asarray(delta_state[key]), np.asarray(eef[key])):
                raise ModelError(f"MoonCake stats delta_state.{key} must exactly equal eef.{key}")


def _verify_loaded_config(policy_server: Any, action_mode: str, stats_path: Path) -> None:
    cfg = policy_server.cfg
    checks = {
        "dataloader.type": "moon_cake",
        "dataloader.action_mode": action_mode,
        "dataloader.normalize_mode": "min-max",
        "dataloader.num_frames": 33,
        "dataloader.video_stride": 4,
        "dataloader.height": 384,
        "dataloader.width": 320,
        "dataloader.multiview": True,
        "dataloader.unify_action": True,
        "dataloader.unify_action_map": ["0-9", "34-43"],
    }
    for path, expected in checks.items():
        actual = _cfg_get(cfg, path)
        if path.endswith("_map") and actual is not None:
            actual = list(actual)
        if actual != expected:
            raise ModelError(f"loaded checkpoint config {path}={actual!r}; expected {expected!r}")
    _verify_stats_file(stats_path, action_mode)
    layout = _cfg_get(cfg, "dataloader.camera_layout")
    if layout is not None:
        try:
            layout_len = len(layout)
        except TypeError as exc:
            raise ModelError(f"loaded checkpoint config camera_layout is not a sequence: {layout!r}") from exc
        if layout_len != 3:
            raise ModelError(f"loaded checkpoint config camera_layout must contain three slots; got {layout!r}")
    policy_server._init_policy()
    state_map = _cfg_get(cfg, "dataloader.unify_state_map")
    if state_map is not None and list(state_map) != ["0-9", "34-43"]:
        raise ModelError(f"loaded checkpoint config unify_state_map={state_map!r}; expected the action map")
    preprocessor = policy_server._obs_preprocessor
    if not preprocessor.multiview or (preprocessor.img_width, preprocessor.img_height) != (320, 384):
        raise ModelError("loaded OpenWAM preprocessor must resolve to multiview 320x384")
    if len(preprocessor.camera_layout) != 3:
        raise ModelError(
            f"loaded OpenWAM preprocessor camera_layout must contain three slots; got {preprocessor.camera_layout!r}"
        )
    if tuple(str(item) for item in preprocessor.camera_layout) != ("head_camera", "left_camera", "right_camera"):
        raise ModelError(
            "loaded OpenWAM preprocessor resolved layout is "
            f"{preprocessor.camera_layout!r}; expected the pinned default layout"
        )
    binary_dims = getattr(getattr(policy_server.engine, "architecture", None), "binary_command_dims", ()) or ()
    if binary_dims:
        raise ModelError(
            f"loaded OpenWAM architecture declares binary command dims {binary_dims!r}; "
            "this front does not apply a command projection"
        )
    normalizer = getattr(policy_server.engine.architecture, "normalizer", None)
    action_map = getattr(normalizer, "_action_dst_index", None)
    state_map = getattr(normalizer, "_state_dst_index", None)
    if action_map is None or state_map is None:
        raise ModelError("loaded OpenWAM model lacks the expected unified action/state normalizer")
    if action_map.tolist() != list(range(10)) + list(range(34, 44)):
        raise ModelError(f"loaded OpenWAM action gather map is {action_map.tolist()!r}; expected raw 20-D map")
    if state_map.tolist() != action_map.tolist():
        raise ModelError("loaded OpenWAM state scatter map differs from the action map")
    if getattr(normalizer, "_unify_dim", None) != 80:
        raise ModelError(
            f"loaded OpenWAM unified normalizer dim is {getattr(normalizer, '_unify_dim', None)!r}; expected 80"
        )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"ok": False, "error": f"no such path {self.path!r}"})
            return
        self._send(200, self.server.front.health())  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/handshake", "/infer"):
            self._send(404, {"ok": False, "error": f"no such path {self.path!r}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            _require(0 < length <= 32 * 1024 * 1024, f"bad Content-Length {length}")
            payload = json.loads(self.rfile.read(length))
            _require(isinstance(payload, dict), "request body must be a JSON object")
            reply = self.server.front.handshake(payload) if path == "/handshake" else self.server.front.infer(payload)  # type: ignore[attr-defined]
        except BusyError as exc:
            self._send(429, {"ok": False, "error": "busy", "message": str(exc)})
            return
        except ProtocolError as exc:
            self._send(400, {"ok": False, "error": "protocol_error", "message": str(exc)})
            return
        except ModelError as exc:
            self._send(500, {"ok": False, "error": "model_error", "message": str(exc)})
            return
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error": "malformed_json", "message": str(exc)})
            return
        except Exception as exc:  # model boundary: report a real server failure
            self._send(500, {"ok": False, "error": "model_error", "message": str(exc)})
            return
        self._send(200, reply)


def serve(front: OpenWAMFront, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.front = front  # type: ignore[attr-defined]
    print(f"[openwam-front] listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def _official_adapter(identity: ModelIdentity, args: argparse.Namespace) -> ModelAdapter:
    root = Path(args.openwam_root).resolve()
    _require(root.is_dir(), f"openwam root does not exist: {root}")
    _require(
        Path(args.ckpt_name).name == Path(identity.checkpoint).name,
        "--ckpt-name does not match the checkpoint identity in service config",
    )
    _verify_artifacts(identity, Path(args.ckpt_dir).resolve(), args.ckpt_name, root, args.patch)
    sys.path.insert(0, str(root))
    third = root / "third_party"
    if third.is_dir():
        sys.path.insert(0, str(third))
    try:
        from omegaconf import OmegaConf

        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config
    except ImportError as exc:
        raise RuntimeError("official OpenWAM environment is missing its model dependencies") from exc
    try:
        imported_from = Path(sys.modules["openwam.deploy.server"].__file__).resolve()
        imported_from.relative_to(root)
    except (KeyError, AttributeError, ValueError) as exc:
        raise ModelError("loaded PolicyServer module is outside the pinned OpenWAM checkout") from exc
    cfg = _load_deploy_yaml(None)
    OmegaConf.update(cfg, "inference.denoise_steps", int(args.denoise_steps), merge=False)
    OmegaConf.update(cfg, "optimization.compile.enabled", bool(args.compile), merge=False)
    OmegaConf.update(cfg, "optimization.dit_cache.enabled", bool(args.dit_cache), merge=False)
    policy_server = build_server_from_config(
        cfg=cfg, ckpt_dir=args.ckpt_dir, device=args.device, ckpt_name=args.ckpt_name
    )
    _verify_loaded_config(policy_server, identity.action_mode, Path(args.ckpt_dir).resolve() / identity.stats)

    class OfficialAdapter:
        def __init__(self) -> None:
            self.identity = identity
            self.policy_server = policy_server

        def reset(self) -> None:
            self.policy_server._init_policy()
            self.policy_server._policy.reset()

        def generate(self, *, image: Any, state: list[float], prompt: str) -> ModelOutput:
            self.policy_server._init_policy()
            policy = self.policy_server._policy
            executor = policy._executor
            buffer = executor._action_buffer
            if buffer:
                buffer.clear()
                raise RuntimeError("OpenWAM action buffer was non-empty before generation")
            completed = False
            try:
                started = time.monotonic()
                obs = {"image": image, "state": state, "prompt": prompt}
                actions = [policy.predict_action(obs)]
                if len(buffer) != ACTION_HORIZON - 1:
                    produced = len(buffer) + 1
                    raise ModelError(
                        f"OpenWAM generated {produced} steps, expected {ACTION_HORIZON}; refusing a second generation"
                    )
                for _ in range(ACTION_HORIZON - 1):
                    actions.append(policy.predict_action(obs))
                if buffer:
                    raise ModelError("OpenWAM generation left actions in its buffer")
                converted = [item.tolist() if hasattr(item, "tolist") else item for item in actions]
                output = ModelOutput(_finite_actions(converted), latency_ms=(time.monotonic() - started) * 1000)
                completed = True
                return output
            finally:
                if not completed:
                    buffer.clear()

    return OfficialAdapter()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="service-owned JSON identity and schema config")
    parser.add_argument("--openwam-root", required=True, help="official OpenWAM checkout")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        help="training patch; integrated mode uses the registered in-repo patch by default, legacy mode uses an external patch",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8849)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dit-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, identity = load_config(args.config)
    front = OpenWAMFront(_official_adapter(identity, args))
    serve(front, args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
