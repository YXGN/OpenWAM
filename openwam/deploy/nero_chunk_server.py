"""HTTP chunk boundary for Nero robots; inference stays in the official engine.

Wire specification: assets/openwam_usage_docs/nero-client-protocol.md.
No robot dependency, alternative normalization, or server action queue.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from openwam.deploy.nero_contract import (
    DEFAULT_INFERENCE_CONFIG,
    conditions_from_observation,
    inference_config,
    validate_contract,
)
from openwam.deploy.obs_preprocess import ObsPreprocessor

WIRE = "openwam_nero/v1"
ROLES = ("head_fpv", "left_hand", "right_hand")
SERVER_ROLES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
CONTRACT = {
    "state": "absolute_body_gripper_flange_eef20",
    "actions": "absolute_body_gripper_flange_eef20",
    "position_unit": "m",
    "rotation": "first_two_matrix_columns_column_major",
    "gripper": "opening_width_m",
    "gripper_bounds": [0.0, 0.1],
    "arm_order": ["left_arm", "right_arm"],
    "image_roles": list(ROLES),
    "image_encoding": "png",
    "action_offset": 1,
    "action_horizon": 32,
    "dt_sec": 1.0 / 60.0,
}


def finite_eef(value, shape, *, observation=False):
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype.kind not in "fi" or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite EEF with shape {shape}")
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("EEF exceeds float32 range")
    for start in (3, 13):
        a, b = arr[..., start : start + 3], arr[..., start + 3 : start + 6]
        if np.any(np.linalg.norm(a, axis=-1) < 1e-8) or np.any(np.linalg.norm(np.cross(a, b), axis=-1) < 1e-8):
            raise ValueError("Degenerate rot6D")
    # Validate, never add a second gripper projection at the transport boundary.
    if observation and (np.any(arr[..., [9, 19]] < 0) or np.any(arr[..., [9, 19]] > np.float32(0.1))):
        raise ValueError("Gripper must be opening width in metres in [0, 0.1]")
    return arr


def validate_schema(schema):
    """Check semantic fields independently; echo validated device constraints."""
    if not isinstance(schema, dict) or schema.get("version") != "velabot.io/v2":
        raise ValueError("Expected complete velabot.io/v2 schema")
    if schema.get("action_dim") != 20 or schema.get("include_base") is not False:
        raise ValueError("Nero requires EEF20 without a base block")
    groups = schema.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("Expected two arm groups")
    layout = []
    for group, name in zip(groups, CONTRACT["arm_order"], strict=True):
        expected = dict(
            group=name,
            arm_format="rot6d",
            arm_dim=9,
            effector_dim=1,
            effector_protocol="agilex.parallel_gripper.width_m/v1",
            effector_names=["opening_width"],
            effector_position_unit="width_m",
            effector_position_limits=[0.0, 0.1],
        )
        if not isinstance(group, dict) or any(group.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Incompatible EEF/width schema for {name}")
        speeds = group.get("effector_position_velocity_max")
        if (
            not isinstance(speeds, list)
            or len(speeds) > 1
            or any(
                isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0 for x in speeds
            )
        ):
            raise ValueError("Invalid effector velocity declaration")
        layout.extend([f"{name}.arm.{i}" for i in range(9)] + [f"{name}.effector.position.opening_width"])
    if schema.get("layout") != layout:
        raise ValueError("EEF slot order mismatch")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def log_timing(request_id, trace):
    """One compact breakdown; durations are disjoint synchronized wall times."""
    stages = trace["stages"]
    order = ("request", "queue", "images", "obs", "setup", "t5", "vae", "denoise", "action", "other", "response")
    fields = " ".join(f"{name}={stages[name]:.1f}" for name in order if name in stages)
    hit = trace.get("t5_cache_hit")
    cache = "" if hit is None else f" t5_cache={'hit' if hit else 'miss'}"
    logging.info(
        "timing_ms id=%r %s total=%.1f%s graph_captures=%d",
        request_id[:12],
        fields,
        trace["total_ms"],
        cache,
        trace["graph_captures"],
    )


class NeroChunkService:
    def __init__(self, engine, cfg, identity):
        self.engine = engine
        self.cfg = cfg
        self.preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        self.identity = identity
        self._lock = threading.Lock()
        self._session = None
        self._logged_prompt = None
        self._prompt_revision = 0

    def request(self, path, payload, *, timing_trace=None):
        request_started = time.monotonic()
        profile = bool(getattr(self.engine, "_profile", False))
        if not isinstance(payload, dict) or payload.get("wire_format") != WIRE:
            raise ValueError(f"Expected wire_format={WIRE}")
        with self._lock:
            acquired = time.monotonic() if profile else None
            if path == "/handshake":
                validate_schema(payload.get("io_schema"))
                if payload.get("contract") != CONTRACT:
                    raise ValueError("Nero input/output contract mismatch")
                self._session = uuid.uuid4().hex
                return dict(
                    ok=True,
                    backend="openwam_nero",
                    wire_format=WIRE,
                    action_dim=20,
                    action_horizon=32,
                    dt_sec=1 / 60,
                    supports_rtc=False,
                    return_hidden=False,
                    image_roles=list(ROLES),
                    io_schema=payload["io_schema"],
                    contract=CONTRACT,
                    checkpoint=self.identity["checkpoint"],
                    identity=self.identity,
                    session_id=self._session,
                )
            if not self._session or payload.get("session_id") != self._session:
                raise ValueError("A current handshake is required")
            if path == "/reset":
                # No action queue or temporal model state. Invalidate in-flight client epochs.
                self._session = uuid.uuid4().hex
                return dict(ok=True, wire_format=WIRE, session_id=self._session, identity=self.identity)
            if path != "/infer":
                raise ValueError("Unknown endpoint")
            request_id = payload.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id is required")
            if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
                raise ValueError("An explicit task prompt is required")
            state = finite_eef(payload.get("state"), (20,), observation=True)
            images = payload.get("images")
            if not isinstance(images, dict) or set(images) != set(SERVER_ROLES):
                raise ValueError("All three named cameras are required")
            decode_started = time.monotonic() if profile else None
            decoded = {}
            for role, encoded in images.items():
                if not isinstance(encoded, str):
                    raise ValueError("Images must be base64 PNG strings")
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.format != "PNG" or im.mode != "RGB":
                        raise ValueError("Expected lossless RGB PNG")
                    decoded[role] = im.copy()
            decode_finished = time.monotonic() if profile else None
            obs = dict(images=decoded, state=state, prompt=payload["prompt"])
            conditions = conditions_from_observation(self.preprocessor, obs, 42)
            runner = getattr(getattr(self.engine, "architecture", None), "_mot_cuda_graph_runner", None)
            captures_before = runner.status(include_signature=False)["capture_count"] if profile and runner else 0
            started = time.monotonic()
            # The official engine owns normalize -> generate -> inverse normalize ->
            # relative pose reconstruction (once, against this request's state).
            result = self.engine.generate(conditions)
            actions = result["actions"]
            if hasattr(actions, "detach"):
                actions = actions.detach().cpu().numpy()
            actions = finite_eef(actions, (32, 20))
            inference_sec = time.monotonic() - started
            response = dict(
                ok=True,
                wire_format=WIRE,
                session_id=self._session,
                request_id=request_id,
                identity=self.identity,
                actions=actions.tolist(),
                action_dim=20,
                action_horizon=32,
                dt_sec=1 / 60,
                anchor_state=state.tolist(),
            )
            response["server_total_sec"] = time.monotonic() - request_started
            prompt = conditions["prompt"]
            changed = prompt != self._logged_prompt
            if changed:
                self._prompt_revision += 1
                self._logged_prompt = prompt
            runner = getattr(getattr(self.engine, "architecture", None), "_mot_cuda_graph_runner", None)
            backend = "eager" if runner is None else runner.status(include_signature=False)["last_backend"]
            # Keep multiline prompts on one log line, without altering model input.
            logging.info(
                "infer id=%r prompt=P%d infer_ms=%.1f server_ms=%.1f backend=%s%s",
                request_id[:12],
                self._prompt_revision,
                inference_sec * 1000,
                response["server_total_sec"] * 1000,
                backend,
                f" text={prompt!r}" if changed else "",
            )
            if profile:
                stages = {
                    "queue": (acquired - request_started) * 1000,
                    "images": (decode_finished - decode_started) * 1000,
                    "obs": (started - decode_finished) * 1000,
                    **result.get("timings_ms", {}),
                }
                total_ms = response["server_total_sec"] * 1000
                remainder = max(0.0, total_ms - sum(stages.values()))
                stages["other"] = stages.get("other", 0.0) + remainder
                trace = {} if timing_trace is None else timing_trace
                trace.update(
                    stages=stages,
                    total_ms=total_ms,
                    t5_cache_hit=result.get("t5_cache_hit"),
                    graph_captures=(runner.status(include_signature=False)["capture_count"] if runner else 0)
                    - captures_before,
                )
                if timing_trace is None:
                    log_timing(request_id, trace)
            return response


def make_http_server(service, host="127.0.0.1", port=8849):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            status = 200
            trace = {} if self.path == "/infer" and getattr(service.engine, "_profile", False) else None
            received = time.monotonic() if trace is not None else None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    self.close_connection = True
                    raise ValueError("Request body must be 1..32 MiB")
                payload = json.loads(self.rfile.read(length))
                if trace is not None:
                    parsed = time.monotonic()
                    result = service.request(self.path, payload, timing_trace=trace)
                else:
                    result = service.request(self.path, payload)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                status, result = 400, dict(ok=False, error=str(exc))
            except Exception:
                logging.exception("Nero inference failed")
                status, result = 500, dict(ok=False, error="Model inference failed; see server log")
            response_started = time.monotonic() if trace is not None else None
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if trace is not None and status == 200:
                finished = time.monotonic()
                stages = trace["stages"]
                stages["request"] = (parsed - received) * 1000
                stages["response"] = (finished - response_started) * 1000
                trace["total_ms"] = (finished - received) * 1000
                stages["other"] += max(0.0, trace["total_ms"] - sum(stages.values()))
                log_timing(payload["request_id"], trace)

        def log_message(self, fmt, *args):
            logging.info(fmt, *args)

        def log_request(self, code="-", size="-"):
            if self.path == "/infer" and code == 200:
                return  # The inference summary already records successful requests.
            super().log_request(code, size)

    return ThreadingHTTPServer((host, port), Handler)


def load_service(
    ckpt_dir, ckpt_name, device="cuda:0", *, fast_path=None, denoise_steps=None, config_path=DEFAULT_INFERENCE_CONFIG
):
    from openwam.deploy.server import build_server_from_config

    deploy = inference_config(denoise_steps, fast_path=fast_path, config_path=config_path)
    root = Path(ckpt_dir).resolve()
    if Path(ckpt_name).name != ckpt_name or not ckpt_name.endswith(".safetensors"):
        raise ValueError("Expected a checkpoint filename")
    training = OmegaConf.load(root / "config.yaml")
    stats = np.load(root / "normalization_stats.npy", allow_pickle=True).item()
    validate_contract(training, stats)
    identity = {"checkpoint": ckpt_name, "model_representation": training.dataloader.action_mode}
    for key, name in (("checkpoint", ckpt_name), ("config", "config.yaml"), ("stats", "normalization_stats.npy")):
        identity[key + "_sha256"] = file_sha256(root / name)
    # Keep synchronous joint inference. No RTC, approximate
    # DiT cache, future video conditioning, or extra classifier-free guidance.
    server = build_server_from_config(deploy, str(root), device, ckpt_name)
    return NeroChunkService(server.engine, server.cfg, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8849)
    parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG, help="Nero inference YAML")
    parser.add_argument("--denoise-steps", type=int, help="Override YAML denoise_steps")
    parser.add_argument("--disable-inference-fast-path", action="store_true", help="Use eager inference for comparison")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    service = load_service(
        args.ckpt_dir,
        args.ckpt_name,
        args.device,
        fast_path=False if args.disable_inference_fast_path else None,
        denoise_steps=args.denoise_steps,
        config_path=args.config,
    )
    server = make_http_server(service, args.host, args.port)
    logging.info(
        "ready http://%s:%d checkpoint=%s mode=%s denoise_steps=%d fast_path=%s",
        *server.server_address[:2],
        args.ckpt_dir / args.ckpt_name,
        service.identity["model_representation"],
        service.cfg.inference.denoise_steps,
        service.cfg.optimization.inference_fast_path.enabled,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
