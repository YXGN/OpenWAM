"""Opt-in observational traces. No robot I/O, replay, or inference decisions.

GPU -> CPU snapshots happen on the inference thread and have measurable cost.
Disk writes use a bounded queue. Loss/errors are explicit; recording never
turns a valid prediction into a robot command failure. A queued trace is not
a durable trace until its manifest exists and says complete.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

import numpy as np
from PIL import Image

_CURRENT = contextvars.ContextVar("openwam_trace", default=None)


def current_capture():
    return _CURRENT.get()


class Capture:
    def __init__(self, level):
        self.trace_id = uuid.uuid4().hex
        self.level = level
        self.arrays = {}
        self.events = []
        self.errors = []
        self.started_ns = time.monotonic_ns()

    def snapshot(self, value, key):
        if hasattr(value, "detach"):
            tensor = value.detach()
            # float32 exactly represents bfloat16 values; record original dtype.
            dtype = str(tensor.dtype)
            if dtype == "torch.bfloat16":
                tensor = tensor.float()
            array = tensor.cpu().numpy().copy()
        elif isinstance(value, (np.ndarray, Image.Image)):
            array = np.array(value, copy=True)
            dtype = str(array.dtype)
        else:
            if isinstance(value, dict):
                return {str(k): self.snapshot(v, key + "/" + str(k)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [self.snapshot(v, key + "/" + str(i)) for i, v in enumerate(value)]
            if isinstance(value, np.generic):
                return value.item()
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            raise TypeError(f"Unsupported trace value at {key}: {type(value)}")
        self.arrays[key] = array
        return {"array": key, "shape": list(array.shape), "dtype": str(array.dtype), "source_dtype": dtype}

    def __call__(self, event, data):
        if event.startswith("step/") and self.level != "steps":
            return
        try:
            self.events.append(
                {
                    "event": event,
                    "monotonic_ns": time.monotonic_ns(),
                    "data": self.snapshot(data, f"{len(self.events):04d}/{event}"),
                }
            )
        except Exception as exc:
            self.errors.append(f"{event}: {type(exc).__name__}: {exc}")
            logging.exception("trace_capture_failed trace_id=%s event=%s", self.trace_id, event)


class TraceRecorder:
    def __init__(self, root, *, level="final", max_pending=2, run_metadata=None):
        if level not in {"final", "steps"} or max_pending < 1:
            raise ValueError("Trace level must be final/steps and max_pending >= 1")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.level = level
        self._queue = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._lock = threading.Lock()
        self.counts = dict(queued=0, complete=0, incomplete=0, dropped=0, write_failed=0)
        meta = dict(
            run_metadata or {},
            schema="openwam.inference_trace/v1",
            pid=os.getpid(),
            level=level,
            max_pending=max_pending,
            wall_time_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            clock_domain="server_process",
            note="No camera exposure timestamps or physical execution inferred from this trace.",
        )
        (self.root / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        self._thread = threading.Thread(target=self._worker, name="openwam-trace-writer", daemon=True)
        self._thread.start()

    def submit(self, capture):
        with self._lock:
            if self._closed:
                self.counts["dropped"] += 1
                logging.error("trace_dropped closed trace_id=%s", capture.trace_id)
                return "dropped_closed"
            try:
                self._queue.put_nowait(capture)
                self.counts["queued"] += 1
                return "queued"
            except queue.Full:
                self.counts["dropped"] += 1
                logging.error("trace_dropped queue_full trace_id=%s", capture.trace_id)
                return "dropped_queue_full"

    def _write(self, capture):
        folder = self.root / capture.trace_id
        folder.mkdir()
        arrays_path = folder / "arrays.npz"
        # CPU/disk compression stays off the model thread.
        np.savez_compressed(arrays_path, **capture.arrays)
        status = "incomplete" if capture.errors else "complete"
        manifest = dict(
            trace_id=capture.trace_id,
            status=status,
            level=capture.level,
            inference_succeeded=any(e["event"] == "response" for e in capture.events),
            has_final_latents=any(e["event"] == "final_latents" for e in capture.events),
            started_ns=capture.started_ns,
            events=capture.events,
            errors=capture.errors,
            arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        )
        temporary = folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(folder / "manifest.json")
        return status

    def _worker(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                try:
                    status = self._write(item)
                except Exception:
                    status = "write_failed"
                    logging.exception("trace_write_failed trace_id=%s", item.trace_id)
                with self._lock:
                    self.counts[status] += 1
                logging.info("trace_status trace_id=%s status=%s", item.trace_id, status)
            finally:
                self._queue.task_done()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        (self.root / "summary.json").write_text(json.dumps(self.counts, indent=2))


def traced_request(method):
    """Thread-local context spans validation, preprocessing, generation and mapping."""

    @wraps(method)
    def wrapper(self, path, payload, **kwargs):
        recorder = getattr(self, "trace_recorder", None)
        if recorder is None or path != "/infer":
            return method(self, path, payload, **kwargs)
        capture = Capture(recorder.level)
        token = _CURRENT.set(capture)
        capture("request", payload)
        logging.info(
            "trace_started trace_id=%s request_id=%r session_id=%r",
            capture.trace_id,
            payload.get("request_id") if isinstance(payload, dict) else None,
            payload.get("session_id") if isinstance(payload, dict) else None,
        )
        response = None
        try:
            response = method(self, path, payload, **kwargs)
            capture("response", response)
            if not any(e["event"] == "final_latents" for e in capture.events):
                capture.errors.append("No final_latents captured: this trace cannot explain video generation")
                logging.error("trace_missing_final_latents trace_id=%s", capture.trace_id)
            return response
        except Exception as exc:
            capture("inference_error", {"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            _CURRENT.reset(token)
            state = recorder.submit(capture)
            if response is not None:
                response["diagnostic_trace"] = {
                    "trace_id": capture.trace_id,
                    "status": state,
                    "capture_errors": len(capture.errors),
                    "capture_elapsed_ms": (time.monotonic_ns() - capture.started_ns) / 1e6,
                }

    return wrapper


def config_snapshot(cfg):
    """Keep training-only interpolations without invoking Hydra resolvers.

    The inference section is independently resolved, since these are the
    settings actually consumed by the running deployment engine.
    """
    from omegaconf import OmegaConf

    return {
        "config": OmegaConf.to_container(cfg, resolve=False),
        "effective_inference": OmegaConf.to_container(cfg.inference, resolve=True),
    }
