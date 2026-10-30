import json
import threading
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openwam.deploy.inference_trace import Capture, TraceRecorder
from tests.test_base_generate_static_indices import _SCHEDULE, _make_probe_arch
from tests.test_nero_chunk_server import bucket as bucket
from tests.test_nero_chunk_server import setup_service


@pytest.mark.parametrize("level", ["final", "steps"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generate_trace_is_observational_and_preserves_padding(level, dtype):
    arch = _make_probe_arch(dtype)
    mask = torch.tensor([True, False, True, False, False, True, False, False])
    args = dict(
        schedule=_SCHEDULE, prompt="pick cylinder", num_frames=5, seed=42, decode_video=False, active_action_mask=mask
    )
    with patch.object(torch.compiler, "cudagraph_mark_step_begin", lambda: None):
        baseline = arch.generate(**args)
        capture = Capture(level)
        recorded = arch.generate(**args, diagnostic_capture=capture)
    np.testing.assert_array_equal(baseline["actions"], recorded["actions"])
    assert not capture.errors
    events = {e["event"]: e["data"] for e in capture.events}
    assert len([k for k in events if k.startswith("step/")]) == (4 if level == "steps" else 0)
    final = capture.arrays[events["final_latents"]["action_latents"]["array"]]
    np.testing.assert_array_equal(final[0], baseline["actions"])
    np.testing.assert_array_equal(final[..., ~mask.numpy()], 0)
    initial = capture.arrays[events["initial"]["action_latents"]["array"]]
    assert not np.array_equal(initial, final)
    assert events["initial"]["action_latents"]["source_dtype"] == str(dtype)


def test_service_records_exact_inputs_raw_and_mapped_action(bucket, tmp_path):
    service, request, _, outputs = setup_service(bucket, gripper_close_offset_m=0.002)
    baseline = service.request("/infer", request)
    service.trace_recorder = TraceRecorder(tmp_path / "trace")
    response = service.request("/infer", request)
    service.trace_recorder.close()
    trace = response.pop("diagnostic_trace")
    assert trace["status"] == "queued"
    for k in baseline:
        if k != "server_total_sec":
            assert baseline[k] == response[k]
    folder = tmp_path / "trace" / trace["trace_id"]
    m = json.loads((folder / "manifest.json").read_text())
    assert m["status"] == "incomplete"  # Fake engine does not produce model latents.
    events = {e["event"]: e["data"] for e in m["events"]}
    assert events["request"] == request
    a = np.load(folder / "arrays.npz", allow_pickle=False)
    np.testing.assert_array_equal(a[events["model_output"]["actions_eef20_before_gripper_mapping"]["array"]], outputs)
    np.testing.assert_array_equal(events["response"]["actions"], baseline["actions"])
    assert events["conditions"]["seed"] == 42


def test_queue_overflow_explicit_and_snapshot_owned(tmp_path):
    recorder = TraceRecorder(tmp_path / "trace", max_pending=1)
    entered = threading.Event()
    release = threading.Event()
    original = recorder._write

    def blocked(c):
        entered.set()
        assert release.wait(3)
        return original(c)

    recorder._write = blocked
    c = Capture("final")
    arr = np.ones(3)
    c("sample", arr)
    arr[:] = 2
    assert recorder.submit(c) == "queued"
    assert entered.wait(3)
    assert recorder.submit(Capture("final")) == "queued"
    assert recorder.submit(Capture("final")) == "dropped_queue_full"
    release.set()
    recorder.close()
    assert recorder.counts["dropped"] == 1
    np.testing.assert_array_equal(np.load(recorder.root / c.trace_id / "arrays.npz")["0000/sample"], 1)


def test_write_failure_is_visible_and_does_not_kill_worker(tmp_path):
    r = TraceRecorder(tmp_path / "trace")
    r._write = lambda c: (_ for _ in ()).throw(OSError("full disk"))
    assert r.submit(Capture("final")) == "queued"
    r.close()
    assert r.counts["write_failed"] == 1


def test_validation_error_is_recorded_and_reraised(bucket, tmp_path):
    service, request, *_ = setup_service(bucket)
    service.trace_recorder = TraceRecorder(tmp_path / "trace")
    request["prompt"] = ""
    with pytest.raises(ValueError):
        service.request("/infer", request)
    service.trace_recorder.close()
    manifest = json.loads(next((tmp_path / "trace").glob("*/manifest.json")).read_text())
    assert manifest["events"][-1]["event"] == "inference_error"


def test_bad_capture_value_marks_incomplete(tmp_path):
    r = TraceRecorder(tmp_path / "trace")
    c = Capture("final")
    c("bad", object())
    r.submit(c)
    r.close()
    assert json.loads((r.root / c.trace_id / "manifest.json").read_text())["status"] == "incomplete"


def test_reader_verifies_hash_and_exports_owned_arrays(tmp_path):
    from openwam.deploy.trace_inspect import read_trace

    r = TraceRecorder(tmp_path / "trace")
    c = Capture("final")
    c("final_latents", {"video_latents": np.ones((1, 2, 3))})
    r.submit(c)
    r.close()
    folder = r.root / c.trace_id
    _, events = read_trace(folder)
    np.testing.assert_array_equal(events["final_latents"]["video_latents"], 1)
    with (folder / "arrays.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_trace(folder)


def test_trace_config_snapshot_does_not_resolve_training_only_hydra_fields():
    from omegaconf import OmegaConf

    from openwam.deploy.inference_trace import config_snapshot

    cfg = OmegaConf.create({"project": {"output_dir": "${now:%Y-%m-%d}"},
                           "inference": {"denoise_steps": 4}})
    snapshot = config_snapshot(cfg)
    assert snapshot["config"]["project"]["output_dir"] == "${now:%Y-%m-%d}"
    assert snapshot["effective_inference"]["denoise_steps"] == 4
    json.dumps(snapshot)
