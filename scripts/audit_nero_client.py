"""Offline real-model HTTP replay through the actual VelaBot codec and decoder.

No hardware connection. Requires a separate VelaBot source checkout; it is not
an OpenWAM package dependency. Reports raw RGB/state/prediction parity.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path[:0] = [str(p) for p in (args.client_root / "packages").glob("*/src")]

    from velabot.contracts.io_schema import IOSchema
    from velabot.contracts.state import CameraFrame
    from velabot.hardware.cameras import encode_at
    from velabot.hardware.profiles import load_descriptor
    from velabot.inference.action_state import SharedActionState
    from velabot.inference.client import negotiate
    from velabot.inference.intents import ActionDecoder
    from velabot.inference.protocol import ProtocolTransport
    from velabot.inference.protocol.openwam import OpenWAMCodec
    from velabot.inference.rotation import pose_matrix_from_rot6d
    from velabot.inference.transport import HttpJsonTransport

    from openwam.dataloader.utils.nero import read_jsonl
    from openwam.deploy.nero_chunk_server import ROLES, SERVER_ROLES, load_service, make_http_server
    from openwam.deploy.nero_contract import conditions_from_observation
    from openwam.evaluation.nero_openloop import current_observation, load_episode

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    descriptor = load_descriptor(args.client_root / "profiles/robots/agilex-nero-dual-arm.yaml")
    schema = IOSchema.from_descriptor(descriptor)
    decoder = ActionDecoder(schema, descriptor, owner="offline_audit")
    service = load_service(args.ckpt_dir, args.ckpt_name)
    http = make_http_server(service, port=0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    codec = OpenWAMCodec(schema)
    transport = ProtocolTransport(HttpJsonTransport(f"http://127.0.0.1:{http.server_port}"), codec)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    episodes = {e["episode_index"]: e for e in read_jsonl(args.dataset / "meta/episodes.jsonl")}
    tasks = {e["task_index"]: e["task"] for e in read_jsonl(args.dataset / "meta/tasks.jsonl")}
    records = []
    outputs = {}
    original = service.engine.generate
    received = []

    def observe(conditions):
        received.append(conditions)
        return original(conditions)

    service.engine.generate = observe
    try:
        negotiated = negotiate(transport, schema)
        assert negotiated.action_horizon == 32
        for eid in (51, 242, 347):
            states, task_ids, _ = load_episode(args.dataset, info, episodes[eid])
            for anchor in (0, (len(states) - 33) // 2, len(states) - 33):
                obs = current_observation(args.dataset, info, eid, anchor, states[anchor], tasks[int(task_ids[anchor])])
                direct_conditions = conditions_from_observation(service.preprocessor, obs, 42)
                direct = np.asarray(original(direct_conditions)["actions"], dtype=np.float32)
                images = {}
                for role, server_role in zip(ROLES, SERVER_ROLES, strict=True):
                    # CameraFrame's raw convention is BGR; the encoder must preserve
                    # the dataset's RGB values across BGR -> PNG -> PIL RGB.
                    rgb = np.asarray(obs["images"][server_role])
                    frame = CameraFrame(role=role, image=rgb[..., ::-1].copy(), host_monotonic_ns=1)
                    images[role] = encode_at(frame, size=(rgb.shape[1], rgb.shape[0]), quality=50, encoding="png")
                payload = dict(images=images, state=obs["state"].tolist(), prompt=obs["prompt"])
                received.clear()
                start = time.perf_counter()
                reply = transport.request("/infer", payload)
                elapsed = time.perf_counter() - start
                assert len(received) == 1, "One request must cause exactly one generation"
                actual_conditions = received[0]
                np.testing.assert_array_equal(
                    np.asarray(actual_conditions["first_frame_image"][0]),
                    np.asarray(direct_conditions["first_frame_image"][0]),
                )
                np.testing.assert_array_equal(actual_conditions["proprio"], direct_conditions["proprio"])
                assert actual_conditions["prompt"] == direct_conditions["prompt"]
                actual = np.asarray(reply["actions"], dtype=np.float32)
                np.testing.assert_array_equal(actual, direct)
                state = SharedActionState()
                state.publish(actual, anchor=obs["state"])
                for row in actual:
                    popped = state.pop()[0]
                    np.testing.assert_array_equal(popped, row)
                    intents = decoder.decode(popped)
                    for offset, pair in ((0, intents[:2]), (10, intents[2:])):
                        np.testing.assert_allclose(
                            pair[0].pose, pose_matrix_from_rot6d(row[offset : offset + 9]), atol=1e-12
                        )
                        np.testing.assert_array_equal(pair[1].value, row[offset + 9 : offset + 10])
                assert state.pop() is None
                name = f"ep{eid}_t{anchor}"
                outputs[name] = actual
                record = dict(
                    episode=eid,
                    anchor=anchor,
                    exact_equal=True,
                    max_abs=0.0,
                    images_equal=True,
                    state_equal=True,
                    decode_equal=True,
                    http_seconds=elapsed,
                    gripper_out_of_bounds=int(np.sum((actual[:, [9, 19]] < 0) | (actual[:, [9, 19]] > 0.1))),
                )
                records.append(record)
                print(record, flush=True)
        result = dict(
            status="passed",
            identity=service.identity,
            client_root=str(args.client_root),
            windows=records,
            scope="loopback HTTP + actual codec + PNG encoder + action queue/decoder; no hardware execution",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(args.output / "predictions.npz", **outputs)
    finally:
        transport.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
