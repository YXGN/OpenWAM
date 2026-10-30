"""Replay UI task changes through the real client, HTTP service and frozen T5.

No robot connection: recorded poses/images feed Soma, and motion ownership
callbacks are stubs. The real PolicySession stop/clear methods invalidate plans.
The browser itself is outside this audit; its actual panel command handler runs.
"""

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace

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

    import torch
    from velabot.app_runtime.policy_compose import codec_for, schema_for
    from velabot.app_runtime.policy_control import PolicyConfig, PolicyController
    from velabot.app_runtime.policy_session import PolicySession
    from velabot.console.contracts import UiIntent
    from velabot.console.panels import CollectPanel, InferencePanel
    from velabot.contracts.clock import ManualClock
    from velabot.contracts.observation import ObservationUpdate
    from velabot.contracts.state import CameraFrame, EffectorFeedback, RobotState
    from velabot.hardware.profiles import load_descriptor
    from velabot.inference.action_state import SharedActionState
    from velabot.inference.client import negotiate
    from velabot.inference.observation import ObservationBuilder
    from velabot.inference.protocol import ProtocolTransport
    from velabot.inference.rotation import pose_matrix_from_rot6d
    from velabot.inference.strategies.receding_horizon import RecedingHorizonStrategy
    from velabot.inference.transport import HttpJsonTransport
    from velabot.soma import Soma

    from openwam.dataloader.utils.nero import read_jsonl
    from openwam.deploy.nero_chunk_server import ROLES, SERVER_ROLES, load_service, make_http_server
    from openwam.evaluation.nero_openloop import current_observation, load_episode
    from openwam.model.video_backbone.wan import encode

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    tasks = {r["task_index"]: r["task"] for r in read_jsonl(args.dataset / "meta/tasks.jsonl")}
    prompts = [tasks[1], tasks[2], tasks[1]]
    info = json.loads((args.dataset / "meta/info.json").read_text())
    episode = next(r for r in read_jsonl(args.dataset / "meta/episodes.jsonl") if r["episode_index"] == 242)
    states, _, _ = load_episode(args.dataset, info, episode)
    anchor = (len(states) - 33) // 2
    obs = current_observation(args.dataset, info, 242, anchor, states[anchor], prompts[0])
    descriptor = load_descriptor(args.client_root / "profiles/robots/agilex-nero-dual-arm.yaml")
    schema = schema_for(descriptor=descriptor, protocol="openwam")
    clock = ManualClock(start_monotonic_ns=1_000_000_000)
    stamp = clock.monotonic_ns()
    frames = {}
    for role, target in zip(ROLES, SERVER_ROLES, strict=True):
        bgr = np.asarray(obs["images"][target])[..., ::-1].copy()
        bgr.setflags(write=False)
        frames[role] = CameraFrame(role=role, image=bgr, host_monotonic_ns=stamp)
    poses, effectors = {}, {}
    for group, offset in (("left_arm", 0), ("right_arm", 10)):
        poses[group] = pose_matrix_from_rot6d(obs["state"][offset : offset + 9])
        effectors[group] = EffectorFeedback(
            group=group, position=obs["state"][offset + 9 : offset + 10], host_monotonic_ns=stamp
        )
    soma = Soma()
    soma.publish(
        ObservationUpdate(
            "recorded:observation",
            source_seq=1,
            read_monotonic_ns=stamp,
            state=RobotState(stamp, task_poses=poses, effectors=effectors),
            cameras=frames,
        )
    )
    builder = ObservationBuilder(
        SimpleNamespace(descriptor=lambda: descriptor),
        SimpleNamespace(required=ROLES),
        schema,
        soma=soma,
        clock=clock,
        camera_roles=ROLES,
        image_encoding="png",
        prompt=prompts[0],
    )
    getter = builder.payload_getter()  # Created once: it must read the updated task each time.
    assert getter() is not None, builder.stats()
    state = SharedActionState()
    strategy = RecedingHorizonStrategy(observation_getter=getter)
    session = SimpleNamespace(
        policy_running=False,
        task=prompts[0],
        policy_owner="offline_policy",
        inference=SimpleNamespace(state=state, pause=lambda: None),
        arbiter=SimpleNamespace(release_all=lambda *a, **k: None),
        control_run=SimpleNamespace(request_owner_stop=lambda *a, **k: None, pause_policy=lambda **k: None),
    )
    # Exercise the real lifecycle code; only robot ownership/output hooks are stubs.
    for name in ("stop_policy", "_clear_chunk", "_set_requester_paused"):
        setattr(session, name, MethodType(getattr(PolicySession, name), session))
    controller = PolicyController(
        session=session,
        builder=builder,
        connect=lambda *a: None,
        apply_rate=lambda *a: None,
        config=PolicyConfig(protocol="openwam", rate_hz=60, task=prompts[0]),
    )
    panel = InferencePanel(policy_session=session, controller=controller)
    recording = CollectPanel(recorder=SimpleNamespace(), session=session)
    reply = recording.handle(UiIntent("record.set_task", {"task": prompts[1]}))
    assert reply.status.value == "succeeded"
    assert getter()["prompt"] == prompts[0] and session.task == prompts[1]
    print("Recorded observation and client UI task routes ready", flush=True)

    service = load_service(args.ckpt_dir, args.ckpt_name)
    http = make_http_server(service, port=0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    transport = ProtocolTransport(
        HttpJsonTransport(f"http://127.0.0.1:{http.server_port}"), codec_for("openwam", schema, descriptor)
    )
    entered, release = threading.Event(), threading.Event()
    received, architecture_prompts, text_calls, records, outputs = [], [], [], [], []
    original_generate = service.engine.generate
    original_architecture = service.engine.architecture.generate
    original_text = encode.encode_text

    def observe_generate(conditions):
        received.append(conditions["prompt"])
        if len(received) == 1:
            entered.set()
            if not release.wait(30):
                raise TimeoutError("Prompt audit did not release old request")
        return original_generate(conditions)

    def observe_architecture(**kwargs):
        architecture_prompts.append(kwargs["prompt"])
        return original_architecture(**kwargs)

    def observe_text(values, **kwargs):
        text_calls.extend(values)
        return original_text(values, **kwargs)

    service.engine.generate = observe_generate
    service.engine.architecture.generate = observe_architecture
    encode.encode_text = observe_text
    try:
        negotiate(transport, schema)
        old = strategy.should_request(state)
        session.policy_running = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(transport.request, "/infer", old.request.observation)
            assert entered.wait(10), "HTTP request did not reach engine"
            refused = panel.handle(UiIntent("inference.set_task", {"task": prompts[1]}))
            assert refused.status.value == "rejected" and getter()["prompt"] == prompts[0]
            generation = state.generation
            stopped = panel.handle(UiIntent("inference.stop"))
            assert stopped.status.value == "succeeded" and state.generation > generation
            changed = panel.handle(UiIntent("inference.set_task", {"task": prompts[1]}))
            assert changed.status.value == "succeeded" and getter()["prompt"] == prompts[1]
            release.set()
            old_reply = future.result(timeout=120)
        old_actions = np.asarray(old_reply["actions"], dtype=np.float32)
        assert not strategy.on_response(state, old, old_actions), "Old prompt response was installed after stop"
        assert state.pop() is None
        outputs.append(old_actions)
        for prompt in prompts[1:]:
            changed = panel.handle(UiIntent("inference.set_task", {"task": prompt}))
            assert changed.status.value == "succeeded"
            decision = strategy.should_request(state)
            assert decision.request.observation["prompt"] == prompt
            reply = transport.request("/infer", decision.request.observation)
            actions = np.asarray(reply["actions"], dtype=np.float32)
            assert strategy.on_response(state, decision, actions)
            outputs.append(actions)
            for row in actions:
                np.testing.assert_array_equal(state.pop()[0], row)
            assert state.pop() is None
        assert received == prompts and architecture_prompts == prompts
        assert text_calls == prompts[:2], "T5 should encode A/B once each and reuse A on return"
        cache = service.engine._prompt_embed_cache
        for prompt in prompts[:2]:
            context, lengths = cache[prompt]
            records.append(
                dict(
                    prompt=prompt,
                    context_shape=list(context.shape),
                    seq_lens=lengths.cpu().tolist(),
                    context_sha256=hashlib.sha256(
                        context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                )
            )
        assert records[0]["context_sha256"] != records[1]["context_sha256"]
        np.testing.assert_array_equal(outputs[0], outputs[2])
        diff = np.abs(outputs[0] - outputs[1])
        assert float(diff.max()) > 0, "Different prompt produced identical actions in this probe"
        result = dict(
            status="passed",
            identity=service.identity,
            episode=242,
            anchor=anchor,
            prompt_sequence=prompts,
            server_received=received,
            architecture_prompts=architecture_prompts,
            t5_actual_encode_calls=text_calls,
            embeddings=records,
            same_observation_and_seed=True,
            repeated_A_exact_equal=True,
            A_vs_B_max_abs=float(diff.max()),
            A_vs_B_changed_elements=int(np.count_nonzero(diff)),
            running_change_rejected=True,
            stop_invalidates_inflight_A=True,
            recording_task_does_not_change_inference_prompt=True,
            scope="Real panel handler/controller/ObservationBuilder/HTTP/model/T5; recorded Soma input; motion hooks stubbed; no browser or robot",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        np.savez_compressed(args.output / "predictions.npz", A=outputs[0], B=outputs[1], A_repeat=outputs[2])
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    finally:
        release.set()
        encode.encode_text = original_text
        service.engine.generate = original_generate
        service.engine.architecture.generate = original_architecture
        transport.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
