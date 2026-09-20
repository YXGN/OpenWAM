"""Compare exact Nero inference outputs and latency with each optimization.

Uses recorded observations and the production loader; never connects a robot.
Each mode starts with fresh optimization caches. Cold latency includes capture;
steady latency is measured after all task prompts have been encoded.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[51, 242, 347])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three steady repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch

    from openwam.dataloader.utils.nero import read_jsonl
    from openwam.deploy.nero_chunk_server import load_service
    from openwam.deploy.nero_contract import conditions_from_observation
    from openwam.evaluation.nero_openloop import current_observation, load_episode, predict_chunk, write_json

    info = json.loads((args.dataset / "meta/info.json").read_text())
    episodes = {row["episode_index"]: row for row in read_jsonl(args.dataset / "meta/episodes.jsonl")}
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(args.dataset / "meta/tasks.jsonl")}
    observations, case_names = [], []
    for eid in args.episodes:
        states, task_ids, _ = load_episode(args.dataset, info, episodes[eid])
        for anchor in (0, (len(states) - 33) // 2, len(states) - 33):
            observations.append(
                current_observation(args.dataset, info, eid, anchor, states[anchor], tasks[int(task_ids[anchor])])
            )
            case_names.append(f"episode_{eid}_anchor_{anchor}")
    # Isolate both cache keys: new prompt with identical cameras/state, then new
    # state with identical cameras/prompt. Return to A detects stale graph buffers.
    prompt_probe = copy.deepcopy(observations[0])
    prompt_probe["prompt"] = next(text for text in tasks.values() if text != prompt_probe["prompt"])
    state_probe = copy.deepcopy(observations[0])
    state_probe["state"] = observations[1]["state"].copy()
    observations.extend([prompt_probe, state_probe, copy.deepcopy(observations[0])])
    case_names.extend(["same_observation_new_prompt", "same_images_new_state", "repeat_first_A"])

    service = load_service(args.ckpt_dir, args.ckpt_name, fast_path=False)
    engine = service.engine
    architecture = engine.architecture
    conditions = [conditions_from_observation(service.preprocessor, obs, 42) for obs in observations]
    variants = {
        "eager": (False, False, False),
        "indices": (True, False, False),
        "context": (True, True, False),
        "graph": (True, True, True),
    }
    report = {
        "identity": service.identity,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "scope": "Recorded observations, official engine; no HTTP/network/robot in timing",
        "cases": [{"name": name, "prompt": obs["prompt"]} for name, obs in zip(case_names, observations, strict=True)],
        "variants": {},
    }
    saved = {}
    reference = None
    for mode, (enabled, context, graph) in variants.items():
        cfg = engine.cfg.optimization
        cfg.inference_fast_path.enabled = enabled
        cfg.request_context_cache.enabled = context
        cfg.cuda_graph.enabled = graph
        engine._init_optimizations()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        predict_chunk(engine, conditions[0])
        cold_sec = time.perf_counter() - started
        # Warm up all texts and the eager kernels before comparing steady times.
        for item in conditions:
            predict_chunk(engine, item)
        predictions = []
        for index, item in enumerate(conditions):
            values = predict_chunk(engine, item)
            predictions.append(values)
            for level, array in zip(("absolute", "native", "normalized"), values, strict=True):
                saved[f"{mode}_{index}_{level}"] = array
        if reference is None:
            reference = predictions
        comparisons = []
        for name, expected, actual in zip(case_names, reference, predictions, strict=True):
            checks = {}
            for level, before, after in zip(("absolute", "native", "normalized"), expected, actual, strict=True):
                checks[level] = {
                    "exact_equal": bool(np.array_equal(before, after)),
                    "max_abs": float(np.max(np.abs(before - after))),
                }
            comparisons.append({"case": name, **checks})
        seconds = []
        stable_equal = True
        for _ in range(args.repeats):
            started = time.perf_counter()
            values = predict_chunk(engine, conditions[0])
            seconds.append(time.perf_counter() - started)
            stable_equal &= all(np.array_equal(a, b) for a, b in zip(predictions[0], values, strict=True))
        runner = architecture._mot_cuda_graph_runner
        cache_status = copy.deepcopy(architecture._request_context_cache_status)
        result = {
            "cold_sec": cold_sec,
            "steady_sec": seconds,
            "steady_median_sec": float(np.median(seconds)),
            "steady_p95_sec": float(np.percentile(seconds, 95)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "graph": None if runner is None else runner.status(),
            "context_cache": cache_status,
            "comparisons": comparisons,
            "repeat_A_exact_equal": all(
                np.array_equal(a, b) for a, b in zip(predictions[0], predictions[-1], strict=True)
            ),
            "steady_repeat_exact_equal": bool(stable_equal),
            "prompt_changes_actions": not np.array_equal(predictions[0][0], predictions[-3][0]),
            "state_changes_actions": not np.array_equal(predictions[0][0], predictions[-2][0]),
        }
        report["variants"][mode] = result
        write_json(args.output / "result.json", report)
        np.savez_compressed(args.output / "predictions.npz", **saved)
        logging.info(
            "%s cold=%.3fs steady_median=%.3fs peak=%.2fGiB exact=%s graph=%s",
            mode,
            cold_sec,
            result["steady_median_sec"],
            result["peak_allocated_gib"],
            all(c[level]["exact_equal"] for c in comparisons for level in ("absolute", "native", "normalized")),
            None if runner is None else runner.status()["last_backend"],
        )

    for mode, result in report["variants"].items():
        assert result["repeat_A_exact_equal"] and result["steady_repeat_exact_equal"], mode
        assert result["prompt_changes_actions"] and result["state_changes_actions"], mode
        for comparison in result["comparisons"]:
            assert all(comparison[level]["exact_equal"] for level in ("absolute", "native", "normalized")), (
                mode,
                comparison,
            )
    graph_status = report["variants"]["graph"]["graph"]
    assert graph_status["captured"] and graph_status["last_backend"] == "cuda_graph", graph_status
    assert graph_status["capture_count"] == 1 and graph_status["replay_count"] > 10, graph_status
    assert report["variants"]["graph"]["context_cache"]["last_reason"] == "cache_hit"
    report["status"] = "passed"
    write_json(args.output / "result.json", report)
    logging.info("All exact-output, prompt/state freshness and actual-replay checks passed")


if __name__ == "__main__":
    main()
