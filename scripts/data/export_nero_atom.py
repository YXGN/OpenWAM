#!/usr/bin/env python
"""Export committed Velabot atom episodes to Nero's current-state LeRobot view.

Raw files are read-only. Reuse the previously deployed Velabot URDF FK engine;
resample complete observations (all cameras + feedback) together at 60 Hz.
No semantic trimming, command-based filtering, or prompt rewriting is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openwam.dataloader.utils.nero import CAMERAS, FK_COLUMNS, FK_NAMES, STATE_NAMES, assemble_eef, sha256  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resample_indices(times_ns):
    times = np.asarray(times_ns, dtype=np.int64)
    if len(times) < 33 or np.any(np.diff(times) <= 0):
        raise ValueError("Need at least 33 strictly increasing source timestamps")
    relative = (times - times[0]) / 1e9
    if np.max(np.diff(relative)) > 0.1:
        raise ValueError("Source gap exceeds 100 ms; requires explicit review")
    grid = np.arange(int(np.floor(relative[-1] * 60)) + 1) / 60
    right = np.searchsorted(relative, grid).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    selected = np.where(abs(relative[left] - grid) <= abs(relative[right] - grid), left, right)
    residual = relative[selected] - grid
    if np.max(abs(residual)) > 0.05:
        raise ValueError("Nearest observation exceeds 50 ms alignment tolerance")
    return selected, grid, residual


def load_fk(velabot_repo, descriptor_path, urdf):
    sys.path[:0] = [str(p) for p in (Path(velabot_repo) / "packages").glob("*/src")]
    from velabot.data.export.fk import build_urdf_fk_export
    from velabot.hardware.profiles import descriptor_from_mapping
    return build_urdf_fk_export(urdf, descriptor_from_mapping(json.loads(Path(descriptor_path).read_text())))


def export_episode(job):
    source, output, velabot_repo, urdf, eid, task_index, offset = job
    source, output = Path(source), Path(output)
    ep = source / "episodes" / f"episode_{eid:06d}"
    manifest_bytes = (ep / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (ep / "COMMITTED").read_text().strip() != manifest["episode_id"] or manifest["state"] != "committed":
        raise ValueError(f"{ep}: not committed")
    # Verify every listed source file, including observations not selected by resampling.
    for entry in manifest["files"]:
        path = (ep / entry["path"]).resolve()
        if not path.is_relative_to(ep.resolve()):
            raise ValueError("Source path escapes episode")
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Source integrity failure: {path}")
    frames = [json.loads(line) for line in (ep / "frames.jsonl").read_text().splitlines() if line.strip()]
    if len(frames) != manifest["frame_count"] or [f["index"] for f in frames] != list(range(len(frames))):
        raise ValueError(f"{ep}: frame count/index mismatch")
    selected, grid, residual = resample_indices([f["sample_monotonic_ns"] for f in frames])
    fk = load_fk(velabot_repo, source / "meta/descriptor.json", urdf)
    states, poses = [], [[], []]
    max_online_error = 0.0
    for f in frames:
        obs = f["observation"]
        state = np.asarray([v for side in ("left_arm", "right_arm")
                            for v in [*obs["groups"][side]["q"], *obs["effectors"][side]["position"]]], dtype=np.float32)
        if state.shape != (16,):
            raise ValueError("Unexpected feedback dimensions")
        states.append(state)
        for arm, chain in enumerate(fk._chains):
            pose = np.asarray(chain.forward(state[arm * 8:arm * 8 + 7]))
            recorded = np.asarray(obs["task_poses"][chain.spec.group])
            expected = np.concatenate((recorded[:3, 3], recorded[:3, 0], recorded[:3, 1]))
            error = float(np.max(abs(pose - expected)))
            max_online_error = max(max_online_error, error)
            if error > 2e-5:
                raise ValueError(f"{ep}/{f['index']}: recorded geometry disagrees with Nero FK: {error}")
            poses[arm].append(pose)
    state = np.asarray(states)[selected]
    pose_arrays = [np.asarray(p, dtype=np.float32)[selected] for p in poses]
    assemble_eef(state, *pose_arrays)
    n = len(selected)
    columns = {"observation.state": state, "action": state, **dict(zip(FK_COLUMNS, pose_arrays))}
    table = {k: pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1), type=pa.float32()), v.shape[1])
             for k, v in columns.items()}
    table.update({k: pa.array(v) for k, v in {
        "timestamp": grid.astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, eid, dtype=np.int64), "task_index": np.full(n, task_index, dtype=np.int64),
        "index": np.arange(offset, offset + n, dtype=np.int64),
        "source_frame_index": selected, "source_sample_monotonic_ns": np.array([f['sample_monotonic_ns'] for f in frames])[selected],
        "source_time_residual_s": residual,
    }.items()})
    path = output / f"data/chunk-000/episode_{eid:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), path)
    for key in CAMERAS:
        role = key.split('.')[-1]
        dest = output / f"videos/chunk-000/{key}/episode_{eid:06d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'image2pipe', '-framerate', '60', '-vcodec', 'mjpeg',
               '-i', '-', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'fast', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-g', '16', '-bf', '0', '-movflags', '+faststart', str(dest)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                for i in selected:
                    cameras = {c['role']: c for c in frames[i]['observation']['cameras']}
                    process.stdin.write((ep / cameras[role]['file']).read_bytes())
                process.stdin.close()
                error = process.stderr.read().decode()
                if process.wait():
                    raise RuntimeError(f"Video encode failed: {error}")
            except BaseException:
                process.kill()
                raise
        # Decode every exported video frame, not only container headers.
        with av.open(str(dest)) as container:
            decoded = 0
            for image in container.decode(video=0):
                if (image.width, image.height) != (640, 360):
                    raise ValueError(f"Unexpected camera size: {dest}")
                decoded += 1
            if decoded != n:
                raise ValueError(f"Video length mismatch: {dest}")
    if (ep / 'manifest.json').read_bytes() != manifest_bytes:
        raise ValueError("Source manifest changed during export")
    report = dict(episode_index=eid, source_episode_id=manifest['episode_id'], task=manifest['task'],
                  source_frames=len(frames), exported_frames=n, source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  source_frames_sha256=sha256(ep / 'frames.jsonl'), source_files_verified=len(manifest['files']),
                  duplicate_samples=n-len(np.unique(selected)), unselected_samples=len(frames)-len(np.unique(selected)),
                  max_time_residual_ms=float(np.max(abs(residual))*1000), max_online_fk_error=max_online_error,
                  max_source_gap_ms=float(np.max(np.diff([f['sample_monotonic_ns'] for f in frames]))/1e6),
                  residual_over_one_period=int(np.count_nonzero(abs(residual)>1/60)),
                  source_duration_s=(frames[-1]['sample_monotonic_ns']-frames[0]['sample_monotonic_ns'])/1e9,
                  empty_command_frames=sum(not f.get('action') for f in frames), all_exported_video_frames_decoded=True)
    write_json(output / f"provenance/episode_{eid:06d}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--velabot-repo', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; never overwrite an export')
    if output.is_relative_to(source):
        raise ValueError('Derivative must be outside raw source')
    eps = sorted(p for p in (source/'episodes').glob('episode_*') if p.is_dir())
    if any(p.name == 'episode_000014' for p in eps):
        raise ValueError('Excluded episode_000014 unexpectedly present; inspect source')
    (output/'meta').mkdir(parents=True)
    (output/'provenance').mkdir()
    descriptor = source/'meta/descriptor.json'
    fk = load_fk(args.velabot_repo, descriptor, 'builtin:nero')
    fk.copy_urdf(output/'meta')
    urdf = str(output/'meta/kinematics.urdf')
    tasks, episodes, jobs, offset = {}, [], [], 0
    for ep in eps:
        m = json.loads((ep/'manifest.json').read_text())
        task = m['task']
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f'Blank prompt: {ep}')
        tid = tasks.setdefault(task, len(tasks))
        times = [json.loads(line)['sample_monotonic_ns'] for line in (ep/'frames.jsonl').read_text().splitlines() if line.strip()]
        n = len(resample_indices(times)[0])
        eid = int(ep.name.split('_')[-1])
        episodes.append(dict(episode_index=eid, tasks=[task], length=n, source_episode_id=m['episode_id']))
        jobs.append((str(source), str(output), args.velabot_repo, urdf, eid, tid, offset))
        offset += n
    provenance = dict(source=str(source), raw_policy='preserve_all_committed_episodes_no_semantic_trimming',
                      excluded_source_ids=['episode_000014'], temporal_policy='60Hz_nearest_complete_observation; no state interpolation',
                      max_source_gap_s=0.1, max_alignment_residual_s=0.05, action_semantics='current_state',
                      velabot_repo=str(Path(args.velabot_repo).resolve()), descriptor_sha256=sha256(descriptor),
                      exporter_sha256=sha256(__file__),
                      quality_flags={'preparation_prefix_untrimmed':[43,68,69,124,125,130],
                                     'recovery_sequence_retained':[140], 'terminal_support_review':[52,60,110,111,120],
                                     'user_confirmed_red_frame_description_unchanged':[131,132,133,134]})
    shutil.copyfile(descriptor, output/'provenance/source_descriptor.json')
    shutil.copyfile(source/'meta/dataset.json', output/'provenance/source_dataset.json')
    provenance['velabot_python_sha256'] = {str(p.relative_to(Path(args.velabot_repo))):sha256(p)
        for p in Path(args.velabot_repo).glob('packages/*/src/velabot/**/*.py')}
    write_json(output/'provenance/build.json', provenance)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(export_episode, jobs):
            reports.append(report)
            print(json.dumps(dict(completed=len(reports), total=len(jobs), **report)), flush=True)
    features = {k:dict(dtype='float32', shape=[16], names=list(STATE_NAMES)) for k in ['observation.state','action']}
    features.update({k:dict(dtype='float32',shape=[9],names=list(FK_NAMES)) for k in FK_COLUMNS})
    features.update({k:dict(dtype='float32' if k=='timestamp' else 'int64',shape=[1],names=None)
                     for k in ['timestamp','frame_index','episode_index','index','task_index']})
    features.update({k:dict(dtype='video',shape=[360,640,3],names=['height','width','channels'],
                           info={'video.fps':60,'video.codec':'h264','video.pix_fmt':'yuv420p'}) for k in CAMERAS})
    info = dict(codebase_version='v2.1',robot_type='agilex_nero_dual_arm', action_semantics='current_state',
                total_episodes=len(episodes),total_frames=offset,total_tasks=len(tasks),total_videos=3*len(episodes),
                total_chunks=1,chunks_size=1000,fps=60,splits={'train':f'0:{max(e["episode_index"] for e in episodes)+1}'},
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                features=features,fk={**fk.metadata(action_source='command'), 'action_semantics':'current_state'},
                notes='Episode IDs preserve raw IDs with 14 absent; explicit meta/episodes.jsonl is authoritative. See provenance.')
    for name, rows in [('episodes',episodes),('tasks',[dict(task_index=i,task=t) for t,i in tasks.items()])]:
        (output/f'meta/{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write_json(output/'meta/info.json',info)
    write_json(output/'provenance/complete.json',dict(complete=True,episodes=len(episodes),frames=offset,
               source_frames=sum(r['source_frames'] for r in reports),windows=offset-32*len(episodes),
               all_source_files_sha256_verified=True,all_video_frames_decoded=True))
    print(f'Export complete: {output}',flush=True)


if __name__ == '__main__':
    main()
