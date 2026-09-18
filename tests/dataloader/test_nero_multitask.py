"""Physical/time/camera contract tests using real v2 parquet and encoded videos."""

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.normalize import Normalizer, NormMode, load_mode_stats
from openwam.dataloader.utils.nero import (
    ACTION_MODE,
    CAMERAS,
    FK_COLUMNS,
    FK_NAMES,
    STATE_NAMES,
    UNIFY_MAP,
    XYZ_DIMS,
    absolute_actions,
    assemble_eef,
    episode_path,
    matrix_rot6d,
    relative_actions,
    rotation_matrix,
    sha256,
    validate_info,
)
from openwam.dataloader.utils.stats_computation.nero_multitask_stats import prepare_dataset
from openwam.dataloader.utils.video_io import decode_video_frames

REPO = Path(__file__).resolve().parents[2]


def write_video(path, count, channel):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=60)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for i in range(count):
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[..., channel] = 80 + i * 3
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    root = tmp_path_factory.mktemp("nero")
    (root / "meta").mkdir()
    (root / "meta/kinematics.urdf").write_text('<robot name="test"/>')
    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex_nero_dual_arm",
        "fps": 60,
        "action_semantics": "current_state",
        "chunks_size": 1000,
        "total_episodes": 3,
        "total_frames": 108,
        "total_tasks": 3,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"shape": [16], "names": list(STATE_NAMES)},
            "action": {"shape": [16], "names": list(STATE_NAMES)},
            **{c: {"shape": [360, 640, 3]} for c in CAMERAS},
            **{c: {"shape": [9], "names": list(FK_NAMES)} for c in FK_COLUMNS},
        },
        "fk": {
            "coordinate_frame": "body",
            "rotation": "rot6d_first_two_rotation_matrix_columns",
            "urdf_sha256": sha256(root / "meta/kinematics.urdf"),
            "chains": [{"side": s, "base_link": "body", "tip_link": f"{s}_gripper_flange"} for s in ("left", "right")],
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, tasks, eefs, total = [], [], [], 0
    for task, (eid, length) in enumerate(((0, 40), (1001, 35), (2, 33))):
        state = np.zeros((length, 16), dtype=np.float32)
        state[:, 7] = np.linspace(0, 0.1, length)
        state[:, 15] = 0.1 - state[:, 7]
        left, right = np.zeros((length, 9), dtype=np.float32), np.zeros((length, 9), dtype=np.float32)
        left[:, 0] = task + np.arange(length) * 0.01
        right[:, 1] = task - np.arange(length) * 0.02
        left[:, 3:] = matrix_rot6d(
            Rotation.from_euler("z", (30 + np.arange(length) * 2)[:, None], degrees=True).as_matrix()
        )
        right[:, 3:] = matrix_rot6d(
            Rotation.from_euler("x", (-20 + np.arange(length) * 3)[:, None], degrees=True).as_matrix()
        )
        data = {
            "observation.state": state.tolist(),
            "action": state.tolist(),
            FK_COLUMNS[0]: left.tolist(),
            FK_COLUMNS[1]: right.tolist(),
            "timestamp": np.arange(length, dtype=np.float32) / 60,
            "index": np.arange(total, total + length),
            "frame_index": np.arange(length),
            "episode_index": [eid] * length,
            "task_index": [task] * length,
        }
        path = episode_path(root, info, eid)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), path)
        prompt = f"task {task}\nsecond instruction"
        episodes.append({"episode_index": eid, "length": length, "tasks": [prompt]})
        tasks.append({"task_index": task, "task": prompt})
        eefs.append(assemble_eef(state, left, right))
        for channel, camera in enumerate(CAMERAS):
            write_video(episode_path(root, info, eid, camera), length, channel)
        total += length
    for name, rows in (("episodes", episodes), ("tasks", tasks)):
        (root / f"meta/{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    output = root / "prepared"
    report = prepare_dataset(root, output)
    cfg = OmegaConf.load(REPO / "configs/dataloader/nero_multitask.yaml")
    cfg.dataset_dir = str(root)
    cfg.normalization_stats_path = str(output / "normalization_stats.npy")
    return root, cfg, eefs, report


def test_noncommuting_relative_rotation_and_roundtrip():
    anchor = np.zeros(20, dtype=np.float32)
    target = np.zeros((2, 20), dtype=np.float32)
    for start in (0, 10):
        r0 = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        rf = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        anchor[start : start + 3] = [1, 2, 3]
        anchor[start + 3 : start + 9] = matrix_rot6d(r0)
        target[:, start : start + 3] = [[1.1, 2.2, 3.3], [1.2, 2.4, 3.6]]
        target[:, start + 3 : start + 9] = matrix_rot6d(rf)
        target[:, start + 9] = [0.05, 0.1]
    relative = relative_actions(target, anchor)
    np.testing.assert_allclose(relative[:, :3], [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]], atol=3e-7)
    np.testing.assert_allclose(rotation_matrix(relative[0, 3:9]), r0.T @ rf, atol=1e-7)
    np.testing.assert_allclose(absolute_actions(relative, anchor), target, atol=3e-7)
    # Identity delta is [1,0,0,0,1,0], never six zero differences.
    zero = relative_actions(anchor[None], anchor)
    np.testing.assert_allclose(zero[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-7)


def test_full_windows_shift_masks_and_task_boundaries(bucket):
    _, cfg, eefs, report = bucket
    reader = build_dataset(cfg)
    assert len(reader) == 8 + 3 + 1 == report["windows"]
    normalizer = Normalizer(NormMode.MIN_MAX, load_mode_stats(cfg.normalization_stats_path, ACTION_MODE))
    for index, ep, offset in ((0, 0, 0), (7, 0, 7), (8, 1, 0), (10, 1, 2), (11, 2, 0)):
        sample = reader[index]
        assert sample["action"].shape == (32, 80)
        assert sample["proprio"].shape == (1, 80)
        assert sample["video_mask"].all() and len(sample["video"]) == 9
        assert sample["action_mask"].sum() == 32 * 20
        assert sample["proprio_mask"].sum() == 20
        assert sample["action_mask"][0].nonzero().flatten().tolist() == UNIFY_MAP
        assert sample["prompt"] == f"task {ep}\nsecond instruction"
        relative = normalizer.unnormalize(sample["action"].numpy()[:, UNIFY_MAP])
        reconstructed = absolute_actions(relative, eefs[ep][offset])
        np.testing.assert_allclose(reconstructed, eefs[ep][offset + 1 : offset + 33], atol=5e-7)
        expected_proprio = normalizer.normalize(eefs[ep][offset : offset + 1])
        np.testing.assert_allclose(sample["proprio"].numpy()[:, UNIFY_MAP], expected_proprio, atol=3e-7)
    with pytest.raises(IndexError):
        reader[len(reader)]


def test_physical_gripper_and_exact_stats_population(bucket):
    _, cfg, _, report = bucket
    payload = np.load(cfg.normalization_stats_path, allow_pickle=True).item()
    # Linear trajectory: maximum future displacement = 32 steps, not 1.
    assert payload[ACTION_MODE]["max"][0] == pytest.approx(0.32, abs=1e-6)
    assert payload[ACTION_MODE]["min"][11] == pytest.approx(-0.64, abs=1e-6)
    assert report["task_windows"] == {0: 8, 1: 3, 2: 1}
    n = Normalizer(NormMode.MIN_MAX, load_mode_stats(cfg.normalization_stats_path, ACTION_MODE))
    widths = np.zeros((3, 20), dtype=np.float32)
    widths[:, [9, 19]] = np.array([0, 0.05, 0.1])[:, None]
    np.testing.assert_allclose(n.normalize(widths)[:, 9], [-1, 0, 1], atol=1e-6)
    assert not np.array_equal(payload[ACTION_MODE]["min"][XYZ_DIMS], payload[f"{ACTION_MODE}_state"]["min"][XYZ_DIMS])


def test_cameras_frame_indices_and_deploy_pixels(bucket):
    from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError

    root, cfg, _, _ = bucket
    reader = build_dataset(cfg)
    info = json.loads((root / "meta/info.json").read_text())
    sample = reader[7]
    prep = ObsPreprocessor.from_cfg(OmegaConf.create({"dataloader": OmegaConf.to_container(cfg)}))
    for j, frame_index in enumerate(range(7, 40, 4)):
        images = {
            key: decode_video_frames(str(episode_path(root, info, 0, camera)), [frame_index], 360, 640)[0]
            for key, camera in zip(("head_camera", "left_wrist_camera", "right_wrist_camera"), CAMERAS)
        }
        deployed = prep.preprocess({"images": images})["image"]
        np.testing.assert_array_equal(np.asarray(sample["video"][j]), np.asarray(deployed))
        pixels = np.asarray(deployed)
        assert np.argmax(pixels[100, 100]) == 0  # head red
        assert np.argmax(pixels[300, 50]) == 1  # left green
        assert np.argmax(pixels[300, 250]) == 2  # right blue
    with pytest.raises(ObsValidationError, match="all three"):
        prep.preprocess({"images": {"head_camera": images["head_camera"]}})


def test_stale_metadata_rejected(bucket):
    root, cfg, _, _ = bucket
    path = root / "meta/tasks.jsonl"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="fingerprint"):
            build_dataset(cfg)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("feature", ["observation.state", "action", *FK_COLUMNS])
def test_reordered_feature_semantics_rejected(bucket, feature):
    root, _, _, _ = bucket
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"][feature]["names"]
    names[0], names[-1] = names[-1], names[0]
    with pytest.raises(ValueError, match="names/order"):
        validate_info(root, info)


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_mode", "delta"),
        ("video_stride", 2),
        ("unify_action_map", ["0-19"]),
        ("camera_layout", list(reversed(CAMERAS))),
    ],
)
def test_incompatible_contract_rejected(bucket, key, value):
    _, cfg, _, _ = bucket
    changed = OmegaConf.create(OmegaConf.to_container(cfg))
    changed[key] = value
    with pytest.raises(ValueError):
        build_dataset(changed)


def test_official_hydra_recipe():
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=nero_multitask"])
    assert cfg.model.architecture.action_dim == cfg.model.architecture.state_dim == 80
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.batch_size == 16 and cfg.training.num_epochs == 5
    assert cfg.training.mixed_precision == "bf16" and cfg.training.zero_stage == 2
    assert cfg.dataloader.action_mode == ACTION_MODE


def test_deploy_normalizer_uses_absolute_state_stats(bucket):
    from openwam.deploy.model_loader import _build_normalizer, repr_contract_from_cfg

    _, dl, eefs, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl), "model": {"architecture": {"action_dim": 80}}})
    n = _build_normalizer(cfg, str(Path(dl.normalization_stats_path).parent))
    reader = build_dataset(dl)
    sample = reader[3]
    np.testing.assert_allclose(n.normalize(eefs[0][3:4]), sample["proprio"].numpy(), atol=3e-7)
    actions = absolute_actions(n.unnormalize(sample["action"].numpy()), eefs[0][3])
    np.testing.assert_allclose(actions, eefs[0][4:36], atol=5e-7)
    assert repr_contract_from_cfg(cfg)["representation"] == "eef"


def test_engine_restores_once_against_generating_state(bucket, monkeypatch):
    from types import SimpleNamespace

    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import _build_normalizer

    _, dl, eefs, _ = bucket
    cfg = OmegaConf.create(
        {
            "dataloader": OmegaConf.to_container(dl),
            "model": {"architecture": {"action_dim": 80}},
            "inference": {"denoise_steps": 1, "num_frames": 33, "video_num_frames": 9},
        }
    )
    normalizer = _build_normalizer(cfg, str(Path(dl.normalization_stats_path).parent))
    anchor = eefs[0][3].copy()
    expected = eefs[0][4:36].copy()
    sample = build_dataset(dl)[3]

    def generate(**kwargs):
        np.testing.assert_allclose(kwargs["proprio"], sample["proprio"].numpy(), atol=3e-7)
        # Simulate an observation changing while a chunk is being generated.
        anchor[0] += 10
        return {"actions": normalizer.unnormalize(sample["action"].numpy()), "video": None}

    architecture = SimpleNamespace(
        normalize_deploy_proprio=normalizer.normalize, generate=generate, video_scheduler=None, action_scheduler=None
    )
    monkeypatch.setattr("openwam.deploy.engine.make_schedule", lambda *args, **kwargs: [])
    engine = JointInferenceEngine(cfg, architecture=architecture)
    result = engine.generate({"proprio": anchor[None]})
    np.testing.assert_allclose(result["actions"], expected, atol=5e-7)
    with pytest.raises(ValueError, match="absolute EEF20"):
        engine.generate({})


def test_bad_gripper_and_rotation_rejected():
    state = np.zeros((1, 16), dtype=np.float32)
    fk = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
    state[0, 7] = 0.2
    with pytest.raises(ValueError, match="Gripper width"):
        assemble_eef(state, fk, fk)
    state[0, 7] = 0.1
    fk[0, 3:] = 0
    with pytest.raises(ValueError, match="rot6d"):
        assemble_eef(state, fk, fk)
