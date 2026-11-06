"""Versioned, complete-window bf16 VAE cache for the fixed Nero TI2V recipe."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

from openwam.dataloader.utils.nero import dataset_fingerprint, sha256

FORMAT = "nero-wan22-window-latents-v1"
SHAPE = (48, 3, 24, 20)
GEOMETRY = (9, 384, 320)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def module_digest(module):
    """Hash the actual execution weights, including names, shapes and dtypes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_contract(dataset):
    if dataset._color_jitter is not None:
        raise ValueError("Fixed video cache forbids color augmentation")
    root = Path(__file__).resolve().parents[2]
    sources = [
        "dataloader/nero_multitask.py",
        "dataloader/nero_multitask_absolute.py",
        "dataloader/bases/lerobot_v3_reader.py",
        "dataloader/utils/video_io.py",
        "dataloader/transforms/multiview.py",
        "model/video_backbone/wan/preprocess.py",
        "model/video_backbone/wan/models/vae.py",
    ]
    import av
    import PIL

    return {
        "dataset_fingerprint": dataset_fingerprint(dataset._dataset_dir, dataset._info, dataset._episodes),
        "windows": len(dataset),
        "cameras": list(dataset._camera_layout),
        "geometry": list(GEOMETRY),
        "raw_window": dataset._num_frames,
        "video_stride": dataset._video_stride,
        "window_stride": dataset._window_stride,
        "code": {p: sha256(root / p) for p in sources},
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "pillow": PIL.__version__,
        "pyav": av.__version__,
    }


class NeroVideoCache:
    """Read only completed shards, without caching labels or normalized state."""

    def __init__(self, directory, dataset):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("format") != FORMAT
            or m.get("complete") is not True
            or m.get("shape") != list(SHAPE)
            or m.get("dtype") != "bfloat16"
        ):
            raise ValueError("Incomplete or incompatible video cache")
        if m["source"] != source_contract(dataset):
            raise ValueError("Stale video cache: dataset/preprocessing/runtime mismatch")
        self.length = len(dataset)
        self.shard_size = int(m["shard_size"])
        if self.shard_size <= 0:
            raise ValueError("Invalid cache shard size")
        self.shards = m["shards"]
        expected = 0
        for sid, record in enumerate(self.shards):
            count = min(self.shard_size, self.length - expected)
            if (
                record["start"] != expected
                or record["count"] != count
                or count <= 0
                or record["file"] != f"shard_{sid:05d}.safetensors"
            ):
                raise ValueError("Video cache has missing/reordered windows")
            path = self.root / record["file"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"Truncated video cache: {path}")
            expected += count
        if expected != self.length:
            raise ValueError("Video cache does not cover the full dataset")
        self._handles = OrderedDict()

    def __getstate__(self):
        return {**self.__dict__, "_handles": OrderedDict()}

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        sid, offset = divmod(int(index), self.shard_size)
        if sid not in self._handles:
            handle = safe_open(self.root / self.shards[sid]["file"], framework="pt", device="cpu")
            handle.__enter__()
            view = handle.get_slice("latents")
            if view.get_shape() != [self.shards[sid]["count"], *SHAPE] or view.get_dtype() != "BF16":
                handle.__exit__(None, None, None)
                raise ValueError("Invalid cached latent tensor")
            self._handles[sid] = handle
            if len(self._handles) > 16:
                self._handles.popitem(last=False)[1].__exit__(None, None, None)
        self._handles.move_to_end(sid)
        return self._handles[sid].get_slice("latents")[offset].clone()

    def validate_backbone(self, backbone):
        from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
        from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant

        if (
            not isinstance(getattr(backbone, "vae", None), WanVideoVAE38)
            or not isinstance(getattr(backbone, "_variant", None), TI2VVariant)
            or backbone.video_encoder is not None
            or backbone.dtype != torch.bfloat16
        ):
            raise ValueError("Nero video cache requires native bf16 Wan2.2 TI2V")
        if any(m.training for m in backbone.vae.modules()) or any(p.requires_grad for p in backbone.vae.parameters()):
            raise ValueError("Video cache requires a frozen, eval-mode VAE")
        if module_digest(backbone.vae) != self.manifest["vae_sha256"]:
            raise ValueError("Video cache VAE weights do not match the loaded model")
        backbone._nero_preencoded_video_enabled = True


def prepare_cached_ti2v(backbone, latents, geometry, text):
    """Same TI2V clean-first-frame conditioning, bypassing only pixels and VAE."""
    from openwam.model.video_backbone.wan import encode

    if not getattr(backbone, "_nero_preencoded_video_enabled", False):
        raise ValueError("Preencoded video requires a validated dataset cache and VAE")
    if (
        tuple(geometry) != GEOMETRY
        or latents.ndim != 5
        or tuple(latents.shape[1:]) != SHAPE
        or latents.dtype != torch.bfloat16
        or len(text) != latents.shape[0]
    ):
        raise ValueError("Incompatible preencoded video shape/dtype/geometry")
    if any(m.training for m in backbone.vae.modules()) or any(p.requires_grad for p in backbone.vae.parameters()):
        raise ValueError("Video cache requires a frozen, eval-mode VAE")
    context, seq_lens = encode.encode_text_for_training(text, backbone=backbone)
    latents = latents.to(device=backbone.device, dtype=backbone.dtype)
    return dict(
        input_latents=latents,
        context=context,
        seq_lens=seq_lens,
        height=GEOMETRY[1],
        width=GEOMETRY[2],
        num_frames=GEOMETRY[0],
        vace_context=None,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=0,
        first_frame_latents=latents[:, :, 0:1].clone(),
        clip_feature=None,
        y=None,
    )
