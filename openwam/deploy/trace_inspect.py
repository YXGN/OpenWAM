"""Verify/export a recorded prediction; optionally decode its ACTUAL saved latent.

Does not rerun denoising, make HTTP requests, or send robot commands.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_trace(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "arrays.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("Trace array hash mismatch")
    arrays = np.load(path, allow_pickle=False)

    def materialize(value):
        if isinstance(value, dict) and "array" in value and "source_dtype" in value:
            arr = arrays[value["array"]]
            if list(arr.shape) != value["shape"] or str(arr.dtype) != value["dtype"]:
                raise ValueError("Array shape/dtype mismatch")
            return arr
        if isinstance(value, dict):
            return {k: materialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [materialize(v) for v in value]
        return value

    try:
        events = {e["event"]: materialize(e["data"]) for e in manifest["events"]}
    finally:
        arrays.close()
    return manifest, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Request directory containing manifest.json")
    parser.add_argument("--export-dir", type=Path, help="New directory for inputs/actions and optional decoded video")
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--ckpt-dir", type=Path)
    parser.add_argument("--ckpt-name")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.decode_video and not (args.ckpt_dir and args.ckpt_name and args.export_dir):
        parser.error("Decode requires --ckpt-dir, --ckpt-name and --export-dir")
    manifest, events = read_trace(args.trace)
    print(
        json.dumps(
            {
                "trace_id": manifest["trace_id"],
                "status": manifest["status"],
                "events": list(events),
                "errors": manifest["errors"],
            },
            indent=2,
        )
    )
    if args.export_dir is None:
        return
    out = args.export_dir
    out.mkdir(parents=True, exist_ok=False)
    request = events.get("request", {})
    # Fixed role names; no request-controlled filesystem paths.
    for role in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        if role in request.get("images", {}):
            (out / (role + ".png")).write_bytes(base64.b64decode(request["images"][role], validate=True))
    if "conditions" in events:
        Image.fromarray(events["conditions"]["first_frame_image"][0]).save(out / "actual_mosaic.png")
    if "model_output" in events:
        np.save(
            out / "actions_before_gripper_mapping.npy", events["model_output"]["actions_eef20_before_gripper_mapping"]
        )
    if "response" in events:
        np.save(out / "actions_returned.npy", np.asarray(events["response"]["actions"], dtype=np.float32))
    (out / "request_metadata.json").write_text(
        json.dumps({k: v for k, v in request.items() if k != "images"}, indent=2)
    )
    if args.decode_video:
        import torch

        from openwam.deploy.nero_chunk_server import load_service
        from openwam.deploy.nero_contract import DEFAULT_INFERENCE_CONFIG

        service = load_service(
            args.ckpt_dir, args.ckpt_name, args.device, fast_path=False, config_path=DEFAULT_INFERENCE_CONFIG
        )
        recorded = json.loads((args.trace.parent / "run.json").read_text())
        if service.identity != recorded["identity"]:
            raise ValueError("Decoder checkpoint/config/stats identity differs from trace")
        if "final_latents" not in events:
            raise ValueError("Trace has no final video latent")
        vb = service.engine.architecture.video_backbone
        latent = torch.from_numpy(events["final_latents"]["video_latents"]).to(
            device=service.engine.architecture.device, dtype=service.engine.architecture.dtype
        )
        with torch.inference_mode():
            frames = vb.decode_video(latent, tiled=events["final_latents"]["tiled"])
        video_dir = out / "generated_video"
        video_dir.mkdir()
        for i, frame in enumerate(frames):
            frame.save(video_dir / f"{i:03d}.png")
        (out / "decode_metadata.json").write_text(
            json.dumps(
                {
                    "source": "actual saved final latent; no denoising replay",
                    "frames": len(frames),
                    "identity": service.identity,
                    "trace_id": manifest["trace_id"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
