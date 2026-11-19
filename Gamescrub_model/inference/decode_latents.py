#!/usr/bin/env python3
"""Decode latents saved by generate_long_multigpu.py --latent-out into an MP4 on one GPU.

The multi-GPU sequence-parallel run keeps the transformer resident on every GPU, so its
full-resolution VAE decode runs out of memory; this decodes the saved final latent on a GPU
that holds only the video VAE, with the pipeline's AUTO_TILING (same as the chunked path).
Video only: the source clips have no audio track.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Bundled LTX-2 code (third_party/ltx2), so no separate LTX-2 checkout is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "ltx2"))

import torch  # noqa: E402

from ltx_core.model.video_vae import AUTO_TILING
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_pipelines.utils.helpers import ensure_tiling_config
from ltx_pipelines.utils.media_io import encode_video


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-prefix", required=True, help="the --latent-out value of the generation run")
    p.add_argument("--video-vae-path", default=os.environ.get("GAMESCRUB_VIDEO_VAE"), help="default: $GAMESCRUB_VIDEO_VAE")
    p.add_argument("--output-path", required=True)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = sorted(glob.glob(f"{glob.escape(args.latent_prefix)}.chunk*.pt"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one saved chunk (single-pass run), found {len(paths)}: {paths}")
    saved = torch.load(paths[0], map_location="cpu")
    device = torch.device("cuda")
    latent = saved["video"].to(device=device, dtype=torch.bfloat16)
    latent_frames = latent.shape[2]
    num_frames = (latent_frames - 1) * 8 + 1
    print(f"latent {tuple(latent.shape)} -> {num_frames} frames at {args.width}x{args.height}", flush=True)

    decoder = VideoDecoder(args.video_vae_path, torch.bfloat16, device)
    tiling = ensure_tiling_config(
        AUTO_TILING,
        video_shape=VideoPixelShape(
            batch=1, frames=num_frames, height=args.height, width=args.width, fps=args.frame_rate
        ),
        vae_checkpoint_path=args.video_vae_path,
        device=device,
        vae_dtype=torch.bfloat16,
    )
    print(f"decode tiling: {tiling}", flush=True)

    started = time.monotonic()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        batches = list(decoder(latent, tiling, generator))
    print(f"decoded {len(batches)} batch(es) in {time.monotonic() - started:.1f}s", flush=True)

    encode_video(
        video=iter(batches),
        fps=args.frame_rate,
        audio=None,
        output_path=args.output_path,
        video_chunks_number=len(batches),
    )
    print(f"wrote {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
