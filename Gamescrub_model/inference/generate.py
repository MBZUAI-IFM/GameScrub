#!/usr/bin/env python3
"""GameScrub inference: remove the game HUD from gameplay videos (one GPU).

Runs the LTX-2.5 two-stage IC-LoRA pipeline with the GameScrub LoRA. The input
video is the IC-LoRA reference; LoRA and reference conditioning are active in BOTH stages:

    stage 1   640x352, 8 denoising steps          (LoRA + reference)
    upsample  x2 latent spatial upsampler
    stage 2   1280x704, 3 refinement steps        (LoRA + reference)

Input, one of:
    --input-video clip.mp4             a single video
    --input-list clips.jsonl           many videos; one JSON object per line:
                                       {"id": "<output name>", "source_video": "<path>"}
                                       (relative paths are resolved against the jsonl's folder)

Output: <output-dir>/<id>.mp4 (+ <id>.json with the settings and run time).
Finished outputs are skipped, so an interrupted run can simply be restarted.

Videos longer than one clip (default 145 frames) need --num-frames set to the video
length and --chunk-frames 145: the video is generated in overlapping 145-frame windows.
See README.md.

Examples:
    python inference/generate.py --input-video clip.mp4 --output-dir out/
    # 4 GPUs, each takes every 4th video of the list:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python inference/generate.py --input-list clips.jsonl \
          --output-dir out/ --rank $i --world-size 4 &
    done; wait
Checkpoint paths come from the flags or from the environment variables in .env.example.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Bundled LTX-2 code (third_party/ltx2), so no separate LTX-2 checkout is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "ltx2"))

import torch  # noqa: E402

CAPTION = "Remove the game HUD overlay from this footage, revealing the clean underlying gameplay video."

CHECKPOINT_ARGS = {  # flag -> environment variable used as its default
    "transformer_path": "GAMESCRUB_TRANSFORMER",
    "text_encoder_path": "GAMESCRUB_TEXT_ENCODER",
    "video_vae_path": "GAMESCRUB_VIDEO_VAE",
    "audio_vae_path": "GAMESCRUB_AUDIO_VAE",
    "spatial_upsampler_path": "GAMESCRUB_SPATIAL_UPSAMPLER",
    "lora": "GAMESCRUB_LORA",
}


def _read_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_video:
        video = args.input_video.resolve()
        return [{"id": args.output_name or video.stem, "source_video": str(video)}]
    rows = []
    with args.input_list.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                source = Path(row["source_video"])
                if not source.is_absolute():
                    source = args.input_list.parent / source
                row["source_video"] = str(source)
                rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{args.input_list}: duplicate ids")
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def _build_pipeline(args: argparse.Namespace):
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.registry import ModelRegistry
    from ltx_pipelines.ic_lora import DEFAULT_IC_LORA_STAGES, ICLoraPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths

    model_paths = ModelPaths.from_split(
        transformer_path=str(args.transformer_path),
        text_encoder_path=str(args.text_encoder_path),
        video_vae_path=str(args.video_vae_path),
        audio_vae_path=str(args.audio_vae_path),
    )
    # GameScrub keeps the LoRA and the reference video in stage 2 as well
    # (the LTX-2 default applies them in stage 1 only).
    stage_1, stage_2 = DEFAULT_IC_LORA_STAGES
    stage_2 = dataclasses.replace(stage_2, apply_ic_lora=True)
    return ICLoraPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=str(args.spatial_upsampler_path),
        loras=[LoraPathStrengthAndSDOps(str(args.lora), args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)],
        registry=ModelRegistry(),  # keeps the transformer loaded between videos
        stages=(stage_1, stage_2),
    )


def _run(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.chunks.layout import ChunkConfig
    from ltx_pipelines.utils.media_io import encode_video

    chunk_config = (
        ChunkConfig(chunk_pixel_frames=args.chunk_frames, next_video_carry_frames=args.chunk_carry_frames)
        if args.chunk_frames
        else None
    )
    assigned = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def done(row: dict[str, Any]) -> bool:
        path = args.output_dir / f"{row['id']}.mp4"
        return path.is_file() and path.stat().st_size > 0

    pending = [row for row in assigned if args.overwrite or not done(row)]
    print(f"rank {args.rank}/{args.world_size}: {len(assigned)} assigned, {len(pending)} to do", flush=True)
    if not pending:
        return 0

    pipeline = _build_pipeline(args)
    for position, row in enumerate(pending, 1):
        case_id, source_video = row["id"], row["source_video"]
        output = args.output_dir / f"{case_id}.mp4"
        partial = output.with_name(f".{case_id}.partial.mp4")
        partial.unlink(missing_ok=True)
        print(f"rank {args.rank} [{position}/{len(pending)}] {case_id}", flush=True)
        started = time.monotonic()

        # no_grad (not inference_mode): the Triton neighborhood-attention fallback in the VAE
        # decoder saves tensors for backward, which inference_mode forbids.
        with torch.no_grad():
            result = pipeline(
                prompt=args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                images=[],
                video_conditioning=[(source_video, 1.0)],
                tiling_config=AUTO_TILING,
                chunk_config=chunk_config,
            )
        encode_video(
            video=result.video,
            fps=args.frame_rate,
            audio=result.audio,
            output_path=str(partial),
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )
        partial.replace(output)

        elapsed = time.monotonic() - started
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "id": case_id,
                    "source_video": source_video,
                    "lora": str(args.lora),
                    "lora_strength": args.lora_strength,
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "num_frames": args.num_frames,
                    "frame_rate": args.frame_rate,
                    "chunk_frames": args.chunk_frames,
                    "chunk_carry_frames": args.chunk_carry_frames if args.chunk_frames else None,
                    "seconds": round(elapsed, 1),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"rank {args.rank}: saved {output} ({elapsed:.1f} s)", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", type=Path, help="one input video")
    src.add_argument("--input-list", type=Path, help='jsonl, one {"id": ..., "source_video": ...} per line')
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-name", help="output file name (without .mp4) for --input-video; default: input name")

    ck = p.add_argument_group("checkpoints (default: environment variables, see .env.example)")
    for name, env in CHECKPOINT_ARGS.items():
        flag = "--" + name.replace("_", "-")
        ck.add_argument(flag, type=Path, default=os.environ.get(env), help=f"default: ${env}")
    p.add_argument("--lora-strength", type=float, default=1.0)

    # Defaults = the GameScrub training/benchmark setting: 1280x704, 145 frames at 30 fps.
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704, help="must be divisible by 64 (use 704, not 720)")
    p.add_argument("--num-frames", type=int, default=145, help="frames to generate; must be 8k+1")
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--chunk-frames", type=int, help="for long videos: window length (8k+1, use 145)")
    p.add_argument("--chunk-carry-frames", type=int, default=25, help="overlap between windows (8k+1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=CAPTION)

    p.add_argument("--limit", type=int, help="only the first N videos of --input-list")
    p.add_argument("--rank", type=int, default=0, help="this process's index when splitting the list over GPUs")
    p.add_argument("--world-size", type=int, default=1, help="number of processes splitting the list")
    p.add_argument("--overwrite", action="store_true", help="regenerate outputs that already exist")
    return p


def main() -> int:
    args = build_parser().parse_args()
    for name, env in CHECKPOINT_ARGS.items():
        value = getattr(args, name)
        if value is None:
            raise SystemExit(f"missing --{name.replace('_', '-')} (or set {env}, see .env.example)")
        if not Path(value).is_file():
            raise SystemExit(f"--{name.replace('_', '-')}: file not found: {value}")
    if args.height % 64 or args.width % 64:
        raise SystemExit("--height and --width must be divisible by 64 (e.g. 1280x704)")
    if (args.num_frames - 1) % 8:
        raise SystemExit("--num-frames must be 8k+1 (e.g. 145)")
    if args.chunk_frames and ((args.chunk_frames - 1) % 8 or (args.chunk_carry_frames - 1) % 8):
        raise SystemExit("--chunk-frames and --chunk-carry-frames must be 8k+1")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("need 0 <= --rank < --world-size")
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    return _run(args, _read_inputs(args))


if __name__ == "__main__":
    raise SystemExit(main())
