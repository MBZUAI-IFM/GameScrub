#!/usr/bin/env python3
"""GameScrub inference for a long video in ONE pass (no chunking), using all GPUs of one node.

The whole video is denoised as one sequence, split across the GPUs with LTX-2 sequence
parallelism, instead of the 145-frame windows of generate.py --chunk-frames.
Needs the compiled ltx-kernels package (see README.md, "Multi-GPU setup").
Tested: 1777 frames (59 s at 30 fps) at 1280x704 on 8x A100 80GB, with --latent-out.

Usage (decode afterwards with decode_latents.py, see README.md):
    python inference/generate_long_multigpu.py --source-video long.mp4 --num-frames 1777 \
        --output-path out/long.mp4 --latent-out out/long_latent

Details:

- transformer: SequenceParallelBuilder (needs the compiled ltx-kernels All2All); bf16 weights,
  no fp8 -- the fp8-cast path needs sm_89+, A100 is sm_80.
- Gemma text encoder: Accelerate-sharded across ranks.
- video VAE decode: distributed 2D spatial tiles (one per rank) + temporal batches.
- reference (HUD clip) VAE encode: tiled along time. The public pipeline encodes the reference of
  an untiled stage in one pass, which OOMs an 80GB GPU for ~1800 frames.

Same GameScrub recipe as generate.py: LoRA + reference conditioning in both stages.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from collections.abc import Iterator
from multiprocessing import SimpleQueue
from pathlib import Path

# Bundled LTX-2 code (third_party/ltx2), so no separate LTX-2 checkout is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "ltx2"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.distributed_builders import AccelerateGemmaBuilder, SequenceParallelBuilder
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.tiling import (
    DimensionSizeConfig,
    DimensionTilingConfig,
    TileCountConfig,
    TileSizeConfig,
    balanced_tile_split,
)
from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.multigpu.runner import MGPURunner, rank_cuda_registry
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.utils.media_io import encode_media_segments
from ltx_pipelines.utils.model_paths import ModelPaths

logger = logging.getLogger(__name__)

_DRIVER_RANK = 0
CAPTION = "Remove the game HUD overlay from this footage, revealing the clean underlying gameplay video."  # same as generate.py


class ICLoraSPRunner(MGPURunner):
    @torch.inference_mode()
    def setup(  # noqa: PLR0913
        self,
        *,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        lora_path: str,
        lora_strength: float,
        vae_queue: SimpleQueue,
        sp_max_tokens: int,
        ref_encode_tile_frames: int,
        ref_encode_tile_overlap: int,
        decode_temporal_batches: int,
        save_latents_only: bool = False,
    ) -> None:
        import ltx_pipelines.ic_lora as ic_lora_mod  # noqa: PLC0415
        from ltx_pipelines.ic_lora import DEFAULT_IC_LORA_STAGES, ICLoraPipeline  # noqa: PLC0415

        # Tiled-in-time reference encode for every stage (runs on each rank, post-spawn).
        ref_tiling = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=ref_encode_tile_frames, overlap=ref_encode_tile_overlap),
            height=DimensionSizeConfig(),
            width=DimensionSizeConfig(),
        )
        ic_lora_mod._concrete_reference_encode_tiling = lambda stage, tiling_config: ref_tiling  # noqa: ARG005

        # Assemble the full-length reference pixels in host RAM, not on the GPU: at stage 2
        # (1280x704 x 1777 frames, bf16) that tensor is ~9.6GB on top of ~67GB resident.
        # VideoEncoder.tiled_encode moves each temporal tile to the GPU itself.
        import ltx_pipelines.iclora_utils as iclora_utils_mod  # noqa: PLC0415

        _orig_preprocess = iclora_utils_mod.video_preprocess

        def _cpu_video_preprocess(frames, height, width, dtype, device):  # noqa: ANN001, ANN202, ARG001
            return _orig_preprocess(frames, height, width, dtype, torch.device("cpu"))

        iclora_utils_mod.video_preprocess = _cpu_video_preprocess

        # --latent-out mode: the SP transformer keeps its ~42GB weights GPU-resident for the
        # whole job, which leaves too little room for a full-resolution VAE decode. Replace the
        # pipeline's decode step with one that saves each chunk's final (post-stage-2) latent
        # from the driver rank; decode_latents.py then decodes it on a clean GPU.
        self._latent_out: str | None = None
        if save_latents_only:
            runner = self

            def _save_latents_instead_of_decoding(chunks, *_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
                for index, chunk in enumerate(chunks):  # every rank iterates: denoise collectives
                    if dist.get_rank() == _DRIVER_RANK and runner._latent_out:
                        torch.save(
                            {
                                "video": chunk.video.detach().cpu(),
                                "audio": None if chunk.audio is None else chunk.audio.detach().cpu(),
                                "prev_video_carry_frames": chunk.layout.prev_video_carry_frames,
                            },
                            f"{runner._latent_out}.chunk{index:03d}.pt",
                        )
                return
                yield  # generator

            ic_lora_mod.decode_chunks = _save_latents_instead_of_decoding

        stage_1, stage_2 = DEFAULT_IC_LORA_STAGES
        stages = (stage_1, dataclasses.replace(stage_2, apply_ic_lora=True))  # GameScrub: LoRA + ref in stage 2

        registry = rank_cuda_registry()
        pipeline = ICLoraPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=[LoraPathStrengthAndSDOps(lora_path, lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)],
            registry=registry,
            quantization=None,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            stages=stages,
        )

        active_weights: dict[str, torch.Tensor] = {}
        stage = pipeline._diffusion_stage
        model_cfg = stage._transformer_builder.model_config().get("transformer", {})
        attn_mgr = AttentionManager(
            max_tokens=sp_max_tokens,
            num_heads=model_cfg["num_attention_heads"],
            head_dim=model_cfg["attention_head_dim"],
            tensor_dtype=pipeline.dtype,
            group=self.groups.transformer_group,
        )
        stage._transformer_builder = SequenceParallelBuilder(
            stage._transformer_builder.with_registry(registry), attn_mgr, dst=active_weights
        )

        pe = pipeline.prompt_encoder
        pe._text_encoder_builder = AccelerateGemmaBuilder.from_root(
            model_paths.text_encoder(), registry, self.groups.gemma_group, _DRIVER_RANK
        )
        pe._enhancer_text_encoder_builder = pe._text_encoder_builder

        h_tiles, w_tiles = balanced_tile_split(dist.get_world_size(self.groups.vae_group))
        pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(  # type: ignore[assignment]
            inner=pipeline.video_decoder._decoder_builder,
            queue=vae_queue,
            vae_group=self.groups.vae_group,
            vae_tiling=TileCountConfig(
                height=DimensionTilingConfig(num_tiles=h_tiles, overlap=4),
                width=DimensionTilingConfig(num_tiles=w_tiles, overlap=4),
            ),
            driver_rank=_DRIVER_RANK,
            registry=registry,
            num_temporal_batches=decode_temporal_batches,
        )
        self._pipeline = pipeline

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        output_path: str,
        source_video: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        latent_out: str | None = None,
    ) -> Iterator[str | None]:
        self._latent_out = latent_out
        chunks, _, _ = self._pipeline.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            video_conditioning=[(source_video, 1.0)],
            tiling_config=None,
            chunk_config=None,  # ONE chunk: the whole clip in a single sequence
        )
        is_driver = dist.get_rank() == _DRIVER_RANK
        if latent_out:
            for _ in chunks:
                pass
            yield latent_out if is_driver else None
            return
        if not is_driver:
            for _ in chunks:
                pass
        else:
            try:
                encode_media_segments(chunks, fps=frame_rate, output_path=output_path, video_chunks_number=None)
            except ValueError as error:
                raise RuntimeError("media encoding failed on the driver rank") from error
        yield output_path if is_driver else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-video", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--transformer-path", default=os.environ.get("GAMESCRUB_TRANSFORMER"), help="default: $GAMESCRUB_TRANSFORMER")
    p.add_argument("--text-encoder-path", default=os.environ.get("GAMESCRUB_TEXT_ENCODER"), help="default: $GAMESCRUB_TEXT_ENCODER")
    p.add_argument("--video-vae-path", default=os.environ.get("GAMESCRUB_VIDEO_VAE"), help="default: $GAMESCRUB_VIDEO_VAE")
    p.add_argument("--audio-vae-path", default=os.environ.get("GAMESCRUB_AUDIO_VAE"), help="default: $GAMESCRUB_AUDIO_VAE")
    p.add_argument("--spatial-upsampler-path", default=os.environ.get("GAMESCRUB_SPATIAL_UPSAMPLER"), help="default: $GAMESCRUB_SPATIAL_UPSAMPLER")
    p.add_argument("--lora", default=os.environ.get("GAMESCRUB_LORA"), help="default: $GAMESCRUB_LORA")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=CAPTION)
    p.add_argument("--sp-max-tokens", type=int, default=450_000)
    p.add_argument("--ref-encode-tile-frames", type=int, default=144)  # must be divisible by 8
    p.add_argument("--ref-encode-tile-overlap", type=int, default=24)
    p.add_argument("--decode-temporal-batches", type=int, default=16)
    p.add_argument(
        "--latent-out",
        help="Save the final latent to <latent-out>.chunkNNN.pt instead of decoding (decode with decode_latents.py)",
    )
    args = p.parse_args()
    for flag in ("transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path",
                 "spatial_upsampler_path", "lora"):
        value = getattr(args, flag)
        if not value or not Path(value).is_file():
            raise SystemExit(f"--{flag.replace('_', '-')}: missing or not a file: {value} (see .env.example)")
    if (args.num_frames - 1) % 8:
        raise SystemExit("--num-frames must be 8k+1 (e.g. 1777)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model_paths = ModelPaths.from_split(
        transformer_path=args.transformer_path,
        text_encoder_path=args.text_encoder_path,
        video_vae_path=args.video_vae_path,
        audio_vae_path=args.audio_vae_path,
    )
    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(ICLoraSPRunner)
    t0 = time.monotonic()
    controller.start(
        model_paths=model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        lora_path=args.lora,
        lora_strength=args.lora_strength,
        vae_queue=vae_queue,
        sp_max_tokens=args.sp_max_tokens,
        ref_encode_tile_frames=args.ref_encode_tile_frames,
        ref_encode_tile_overlap=args.ref_encode_tile_overlap,
        decode_temporal_batches=args.decode_temporal_batches,
        save_latents_only=bool(args.latent_out),
    )
    print(f"fleet started in {time.monotonic() - t0:.1f}s", flush=True)
    try:
        t1 = time.monotonic()
        for _ in controller.stream(
            output_path=args.output_path,
            source_video=args.source_video,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            latent_out=args.latent_out,
        ):
            pass
        print(f"saved {args.latent_out or args.output_path} in {time.monotonic() - t1:.1f}s", flush=True)
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
