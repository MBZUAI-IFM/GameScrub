"""Stage 4: masked-latent diffusion conditioned on the ProPainter pre-fill +
caption, then a hard-cut pixel composite back onto the original (HUD) clip.

LTX-2.5's distilled transformer is used unmodified (no LoRA, no architecture
change). The only custom part is `CellHitRegionMask`, which lets the
diffusion process denoise only the latent cells under the HUD mask and
freezes everything else at the ProPainter pre-fill -- so the model only has
to regenerate the HUD region.

Uses the LTX-2 code bundled in third_party/ltx2 (commit e98f6d9).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

from . import config


def _ltx_imports():
    """Import the bundled LTX-2 packages lazily, so this module can be
    imported (e.g. for --help) without loading torch-heavy code."""
    if str(config.LTX2_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(config.LTX2_CODE_DIR))
    import ltx_core.components.noisers as _noisers  # noqa: F401
    from ltx_core.components.patchifiers import get_pixel_coords
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_core.types import SpatioTemporalScaleFactors, VideoPixelShape
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import (
        ensure_tiling_config, get_device, tiling_scale_factors_for_vae, video_latent_from_file,
    )
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ModalitySpec, VideoAudio
    from ltx_core.components.noisers import GaussianNoiser
    return dict(
        get_pixel_coords=get_pixel_coords, ModelRegistry=ModelRegistry, AUTO_TILING=AUTO_TILING,
        get_video_chunks_number=get_video_chunks_number, SpatioTemporalScaleFactors=SpatioTemporalScaleFactors,
        VideoPixelShape=VideoPixelShape, ICLoraPipeline=ICLoraPipeline, DiffusionStage=DiffusionStage,
        DISTILLED_SIGMAS=DISTILLED_SIGMAS, SimpleDenoiser=SimpleDenoiser, ensure_tiling_config=ensure_tiling_config,
        get_device=get_device, tiling_scale_factors_for_vae=tiling_scale_factors_for_vae,
        video_latent_from_file=video_latent_from_file, encode_video=encode_video,
        get_videostream_metadata=get_videostream_metadata, ModelPaths=ModelPaths, ModalitySpec=ModalitySpec,
        VideoAudio=VideoAudio, GaussianNoiser=GaussianNoiser,
    )


def temporal_dilate_mask(mask: torch.Tensor, window_frames: int) -> torch.Tensor:
    """For each pixel independently, extend "masked" status forward/backward
    across +/-window_frames -- closes gaps where a HUD element is on screen
    the whole time but only detected starting partway through the clip."""
    arr = mask.numpy()
    size = 2 * window_frames + 1
    return torch.from_numpy(maximum_filter1d(arr, size=size, axis=0, mode="nearest"))


def build_mask_from_video(mask_path: Path, n_frames: int, w: int, h: int, temporal_window: int) -> torch.Tensor:
    import subprocess
    raw = subprocess.run(
        [config.FFMPEG_BIN, "-i", str(mask_path), "-frames:v", str(n_frames),
         "-vf", f"scale={w}:{h}:flags=neighbor", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    ).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = arr.size // (h * w)
    gray = arr[: n * h * w].reshape(n, h, w)
    mask_bin = (gray > 127).astype(np.float32)
    return temporal_dilate_mask(torch.from_numpy(mask_bin), window_frames=temporal_window)


@dataclass(frozen=True)
class CellHitRegionMask:
    """Latent-space conditioning: only the latent cells whose receptive field
    overlaps the (temporally-dilated) pixel mask are left free to denoise;
    every other cell is frozen at its ProPainter-conditioned initial value."""
    cell_hits: torch.Tensor  # (F_latent, H_latent, W_latent) bool

    def apply_to(self, latent_state, latent_tools):
        dep = _ltx_imports()
        # LTX-2 wraps the mask in a DenoiseMask that conditioning edits on the
        # host via `.tensor`, before the stage compacts/uploads it to the GPU.
        dmask = latent_state.denoise_mask.tensor
        dev = dmask.device
        coords = latent_tools.patchifier.get_patch_grid_bounds(latent_tools.target_shape, device=dev)
        scale_factors = getattr(latent_tools, "scale_factors", dep["SpatioTemporalScaleFactors"].default())
        px = dep["get_pixel_coords"](coords, scale_factors, causal_fix=getattr(latent_tools, "causal_fix", True))
        t_start, t_end = px[:, 0].unbind(dim=-1)
        y_start = px[:, 1, :, 0]
        x_start = px[:, 2, :, 0]
        hits = self.cell_hits.to(dev)
        f_ = hits.shape[0]
        cs_ = hits.to(torch.int32).cumsum(0)
        cs = torch.cat([torch.zeros_like(cs_[:1]), cs_], dim=0)
        f0 = t_start.round().long().clamp(0, f_ - 1)
        f1 = t_end.round().long().clamp(0, f_)
        f1 = torch.maximum(f1, f0 + 1)
        cy = (y_start / scale_factors.height).long().clamp(0, hits.shape[1] - 1)
        cx = (x_start / scale_factors.width).long().clamp(0, hits.shape[2] - 1)
        sp = ((cs[f1, cy, cx] - cs[f0, cy, cx]) > 0).float()
        mask_val = sp.to(dmask.dtype)
        state = latent_state.clone()
        if dmask.dim() == 3:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.tensor.copy_(mask_val)
        return state


def _load_mp4(ffmpeg_bin, path, n_frames, h, w):
    import subprocess
    raw = subprocess.run(
        [ffmpeg_bin, "-i", str(path), "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    ).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = arr.size // (h * w * 3)
    frames01 = arr[: n * h * w * 3].reshape(n, h, w, 3).astype(np.float32) / 255.0
    return torch.from_numpy(frames01).permute(0, 3, 1, 2)


def _write_mp4(ffmpeg_bin, frames01, out_path, w, h, fps):
    import subprocess
    frames_u8 = np.clip(frames01 * 255.0, 0, 255).astype(np.uint8)
    cmd = [ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, input=frames_u8.tobytes(), check=True, capture_output=True)


class DiffusionModel:
    """Loads the LTX-2.5 transformer/text-encoder/VAEs once; call
    `.run(...)` per clip. Loading is the expensive part (several minutes),
    so reuse one instance across multiple clips."""

    def __init__(self, gpu_id: str = "0"):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self._dep = _ltx_imports()
        self.device = self._dep["get_device"]()
        self.dtype = torch.bfloat16

        model_paths = self._dep["ModelPaths"].from_split(
            transformer_path=str(config.LTX25_TRANSFORMER),
            text_encoder_path=str(config.LTX25_TEXT_ENCODER),
            video_vae_path=str(config.LTX25_VIDEO_VAE),
            audio_vae_path=str(config.LTX25_AUDIO_VAE),
        )
        registry = self._dep["ModelRegistry"](cache_weights=True, cache_models=True)
        self.pipeline = self._dep["ICLoraPipeline"](
            model_paths=model_paths,
            spatial_upsampler_path=str(config.LTX25_SPATIAL_UPSAMPLER),
            loras=(),
            registry=registry,
        )
        self.stage = self._dep["DiffusionStage"].from_checkpoint(
            model_paths.transformer(), self.pipeline.dtype, self.pipeline.device, loras=(), registry=registry,
        )
        self.sigmas = self._dep["DISTILLED_SIGMAS"].to(dtype=torch.float32, device=self.device)

    def run(self, propainter_clip: Path, mask_video_path: Path, prompt: str,
            hud_clip_path: Path, out_stage1: Path, out_composite: Path) -> None:
        dep = self._dep
        device, dtype = self.device, self.dtype

        cap = cv2.VideoCapture(str(hud_clip_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        wc, hc = w // config.LATENT_SCALE, h // config.LATENT_SCALE

        mask_pixel = build_mask_from_video(mask_video_path, n_frames, w, h, config.TEMPORAL_WINDOW)
        mask_bool_np = (mask_pixel.numpy() > 0.5)
        cell_hits_np = np.stack([
            (cv2.resize(mask_bool_np[f].astype(np.float32), (wc, hc), interpolation=cv2.INTER_AREA) > 0.02)
            for f in range(n_frames)
        ])
        cell_hits = torch.from_numpy(cell_hits_np)

        with torch.inference_mode():
            (ctx_p,) = self.pipeline.prompt_encoder([prompt])
            v_context, a_context = ctx_p.video_encoding, ctx_p.audio_encoding
            stage1_shape = dep["get_videostream_metadata"](str(propainter_clip))
            initial_video_latent = self.pipeline.image_conditioner(
                lambda enc: dep["video_latent_from_file"](
                    video_encoder=enc, file_path=str(propainter_clip), output_shape=stage1_shape,
                    dtype=dtype, device=device,
                )
            )
            cell_mask = CellHitRegionMask(cell_hits=cell_hits)
            video_state, _ = self.stage(
                denoiser=dep["SimpleDenoiser"](v_context=v_context, a_context=a_context),
                sigmas=self.sigmas,
                noiser=dep["GaussianNoiser"](generator=torch.Generator(device=device).manual_seed(config.SEED)),
                # The ProPainter latent is the clean latent: the noiser fully noises only
                # cells CellHitRegionMask marks (denoise_mask=1), the rest stay frozen at it.
                modalities=dep["VideoAudio"](
                    dep["ModalitySpec"](latent=initial_video_latent, conditioning_fps=float(stage1_shape.fps),
                                        context=v_context, conditionings=[cell_mask], frozen=False),
                    None,
                ),
            )
            stage1_latent = video_state.latent.detach().clone()

            stage1_tiling = dep["ensure_tiling_config"](
                dep["AUTO_TILING"], scale_factors=dep["tiling_scale_factors_for_vae"](self.pipeline.video_decoder.checkpoint_path),
                vae_checkpoint_path=self.pipeline.video_decoder.checkpoint_path,
                video_shape=dep["VideoPixelShape"](batch=1, frames=n_frames, width=w, height=h, fps=float(config.NATIVE_FPS)),
                diffvae_optimization=self.pipeline.video_decoder.diffvae_optimization, device=device,
            )
            gen = torch.Generator(device=device).manual_seed(config.SEED)
            decoded = list(self.pipeline.video_decoder(stage1_latent, stage1_tiling, gen))
            decoded_cat = torch.cat(decoded, dim=0)
            n_chunks = dep["get_video_chunks_number"](n_frames, stage1_tiling)
            dep["encode_video"](video=iter([decoded_cat]), fps=config.NATIVE_FPS, audio=None,
                                output_path=str(out_stage1), video_chunks_number=n_chunks)

        hud_px01 = _load_mp4(config.FFMPEG_BIN, hud_clip_path, n_frames, h, w)
        generated_px01 = _load_mp4(config.FFMPEG_BIN, out_stage1, n_frames, h, w)
        mask_px = np.stack([
            cv2.resize(cell_hits_np[f].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for f in range(n_frames)
        ])
        n = min(hud_px01.shape[0], generated_px01.shape[0], mask_px.shape[0])
        hud_n, gen_n, mask_n = hud_px01[:n], generated_px01[:n], mask_px[:n]
        w_hard = mask_n[:, :, :, None].astype(np.float32)
        composite01 = (gen_n.permute(0, 2, 3, 1).numpy() * w_hard
                       + hud_n.permute(0, 2, 3, 1).numpy() * (1 - w_hard))
        _write_mp4(config.FFMPEG_BIN, composite01, out_composite, w, h, config.NATIVE_FPS)
