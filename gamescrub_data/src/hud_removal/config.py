"""All paths and settings the pipeline needs.

Fill in the checkpoint paths and the API key in `.env` (copy `.env.example`),
then load it into your shell before running anything:

    set -a; source .env; set +a

Every value below reads the matching environment variable; the defaults are
placeholders that must be replaced.
"""
from __future__ import annotations

import os
from pathlib import Path

PLACEHOLDER = "PATH TO CHECKPOINT"

# --- bundled third-party code (shipped in third_party/, nothing to clone) -----
ROOT_DIR = Path(__file__).resolve().parents[2]
LTX2_CODE_DIR = ROOT_DIR / "third_party" / "ltx2"
PROPAINTER_CODE_DIR = ROOT_DIR / "third_party" / "propainter"


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, PLACEHOLDER))


# --- checkpoints ---------------------------------------------------------------
# LTX-2.5 (Hugging Face: Lightricks/LTX-2.5), one file each:
LTX25_TRANSFORMER = _env_path("LTX25_TRANSFORMER")              # PATH TO CHECKPOINT: ltx-2.5-22b-distilled-transformer-bf16.safetensors
LTX25_TEXT_ENCODER = _env_path("LTX25_TEXT_ENCODER")            # PATH TO CHECKPOINT: gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
LTX25_VIDEO_VAE = _env_path("LTX25_VIDEO_VAE")                  # PATH TO CHECKPOINT: ltx-2.5-video-vae-conv-bf16.safetensors  (the "-conv-" variant)
LTX25_AUDIO_VAE = _env_path("LTX25_AUDIO_VAE")                  # PATH TO CHECKPOINT: ltx-2.5-audio-vae-bf16.safetensors
LTX25_SPATIAL_UPSAMPLER = _env_path("LTX25_SPATIAL_UPSAMPLER")  # PATH TO CHECKPOINT: ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

# ProPainter (github.com/sczhou/ProPainter, release v0.1.0): a folder holding
# ProPainter.pth, raft-things.pth and recurrent_flow_completion.pth.
PROPAINTER_WEIGHTS_DIR = _env_path("PROPAINTER_WEIGHTS_DIR")    # PATH TO CHECKPOINT (folder)
PROPAINTER_WEIGHT_FILES = ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")

# Python interpreter of the separate ProPainter environment (Python 3.8, see
# requirements-propainter.txt) -- not the environment running this package.
PROPAINTER_PYTHON = os.environ.get("PROPAINTER_PYTHON", "PATH TO PROPAINTER PYTHON")


def _default_ffmpeg() -> str:
    # Use a system ffmpeg if there is one, otherwise the static binary bundled
    # with the imageio-ffmpeg pip package.
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or _default_ffmpeg()

# --- captioning (any OpenAI-compatible chat-completions API with a vision model)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
CAPTION_MODEL = os.environ.get("HUD_CAPTION_MODEL", "")  # name of a vision-capable chat model
CAPTION_REASONING_EFFORT = os.environ.get("HUD_CAPTION_REASONING", "medium")


def api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


# --- fixed recipe constants ------------------------------------------------------
SEED = 42
LATENT_SCALE = 32  # pixels per latent cell, both axes
TEMPORAL_WINDOW = 16  # +/- frames a box is temporally dilated across in latent space
NATIVE_FPS = 30
STAGE_W, STAGE_H = 640, 352  # working resolution (both divisible by 32)
N_FRAMES = 145  # 8k+1 frames (~4.83 s at 30 fps); clips are trimmed/padded to this
