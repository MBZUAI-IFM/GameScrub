# GameScrub data pipeline: HUD removal from boxes

This pipeline removes the HUD (minimap, health bar, crosshair, subtitles, markers, ...)
from a gameplay clip. You give it the clip and the per-frame bounding boxes of the HUD
elements, and it returns the same clip with those regions repainted.

It produces paired data: the clip with its HUD (`hud_clip.mp4`) and the same clip
without it (`composite.mp4`). Pairs like these are what the companion
[GameScrub model](../Gamescrub_model) learns from. That model then removes the HUD
from a video on its own, with no boxes.

| | This repo (`gamescrub_data`) | `Gamescrub_model` |
|---|---|---|
| Input | clip + HUD bounding boxes | clip only |
| Method | ProPainter pre-fill + masked LTX-2.5 diffusion (base model, no LoRA) | LTX-2.5 + GameScrub IC-LoRA |
| Output | 640x352, 145 frames, 30 fps | 1280x704, 30 fps |
| Use | building (HUD, clean) training pairs | inference on new videos |

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running](#running)
- [Input format](#input-bboxesjson)
- [Outputs](#outputs)
- [Configuration reference](#configuration-reference)
- [Using it from Python](#using-it-from-python)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Licenses](#licenses)

## How it works

```
clip.mp4 + bboxes.json
   │
[1] mask prep      resize to 640x352, trim/pad to 145 frames, draw boxes   -> hud_clip.mp4, mask.mp4, mask_frames/
   │
[2] ProPainter     flow-guided pre-fill of the boxed regions               -> inpaint_out.mp4
   │
[3] caption        a vision LLM describes the clip with the HUD blacked out -> caption.json
   │
[4] diffusion      LTX-2.5 regenerates only the latent cells under the boxes, starting
   │               from [2] and guided by [3]; the result is pasted back onto hud_clip.mp4
   ▼
composite.mp4      HUD removed
```

### Stage 1: mask prep (`src/hud_removal/mask_utils.py`)

- Decodes `clip.mp4` with OpenCV, resizes every frame to **640x352** (`INTER_AREA`) and
  keeps the first **145 frames**. A shorter clip is padded by repeating its last frame.
- Draws each box from `bboxes.json` as a filled white rectangle. The boxes are normalized
  coordinates, so they work at any input resolution.
- Writes `hud_clip.mp4` (H.264, CRF 16), `mask.mp4` (lossless, CRF 0) and the same mask as
  PNGs in `mask_frames/` (ProPainter reads masks as images).

### Stage 2: ProPainter pre-fill (`src/hud_removal/propainter_run.py`)

[ProPainter](https://github.com/sczhou/ProPainter) fills the masked regions using optical
flow, bringing in pixels from other frames where the background is visible. Its result is
temporally consistent but often blurry where nothing could be borrowed.

ProPainter needs an older stack (Python 3.8, torch 2.4), so it runs as a subprocess in its
own environment (`PROPAINTER_PYTHON`). Its code is bundled in `third_party/propainter/`.

This pre-fill is required. If the diffusion stage starts from the raw clip instead, it
redraws the HUD elements in place.

### Stage 3: captioning (`src/hud_removal/caption.py`)

LTX-2.5 is a text-conditioned video model, so it needs a description of the clip.

- 8 frames are sampled evenly from `hud_clip.mp4`, the HUD mask is painted solid black on
  each, and they are sent to a vision chat model through any **OpenAI-compatible**
  `/chat/completions` API.
- The system prompt asks for a 150-220 word caption in the style of LTX-2 training
  captions. It tells the model to describe the masked regions as the natural continuation
  of the scene and never to mention the HUD, boxes or black patches.
- The result is saved to `caption.json` (`prompt`, model name, sampled frame times, token
  count).

### Stage 4: masked diffusion and compositing (`src/hud_removal/diffusion.py`)

The LTX-2.5 22B distilled transformer is used unmodified (no LoRA, no fine-tuning).

1. The ProPainter clip is encoded by the LTX-2.5 video VAE. This latent is the starting
   point.
2. The pixel mask is **temporally dilated by ±16 frames**: if a pixel is HUD in any frame
   within 16 frames, it is masked. This closes gaps where the detector missed an element
   on some frames.
3. `CellHitRegionMask` converts the pixel mask into a mask over latent cells (each cell
   covers 32x32 pixels and a few frames). A cell is "hit" if more than 2% of its area is
   masked in any frame it covers.
4. Diffusion runs on the distilled sigma schedule with seed 42. Only the hit cells are
   noised and denoised. All other cells stay fixed at the ProPainter latent, so the model
   only regenerates the HUD regions, in the context of the unchanged scene. The prompt
   from stage 3 conditions the generation.
5. The latent is decoded to `stage1_output.mp4`.
6. **Hard composite**: inside the hit cells (upscaled back to pixels) the generated pixels
   are used, and everywhere else the pixels come from `hud_clip.mp4`. The result is
   `composite.mp4`.

So the repainted area is the boxes rounded out to the 32-pixel latent grid and widened by
±16 frames. Everything outside it comes from `hud_clip.mp4` (re-encoded at CRF 16, so it
is visually but not bit-for-bit identical).

### Batching and memory

ProPainter and the loaded diffusion model do not fit on one 80 GB GPU together. The batch
script therefore runs stages 1-3 for **all** clips first, then loads the diffusion model
**once** and runs stage 4 for every clip. Loading the model takes several minutes, so
batching matters.

## Repository layout

```
gamescrub_data/
├── README.md
├── .env.example                  settings template (checkpoint paths, API key) -> copy to .env
├── requirements.txt              main environment, Python 3.10, exact versions (stages 1, 3, 4)
├── requirements-propainter.txt   ProPainter environment, Python 3.8 (stage 2)
├── docs/
│   └── bboxes_schema.md          bboxes.json format
├── scripts/
│   ├── check_setup.py            checks environments, GPU, checkpoints and API settings
│   ├── run_pipeline.py           one clip
│   └── run_pipeline_batch.py     many clips from a JSON manifest, model loaded once
├── src/hud_removal/
│   ├── config.py                 all settings (read from environment) and fixed constants
│   ├── pipeline.py               chains the stages: prepare() = 1-3, diffuse() = 4
│   ├── mask_utils.py             stage 1
│   ├── propainter_run.py         stage 2
│   ├── caption.py                stage 3
│   └── diffusion.py              stage 4 (DiffusionModel, CellHitRegionMask, compositing)
└── third_party/
    ├── ltx2/                     LTX-2 code (ltx_core, ltx_pipelines), unmodified, commit e98f6d9
    └── propainter/               ProPainter code, commit 3267825 (one file modified, see NOTICE.md)
```

All code needed to run the pipeline is in this folder. Only the model checkpoints are
downloaded separately.

## Requirements

- Linux x86_64 with an NVIDIA GPU with **80 GB** of memory (A100 80GB or H100). The
  diffusion stage peaks at about 71 GB.
- An NVIDIA driver that supports **CUDA 13.2** (the PyTorch wheels in `requirements.txt`
  are built for it).
- About 75 GB of disk for the checkpoints.
- An API key for an OpenAI-compatible chat API with a vision-capable model (stage 3).
- [`uv`](https://docs.astral.sh/uv/), or another way to get Python 3.10 and Python 3.8.
- `ffmpeg`: a system `ffmpeg` is used if it is on `PATH`. Otherwise the static binary from
  the `imageio-ffmpeg` package is used, so nothing extra is needed.

## Setup

Run all commands from this folder.

### 1. Main environment (Python 3.10)

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --no-deps --index-strategy unsafe-best-match -r requirements.txt
```

Both flags are required. The file pins every package (including dependencies) at an exact
version, and the packages come from several indexes (PyPI, PyTorch, NATTEN). `--no-deps`
also lets `nvidia-cudnn-cu13` stay at a newer version than torch declares, which avoids a
cuDNN symbol crash.

### 2. ProPainter environment (Python 3.8)

```bash
uv venv --python 3.8 propainter-env
uv pip install --python propainter-env/bin/python -r requirements-propainter.txt
```

You never activate this environment. The pipeline calls its Python directly.

### 3. Checkpoints

| Checkpoint | Source | Size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | Hugging Face `Lightricks/LTX-2.5` (`diffusion_models/`) | 42 GB |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `Lightricks/LTX-2.5` (`text_encoders/`) | 26 GB |
| `ltx-2.5-video-vae-conv-bf16.safetensors` | `Lightricks/LTX-2.5` (`vae/`) | 1.5 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5` (`vae/`) | 0.4 GB |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `Lightricks/LTX-2.5` (`latent_upscale_models/`) | 1 GB |
| `ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` | github.com/sczhou/ProPainter, release `v0.1.0` | 0.2 GB |

- The video VAE must be the **`-conv-`** variant.
- The audio VAE and spatial upscaler are required by the LTX-2 pipeline loader, even
  though this pipeline generates no audio and does not upscale.
- The LTX-2.5 repository is gated: accept its terms on Hugging Face first.

```bash
CKPT="/path/to/checkpoints"   # choose a folder with ~75 GB free

hf auth login
hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir "$CKPT"

mkdir -p "$CKPT/propainter"
for f in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do
  curl -L -o "$CKPT/propainter/$f" "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/$f"
done
```

### 4. Settings

```bash
cp .env.example .env
```

Edit `.env`: set every checkpoint path, `PROPAINTER_PYTHON` (the absolute path of
`propainter-env/bin/python` from step 2), your API key and the caption model name. See the
[configuration reference](#configuration-reference). `.env` is in `.gitignore`. Never
commit it.

### 5. Check

```bash
source .venv/bin/activate
set -a; source .env; set +a
python scripts/check_setup.py
```

This checks the Python version, packages, ffmpeg, GPU, bundled LTX-2 code, all
checkpoints, the ProPainter environment and the captioning settings, and lists every
problem in one pass. Fix anything marked `[X]` before running.

## Running

Before running anything, activate the main environment and load `.env` in every new shell:

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

**One clip:**

```bash
python scripts/run_pipeline.py --clip clip.mp4 --bboxes bboxes.json --out-dir out/clip --gpu 0
```

**Many clips** (diffusion model loaded once):

```bash
python scripts/run_pipeline_batch.py --manifest clips.json --gpu 0
```

`clips.json` is a list of jobs:

```json
[
  {"clip": "data/a.mp4", "bboxes": "data/a_bboxes.json", "out_dir": "out/a"},
  {"clip": "data/b.mp4", "bboxes": "data/b_bboxes.json", "out_dir": "out/b"}
]
```

- A clip that fails is reported and skipped. The script exits with code 1 if any clip
  failed, and prints which ones.
- **Several GPUs:** split the manifest into one file per GPU and run one process per GPU
  (`--gpu 0`, `--gpu 1`, ...).
- **Resuming:** every stage skips itself if its output already exists. Re-run the same
  command after an interruption. To recompute a stage, delete its output file.

### Timing

On one A100 80GB, loading the diffusion model takes several minutes (once per process).
After that, each clip takes about 1 minute: ~35-40 s for ProPainter, plus captioning,
diffusion and compositing.

## Input: `bboxes.json`

Per-frame HUD boxes in normalized `[0, 1]` coordinates:

```json
{
  "schema": "per_frame_bbox_v1",
  "frames": [
    {"frame_index": 0, "bboxes_xyxy_norm": [[0.02, 0.03, 0.18, 0.09], [0.80, 0.85, 0.97, 0.98]]},
    {"frame_index": 1, "bboxes_xyxy_norm": []}
  ]
}
```

- `frame_index` is 0-based, in the clip's own frame order. A frame with no entry has no
  boxes.
- Each box is `[x1, y1, x2, y2]`, one per HUD element. Several boxes per frame are normal.
- Boxes can come from any detector or from manual annotation.
- Boxes are used exactly as given, with no padding. **Make each box fully cover its HUD
  element.** Any HUD pixels outside the repainted area stay in the output.

Full description: [`docs/bboxes_schema.md`](docs/bboxes_schema.md).

## Outputs

Written to `--out-dir`:

| File | Stage | Content |
|---|---|---|
| `hud_clip.mp4` | 1 | input at 640x352, 145 frames, 30 fps, HUD visible |
| `mask.mp4`, `mask_frames/*.png` | 1 | HUD mask (white = boxes) |
| `inpaint_out.mp4` | 2 | ProPainter pre-fill |
| `caption.json` | 3 | caption used for diffusion |
| `stage1_output.mp4` | 4 | raw diffusion output, before compositing |
| `composite.mp4` | 4 | **final result**: HUD removed |

For a training pair, use `hud_clip.mp4` (input) and `composite.mp4` (target). Both have the
same size, length and frame alignment.

## Configuration reference

All settings are read from environment variables, which is why `.env` must be loaded
first. They are defined in `src/hud_removal/config.py`.

| Variable | Required | Meaning |
|---|---|---|
| `LTX25_TRANSFORMER` | yes | path to `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |
| `LTX25_TEXT_ENCODER` | yes | path to `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| `LTX25_VIDEO_VAE` | yes | path to `ltx-2.5-video-vae-conv-bf16.safetensors` |
| `LTX25_AUDIO_VAE` | yes | path to `ltx-2.5-audio-vae-bf16.safetensors` |
| `LTX25_SPATIAL_UPSAMPLER` | yes | path to `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| `PROPAINTER_WEIGHTS_DIR` | yes | folder with `ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` |
| `PROPAINTER_PYTHON` | yes | Python of the ProPainter environment (`.../propainter-env/bin/python`) |
| `OPENAI_API_KEY` | yes | API key for the captioning endpoint |
| `HUD_CAPTION_MODEL` | yes | name of a vision-capable chat model |
| `OPENAI_BASE_URL` | no | OpenAI-compatible endpoint (default `https://api.openai.com/v1`) |
| `HUD_CAPTION_REASONING` | no | `reasoning_effort` for reasoning models (default `medium`). Sent only when the model name starts with `gpt-5`, `gpt-6` or `o` |
| `FFMPEG_BIN` | no | ffmpeg binary to use (default: system `ffmpeg`, else the `imageio-ffmpeg` binary) |

Constants that define the recipe (in `config.py`; change them only if you know why):

| Constant | Value | Meaning |
|---|---|---|
| `STAGE_W`, `STAGE_H` | 640, 352 | working resolution (multiples of 32) |
| `N_FRAMES` | 145 | frames per clip (8k+1, as LTX requires; ~4.8 s at 30 fps) |
| `NATIVE_FPS` | 30 | frame rate written to all outputs |
| `LATENT_SCALE` | 32 | pixels per latent cell |
| `TEMPORAL_WINDOW` | 16 | ± frames the mask is dilated over |
| `SEED` | 42 | diffusion and decoding seed |

## Using it from Python

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
from hud_removal.pipeline import prepare, diffuse, run_pipeline
from hud_removal.diffusion import DiffusionModel

# whole pipeline, one clip
run_pipeline(Path("clip.mp4"), Path("bboxes.json"), Path("out/clip"), gpu_id="0")

# or split, to load the model once for many clips
jobs = [("a.mp4", "a_bboxes.json", "out/a"), ("b.mp4", "b_bboxes.json", "out/b")]
prepared = [prepare(Path(c), Path(b), Path(o)) for c, b, o in jobs]   # stages 1-3
model = DiffusionModel(gpu_id="0")                                    # several minutes
for p in prepared:
    diffuse(p, model)                                                 # stage 4
```

Set `CUDA_VISIBLE_DEVICES` before importing torch (the scripts do this for you), and load
`.env` into the environment first.

## Limitations

- **Fixed size and length.** Output is always 640x352 and 145 frames. Longer clips are
  trimmed to their first 145 frames. Shorter clips are padded by repeating the last frame,
  so the end of the output is frozen.
- **No frame-rate conversion.** Frames are taken in order and written at 30 fps. A clip
  that is not 30 fps plays at the wrong speed (for example, a 60 fps clip plays at half
  speed and covers only ~2.4 s of the source). Convert clips to 30 fps first, for example
  `ffmpeg -i in.mp4 -r 30 out.mp4`.
- **Aspect ratio.** Clips are resized to 640x352 (~16:9) without cropping, so other aspect
  ratios are stretched.
- **No audio.** Outputs have no audio track.
- **Box quality limits output quality.** HUD pixels outside the repainted area stay in the
  output. Translucent HUD elements also need boxes.
- **Large or busy regions.** Large boxes over complex, fast-moving backgrounds are the
  hardest case. Check `composite.mp4` before using a pair for training.
- **Determinism.** The seed is fixed, but the caption comes from an external LLM, so
  re-captioning can change the result. Keep `caption.json` to reproduce a run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check_setup.py` reports a path that literally reads `PATH TO CHECKPOINT` | `.env` is not loaded in this shell: `set -a; source .env; set +a` |
| `No API key found` / `No caption model set` | fill `OPENAI_API_KEY` / `HUD_CAPTION_MODEL` in `.env` and reload it |
| `OpenAI 400: ... reasoning_effort ...` | your model does not accept `reasoning_effort` but its name starts with `o`. Use another endpoint/model name, or remove that line in `caption.py` |
| `ProPainter failed` | run `scripts/check_setup.py`. Usually `PROPAINTER_PYTHON` points at the wrong environment or the ProPainter weights are missing |
| CUDA out of memory in stage 4 | use an 80 GB GPU with nothing else running on it. Do not run ProPainter on the same GPU at the same time |
| A stage does not rerun after you changed something | outputs are reused if present: delete that stage's output (see [Outputs](#outputs)) |
| cuDNN symbol error at import | reinstall with `--no-deps` exactly as in setup step 1 |

## Licenses

The code outside `third_party/` is released under the [MIT License](../LICENSE).

The bundled third-party code keeps its own licenses:

- **`third_party/ltx2/`**: LTX-2 code by Lightricks, LTX Community License (see
  `third_party/ltx2/NOTICE.md` and `LICENSE*`). Use of this code and of the LTX-2.5
  weights is subject to the use restrictions in Section 4 and Attachment A of
  `third_party/ltx2/LICENSE-2_x`.
- **`third_party/propainter/`**: ProPainter code by S-Lab, NTU, under the **S-Lab License
  1.0, non-commercial use only** (see `third_party/propainter/NOTICE.md`).

Because ProPainter is non-commercial, the pipeline as a whole can only be used for
non-commercial purposes unless you get permission from ProPainter's authors. The captioning
stage sends video frames to the API provider you configure, under that provider's terms.

## Acknowledgements

- [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks
- [ProPainter](https://github.com/sczhou/ProPainter) by Shangchen Zhou et al.
- [RAFT](https://github.com/princeton-vl/RAFT) optical flow (bundled inside ProPainter)
