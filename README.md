# GameScrub

GameScrub removes the game HUD (minimap, health bar, crosshair, subtitles, markers, ...)
from gameplay video and returns the clean underlying footage.

This repository has two parts:

| Folder | What it is | Input | Output |
|---|---|---|---|
| [`gamescrub_data/`](gamescrub_data) | **Data pipeline.** Removes the HUD given per-frame HUD bounding boxes, using ProPainter plus masked LTX-2.5 diffusion. Used to build (HUD clip, clean clip) pairs. | clip + `bboxes.json` | 640x352, 145 frames, 30 fps |
| [`Gamescrub_model/`](Gamescrub_model) | **Model inference.** The GameScrub IC-LoRA on LTX-2.5 22B. Finds and removes the HUD by itself, with no boxes, masks or captions. | clip only | 1280x704, 30 fps, any length |

```
                     ┌──────────────────── gamescrub_data ─────────────────────┐
gameplay clip ──────►│ boxes → ProPainter pre-fill → caption → masked LTX-2.5  │──► (HUD clip, clean clip) pairs
+ HUD boxes          └─────────────────────────────────────────────────────────┘          │
                                                                                         ▼  training data
                     ┌──────────────────── Gamescrub_model ────────────────────┐
any gameplay clip ──►│ LTX-2.5 22B distilled + GameScrub IC-LoRA (2 stages)    │──► clean clip
                     └─────────────────────────────────────────────────────────┘
```

Each folder is self-contained, with its own environment, settings and README. It includes
all the code it needs (pinned LTX-2 code in `third_party/`). Only model checkpoints are
downloaded separately.

## Which one do I need?

- **To remove the HUD from your videos:** use [`Gamescrub_model/`](Gamescrub_model). It
  needs only the video.
- **To build paired training data**, or to remove the HUD when you already have accurate
  HUD boxes: use [`gamescrub_data/`](gamescrub_data).

## Results (GameScrub model)

200-clip benchmark: 100 synthetic clips (HUD rendered onto clean footage) and 100
in-the-wild clips (real gameplay with real HUD).

| Metric | Overall | Synthetic | Wild |
|---|---|---|---|
| HUD removal (%) ↑ | 92.0 | 93.9 | 90.1 |
| Artifact-adjusted removal (%) ↑ | 77.1 | 82.2 | 72.0 |
| Background preservation (%) ↑ | 95.3 | 98.3 | 92.3 |
| Background LPIPS ↓ | 0.140 | 0.132 | 0.148 |
| Background MAE (0-255) ↓ | 5.57 | 5.47 | 5.66 |
| HUD detector reduction (%) ↑ | 94.3 | 97.3 | 91.2 |

On one A100 80GB: about 81 s per 5 s clip (1280x704, 145 frames), excluding a one-time
model load of about 5 min. Metric definitions, speed and memory:
[`Gamescrub_model/docs/RESULTS.md`](Gamescrub_model/docs/RESULTS.md).

## How each part works

### Data pipeline (`gamescrub_data/`)

1. **Mask prep:** resize the clip to 640x352, trim or pad it to 145 frames, and draw the
   HUD boxes into a mask.
2. **ProPainter:** flow-guided video inpainting pre-fills the boxed regions with
   background borrowed from other frames.
3. **Caption:** a vision LLM (any OpenAI-compatible API) describes the clip with the HUD
   blacked out, without mentioning the HUD.
4. **Masked diffusion:** the base LTX-2.5 22B distilled model (no LoRA) starts from the
   ProPainter result and regenerates only the latent cells under the (temporally dilated)
   boxes, guided by the caption. The result is pasted back onto the original clip, so
   everything outside the repainted area is kept.

Details: [`gamescrub_data/README.md`](gamescrub_data/README.md).

### Model (`Gamescrub_model/`)

- An **IC-LoRA** (in-context LoRA) on LTX-2.5 22B distilled. The HUD clip is VAE-encoded
  and appended to the sequence as a reference, and the model generates the clean version
  under a fixed prompt.
- **Two stages:** 8 denoising steps at 640x352, a x2 latent upsample, then 3 refinement
  steps at 1280x704. Unlike the stock LTX-2 IC-LoRA pipeline, the LoRA and reference are
  also kept in stage 2, which is what preserves the background.
- **Long videos:** either 145-frame windows with 25-frame blended overlap on one GPU, or
  the whole video as one sequence split across 8 GPUs (sequence parallelism).

Details: [`Gamescrub_model/README.md`](Gamescrub_model/README.md) and
[`Gamescrub_model/docs/HOW_IT_WORKS.md`](Gamescrub_model/docs/HOW_IT_WORKS.md).

## Requirements (both parts)

- Linux x86_64 with an NVIDIA GPU with **80 GB** of memory (A100 80GB or H100).
- An NVIDIA driver that supports CUDA 13.2.
- [`uv`](https://docs.astral.sh/uv/) (or another way to get Python 3.10, and Python 3.8
  for the data pipeline's ProPainter environment).
- About 75 GB of disk for the LTX-2.5 checkpoints. The two parts can share them, except
  for the video VAE (below).
- Data pipeline only: an API key for an OpenAI-compatible vision chat model.
- Multi-GPU long-video inference only: a node with 8 GPUs connected by NVLink.

## Quick start

Each part is set up **from inside its own folder**, in its own environment.

**Model** (remove the HUD from a video):

```bash
cd Gamescrub_model
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install --no-deps --index-strategy unsafe-best-match -r requirements.txt
# download checkpoints (README step 2), then:
cp .env.example .env            # fill in the checkpoint paths
set -a; source .env; set +a
python scripts/check_setup.py
bash scripts/run_one_video.sh clip.mp4 out/
```

**Data pipeline** (clip + boxes → clean clip):

```bash
cd gamescrub_data
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install --no-deps --index-strategy unsafe-best-match -r requirements.txt
uv venv --python 3.8 propainter-env
uv pip install --python propainter-env/bin/python -r requirements-propainter.txt
# download checkpoints (README step 3), then:
cp .env.example .env            # checkpoint paths, PROPAINTER_PYTHON, API key, caption model
set -a; source .env; set +a
python scripts/check_setup.py
python scripts/run_pipeline.py --clip clip.mp4 --bboxes bboxes.json --out-dir out/clip --gpu 0
```

## Differences between the two parts

Set up the two parts separately. Do not share an environment or `.env` file between them.

| | `gamescrub_data` | `Gamescrub_model` |
|---|---|---|
| Bundled LTX-2 code | commit `e98f6d9` | commit `c1b4675` |
| Video VAE | `ltx-2.5-video-vae-conv-bf16.safetensors` (**`-conv-`** variant) | `ltx-2.5-video-vae-bf16.safetensors` (**not** `-conv-`) |
| LoRA | none (base model) | `gamescrub_lora.safetensors` ([Hugging Face](https://huggingface.co/mohammadareebqazi/GameScrub)) |
| Env variable names | `LTX25_*`, `PROPAINTER_*`, `OPENAI_*`, `HUD_CAPTION_*` | `GAMESCRUB_*` |
| Extra environment | Python 3.8 for ProPainter | optional `ltx-kernels` build for multi-GPU |
| Resizing | stretched to 640x352 | aspect kept, center-cropped to 1280x704 |
| Length | exactly 145 frames (trimmed or padded) | 145 by default; any 8k+1 length with windows or multi-GPU |

The transformer, text encoder, audio VAE and spatial upsampler are the same files in both
parts, so you can download them once and point both `.env` files at them.

**Common to both:** frames are read in order with **no frame-rate conversion**, and output
is written at 30 fps. Convert other frame rates first (`ffmpeg -i in.mp4 -r 30 out.mp4`),
or the output plays at the wrong speed. Outputs have no audio.

## Repository layout

```
.
├── README.md                     this file
├── gamescrub_data/               data pipeline (boxes → clean clip)
│   ├── README.md
│   ├── src/hud_removal/          the four stages
│   ├── scripts/                  run_pipeline.py, run_pipeline_batch.py, check_setup.py
│   ├── docs/bboxes_schema.md     input box format
│   └── third_party/              LTX-2 (e98f6d9), ProPainter (3267825)
└── Gamescrub_model/              model inference (clip → clean clip)
    ├── README.md
    ├── inference/                generate.py, generate_long_multigpu.py, decode_latents.py
    ├── scripts/                  shell wrappers, check_setup.py, helpers
    ├── docs/                     HOW_IT_WORKS.md, RESULTS.md
    └── third_party/              LTX-2 (c1b4675), ltx-kernels
```

## Licenses

The original GameScrub code in this repository is released under the
[MIT License](LICENSE). That covers everything outside the `third_party/` folders.

The bundled third-party code keeps its own licenses:

- **LTX-2 code and LTX-2.5 weights** (`*/third_party/ltx2/`): LTX Community License,
  including the use restrictions in Section 4 and Attachment A of `LICENSE-2_x`. The
  GameScrub LoRA is a derivative of LTX-2.5 and is subject to the same license.
- **ProPainter** (`gamescrub_data/third_party/propainter/`): S-Lab License 1.0,
  **non-commercial use only**. The data pipeline as a whole is therefore limited to
  non-commercial use.
- **ltx-kernels** (`Gamescrub_model/third_party/ltx-kernels/`): Apache-2.0.

See the `NOTICE.md` files in each `third_party/` folder for exact sources and commits.

## Acknowledgements

- [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks
- [ProPainter](https://github.com/sczhou/ProPainter) by Shangchen Zhou et al.
- [RAFT](https://github.com/princeton-vl/RAFT) optical flow
