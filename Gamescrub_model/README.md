# GameScrub: HUD removal model (inference)

GameScrub removes the game HUD (minimap, health bar, crosshair, subtitles, markers, ...)
from gameplay video. It needs only the video: no masks, no bounding boxes, no captions.

```
gameplay clip with HUD  ──►  GameScrub  ──►  same clip without HUD (1280x704, 30 fps)
```

It is an IC-LoRA (in-context LoRA) on the LTX-2.5 22B distilled video model.
The HUD clip is given to the model as a reference video, and the model generates the
clean clip. This folder contains everything needed to run it: the inference scripts and
the required LTX-2 code (`third_party/`). Only the checkpoints are downloaded separately.

How it works: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).
Benchmark results and speed: [`docs/RESULTS.md`](docs/RESULTS.md).

## Contents

```
inference/generate.py                 main script: one video or a list of videos, one GPU per process
inference/generate_long_multigpu.py   long video in one pass, split across all GPUs of a node
inference/decode_latents.py           decoding step for generate_long_multigpu.py
scripts/run_one_video.sh              one clip, one GPU
scripts/run_folder_multigpu.sh        a folder of clips, spread over all GPUs of a machine
scripts/run_long_video.sh             a long video on one GPU (145-frame windows)
scripts/run_long_video_multigpu.sh    a long video in one pass on 8 GPUs
scripts/check_setup.py                checks environment, GPU and checkpoints
scripts/make_input_list.py            folder of videos -> input list for generate.py
scripts/count_frames.py               valid --num-frames for a video
third_party/ltx2/                     LTX-2 code (ltx_core, ltx_pipelines), pinned version
third_party/ltx-kernels/              LTX-2 CUDA kernels (multi-GPU only)
requirements.txt                      exact package versions (Python 3.10)
requirements-multigpu.txt             extra packages for the multi-GPU script
.env.example                          checkpoint paths to fill in
```

## Requirements

- Linux x86_64 with an NVIDIA GPU with **80 GB** of memory (A100 80GB or H100).
  One 5 s clip peaks at about 62 GB of reserved GPU memory.
- An NVIDIA driver that supports CUDA 13.2 (the PyTorch wheels are built for CUDA 13.2).
- About 75 GB of disk for the checkpoints and 10 GB for the Python environment.
- [`uv`](https://docs.astral.sh/uv/) (or any other way to get Python 3.10 and pip).
- For multi-GPU long-video inference only: a node with 8 GPUs connected by NVLink.

## Setup

### 1. Python environment (Python 3.10)

From this folder:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --no-deps --index-strategy unsafe-best-match -r requirements.txt
```

Both flags are required. The file pins every package at an exact version, and the
packages come from both PyPI and the PyTorch index. `--no-deps` is needed because one
package (cuDNN) is deliberately newer than what PyTorch declares.

The LTX-2 code in `third_party/ltx2` is not installed. The scripts add it to the
Python path themselves.

### 2. Checkpoints

| File | Where to get it | Size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | Hugging Face `Lightricks/LTX-2.5`, folder `diffusion_models/` | 42 GB |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `Lightricks/LTX-2.5`, `text_encoders/` | 26 GB |
| `ltx-2.5-video-vae-bf16.safetensors` | `Lightricks/LTX-2.5`, `vae/` | 1.5 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5`, `vae/` | 0.4 GB |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `Lightricks/LTX-2.5`, `latent_upscale_models/` | 1 GB |
| `gamescrub_lora.safetensors` | Hugging Face [`mohammadareebqazi/GameScrub`](https://huggingface.co/mohammadareebqazi/GameScrub) | 1.3 GB |

The LTX-2.5 repository is gated: accept its terms on Hugging Face first. Then:

```bash
hf auth login
hf download mohammadareebqazi/GameScrub gamescrub_lora.safetensors --local-dir "PATH TO CHECKPOINTS"
HF_HUB_DISABLE_XET=1 hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir "PATH TO CHECKPOINTS"
```

(`HF_HUB_DISABLE_XET=1` avoids downloads stalling on some clusters; it can be left out
if downloads work without it.)

Use `ltx-2.5-video-vae-bf16.safetensors`, the VAE all GameScrub results were produced with,
not the `-conv-` variant.

### 3. Settings

```bash
cp .env.example .env
# edit .env: replace every "PATH TO CHECKPOINT" with the full path of that file
```

Load it before running anything (the `scripts/*.sh` wrappers do this themselves):

```bash
set -a; source .env; set +a
```

Each path can also be given as a flag instead (`--transformer-path`, `--text-encoder-path`,
`--video-vae-path`, `--audio-vae-path`, `--spatial-upsampler-path`, `--lora`).

### 4. Check

On a GPU machine:

```bash
source .venv/bin/activate
set -a; source .env; set +a
python scripts/check_setup.py
```

Fix anything marked `[X]` before running. (A warning about Matplotlib from
`colour` can be ignored.)

## Run

Always activate the environment first (`source .venv/bin/activate`).

### One clip

```bash
bash scripts/run_one_video.sh clip.mp4 out/
# -> out/clip.mp4 (HUD removed) and out/clip.json (settings, run time)
```

This processes the first **145 frames** (4.8 s at 30 fps), the length the model was
trained on. Takes about 80 s on an A100, plus about 5 min to load the model the
first time.

### Many clips

```bash
bash scripts/run_folder_multigpu.sh videos/ out/
```

Runs one process per GPU (all visible GPUs, or `GPUS="0 1"`); each process loads the
model once and then works through its share of the clips. Logs are in `out/logs/`.
Running the same command again skips clips that are already done.

The same with `generate.py` directly, for example with a list file:

```bash
python scripts/make_input_list.py videos/ clips.jsonl     # {"id": ..., "source_video": ...} per line
CUDA_VISIBLE_DEVICES=0 python inference/generate.py --input-list clips.jsonl --output-dir out/ --rank 0 --world-size 2 &
CUDA_VISIBLE_DEVICES=1 python inference/generate.py --input-list clips.jsonl --output-dir out/ --rank 1 --world-size 2 &
wait
```

### Long videos

Two options:

1. **One GPU, in windows** (simplest):
   ```bash
   bash scripts/run_long_video.sh long.mp4 out/
   ```
   The video is generated in 145-frame windows; consecutive windows overlap by 25 frames,
   which are blended. A 59 s video (1777 frames) takes about 20 min on one A100.
   With `generate.py`: `--num-frames N --chunk-frames 145 --chunk-carry-frames 25`, where N is
   the video length rounded down to 8k+1 (`python scripts/count_frames.py long.mp4`).

2. **One pass on 8 GPUs** (no windows: the whole video is one sequence, so there are no
   window seams). Needs the multi-GPU setup below.
   ```bash
   bash scripts/run_long_video_multigpu.sh long.mp4 out/
   ```
   A 59 s video takes about 10 min to generate plus 3.5 min to decode on 8 A100 80GB.
   Tested up to 1777 frames; much longer videos may run out of memory.

### Multi-GPU setup (only for option 2)

Build the bundled CUDA kernels once, inside the environment. This uses the CUDA compiler
from `requirements-multigpu.txt`, so no system CUDA toolkit is needed. Set
`TORCH_CUDA_ARCH_LIST` to your GPU (`8.0` = A100, `9.0` = H100):

```bash
source .venv/bin/activate
uv pip install --no-deps -r requirements-multigpu.txt
TORCH_CUDA_ARCH_LIST=8.0 uv pip install --no-deps --no-build-isolation -e third_party/ltx-kernels
python scripts/check_setup.py --multigpu
```

### Input videos

- Any resolution and aspect ratio. Each frame is resized and **center-cropped** to
  1280x704 (16:9 footage loses 8 px at the top and bottom).
- Frames are read in order with no frame-rate conversion, and the output is written at
  30 fps. Convert other frame rates to 30 fps first, or the output plays at the wrong speed:
  `ffmpeg -i in.mp4 -r 30 -c:v libx264 -crf 15 in_30fps.mp4`.
- The video needs at least `--num-frames` frames (default 145). Extra frames are ignored.
- The output has no audio.

### Options of `inference/generate.py`

| Flag | Default | Meaning |
|---|---|---|
| `--input-video` / `--input-list` | | one video, or a jsonl list of `{"id", "source_video"}` |
| `--output-dir` | | outputs go to `<output-dir>/<id>.mp4` and `<id>.json` |
| `--num-frames` | 145 | frames to generate (8k+1) |
| `--chunk-frames`, `--chunk-carry-frames` | off, 25 | window length and overlap for long videos (8k+1) |
| `--width`, `--height` | 1280, 704 | output size (multiples of 64; the model was trained at 1280x704) |
| `--frame-rate` | 30 | output frame rate |
| `--seed` | 42 | random seed (results are reproducible for a fixed seed) |
| `--prompt` | fixed instruction | the prompt the model was trained with; keep it |
| `--lora-strength` | 1.0 | LoRA scale |
| `--rank`, `--world-size` | 0, 1 | split a list across processes |
| `--overwrite` | off | redo videos whose output exists |

## Troubleshooting

- **CUDA out of memory**: needs an 80 GB GPU with nothing else on it. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in `.env.example`).
- **`height=720` / not divisible by 64**: use 704. Two-stage generation needs multiples of 64.
- **`--num-frames must be 8k+1`**: use 145, 289, ... (`scripts/count_frames.py` prints the right value).
- **Multi-GPU run hangs or fails at start**: all 8 GPUs need working peer-to-peer
  access (NVLink). Check with `nvidia-smi topo -m`; try another node.
- **`ImportError: libc10.so` when importing `all2all_cpp` by hand**: import `torch` first.

## License

The code outside `third_party/` is released under the [MIT License](../LICENSE).

The LTX-2 code and the LTX-2.5 weights are under the LTX Community License
(`third_party/ltx2/LICENSE*`, see `third_party/NOTICE.md`), including its use
restrictions. The LoRA is a derivative of LTX-2.5 and is subject to the same license.
