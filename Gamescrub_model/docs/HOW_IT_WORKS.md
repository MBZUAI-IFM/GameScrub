# How GameScrub inference works

## Model

- **Base**: LTX-2.5 22B *distilled* video diffusion transformer (bf16), Gemma-4 12B text
  encoder, LTX-2.5 video VAE, x2 latent spatial upsampler.
- **GameScrub LoRA**: an IC-LoRA (in-context LoRA), trained on pairs of (gameplay clip with
  HUD -> the same clip with the HUD removed), 145 frames at 1280x704, 30 fps, with the fixed
  prompt below.
- **Input = reference video**: in an IC-LoRA the conditioning video is encoded by the VAE and
  its latent tokens are appended to the sequence being denoised (at the same positions in
  space and time). The model attends to it and generates the clean version. No mask or HUD
  location is given; the model finds the HUD by itself.
- **Prompt** (fixed, the one used in training):
  `Remove the game HUD overlay from this footage, revealing the clean underlying gameplay video.`
- No classifier-free guidance (distilled model), so each step is a single forward pass.

## Two-stage generation

```
input video ──resize + center-crop──► 1280x704 (stage 2 reference)
            └─────────────────────── 640x352   (stage 1 reference)

stage 1   640x352,  8 distilled denoising steps from noise        LoRA + reference
   │
upsample  x2 latent spatial upsampler (in latent space)
   │
stage 2   1280x704, 3 refinement steps, starting at sigma 0.909   LoRA + reference
   │
VAE decode (tiled) ──► output .mp4, 30 fps
```

The LTX-2 default IC-LoRA pipeline applies the LoRA and the reference in stage 1 only.
**GameScrub keeps both in stage 2 too** (`apply_ic_lora=True` for stage 2 in
`inference/generate.py`). This matters: without the reference, stage 2 (starting at 91%
noise) regenerates the scene instead of refining it, and background preservation collapses.

## Long videos

The model was trained on 145-frame clips. Longer videos are handled in one of two ways:

- **Windows** (`generate.py --chunk-frames 145 --chunk-carry-frames 25`, one GPU): LTX-2's
  `ChunkConfig`. The video is generated window by window; each window re-renders the last 25
  frames of the previous one and the overlap is blended. Memory stays at single-clip level.
- **One pass** (`generate_long_multigpu.py`, 8 GPUs): the whole video is a single sequence,
  split across GPUs with LTX-2 sequence parallelism (All2All attention, from `ltx-kernels`).
  Changes from the stock LTX-2 multi-GPU path, needed to fit 1777 frames on A100 80GB:
  - the reference video is VAE-encoded in 144-frame temporal tiles (overlap 24) instead of in one pass;
  - the full-resolution reference pixels are kept in CPU memory, not on the GPU;
  - with `--latent-out`, the final latent is saved and decoded afterwards on one free GPU
    (`decode_latents.py`), because the transformer weights stay resident on every GPU during
    the run and leave too little memory for a full-resolution decode.
  - bf16 weights (no fp8: the fp8 path needs compute capability 8.9+, the A100 is 8.0).

## Frame and size constraints

- Width and height must be multiples of 64 (two-stage pipeline: stage 1 runs at half size,
  and the VAE downsamples by 32). Hence 1280x704 rather than 1280x720.
- Frame counts must be 8k+1 (the VAE compresses time by 8, plus the first frame).
- Inputs are resized (keeping the aspect ratio) and center-cropped to the target size, and
  frames are taken in order without frame-rate conversion.

## Code

All model code is the public LTX-2 code in `third_party/ltx2`, pinned to commit `c1b4675`
(see `third_party/NOTICE.md`). GameScrub adds only the scripts in `inference/`:

| Script | What it adds on top of LTX-2 |
|---|---|
| `generate.py` | builds `ICLoraPipeline` with the GameScrub stage config, batch/list handling, sharding across processes, resumable outputs, run metadata |
| `generate_long_multigpu.py` | sequence-parallel runner for `ICLoraPipeline` (based on LTX-2's `distilled_mgpu`), tiled reference encode, CPU-side reference pixels, latent saving |
| `decode_latents.py` | tiled VAE decode of a saved latent on one GPU |
