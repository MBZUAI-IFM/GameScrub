"""Chain the four stages.

Stages 1-3 (mask prep, ProPainter, caption) and stage 4 (diffusion) are
exposed separately so a batch can run stages 1-3 for every clip first and
only then load the large diffusion model -- ProPainter and the loaded
diffusion model do not fit on one 80GB GPU at the same time.
Every stage skips itself if its output already exists, so an interrupted
run can simply be re-launched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import caption, config, diffusion, mask_utils, propainter_run


@dataclass
class PreparedClip:
    out_dir: Path
    hud_clip: Path
    mask_video: Path
    propainter_clip: Path
    prompt: str


def prepare(clip_path: Path, bboxes_path: Path, out_dir: Path) -> PreparedClip:
    """Stages 1-3 for one clip (no diffusion model needed)."""
    out_dir, clip_path, bboxes_path = Path(out_dir), Path(clip_path), Path(bboxes_path)

    print(f"[1/4] preparing hud_clip + mask -> {out_dir}", flush=True)
    hud_clip_path, mask_video_path, mask_frames_dir = mask_utils.prepare_clip(clip_path, bboxes_path, out_dir)

    print("[2/4] running ProPainter pre-fill...", flush=True)
    propainter_clip = propainter_run.run_propainter(hud_clip_path, mask_frames_dir, out_dir)

    caption_path = out_dir / "caption.json"
    if caption_path.exists():
        print("[3/4] caption.json already exists, reusing it", flush=True)
        caption_result = json.loads(caption_path.read_text())
    else:
        print("[3/4] captioning (HUD region blacked out)...", flush=True)
        mask_bool = mask_utils.load_mask_bool_from_video(mask_video_path, config.N_FRAMES, config.STAGE_H, config.STAGE_W)
        caption_result = caption.caption_clip(hud_clip_path, mask_bool)
        caption_path.write_text(json.dumps(caption_result, indent=2))

    return PreparedClip(out_dir, hud_clip_path, mask_video_path, propainter_clip, caption_result["prompt"])


def diffuse(prepared: PreparedClip, model: diffusion.DiffusionModel) -> Path:
    """Stage 4 for one prepared clip. Returns the final composite path."""
    composite_out = prepared.out_dir / "composite.mp4"
    if composite_out.exists():
        print(f"[4/4] {composite_out} already exists, skipping", flush=True)
        return composite_out
    print("[4/4] running masked diffusion + compositing...", flush=True)
    model.run(prepared.propainter_clip, prepared.mask_video, prepared.prompt, prepared.hud_clip,
              prepared.out_dir / "stage1_output.mp4", composite_out)
    print(f"done -> {composite_out}", flush=True)
    return composite_out


def run_pipeline(clip_path: Path, bboxes_path: Path, out_dir: Path,
                 diffusion_model: diffusion.DiffusionModel | None = None,
                 gpu_id: str = "0") -> Path:
    """All four stages for a single clip. Returns the final composite path."""
    prepared = prepare(clip_path, bboxes_path, out_dir)
    model = diffusion_model or diffusion.DiffusionModel(gpu_id=gpu_id)
    return diffuse(prepared, model)
