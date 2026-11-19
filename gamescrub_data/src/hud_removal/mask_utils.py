"""Stage 1: clip.mp4 + bboxes.json -> hud_clip.mp4 + mask.mp4 (+ mask_frames/).

Input contract (see docs/bboxes_schema.md):
    bboxes.json = {
      "schema": "per_frame_bbox_v1",
      "frames": [
        {"frame_index": 0, "bboxes_xyxy_norm": [[x1, y1, x2, y2], ...]},
        ...
      ]
    }
Coordinates are normalized to [0, 1], so a resize needs no offset correction
-- the boxes still land in the right place after hud_clip/mask are resized
to the diffusion model's working resolution.

Outputs, written into `out_dir`:
    hud_clip.mp4   - clip resized to (config.STAGE_W, config.STAGE_H),
                     trimmed/padded to config.N_FRAMES, HUD still visible
    mask.mp4       - same size/length, grayscale, 255 = HUD box / to repaint
    mask_frames/   - the same mask as per-frame PNGs (ProPainter wants this)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config


def _read_bboxes_by_frame(bboxes_path: Path) -> dict:
    data = json.loads(bboxes_path.read_text())
    return {f["frame_index"]: f for f in data["frames"]}


def prepare_clip(clip_path: Path, bboxes_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    """Build hud_clip.mp4, mask.mp4 and mask_frames/ for one clip. Returns
    (hud_clip_path, mask_video_path, mask_frames_dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hud_clip_path = out_dir / "hud_clip.mp4"
    mask_video_path = out_dir / "mask.mp4"
    mask_png_dir = out_dir / "mask_frames"
    if hud_clip_path.exists() and mask_video_path.exists() and mask_png_dir.exists():
        return hud_clip_path, mask_video_path, mask_png_dir

    by_idx = _read_bboxes_by_frame(bboxes_path)
    dst_w, dst_h, n_frames, fps = config.STAGE_W, config.STAGE_H, config.N_FRAMES, config.NATIVE_FPS

    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode any frames from {clip_path}")
    while len(frames) < n_frames:  # pad with the last frame if the clip is a hair short
        frames.append(frames[-1].copy())

    mask_png_dir.mkdir(exist_ok=True)
    p_orig = subprocess.Popen(
        [config.FFMPEG_BIN, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{dst_w}x{dst_h}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(hud_clip_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    p_mask = subprocess.Popen(
        [config.FFMPEG_BIN, "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{dst_w}x{dst_h}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mask_video_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    for fi, frame in enumerate(frames):
        p_orig.stdin.write(frame.tobytes())
        rec = by_idx.get(fi)
        mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
        if rec:
            for box in rec.get("bboxes_xyxy_norm", []):
                x1, y1, x2, y2 = box
                cv2.rectangle(mask, (int(x1 * dst_w), int(y1 * dst_h)),
                              (int(x2 * dst_w), int(y2 * dst_h)), 255, -1)
        p_mask.stdin.write(mask.tobytes())
        cv2.imwrite(str(mask_png_dir / f"{fi:05d}.png"), mask)

    p_orig.stdin.close(); p_orig.wait()
    p_mask.stdin.close(); p_mask.wait()
    if p_orig.returncode or p_mask.returncode:
        raise RuntimeError(f"ffmpeg failed while writing hud_clip/mask for {clip_path}")

    return hud_clip_path, mask_video_path, mask_png_dir


def load_mask_bool_from_video(mask_video_path: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    """Decode mask.mp4 back into a (n_frames, h, w) bool array, for stages
    (captioning) that need the mask as pixels rather than as a video file."""
    cap = cv2.VideoCapture(str(mask_video_path))
    out = np.zeros((n_frames, h, w), dtype=bool)
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame[:, :, 0] if frame.ndim == 3 else frame
        out[fi] = gray > 127
    cap.release()
    return out
