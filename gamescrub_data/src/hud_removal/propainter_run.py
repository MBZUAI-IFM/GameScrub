"""Stage 2: flow-guided pre-fill of the masked region via ProPainter.

ProPainter's inference code is bundled in third_party/propainter. It needs
its own Python 3.8 environment (see requirements-propainter.txt), so it is
run as a subprocess with that environment's interpreter
(config.PROPAINTER_PYTHON) rather than imported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import config


def run_propainter(hud_clip_path: Path, mask_frames_dir: Path, out_dir: Path) -> Path:
    """Run ProPainter on hud_clip_path using the per-frame masks in
    mask_frames_dir. Writes and returns out_dir/inpaint_out.mp4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out = out_dir / "inpaint_out.mp4"
    if final_out.exists():
        return final_out

    weights_dir = config.PROPAINTER_WEIGHTS_DIR
    missing = [f for f in config.PROPAINTER_WEIGHT_FILES if not (weights_dir / f).is_file()]
    if missing:
        raise RuntimeError(
            f"ProPainter checkpoint(s) {missing} not found in PROPAINTER_WEIGHTS_DIR={weights_dir}. "
            "See README.md (Checkpoints)."
        )

    # ProPainter writes <output>/<input video stem>/inpaint_out.mp4; use a
    # per-clip temp folder so parallel runs never collide.
    tmp_dir = out_dir / "_propainter_tmp"
    cmd = [
        config.PROPAINTER_PYTHON, "inference_propainter.py",
        "--video", str(hud_clip_path.resolve()),
        "--mask", str(mask_frames_dir.resolve()),
        "--output", str(tmp_dir.resolve()),
        "--weights_dir", str(weights_dir.resolve()),
    ]
    r = subprocess.run(cmd, cwd=str(config.PROPAINTER_CODE_DIR), env=dict(os.environ),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ProPainter failed (command: {' '.join(cmd)}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    produced = tmp_dir / hud_clip_path.stem / "inpaint_out.mp4"
    if not produced.exists():
        raise RuntimeError(f"ProPainter ran but {produced} is missing")
    shutil.move(str(produced), str(final_out))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_out
