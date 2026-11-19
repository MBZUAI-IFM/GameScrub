#!/usr/bin/env python3
"""Run the full HUD-removal pipeline on one clip.

    python scripts/run_pipeline.py \\
        --clip path/to/clip.mp4 \\
        --bboxes path/to/bboxes.json \\
        --out-dir out/my_clip \\
        --gpu 0

See README.md for setup and docs/bboxes_schema.md for the bboxes.json format.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, type=Path, help="input clip (HUD visible)")
    p.add_argument("--bboxes", required=True, type=Path, help="per-frame HUD boxes (see docs/bboxes_schema.md)")
    p.add_argument("--out-dir", required=True, type=Path, help="folder for all intermediate + final outputs")
    p.add_argument("--gpu", default="0", help="CUDA device id (used by ProPainter and diffusion)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # before torch / ProPainter start
    from hud_removal.pipeline import run_pipeline

    run_pipeline(args.clip, args.bboxes, args.out_dir, gpu_id=args.gpu)


if __name__ == "__main__":
    main()
