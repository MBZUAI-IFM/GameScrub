#!/usr/bin/env python3
"""Run the pipeline over many clips on one GPU.

Runs stages 1-3 (mask prep, ProPainter, caption) for every clip first, then
loads the diffusion model once and runs stage 4 for every clip. Expects a
JSON manifest:

    [
      {"clip": "path/to/a.mp4", "bboxes": "path/to/a_bboxes.json", "out_dir": "out/a"},
      {"clip": "path/to/b.mp4", "bboxes": "path/to/b_bboxes.json", "out_dir": "out/b"}
    ]

    python scripts/run_pipeline_batch.py --manifest clips.json --gpu 0

To use several GPUs, split the manifest and run one process per GPU.
A clip that fails is reported and skipped; re-running the same command
resumes, because every stage skips outputs that already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gpu", default="0", help="CUDA device id (used by ProPainter and diffusion)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # before torch / ProPainter start
    from hud_removal.diffusion import DiffusionModel
    from hud_removal.pipeline import diffuse, prepare

    items = json.loads(args.manifest.read_text())
    failed: list[str] = []

    prepared = []
    for item in items:
        try:
            prepared.append(prepare(Path(item["clip"]), Path(item["bboxes"]), Path(item["out_dir"])))
        except Exception:
            failed.append(item["out_dir"])
            print(f"{item['out_dir']}: FAILED in stages 1-3\n{traceback.format_exc()}", flush=True)

    if prepared:
        print(f"loading diffusion model on gpu {args.gpu}...", flush=True)
        model = DiffusionModel(gpu_id=args.gpu)
        for pc in prepared:
            try:
                diffuse(pc, model)
            except Exception:
                failed.append(str(pc.out_dir))
                print(f"{pc.out_dir}: FAILED in stage 4\n{traceback.format_exc()}", flush=True)

    print(f"batch done: {len(items) - len(failed)} ok, {len(failed)} failed", flush=True)
    for f in failed:
        print(f"  failed: {f}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
