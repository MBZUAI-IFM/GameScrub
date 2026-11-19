#!/usr/bin/env python3
"""Write an --input-list jsonl for inference/generate.py from a folder of videos.

    python scripts/make_input_list.py VIDEO_FOLDER clips.jsonl

One line per video file (*.mp4, *.mov, *.mkv, *.webm, *.avi), searched recursively:
    {"id": "<file name without extension>", "source_video": "<absolute path>"}
Ids must be unique, because they become the output file names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    folder, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS and not p.name.startswith("."))
    if not videos:
        raise SystemExit(f"no videos found in {folder}")
    ids = [p.stem for p in videos]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SystemExit(f"duplicate file names (ids must be unique): {duplicates[:10]}")
    with out.open("w", encoding="utf-8") as handle:
        for p in videos:
            handle.write(json.dumps({"id": p.stem, "source_video": str(p)}) + "\n")
    print(f"wrote {len(videos)} videos to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
