#!/usr/bin/env bash
# Remove the HUD from a long video (any length) on ONE GPU, in overlapping 145-frame windows.
# Usage: bash scripts/run_long_video.sh INPUT.mp4 OUTPUT_DIR [extra generate.py flags]
# Generates the largest 8k+1 frame count that fits in the video. ~20 min for 59 s on an A100.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] || { sed -n 2,4p "$0"; exit 2; }
INPUT=$1; OUT=$2; shift 2
set -a; source "$ROOT/.env"; set +a
N=$(python "$ROOT/scripts/count_frames.py" "$INPUT")
echo "$INPUT: generating $N frames"
python "$ROOT/inference/generate.py" --input-video "$INPUT" --output-dir "$OUT" \
  --num-frames "$N" --chunk-frames 145 --chunk-carry-frames 25 "$@"
