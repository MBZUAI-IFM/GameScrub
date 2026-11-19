#!/usr/bin/env bash
# Remove the HUD from one video (first 145 frames = ~4.8 s at 30 fps) on one GPU.
# Usage: bash scripts/run_one_video.sh INPUT.mp4 OUTPUT_DIR [extra generate.py flags]
# Checkpoints come from .env (see .env.example).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] || { sed -n 2,4p "$0"; exit 2; }
INPUT=$1; OUT=$2; shift 2
set -a; source "$ROOT/.env"; set +a
python "$ROOT/inference/generate.py" --input-video "$INPUT" --output-dir "$OUT" "$@"
