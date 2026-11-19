#!/usr/bin/env bash
# Remove the HUD from a long video in ONE pass (no windows), split across all GPUs of one node
# with sequence parallelism, then decode on one GPU. Needs the multi-GPU setup (README.md).
# Usage: bash scripts/run_long_video_multigpu.sh INPUT.mp4 OUTPUT_DIR [extra flags]
# Tested: 1777 frames (59 s) at 1280x704 on 8x A100 80GB: ~10 min generation + ~3.5 min decode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] || { sed -n 2,5p "$0"; exit 2; }
INPUT=$1; OUT=$2; shift 2
set -a; source "$ROOT/.env"; set +a
NAME=$(basename "${INPUT%.*}")
N=$(python "$ROOT/scripts/count_frames.py" "$INPUT")
mkdir -p "$OUT"
echo "$INPUT: generating $N frames"
python "$ROOT/inference/generate_long_multigpu.py" --source-video "$INPUT" --num-frames "$N" \
  --output-path "$OUT/$NAME.mp4" --latent-out "$OUT/${NAME}_latent" "$@"
echo "decoding on one GPU"
CUDA_VISIBLE_DEVICES=${DECODE_GPU:-0} python "$ROOT/inference/decode_latents.py" \
  --latent-prefix "$OUT/${NAME}_latent" --output-path "$OUT/$NAME.mp4"
rm -f "$OUT/${NAME}_latent".chunk*.pt
echo "wrote $OUT/$NAME.mp4"
