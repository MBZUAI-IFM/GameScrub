#!/usr/bin/env bash
# Remove the HUD from every video in a folder, splitting the videos over all GPUs of this
# machine (one independent process per GPU; each GPU loads its own copy of the model).
# Usage: bash scripts/run_folder_multigpu.sh VIDEO_FOLDER OUTPUT_DIR [extra generate.py flags]
# GPUs: all visible ones, or set GPUS="0 1 2 3". Re-running skips finished videos.
# Logs: OUTPUT_DIR/logs/gpu<N>.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] || { sed -n 2,6p "$0"; exit 2; }
IN=$1; OUT=$2; shift 2
set -a; source "$ROOT/.env"; set +a
mkdir -p "$OUT/logs"
python "$ROOT/scripts/make_input_list.py" "$IN" "$OUT/input_list.jsonl"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
echo "running on $N GPU(s): ${GPU_ARR[*]}"
pids=()
for RANK in "${!GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_ARR[$RANK]} python "$ROOT/inference/generate.py" \
    --input-list "$OUT/input_list.jsonl" --output-dir "$OUT" --rank "$RANK" --world-size "$N" "$@" \
    > "$OUT/logs/gpu${GPU_ARR[$RANK]}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "done (exit $status); outputs in $OUT"
exit $status
