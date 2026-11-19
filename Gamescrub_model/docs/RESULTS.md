# GameScrub: results

All numbers were produced with the settings in this folder (default flags of
`inference/generate.py`: 1280x704, 145 frames, 30 fps, seed 42).

## Benchmark

200 clips: 100 synthetic (HUD rendered onto clean footage) and 100 in-the-wild
(real gameplay with real HUD).

| Metric | Overall | Synthetic | Wild |
|---|---|---|---|
| HUD removal (%) ↑ | 92.0 | 93.9 | 90.1 |
| Artifact-adjusted removal (%) ↑ | 77.1 | 82.2 | 72.0 |
| Background preservation (%) ↑ | 95.3 | 98.3 | 92.3 |
| Background LPIPS ↓ | 0.140 | 0.132 | 0.148 |
| Background MAE (0-255) ↓ | 5.57 | 5.47 | 5.66 |
| HUD detector reduction (%) ↑ | 94.3 | 97.3 | 91.2 |

- HUD removal, artifact-adjusted removal and background preservation are scored by an LLM judge.
- LPIPS and MAE compare output and input outside the HUD region.
- HUD detector reduction is the drop in detections from a dedicated HUD detector.

## Speed and memory (one NVIDIA A100-SXM4-80GB)

| | |
|---|---|
| Model time per 5 s clip (145 frames, 1280x704), excluding model loading | 81 s |
| Peak GPU memory, allocated / reserved | 40 GiB / 62 GiB |
| GPU-hours per hour of footage | 16.8 |
| First-time model loading | ~5 min |
| 59 s video (1777 frames), one GPU, 145-frame windows | ~20 min |
| 59 s video, one pass on 8 A100s | ~10 min generation + ~3.5 min decode |
