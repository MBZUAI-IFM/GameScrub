#!/usr/bin/env python3
"""Check that everything GameScrub inference needs is in place. Run after setup:

    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/check_setup.py [--multigpu]

Prints [OK] / [X] per item; fix every [X] before running inference.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "ltx2"))

CHECKPOINTS = {  # env var -> (expected file name, approximate size in GB)
    "GAMESCRUB_TRANSFORMER": ("ltx-2.5-22b-distilled-transformer-bf16.safetensors", 42.0),
    "GAMESCRUB_TEXT_ENCODER": ("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", 26.3),
    "GAMESCRUB_VIDEO_VAE": ("ltx-2.5-video-vae-bf16.safetensors", 1.5),
    "GAMESCRUB_AUDIO_VAE": ("ltx-2.5-audio-vae-bf16.safetensors", 0.36),
    "GAMESCRUB_SPATIAL_UPSAMPLER": ("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 1.0),
    "GAMESCRUB_LORA": ("gamescrub_lora.safetensors", 1.3),
}
failures = 0


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str, hint: str) -> None:
    global failures
    failures += 1
    print(f"[X]  {msg}\n       -> {hint}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--multigpu", action="store_true", help="also check the multi-GPU (ltx-kernels) setup")
    args = parser.parse_args()

    if sys.version_info[:2] == (3, 10):
        ok(f"Python {sys.version.split()[0]}")
    else:
        fail(f"Python {sys.version.split()[0]}", "use Python 3.10 (the pinned wheels are for 3.10)")

    try:
        import torch

        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gb = props.total_memory / 1024**3
                line = f"GPU {i}: {props.name}, {gb:.0f} GB"
                ok(line) if gb >= 70 else fail(line, "need an 80 GB GPU (A100/H100 80GB); peak use is ~62 GB")
        else:
            fail("no CUDA GPU visible", "run on a GPU node; check nvidia-smi and the driver (CUDA 13.2 wheels)")
    except Exception as error:  # noqa: BLE001
        fail(f"import torch: {error}", "install requirements.txt (see README.md)")

    try:
        import ltx_pipelines.ic_lora  # noqa: F401

        ok("bundled LTX-2 code imports (third_party/ltx2)")
    except Exception as error:  # noqa: BLE001
        fail(f"import ltx_pipelines.ic_lora: {error}", "install requirements.txt with --no-deps (see README.md)")

    for env, (name, size_gb) in CHECKPOINTS.items():
        value = os.environ.get(env)
        if not value or value == "PATH TO CHECKPOINT":
            fail(f"{env} not set", "fill in .env and load it: set -a; source .env; set +a")
            continue
        path = Path(value)
        if not path.is_file():
            fail(f"{env}: no file at {path}", f"download {name} (see README.md)")
            continue
        got = path.stat().st_size / 1e9
        if abs(got - size_gb) > 0.2 * size_gb:
            fail(f"{env}: {path} is {got:.2f} GB, expected ~{size_gb} GB", f"is this {name}? re-download it")
        else:
            ok(f"{env}: {path.name} ({got:.1f} GB)")

    if args.multigpu:
        try:
            import all2all_cpp  # noqa: F401
            import ltx_kernels  # noqa: F401

            ok("ltx-kernels built (all2all_cpp imports)")
        except Exception as error:  # noqa: BLE001
            fail(f"ltx-kernels: {error}", "build it, see README.md 'Multi-GPU setup'")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} problem(s) found.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
