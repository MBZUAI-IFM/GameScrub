#!/usr/bin/env python3
"""Check the setup before running a real clip.

    set -a; source .env; set +a
    python scripts/check_setup.py

Checks every piece independently and prints everything that is still
missing in one pass.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

problems: list[str] = []


def ok(msg: str) -> None:
    print("  [OK] " + msg)


def fail(msg: str, hint: str) -> None:
    print("  [X]  " + msg)
    print(f"        -> {hint}")
    problems.append(msg)


def check_python_and_deps() -> None:
    print("\n[1/6] Main environment (requirements.txt)")
    if sys.version_info[:2] != (3, 10):
        fail(f"running Python {sys.version_info.major}.{sys.version_info.minor}",
             "use the Python 3.10 environment from README step 1 (not the ProPainter one)")
    else:
        ok("Python 3.10")
    for mod in ("numpy", "cv2", "scipy", "av", "PIL", "imageio_ffmpeg", "transformers", "safetensors", "einops"):
        try:
            __import__(mod)
            ok(f"{mod} importable")
        except Exception as e:
            fail(f"{mod} not importable ({e})", "pip install -r requirements.txt (README step 1)")


def check_ffmpeg() -> None:
    print("\n[2/6] ffmpeg")
    try:
        from hud_removal import config
        ok(f"using {config.FFMPEG_BIN}")
    except Exception as e:
        fail(f"could not resolve ffmpeg ({e})", "pip install -r requirements.txt (provides imageio-ffmpeg)")


def check_torch_gpu() -> None:
    print("\n[3/6] torch + GPU")
    try:
        import torch
        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            ok(f"CUDA available: {names}")
        else:
            fail("torch.cuda.is_available() is False", "run on a machine with an NVIDIA GPU + driver (nvidia-smi)")
    except Exception as e:
        fail(f"torch not importable ({e})", "pip install -r requirements.txt (README step 1)")


def check_ltx() -> None:
    print("\n[4/6] LTX-2 code (bundled) + LTX-2.5 checkpoints")
    from hud_removal import config
    try:
        from hud_removal.diffusion import _ltx_imports
        _ltx_imports()
        ok(f"bundled LTX-2 code imports ({config.LTX2_CODE_DIR})")
    except Exception as e:
        fail(f"bundled LTX-2 code does not import ({type(e).__name__}: {e})",
             "make sure requirements.txt is installed in this Python 3.10 environment")
    for var, path in [
        ("LTX25_TRANSFORMER", config.LTX25_TRANSFORMER), ("LTX25_TEXT_ENCODER", config.LTX25_TEXT_ENCODER),
        ("LTX25_VIDEO_VAE", config.LTX25_VIDEO_VAE), ("LTX25_AUDIO_VAE", config.LTX25_AUDIO_VAE),
        ("LTX25_SPATIAL_UPSAMPLER", config.LTX25_SPATIAL_UPSAMPLER),
    ]:
        if path.is_file():
            ok(f"{var}: {path.name} ({path.stat().st_size / 1e9:.1f} GB)")
        else:
            fail(f"{var} not found: {path}", "set the path in .env (README step 3)")


def check_propainter() -> None:
    print("\n[5/6] ProPainter (bundled code, own environment, checkpoints)")
    from hud_removal import config
    if (config.PROPAINTER_CODE_DIR / "inference_propainter.py").is_file():
        ok(f"bundled ProPainter code found ({config.PROPAINTER_CODE_DIR})")
    else:
        fail("third_party/propainter is incomplete", "re-copy this folder")
    for fname in config.PROPAINTER_WEIGHT_FILES:
        if (config.PROPAINTER_WEIGHTS_DIR / fname).is_file():
            ok(f"PROPAINTER_WEIGHTS_DIR/{fname} found")
        else:
            fail(f"{fname} not found in PROPAINTER_WEIGHTS_DIR={config.PROPAINTER_WEIGHTS_DIR}",
                 "set PROPAINTER_WEIGHTS_DIR in .env (README step 3)")
    py = config.PROPAINTER_PYTHON
    if not (shutil.which(py) or Path(py).is_file()):
        fail(f"PROPAINTER_PYTHON={py} not found", "set it to the ProPainter environment's python (README step 2)")
        return
    r = subprocess.run([py, "-c", "import sys, torch, cv2, numpy, scipy, imageio, einops; "
                            "print('%d.%d' % sys.version_info[:2], torch.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        ok(f"ProPainter environment works (Python, torch: {r.stdout.strip()})")
    else:
        fail("ProPainter environment is missing packages",
             "pip install -r requirements-propainter.txt in that environment (README step 2)")


def check_captioning() -> None:
    print("\n[6/6] Captioning API")
    from hud_removal import config
    if config.api_key():
        ok(f"API key set (base_url={config.OPENAI_BASE_URL})")
    else:
        fail("OPENAI_API_KEY is empty", "paste your API key into .env")
    if config.CAPTION_MODEL:
        ok(f"caption model: {config.CAPTION_MODEL}")
    else:
        fail("HUD_CAPTION_MODEL is empty", "put a vision-capable chat model name in .env")


def main() -> None:
    check_python_and_deps()
    check_ffmpeg()
    check_torch_gpu()
    check_ltx()
    check_propainter()
    check_captioning()
    print()
    if problems:
        print(f"{len(problems)} problem(s) -- fix the items marked [X] above and re-run.")
        sys.exit(1)
    print("All checks passed. Try scripts/run_pipeline.py on one clip.")


if __name__ == "__main__":
    main()
