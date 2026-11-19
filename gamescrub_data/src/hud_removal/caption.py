"""Stage 3: caption the clip the way the diffusion model will actually see
it -- HUD region(s) blacked out -- via an OpenAI-compatible vision chat
model. Needs OPENAI_API_KEY and HUD_CAPTION_MODEL (see .env.example).
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import config

# Rules 1-7 are the official LTX-2 training-caption style rules; only the
# framing (describe THIS clip from sampled frames) and the repaint note
# below are specific to this pipeline.
CAPTION_SYSTEM = """You are shown {n} frames sampled in chronological order from a {dur:.1f}-second single-take video clip. Write ONE caption describing THIS clip, in the EXACT style of the training captions used for the LTX-2 video model. The caption will guide a detail-preserving generation of the same clip, so it must be faithful: describe only what is actually visible in the frames and how it actually changes across them. Never invent events, objects, text or camera moves that are not visible.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…", "The video shows…".
2. Objective, observable description only. Do not infer emotions or intentions.
3. Full visual detail: environment, character appearance, spatial positioning of all elements.
4. Precise motion and cinematic description woven into the prose: shot type, camera motion, camera viewpoint.
5. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "By the end…".
6. Visual medium and condition, briefly, near the end (digital video, animation, etc. -- whatever is actually visible).
7. One single continuous paragraph, 150-220 words, never exceeding 300 words. No bullet points, no labels.

Audio: {audio_rule}

Return JSON: {{"prompt": "<the caption>"}}"""

AUDIO_RULE_SILENT = ("the clip has no audio track; end the caption with a short statement that there is no sound "
                     "(no speech, no music, no ambient audio).")
AUDIO_RULE_UNKNOWN = ("the clip has an audio track but it was NOT provided to you; do not describe or invent any "
                      "sounds, speech or music -- describe the visuals only.")

REPAINT_NOTE = """

REPAINT CONTEXT: one or more regions of every frame (marked for you, tracked through the whole clip) are BAD and will be repainted; everything else stays exactly as it is. Do NOT describe, guess or hint at the current content of the marked region(s) -- never mention a defect, a box, or black/solid-colored patches anywhere in the caption. Describe the region(s) as the natural continuation of the surrounding scene: same setting, same lighting, same camera, no new people/objects the rest of the frame contradicts."""


def _video_meta(path: str) -> tuple[int, int, int, float]:
    import av
    with av.open(path) as c:
        s = next(x for x in c.streams if x.type == "video")
        n = s.frames or 0
        if n == 0:
            n = sum(1 for _ in c.decode(s))
        return s.codec_context.width, s.codec_context.height, n, float(s.average_rate or 24)


def _has_audio(path: str) -> bool:
    try:
        import av
        with av.open(path) as c:
            return any(s.type == "audio" for s in c.streams)
    except Exception:
        return False


def _frames_at(path: str, indices: list[int]):
    """Yield (index, numpy RGB array) for the requested frame indices."""
    import av
    want = sorted(set(indices))
    with av.open(path) as c:
        s = next(x for x in c.streams if x.type == "video")
        for i, fr in enumerate(c.decode(s)):
            if i > want[-1]:
                break
            if i in want:
                yield i, fr.to_ndarray(format="rgb24")


def _jpeg_b64(rgb: np.ndarray, width: int = 832, quality: int = 85) -> str:
    from PIL import Image
    img = Image.fromarray(rgb)
    if img.width > width:
        img = img.resize((width, max(2, round(img.height * width / img.width) // 2 * 2)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _post_json(path: str, body: dict, key: str, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        config.OPENAI_BASE_URL + path, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI {e.code}: {e.read().decode(errors='replace')[:400]}") from e


def _chat_json(body: dict, key: str) -> dict:
    d = _post_json("/chat/completions", body, key)
    ch = d["choices"][0]
    txt = (ch["message"].get("content") or "").strip().strip("`")
    if txt.startswith("json"):
        txt = txt[4:].strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < i:
        raise RuntimeError(f"model returned no JSON (finish_reason={ch.get('finish_reason')}): {txt[:200]!r}")
    js = json.loads(txt[i:j + 1])
    js["_usage"] = d.get("usage") or {}
    return js


def _sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def caption_clip(video_path: Path, mask_pixel_bool: np.ndarray) -> dict:
    """Caption `video_path` as the diffusion model will see it: every sampled
    frame has `mask_pixel_bool[frame_index]` (H, W) blacked out first, and the
    model is told that region is a to-be-repainted placeholder, not real
    content. Returns {"prompt": ..., "model": ..., "frames": ..., ...}."""
    key = config.api_key()
    if not key:
        raise RuntimeError(
            "No API key found. Paste your API key into OPENAI_API_KEY in .env "
            "and load it (set -a; source .env; set +a)."
        )
    if not config.CAPTION_MODEL:
        raise RuntimeError(
            "No caption model set. Put the name of a vision-capable chat model "
            "in HUD_CAPTION_MODEL in .env."
        )
    w, h, n, fps = _video_meta(str(video_path))
    k = 8 if n >= 8 else max(1, n)
    idx = _sample_indices(n, k)

    content = [{"type": "text", "text": f"{len(idx)} frames in temporal order (t = " +
                ", ".join(f"{i / fps:.1f}" for i in idx) + " s):"}]
    got = 0
    for i, frame in _frames_at(str(video_path), idx):
        arr = frame.copy()
        m = mask_pixel_bool[min(i, mask_pixel_bool.shape[0] - 1)]
        arr[m] = 0  # solid black -- this is what the model is actually conditioned on
        got += 1
        content += [
            {"type": "text", "text": f"Frame {got} (t={i / fps:.1f}s):"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _jpeg_b64(arr), "detail": "low"}},
        ]
    if got == 0:
        raise RuntimeError(f"could not decode any frames from {video_path}")

    audio_rule = AUDIO_RULE_UNKNOWN if _has_audio(str(video_path)) else AUDIO_RULE_SILENT
    system = CAPTION_SYSTEM.format(n=got, dur=(n - 1) / fps, audio_rule=audio_rule) + REPAINT_NOTE

    body = {
        "model": config.CAPTION_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 8000,
    }
    if config.CAPTION_MODEL.startswith(("gpt-5", "gpt-6", "o")):
        body["reasoning_effort"] = config.CAPTION_REASONING_EFFORT

    js = _chat_json(body, key)
    prompt = str(js.get("prompt", "")).strip()
    if not prompt:
        raise RuntimeError("model response had no 'prompt' field")
    return {
        "prompt": prompt, "model": config.CAPTION_MODEL, "frames": got,
        "frame_times": [round(i / fps, 2) for i in idx], "tokens": js["_usage"].get("total_tokens"),
    }
