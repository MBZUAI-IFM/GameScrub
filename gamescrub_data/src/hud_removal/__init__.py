"""HUD-removal pipeline: clip + per-frame HUD boxes -> clean, HUD-free clip.

Four stages, each in its own module:
  1. mask_utils     - build hud_clip.mp4 + mask.mp4 from clip.mp4 + bboxes.json
  2. propainter_run - flow-guided pre-fill of the masked region (ProPainter)
  3. caption        - caption the clip as the diffusion model will see it
                      (HUD region blacked out), via an OpenAI-compatible
                      vision chat model
  4. diffusion      - masked-latent diffusion conditioned on the ProPainter
                      pre-fill + caption, then a hard-cut pixel composite
                      back onto the original clip

`pipeline.run_pipeline(...)` chains all four for a single clip.
"""
