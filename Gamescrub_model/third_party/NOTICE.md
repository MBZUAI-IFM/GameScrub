# Bundled third-party code

## `ltx2/` (LTX-2)

`ltx2/ltx_core/` and `ltx2/ltx_pipelines/` are copied unmodified from
https://github.com/Lightricks/LTX-2 at commit
`c1b4675863c07c59c92e502446cccdaedbe5889c` (2026-11-03)
(`packages/ltx-core/src/ltx_core` and `packages/ltx-pipelines/src/ltx_pipelines`).
The GameScrub scripts depend on the API of this exact version; other LTX-2
versions may not work.

This code, and the LTX-2.5 weights it loads, are licensed under the LTX Community
License Agreements in `ltx2/` (`LICENSE`, `LICENSE-2`, `LICENSE-2_x`). Use is
subject to the use-based restrictions in Section 4 and Attachment A of
`LICENSE-2_x`, in their entirety, including all use restrictions and acceptable
use policies.

## `ltx-kernels/`

Source of the `ltx-kernels` package (`packages/ltx-kernels`, same commit,
Apache-2.0, see its `pyproject.toml`). Only needed for multi-GPU inference
(`inference/generate_long_multigpu.py`); it must be compiled once, see the
README's "Multi-GPU setup".
