#!/usr/bin/env python3
"""Print the largest 8k+1 frame count <= the number of frames in a video (for --num-frames).

    python scripts/count_frames.py video.mp4
"""
import sys

import av


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with av.open(sys.argv[1]) as container:
        stream = container.streams.video[0]
        n = stream.frames or sum(1 for _ in container.decode(stream))
    if n < 9:
        raise SystemExit(f"{sys.argv[1]}: only {n} frames")
    print((n - 1) // 8 * 8 + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
