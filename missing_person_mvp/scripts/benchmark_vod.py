from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from missing_person_mvp.utils.video import iter_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure VOD frame sampling throughput.")
    parser.add_argument("video_path", help="Path to a CCTV VOD file")
    parser.add_argument("--start", default="00:00:00", help="Start time HH:MM:SS")
    parser.add_argument("--end", default="01:00:00", help="End time HH:MM:SS")
    parser.add_argument("--sample-every", type=int, default=3, help="Analyze every Nth frame")
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        raise FileNotFoundError(args.video_path)

    started = time.perf_counter()
    frames = 0
    last_sec = 0.0
    for _frame, sec in iter_frames(args.video_path, args.start, args.end, args.sample_every):
        frames += 1
        last_sec = sec
    elapsed = time.perf_counter() - started
    fps = frames / elapsed if elapsed else 0.0
    print(f"sampled_frames={frames}")
    print(f"last_second={last_sec:.2f}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"sampled_fps={fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
