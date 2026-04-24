#!/usr/bin/env python3
"""
Create VID2E upsampling manifests from a nuScenes camera sweep directory.

Input example:
  /path/to/v1.0-mini/sweeps/CAM_FRONT
    n008-...__CAM_FRONT__1533151603612404.jpg
    n008-...__CAM_FRONT__1533151603662404.jpg
    n015-...__CAM_FRONT__1538448745162460.jpg
    ...

Output example:
  /path/to/nuscenes_cam_front_manifests
    n008-2018-08-01-15-16-36-0400/
      frames.txt
      fps.txt
    n015-2018-10-02-10-50-40+0800/
      frames.txt
      fps.txt

`frames.txt` format:
  <relative_time_seconds> <absolute_image_path>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median


_NU_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)__(?P<camera>CAM_[A-Z0-9_]+)__(?P<timestamp>\d+)\.(?P<ext>[A-Za-z0-9]+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frames.txt manifests from nuScenes camera image filenames."
    )
    parser.add_argument(
        "--cam_dir",
        required=True,
        help="Path to nuScenes sweeps camera directory (e.g. sweeps/CAM_FRONT).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path where sequence manifest folders will be generated.",
    )
    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
        help="Camera name to include from filename pattern (default: CAM_FRONT).",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        help="Minimum frame count per sequence to emit (default: 2).",
    )
    parser.add_argument(
        "--no_fps",
        action="store_true",
        help="Do not emit fps.txt next to frames.txt.",
    )
    return parser.parse_args()


def _collect_sequences(cam_dir: Path, camera: str):
    groups: dict[str, list[tuple[int, Path]]] = {}

    for path in sorted(cam_dir.iterdir()):
        if not path.is_file():
            continue
        match = _NU_FILENAME_RE.match(path.name)
        if not match:
            continue
        if match.group("camera") != camera:
            continue
        prefix = match.group("prefix")
        ts = int(match.group("timestamp"))
        groups.setdefault(prefix, []).append((ts, path.resolve()))

    for prefix in groups:
        groups[prefix].sort(key=lambda x: x[0])
    return groups


def _write_manifest(
    sequence_dir: Path,
    rows: list[tuple[int, Path]],
    write_fps: bool,
):
    sequence_dir.mkdir(parents=True, exist_ok=True)
    start_ts = rows[0][0]
    rel_times = [(ts - start_ts) / 1_000_000.0 for ts, _ in rows]

    frames_path = sequence_dir / "frames.txt"
    with frames_path.open("w") as f:
        for rel_t, (_, img_path) in zip(rel_times, rows):
            f.write(f"{rel_t:.6f} {img_path}\n")

    if write_fps and len(rel_times) >= 2:
        deltas = [b - a for a, b in zip(rel_times[:-1], rel_times[1:]) if b > a]
        if deltas:
            fps = 1.0 / median(deltas)
            with (sequence_dir / "fps.txt").open("w") as f:
                f.write(f"{fps:.6f}\n")


def main():
    args = _parse_args()
    cam_dir = Path(args.cam_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not cam_dir.is_dir():
        raise FileNotFoundError(f"--cam_dir does not exist: {cam_dir}")

    sequences = _collect_sequences(cam_dir, args.camera)
    if not sequences:
        raise RuntimeError(
            f"No files matched camera={args.camera} in {cam_dir}. "
            "Expected filenames like '<prefix>__CAM_FRONT__<timestamp>.jpg'."
        )

    kept = 0
    dropped = 0
    total_frames = 0
    for prefix, rows in sorted(sequences.items()):
        if len(rows) < args.min_frames:
            dropped += 1
            continue
        sequence_dir = out_dir / prefix
        _write_manifest(sequence_dir, rows, write_fps=not args.no_fps)
        kept += 1
        total_frames += len(rows)

    if kept == 0:
        raise RuntimeError(
            f"No sequence has at least {args.min_frames} frames. "
            "Try a smaller --min_frames."
        )

    print(f"Generated manifests in: {out_dir}")
    print(f"camera: {args.camera}")
    print(f"sequences_kept: {kept}")
    print(f"sequences_dropped(min_frames): {dropped}")
    print(f"total_frames: {total_frames}")
    print("")
    print("Next step:")
    print(
        f"python upsampling/upsample.py --input_dir {out_dir} "
        "--output_dir /path/to/upsampled --device=auto --tf_log_level=3 --quiet"
    )


if __name__ == "__main__":
    main()
