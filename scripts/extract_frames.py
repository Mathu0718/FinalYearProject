#!/usr/bin/env python3
"""
Extract frames from a video for labeling (e.g. every N seconds).
Output: data/traffic_signs/train/images/frame_000001.jpg, ...
Run from project root: python scripts/extract_frames.py videosample/sample.mp4
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for labeling")
    parser.add_argument("video", type=str, help="Path to video file (e.g. videosample/sample.mp4)")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Save one frame every N seconds (default 2)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(project_root / "data" / "traffic_signs" / "train" / "images"),
        help="Output folder for frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of frames to extract (default 500)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = project_root / video_path
    if not video_path.exists():
        print("Video not found:", video_path)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video:", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval_frames = max(1, int(fps * args.interval))
    count = 0
    saved = 0

    while saved < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval_frames == 0:
            name = out_dir / f"frame_{saved + 1:06d}.jpg"
            cv2.imwrite(str(name), frame)
            saved += 1
            if saved % 50 == 0:
                print("Saved", saved, "frames")
        count += 1

    cap.release()
    print("Done. Saved", saved, "frames to", out_dir)
    print("Next: label these images in YOLO format (see TRAINING.md), then run training.")


if __name__ == "__main__":
    main()
