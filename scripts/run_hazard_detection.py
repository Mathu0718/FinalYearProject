#!/usr/bin/env python3
"""
Phase 2 – Standalone runner for Hazard Detection (YOLOv8).
By default shows live in a window. If you have no display/GUI, use --headless to write a video file.
Run from project root:
  python scripts/run_hazard_detection.py                      # webcam or videosample/sample.mp4
  python scripts/run_hazard_detection.py videosample/xx.mp4 # video file
  python scripts/run_hazard_detection.py --headless           # no window, write to videosample/output_detection.mp4
  python scripts/run_hazard_detection.py --all               # all COCO objects
"""

import sys
from pathlib import Path

# Ensure project root is on path when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.hazard_detector import run_video

DEFAULT_SAMPLE = project_root / "videosample" / "sample.mp4"
DEFAULT_OUTPUT = project_root / "videosample" / "output_detection.mp4"


def _parse_args():
    raw = sys.argv[1:]
    sign_only = "--all" not in raw
    model = "yolov8n.pt"
    headless = "--headless" in raw
    output_path = None

    if "--model" in raw:
        i = raw.index("--model")
        if i + 1 < len(raw):
            model = raw[i + 1]
            raw = raw[:i] + raw[i + 2:]
    if "--output" in raw:
        i = raw.index("--output")
        if i + 1 < len(raw):
            output_path = raw[i + 1]
            raw = raw[:i] + raw[i + 2:]

    raw = [a for a in raw if a not in ("--all", "--headless")]
    source = raw[0] if raw else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if headless and output_path is None:
        output_path = str(DEFAULT_OUTPUT)

    return source, sign_only, model, headless, output_path


if __name__ == "__main__":
    source, sign_only, model_path, headless, output_path = _parse_args()

    def do_run(vid_source, use_headless=None, use_output=None):
        run_video(
            vid_source,
            sign_only=sign_only,
            model_path=model_path,
            headless=use_headless if use_headless is not None else headless,
            output_path=use_output if use_output is not None else output_path,
        )

    if source == 0:
        try:
            do_run(0)
        except RuntimeError as e:
            err = str(e).lower()
            if "webcam" in err and DEFAULT_SAMPLE.exists():
                print("Using project sample:", DEFAULT_SAMPLE)
                try:
                    do_run(str(DEFAULT_SAMPLE))
                except RuntimeError as e2:
                    if "display not available" in str(e2).lower() or "no gui" in str(e2).lower():
                        print("\nNo display available. Writing output to file instead.")
                        do_run(str(DEFAULT_SAMPLE), use_headless=True, use_output=str(DEFAULT_OUTPUT))
                        print("Done. Play this file to see detection:", DEFAULT_OUTPUT)
                    else:
                        raise
            elif "display not available" in err or "no gui" in err:
                print("\nNo display available. Run with --headless to write output to a video file.")
                raise
            else:
                raise
    else:
        try:
            do_run(source)
        except RuntimeError as e:
            if "display not available" in str(e).lower() or "no gui" in str(e).lower():
                print("\nNo display available. Writing output to file instead.")
                do_run(source, use_headless=True, use_output=str(DEFAULT_OUTPUT))
                print("Done. Play this file to see detection:", DEFAULT_OUTPUT)
            else:
                raise
