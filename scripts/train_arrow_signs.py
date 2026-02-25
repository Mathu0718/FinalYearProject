#!/usr/bin/env python3
"""
One-shot: download U-turn + arrow left/right dataset from Roboflow and train.
Signs on left/right of road (dashboard front view). Requires ROBOFLOW_API_KEY.
Run from project root: python scripts/train_arrow_signs.py
"""

import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("Set ROBOFLOW_API_KEY (free at https://app.roboflow.com/settings/api)")
        sys.exit(1)

    # Step 1: Download and prepare dataset
    print("=== Step 1: Download dataset (U-turn, arrow left, arrow right) ===\n")
    code = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "download_arrow_signs_dataset.py")],
        cwd=str(project_root),
    ).returncode
    if code != 0:
        sys.exit(code)

    # Step 2: Train
    print("\n=== Step 2: Training YOLOv8 on arrow signs ===\n")
    data_yaml = project_root / "data" / "traffic_signs" / "data_arrows.yaml"
    train_script = project_root / "training" / "train_traffic_signs.py"
    code = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--data", str(data_yaml),
            "--epochs", "80",
            "--name", "arrow_signs",
        ],
        cwd=str(project_root),
    ).returncode
    if code != 0:
        sys.exit(code)

    best = project_root / "runs" / "detect" / "arrow_signs" / "weights" / "best.pt"
    print("\nDone. Run on your dashboard video with:")
    print("  python scripts/run_hazard_detection.py --model", best, "videosample/sample.mp4")


if __name__ == "__main__":
    main()
