#!/usr/bin/env python3
"""
Train YOLOv8 on a custom traffic sign dataset (YOLO format).
Run from project root: python training/train_traffic_signs.py
Optional: python training/train_traffic_signs.py --data data/traffic_signs/data.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO

# Supported image extensions for YOLO
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _print_dataset_help(data_path: Path) -> None:
    print("\nTo get data for arrow signs (U-turn, <, >):")
    print("  1. Set ROBOFLOW_API_KEY (free at https://app.roboflow.com/settings/api)")
    print("  2. Run: python scripts/download_arrow_signs_dataset.py")
    print("  3. Then run this training command again.")
    print("\nOr add your own images to data/traffic_signs/train/images and")
    print("YOLO-format .txt labels to data/traffic_signs/train/labels. See TRAINING.md.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for traffic sign detection")
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "data" / "traffic_signs" / "data.yaml"),
        help="Path to data.yaml (dataset config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (pixels)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (-1 = auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (yolov8n.pt, yolov8s.pt, etc.)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(project_root / "runs" / "detect"),
        help="Project folder for saving runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="traffic_signs",
        help="Run name (folder under project)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print("Dataset not found:", data_path)
        print("Create data/traffic_signs/ and add images + labels. See TRAINING.md.")
        sys.exit(1)

    # Resolve dataset root (parent of data.yaml) and check train/images has images
    data_root = data_path.resolve().parent
    train_img = data_root / "train" / "images"
    if not train_img.exists():
        print("Train images folder not found:", train_img)
        _print_dataset_help(data_path)
        sys.exit(1)
    imgs = list(train_img.glob("*.*"))
    imgs = [f for f in imgs if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")]
    if not imgs:
        print("No images in", train_img)
        _print_dataset_help(data_path)
        sys.exit(1)

    print("Loading base model:", args.model)
    model = YOLO(args.model)

    print("Training on:", data_path)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("Training finished. Best weights:", best_weights)
    print("Run detection with: python scripts/run_hazard_detection.py --model", best_weights, "videosample/sample.mp4")


if __name__ == "__main__":
    main()
