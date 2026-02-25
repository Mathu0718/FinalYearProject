#!/usr/bin/env python3
"""
Download a traffic sign dataset from Roboflow and prepare for U-turn + arrow left/right training.
Maps dataset classes to: u_turn (0), arrow_left (1), arrow_right (2).
Requires: pip install roboflow, and set env ROBOFLOW_API_KEY (get free key at roboflow.com).
Run from project root: python scripts/download_arrow_signs_dataset.py
"""

import os
import re
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
out_train_img = project_root / "data" / "traffic_signs" / "train" / "images"
out_train_lbl = project_root / "data" / "traffic_signs" / "train" / "labels"
out_val_img = project_root / "data" / "traffic_signs" / "val" / "images"
out_val_lbl = project_root / "data" / "traffic_signs" / "val" / "labels"

# Map Roboflow dataset class names -> our class index (0=u_turn, 1=arrow_left, 2=arrow_right)
NAME_TO_OUR_ID = {
    "u_turn": 0,
    "u-turn": 0,
    "uturn": 0,
    "u-trun": 0,  # typo in some datasets
    "do_not_turn_l": 1,
    "turn_left": 1,
    "left_arrow": 1,
    "arrow_left": 1,
    "do_not_turn_r": 2,
    "turn_right": 2,
    "right_arrow": 2,
    "arrow_right": 2,
}


def normalize_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def our_class_id(their_name: str) -> int | None:
    n = normalize_name(their_name)
    for key, cid in NAME_TO_OUR_ID.items():
        if normalize_name(key) == n or key.replace("_", "") == n.replace("_", ""):
            return cid
    if "left" in n and "turn" in n:
        return 1
    if "right" in n and "turn" in n:
        return 2
    if "u" in n and "turn" in n:
        return 0
    return None


def remap_label_file(
    src_path: Path,
    dst_path: Path,
    their_names: list[str],
) -> bool:
    """Rewrite label file keeping only u_turn/arrow_left/arrow_right, remap to 0,1,2. Return True if any line kept."""
    if not src_path.exists():
        return False
    kept = []
    for line in src_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= len(their_names):
                continue
            name = their_names[cls_id]
            new_id = our_class_id(name)
            if new_id is None:
                continue
            parts[0] = str(new_id)
            kept.append(" ".join(parts))
        except (ValueError, IndexError):
            continue
    if not kept:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(kept) + "\n")
    return True


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("Set ROBOFLOW_API_KEY (get a free key at https://app.roboflow.com/settings/api)")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        sys.exit(1)

    # Roboflow 100 – road signs (has u_turn, do_not_turn_l, do_not_turn_r, etc.)
    workspace_name = "roboflow-100"
    project_name = "road-signs-6ih4y"
    download_dir = project_root / "data" / "roboflow_download"

    print("Downloading dataset from Roboflow (this may take a minute)...")
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_name)
    project = workspace.project(project_name)
    version = project.version(1)
    version.download("yolov8", location=str(download_dir))

    # Find the downloaded folder (e.g. road-signs-6ih4y-1)
    subdirs = [d for d in download_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("Download folder empty or unexpected structure:", download_dir)
        sys.exit(1)
    data_root = subdirs[0]

    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_root / "data.yml"
    if not data_yaml.exists():
        print("No data.yaml found in", data_root)
        sys.exit(1)

    # Parse class names from data.yaml
    text = data_yaml.read_text()
    names_match = re.search(r"names\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if names_match:
        names_str = names_match.group(1)
        their_names = [s.strip().strip('"').strip("'") for s in names_str.split(",")]
    else:
        # try nc: 21 names: {0: x, 1: y, ...}
        their_names = []
        for m in re.finditer(r"(\d+)\s*:\s*([^\s\n]+)", text):
            idx, name = int(m.group(1)), m.group(2).strip().strip('"').strip("'")
            while len(their_names) <= idx:
                their_names.append("")
            their_names[idx] = name
        if not their_names:
            print("Could not parse class names from data.yaml")
            sys.exit(1)

    print("Dataset classes:", their_names)
    our_set = set()
    for n in their_names:
        cid = our_class_id(n)
        if cid is not None:
            our_set.add((cid, ["u_turn", "arrow_left", "arrow_right"][cid]))
    print("Mapped to our classes:", dict(our_set))

    for split, out_img, out_lbl in [("train", out_train_img, out_train_lbl), ("valid", out_val_img, out_val_lbl)]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.exists():
            img_dir = data_root / split / "img"
        if not lbl_dir.exists():
            lbl_dir = data_root / split / "label"
        if split == "valid" and not img_dir.exists():
            img_dir = data_root / "val" / "images"
            lbl_dir = data_root / "val" / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            out_lbl_path = out_lbl / f"{stem}.txt"
            if remap_label_file(lbl_path, out_lbl_path, their_names):
                shutil.copy2(img_path, out_img / img_path.name)
                count += 1
        print(f"{split}: copied {count} images with u_turn/arrow_left/arrow_right labels.")

    # Ensure we have data_arrows.yaml
    arrows_yaml = project_root / "data" / "traffic_signs" / "data_arrows.yaml"
    if not arrows_yaml.exists():
        arrows_yaml.write_text("""path: .
train: train/images
val: val/images
nc: 3
names:
  0: u_turn
  1: arrow_left
  2: arrow_right
""")
    print("Dataset ready. Train with:")
    print("  python training/train_traffic_signs.py --data data/traffic_signs/data_arrows.yaml --epochs 80")


if __name__ == "__main__":
    main()
