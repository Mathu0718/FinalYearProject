# Train Custom Traffic Sign Detector

The default model (YOLOv8 on COCO) only detects **traffic light** and **stop sign**. To detect more signals (speed limit, yield, warning, U-turn, arrows, etc.) in your videos, use the flows below.

---

## U-turn and arrow (< >) signs (dashboard front view)

For **road signs on the left and right** of a front-view dashboard video (U-turn, &gt;, &lt;), you can train on a public dataset in one go:

1. **Get a free Roboflow API key:** https://app.roboflow.com/settings/api  
2. **Install and set key:**
   ```bash
   pip install roboflow
   set ROBOFLOW_API_KEY=your_key_here
   ```
   (On Linux/macOS: `export ROBOFLOW_API_KEY=your_key_here`)
3. **Download dataset and train** (one command):
   ```bash
   python scripts/train_arrow_signs.py
   ```
   This downloads a road-sign dataset, keeps only **u_turn**, **arrow_left** (&lt;), **arrow_right** (&gt;), and trains YOLOv8. Best weights: `runs/detect/arrow_signs/weights/best.pt`.

4. **Run on your video:**
   ```bash
   python scripts/run_hazard_detection.py --model runs/detect/arrow_signs/weights/best.pt videosample/sample.mp4
   ```

To only download the dataset (then train yourself):
```bash
python scripts/download_arrow_signs_dataset.py
python training/train_traffic_signs.py --data data/traffic_signs/data_arrows.yaml --epochs 80 --name arrow_signs
```

---

## 1. Dataset layout (manual labeling)

Use this structure (already created):

```
data/traffic_signs/
├── data.yaml          # dataset config (edit class names here)
├── train/
│   ├── images/        # put training images here
│   └── labels/        # one .txt per image, YOLO format
└── val/
    ├── images/        # validation images
    └── labels/        # validation labels
```

---

## 2. Get frames from your video

Extract frames so you can label them:

```bash
python scripts/extract_frames.py videosample/sample.mp4
```

Options:
- `--interval 2` – one frame every 2 seconds (default)
- `--interval 1` – one frame per second (more images)
- `--max-frames 300` – stop after 300 frames
- `--out-dir data/traffic_signs/val/images` – extract to validation set

Example: 200 train frames + 50 val frames:

```bash
python scripts/extract_frames.py videosample/sample.mp4 --interval 2 --max-frames 200
python scripts/extract_frames.py videosample/sampleone.mp4 --interval 3 --max-frames 50 --out-dir data/traffic_signs/val/images
```

Then move some train images into `val/images` (e.g. 20%) and create matching `val/labels/`.

---

## 3. Label images (YOLO format)

Each image needs a `.txt` file with the **same name** in the `labels` folder.

**Format:** one line per object:
```
class_id x_center y_center width height
```
All values **normalized** (0–1): x_center, y_center, width, height are divided by image width/height.

**Example:** `train/images/frame_000001.jpg` → `train/labels/frame_000001.txt`:
```
0 0.45 0.32 0.08 0.12
2 0.70 0.28 0.06 0.10
```
Here: class 0 = traffic_light, class 2 = speed_limit (see `data.yaml`).

### Labeling tools

- **labelImg** (YOLO export): https://github.com/HumanSignal/labelImg  
- **Roboflow** (upload images, label in browser, export YOLOv8): https://roboflow.com  
- **CVAT**: https://www.cvat.ai  

In `data/traffic_signs/data.yaml` set **class names** and **nc** to match your labels (same order as class_id 0, 1, 2, …).

---

## 4. Edit data.yaml

Open `data/traffic_signs/data.yaml`:

- **nc:** number of classes (e.g. 8).
- **names:** list of names for each class index (0, 1, 2, …).

Example for 5 sign types:

```yaml
nc: 5
names:
  0: traffic_light
  1: stop
  2: speed_limit
  3: yield
  4: warning
```

Paths `train: train/images` and `val: val/images` are relative to the folder containing `data.yaml`.

---

## 5. Run training

From project root:

```bash
python training/train_traffic_signs.py
```

Optional arguments:

```bash
python training/train_traffic_signs.py --data data/traffic_signs/data.yaml --epochs 80 --batch 8
```

- **--epochs** – more epochs often help (e.g. 80–100).
- **--batch** – reduce to 4 if you run out of memory.
- **--model yolov8s.pt** – slightly larger, often more accurate.

When training finishes, best weights are saved at:

```
runs/detect/traffic_signs/weights/best.pt
```

---

## 6. Run detection with your model

Use the trained weights on a video:

```bash
python scripts/run_hazard_detection.py --model runs/detect/traffic_signs/weights/best.pt videosample/sample.mp4
```

No need for `--all`; the custom model already outputs only your sign classes.

---

## Quick checklist

1. Extract frames: `python scripts/extract_frames.py videosample/sample.mp4`
2. Label images (YOLO format) and put `.txt` in `train/labels/` and `val/labels/`.
3. Edit `data/traffic_signs/data.yaml` (nc + names).
4. Train: `python training/train_traffic_signs.py --epochs 50`
5. Run: `python scripts/run_hazard_detection.py --model runs/detect/traffic_signs/weights/best.pt videosample/sample.mp4`
