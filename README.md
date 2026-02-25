# AI-Based Road Safety System

Multi-phase project providing:

1. **Hazard Detection** (YOLOv8)  
2. **Driver Monitoring** (Drowsiness / Distraction)  
3. **Risk Advisory Model** (ML-based)  
4. **Weather API Integration**  
5. **Emergency Alert Simulation**  

---

## Phase 1 – Environment Setup

- **Folder structure:** See [SETUP.md](SETUP.md#2-folder-structure).  
- **Install:** Follow [SETUP.md](SETUP.md) (venv, `pip install -r requirements.txt`).  
- **Verify:** Run `python scripts/verify_installation.py` from the project root.  

## Phase 2 – Hazard Detection (Current)

- **Module:** `models/hazard_detector.py` — YOLOv8 pretrained or custom model, bounding boxes on video.  
- **Run:** `python scripts/run_hazard_detection.py` or `python scripts/run_hazard_detection.py videosample/sample.mp4`.  
- **Custom traffic signs:** Train on your own labels to detect more signal types. See **[TRAINING.md](TRAINING.md)** for dataset setup, labeling, and `--model runs/detect/traffic_signs/weights/best.pt`.

---

## Requirements

- Python 3.10+  
- See `requirements.txt` for packages (Ultralytics YOLOv8, OpenCV, PyTorch, Scikit-learn, Joblib, Requests, NumPy, Pandas).  

---

## Project Root

Run commands from the `road_safety_ai_system` directory (where `requirements.txt` is located).
