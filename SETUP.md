# Road Safety AI System – Setup Instructions

## Phase 1 – Environment Setup

Follow these steps to prepare the environment. Complete Phase 1 before moving to Phase 2.

---

## 1. Prerequisites

- **Python 3.10 or higher** (3.11 or 3.12 recommended)
- **pip** (Python package installer)
- **Git** (optional; for cloning the project)

Check your Python version:

```bash
python --version
```

On some systems use `python3`:

```bash
python3 --version
```

---

## 2. Folder Structure

After Phase 1, the project layout is:

```
road_safety_ai_system/
├── config/                 # Configuration (Phase 2+)
│   └── __init__.py
├── data/                   # Datasets, models cache (Phase 2+)
│   └── .gitkeep
├── models/                 # Hazard detector, driver monitor, advisory (Phase 2+)
│   ├── __init__.py
│   └── hazard_detector.py  # Phase 2: YOLOv8 hazard detection
├── services/               # Weather API, alert service (Phase 2+)
│   └── __init__.py
├── training/               # YOLO training scripts (Phase 2+)
│   └── __init__.py
├── utils/                  # Video stream, helpers (Phase 2+)
│   └── __init__.py
├── scripts/                # Verification and utility scripts
│   ├── verify_installation.py
│   └── run_hazard_detection.py   # Phase 2: run hazard detection on webcam
├── requirements.txt
├── SETUP.md                # This file
└── README.md
```

---

## 3. Create a Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated.

**Windows (PowerShell):**

```powershell
cd c:\Kato\Collage\road_safety_ai_system
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
cd c:\Kato\Collage\road_safety_ai_system
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
cd /path/to/road_safety_ai_system
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your prompt when the environment is active.

---

## 4. Install Dependencies

With the virtual environment activated (or using your system Python):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installation may take several minutes (PyTorch and Ultralytics are large).

---

## 5. Verify Installation

Run the verification script from the **project root** (`road_safety_ai_system`):

```bash
python scripts/verify_installation.py
```

Or as a module:

```bash
python -m scripts.verify_installation
```

**Expected output:** All lines should show `OK` and end with:

```
SUCCESS: All checks passed. Environment is ready for Phase 2.
```

If any check shows `FAIL`, install the missing package (e.g. `pip install <package>`) and run the script again.

---

## 6. Troubleshooting

| Issue | Action |
|-------|--------|
| `python` not found | Use `python3` or add Python to PATH. |
| `pip` not found | Run `python -m pip install -r requirements.txt`. |
| PyTorch / CUDA errors | For CPU-only: `pip install torch torchvision`. For GPU, see [pytorch.org](https://pytorch.org). |
| OpenCV import error | Ensure `opencv-python` is installed: `pip install opencv-python`. |
| Permission errors on venv | Create venv in a user-writable directory or run without venv. |

---

## Next Step

After verification passes, **confirm Phase 1 is complete** and then provide the **Phase 2** requirements to continue.

---

## Phase 2 – Hazard Detection (run independently)

From project root, run YOLOv8 on your webcam with bounding boxes:

```bash
python scripts/run_hazard_detection.py
```

Or run the module directly:

```bash
python -m models.hazard_detector
```

Press **q** in the OpenCV window to quit. On first run, the pretrained model `yolov8n.pt` is downloaded automatically.
