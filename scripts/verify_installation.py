#!/usr/bin/env python3
"""
Phase 1 – Installation verification script for Road Safety AI System.
Run this after installing requirements to confirm the environment is ready.
Usage: python scripts/verify_installation.py
       or from project root: python -m scripts.verify_installation
"""

import sys

def check_python_version():
    """Require Python 3.10+."""
    if sys.version_info < (3, 10):
        print(f"FAIL: Python 3.10+ required. Current: {sys.version}")
        return False
    print(f"OK   Python {sys.version.split()[0]}")
    return True

def check_import(module_name, package_name=None):
    """Try importing a module; return True on success."""
    name = package_name or module_name
    try:
        __import__(module_name)
        print(f"OK   {name}")
        return True
    except ImportError as e:
        print(f"FAIL {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Road Safety AI System – Installation Verification")
    print("=" * 60)

    all_ok = True

    if not check_python_version():
        all_ok = False

    print("\nChecking required packages:")
    all_ok &= check_import("ultralytics", "ultralytics (YOLOv8)")
    all_ok &= check_import("cv2", "opencv-python")
    all_ok &= check_import("torch", "torch")
    all_ok &= check_import("torchvision", "torchvision")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("joblib", "joblib")
    all_ok &= check_import("requests", "requests")
    all_ok &= check_import("numpy", "numpy")
    all_ok &= check_import("pandas", "pandas")

    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS: All checks passed. Environment is ready for Phase 2.")
    else:
        print("FAILED: Fix the missing packages above, then run this script again.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
