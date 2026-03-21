"""
Step 3: Train YOLOv8x on the grocery shelf detection dataset.

Two-phase training:
  1. Train on 90/10 split to find best epoch
  2. Train on all data for best_epoch * 1.15

Usage:
  python train_yolo.py              # Phase 1: validation run
  python train_yolo.py --final      # Phase 2: full training
  python train_yolo.py --final --epochs 150  # Override epoch count
"""

import argparse
from pathlib import Path

# Monkeypatch torch.load for ultralytics 8.1.0 compatibility with PyTorch 2.6+
# (PyTorch 2.6 changed weights_only default to True, breaking ultralytics 8.1.0)
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"


def train_validation():
    """Phase 1: Train on 90/10 split to validate approach and find best epoch."""
    print("=" * 60)
    print("Phase 1: Validation training (90/10 split)")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    results = model.train(
        data=str(DATA_YAML),
        imgsz=1280,
        batch=4,  # Fits in 24 GB VRAM with YOLOv8x + mosaic at 1280
        epochs=300,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,  # Disabled: creates ghost products on dense shelves
        copy_paste=0.1,
        close_mosaic=30,
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name="yolov8x_val",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )

    return results


def train_final(epochs=None):
    """Phase 2: Train on all data for the specified number of epochs."""
    print("=" * 60)
    print(f"Phase 2: Final training on all data ({epochs} epochs)")
    print("=" * 60)

    # For final training, we need a data.yaml pointing to all images
    # Create a temporary data.yaml with train pointing to all images
    all_data_yaml = DATASET_DIR / "data_all.yaml"

    # Read original yaml and modify
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(all_data_yaml, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("val:"):
                # Point val to train as well (we don't need validation for final)
                f.write("val: train/images\n")
            else:
                f.write(line)

    model = YOLO("yolov8x.pt")

    results = model.train(
        data=str(all_data_yaml),
        imgsz=1280,
        batch=4,  # Fits in 24 GB VRAM with YOLOv8x + mosaic at 1280
        epochs=epochs or 200,
        patience=0,  # No early stopping for final run
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,  # Disabled: creates ghost products on dense shelves
        copy_paste=0.1,
        close_mosaic=30,
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name="yolov8x_final",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )

    return results


def export_best_weights():
    """Copy best weights to solution root for submission."""
    import shutil
    # Check for final run first, then validation run
    for name in ["yolov8x_final", "yolov8x_val"]:
        best = ROOT / "runs" / name / "weights" / "best.pt"
        if best.exists():
            dst = ROOT / "best.pt"
            shutil.copy2(best, dst)
            print(f"Exported {best} -> {dst}")
            print(f"Size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
            return
    print("WARNING: No best.pt found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Run final training on all data")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for final training")
    parser.add_argument("--export", action="store_true", help="Export best weights to solution root")
    args = parser.parse_args()

    if args.export:
        export_best_weights()
    elif args.final:
        train_final(args.epochs)
        export_best_weights()
    else:
        train_validation()
        export_best_weights()


if __name__ == "__main__":
    main()
