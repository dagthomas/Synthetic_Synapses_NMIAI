"""
YOLOv8x training v2 — improved augmentation for better generalization.

Changes from v1:
- 200 epochs (was 120) — more training time for small dataset
- copy_paste=0.2 (was 0.1) — more augmentation diversity
- degrees=10 — slight rotation for tilted products
- translate=0.15 — more position variation
- scale=0.6 — wider scale range (0.4-1.6x)
- erasing=0.3 — random erasing for occlusion robustness
- close_mosaic=40 (was 30) — more mosaic before pure training
- weight_decay through optimizer config

Usage:
  python train_yolo_v2.py --final --epochs 200
"""

import argparse
from pathlib import Path

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


def train_final(epochs=200):
    """Train on all data with improved augmentation."""
    print("=" * 60)
    print(f"YOLOv8x v2: Final training ({epochs} epochs, improved augmentation)")
    print("=" * 60)

    all_data_yaml = DATASET_DIR / "data_all.yaml"

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(all_data_yaml, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("val:"):
                f.write("val: train/images\n")
            else:
                f.write(line)

    model = YOLO("yolov8x.pt")

    results = model.train(
        data=str(all_data_yaml),
        imgsz=1280,
        batch=4,
        epochs=epochs,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        # Augmentation — more aggressive for generalization
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.2,       # Was 0.1 — more diversity
        close_mosaic=40,      # Was 30 — longer mosaic phase
        degrees=10.0,         # Slight rotation
        translate=0.15,       # Position variation
        scale=0.6,            # Scale variation (0.4x-1.6x)
        fliplr=0.5,           # Horizontal flip
        erasing=0.3,          # Random erasing for robustness
        # Training
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name="yolov8x_v2",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )

    return results


def export_best_weights():
    """Strip optimizer and copy best weights to solution root."""
    best = ROOT / "runs" / "yolov8x_v2" / "weights" / "best.pt"
    if best.exists():
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        ckpt["optimizer"] = None
        dst = ROOT / "best_v2.pt"
        torch.save(ckpt, str(dst))
        print(f"Exported {best} -> {dst}")
        print(f"Size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("WARNING: No best.pt found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.export:
        export_best_weights()
    elif args.final:
        train_final(args.epochs)
        export_best_weights()
    else:
        print("Use --final to train on all data")


if __name__ == "__main__":
    main()
