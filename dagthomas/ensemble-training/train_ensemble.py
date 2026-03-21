"""
Train multiple YOLO variants for ensembling.

Trains different model sizes and random seeds sequentially to produce
diverse detectors that complement each other when ensembled.

Variants:
  1. YOLOv8x  seed=0   (largest, strongest baseline)
  2. YOLOv8x  seed=42  (same arch, different augmentation order)
  3. YOLOv8l  seed=0   (lighter, catches different patterns)
  4. YOLOv8x  seed=7   (third diverse seed for the strongest arch)

Each variant trains on 100% of the data (no early stopping).
All use mixup=0.0 (ghost products hurt precision on dense shelves).

Usage:
  python train_ensemble.py                          # Train all variants
  python train_ensemble.py --variants 0 1           # Train only variants 0 and 1
  python train_ensemble.py --epochs 150             # Override epoch count
  python train_ensemble.py --resume yolov8x_seed0   # Resume a specific run
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
DATA_ALL_YAML = DATASET_DIR / "data_all.yaml"
DATA_VAL_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs"

# ── Variant definitions ──────────────────────────────────────────────
VARIANTS = [
    {
        "name": "yolov8x_seed0",
        "model": "yolov8x.pt",
        "seed": 0,
        "imgsz": 1280,
        "batch": 4,
        "epochs": 120,
    },
    {
        "name": "yolov8x_seed42",
        "model": "yolov8x.pt",
        "seed": 42,
        "imgsz": 1280,
        "batch": 4,
        "epochs": 120,
    },
    {
        "name": "yolov8l_seed0",
        "model": "yolov8l.pt",
        "seed": 0,
        "imgsz": 1280,
        "batch": 6,     # lighter model → larger batch
        "epochs": 150,   # lighter model → more epochs
    },
    {
        "name": "yolov8x_seed7",
        "model": "yolov8x.pt",
        "seed": 7,
        "imgsz": 1280,
        "batch": 4,
        "epochs": 120,
    },
]


def train_variant(variant, epoch_override=None, use_val_split=False, resume=False):
    """Train a single YOLO variant."""
    name = variant["name"]
    epochs = epoch_override or variant["epochs"]
    data_yaml = DATA_VAL_YAML if use_val_split else DATA_ALL_YAML

    print("=" * 70)
    print(f"Training: {name}")
    print(f"  Model: {variant['model']}, Seed: {variant['seed']}")
    print(f"  Epochs: {epochs}, ImgSz: {variant['imgsz']}, Batch: {variant['batch']}")
    print(f"  Data: {data_yaml}")
    print("=" * 70)

    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_data.py first.")
        return None

    if resume:
        last_pt = RUNS_DIR / name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh")
            model = YOLO(variant["model"])
    else:
        model = YOLO(variant["model"])

    results = model.train(
        data=str(data_yaml),
        imgsz=variant["imgsz"],
        batch=variant["batch"],
        epochs=epochs,
        patience=0 if not use_val_split else 50,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        close_mosaic=30,
        amp=True,
        workers=8,
        seed=variant["seed"],
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
        resume=resume and (RUNS_DIR / name / "weights" / "last.pt").exists(),
    )

    # Strip optimizer from best weights
    best_pt = RUNS_DIR / name / "weights" / "best.pt"
    if best_pt.exists():
        stripped = ROOT / "weights" / f"{name}.pt"
        stripped.parent.mkdir(parents=True, exist_ok=True)
        ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        ckpt["optimizer"] = None
        torch.save(ckpt, str(stripped))
        size_mb = stripped.stat().st_size / 1024 / 1024
        print(f"Stripped weights saved: {stripped} ({size_mb:.1f} MB)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, nargs="*", default=None,
                        help="Which variant indices to train (default: all). E.g. --variants 0 2")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for all variants")
    parser.add_argument("--val-split", action="store_true",
                        help="Use 90/10 split instead of all-data (for validation runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a specific variant by name (e.g. yolov8x_seed0)")
    parser.add_argument("--list", action="store_true",
                        help="List available variants and exit")
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for i, v in enumerate(VARIANTS):
            print(f"  [{i}] {v['name']}  model={v['model']}  seed={v['seed']}  "
                  f"epochs={v['epochs']}  batch={v['batch']}")
        return

    if args.resume:
        variant = next((v for v in VARIANTS if v["name"] == args.resume), None)
        if variant is None:
            print(f"Unknown variant: {args.resume}")
            print("Available:", [v["name"] for v in VARIANTS])
            return
        train_variant(variant, epoch_override=args.epochs,
                      use_val_split=args.val_split, resume=True)
        return

    indices = args.variants if args.variants is not None else list(range(len(VARIANTS)))

    for i in indices:
        if i < 0 or i >= len(VARIANTS):
            print(f"WARNING: Variant index {i} out of range, skipping")
            continue
        train_variant(VARIANTS[i], epoch_override=args.epochs,
                      use_val_split=args.val_split)

    # Summary
    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    weights_dir = ROOT / "weights"
    if weights_dir.exists():
        for pt in sorted(weights_dir.glob("*.pt")):
            size_mb = pt.stat().st_size / 1024 / 1024
            print(f"  {pt.name}: {size_mb:.1f} MB")
        total_mb = sum(f.stat().st_size for f in weights_dir.glob("*.pt")) / 1024 / 1024
        print(f"  Total: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
