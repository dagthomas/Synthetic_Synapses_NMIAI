"""
Overnight cloud training — exact replica of the winning local v1 hyperparameters.
Trains multiple seeds sequentially for ensemble diversity.

Usage:
  python cloud_train_overnight.py --seeds 42,123    # VM1
  python cloud_train_overnight.py --seeds 7,99      # VM2

Each model takes ~2-3 hours on A100. Two seeds = ~5-6 hours total.
"""

import argparse
import time
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

# Verify data exists
assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"

# Merge ALL images into a single training directory for maximum data usage.
# The VMs may have split train/val — we want to train on everything.
import shutil

merged_dir = DATASET_DIR / "merged" / "images"
merged_labels = DATASET_DIR / "merged" / "labels"
merged_dir.mkdir(parents=True, exist_ok=True)
merged_labels.mkdir(parents=True, exist_ok=True)

img_count = 0
for src_name in ["train", "val", "all_images"]:
    img_src = DATASET_DIR / src_name / "images" if src_name != "all_images" else DATASET_DIR / src_name
    lbl_src = DATASET_DIR / src_name / "labels" if src_name != "all_images" else None

    if img_src.exists():
        for f in img_src.glob("*.jpg"):
            dst = merged_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                img_count += 1
                # Also copy matching label
                if lbl_src:
                    lbl = lbl_src / f.with_suffix(".txt").name
                    if lbl.exists():
                        shutil.copy2(lbl, merged_labels / lbl.name)
                else:
                    # Try train/labels and val/labels as fallback
                    for lb_dir in [DATASET_DIR / "train" / "labels", DATASET_DIR / "val" / "labels"]:
                        lbl = lb_dir / f.with_suffix(".txt").name
                        if lbl.exists():
                            shutil.copy2(lbl, merged_labels / lbl.name)
                            break

total_imgs = len(list(merged_dir.glob("*.jpg")))
total_lbls = len(list(merged_labels.glob("*.txt")))
print(f"Merged dataset: {total_imgs} images, {total_lbls} labels (added {img_count} new)")
assert total_imgs >= 100, f"Expected 100+ images, got {total_imgs}"

# Create/update data_all.yaml to point to merged directory
DATA_ALL_YAML = DATASET_DIR / "data_all.yaml"
# Read original data.yaml for class names
with open(DATASET_DIR / "data.yaml", "r", encoding="utf-8") as f:
    lines = f.readlines()
with open(DATA_ALL_YAML, "w", encoding="utf-8") as f:
    for line in lines:
        if line.startswith("train:"):
            f.write("train: merged/images\n")
        elif line.startswith("val:"):
            f.write("val: merged/images\n")
        else:
            f.write(line)
print(f"Updated {DATA_ALL_YAML} to use merged/ directory")


def train_one(seed: int, epochs: int = 120):
    """Train one YOLOv8x model with the exact winning v1 hyperparameters."""
    name = f"yolov8x_seed{seed}"
    print("=" * 60)
    print(f"Training {name} — seed={seed}, epochs={epochs}")
    print(f"Data: {DATA_ALL_YAML}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    # === EXACT COPY of winning local v1 hyperparameters ===
    results = model.train(
        data=str(DATA_ALL_YAML),
        imgsz=1280,
        batch=4,
        epochs=epochs,
        patience=0,            # No early stopping — train full duration
        optimizer="AdamW",
        lr0=0.001,             # Standard LR (NOT 0.0005)
        lrf=0.01,              # Standard final LR (NOT 0.005)
        cos_lr=True,
        mosaic=1.0,            # Mosaic ON (critical for small dataset)
        mixup=0.0,             # OFF — creates ghost products
        copy_paste=0.1,        # Mild copy-paste
        close_mosaic=30,       # Disable mosaic last 30 epochs
        degrees=0.0,           # NO ROTATION — products are always upright!
        scale=0.5,             # Standard scale (NOT 0.6)
        translate=0.1,         # Standard translate (NOT 0.15)
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.4,
        amp=True,
        workers=8,
        deterministic=True,
        seed=seed,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )

    # Strip optimizer from best.pt (785MB → 131MB)
    best_path = ROOT / "runs" / name / "weights" / "best.pt"
    if best_path.exists():
        stripped_path = ROOT / f"best_{name}.pt"
        c = torch.load(str(best_path), map_location="cpu", weights_only=False)
        c["optimizer"] = None
        torch.save(c, str(stripped_path))
        size_mb = stripped_path.stat().st_size / 1024 / 1024
        print(f"Stripped: {stripped_path} ({size_mb:.0f} MB)")
    else:
        print(f"WARNING: {best_path} not found!")

    print(f"Finished {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123",
                        help="Comma-separated seeds (default: 42,123)")
    parser.add_argument("--epochs", type=int, default=120,
                        help="Epochs per model (default: 120)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Overnight training: {len(seeds)} models, seeds={seeds}, epochs={args.epochs}")
    print(f"Estimated time: {len(seeds) * 2.5:.1f} hours")
    print()

    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 60}")
        print(f"# Model {i+1}/{len(seeds)}: seed={seed}")
        print(f"{'#' * 60}\n")
        train_one(seed, args.epochs)

    # Final summary
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    for seed in seeds:
        name = f"yolov8x_seed{seed}"
        stripped = ROOT / f"best_{name}.pt"
        if stripped.exists():
            size_mb = stripped.stat().st_size / 1024 / 1024
            print(f"  ✓ {stripped.name} ({size_mb:.0f} MB)")
        else:
            print(f"  ✕ {stripped.name} — NOT FOUND")

    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
