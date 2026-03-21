"""
V4 Training — Lessons learned from eval:

VM1 (1536px, long training, heavy augmentation):
  - 300 epochs (was 120-150), patience=50
  - copy_paste=0.3 (was 0.1-0.2) — critical for dense shelves
  - Multi-scale: imgsz=1536 with scale=0.7 (sees 1075-1536px effective)
  - close_mosaic=20 (keep mosaic even longer on 248 images)
  - erasing=0.5, mixup=0.15 (more augmentation diversity)

VM2 (640px specialist + classifier retraining):
  - 640px model for multi-scale ensemble diversity
  - Different receptive field catches what 1536px misses
  - Faster training = more epochs in same time

Usage:
  python cloud_train_v4.py --mode long --seeds 42,7        # VM1: 1536px 300ep
  python cloud_train_v4.py --mode multiscale --seeds 42,7  # VM2: 640px + 1024px
"""

import argparse
import shutil
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

assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"

# Merge images
merged_dir = DATASET_DIR / "merged" / "images"
merged_labels = DATASET_DIR / "merged" / "labels"
merged_dir.mkdir(parents=True, exist_ok=True)
merged_labels.mkdir(parents=True, exist_ok=True)

for src_name in ["train", "val"]:
    img_src = DATASET_DIR / src_name / "images"
    lbl_src = DATASET_DIR / src_name / "labels"
    if img_src.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in img_src.glob(ext):
                dst = merged_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
    if lbl_src.exists():
        for f in lbl_src.glob("*.txt"):
            dst = merged_labels / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

total_imgs = len(list(merged_dir.glob("*.*")))
total_lbls = len(list(merged_labels.glob("*.txt")))
print(f"Dataset: {total_imgs} images, {total_lbls} labels")
assert total_imgs >= 200, f"Expected 200+ images, got {total_imgs}"

# Ensure data_all.yaml
if not DATA_ALL_YAML.exists():
    with open(DATASET_DIR / "data.yaml", "r") as f:
        lines = f.readlines()
    with open(DATA_ALL_YAML, "w") as f:
        for line in lines:
            if line.startswith("train:"):
                f.write("train: merged/images\n")
            elif line.startswith("val:"):
                f.write("val: merged/images\n")
            else:
                f.write(line)


def strip_and_save(name):
    """Strip optimizer from best.pt to reduce size."""
    best_path = ROOT / "runs" / name / "weights" / "best.pt"
    if best_path.exists():
        stripped_path = ROOT / f"best_{name}.pt"
        c = torch.load(str(best_path), map_location="cpu", weights_only=False)
        c["optimizer"] = None
        torch.save(c, str(stripped_path))
        size_mb = stripped_path.stat().st_size / 1024 / 1024
        print(f"Stripped: {stripped_path} ({size_mb:.0f} MB)")
        return stripped_path
    else:
        print(f"WARNING: {best_path} not found!")
        return None


def train_long(seed: int, epochs: int = 300):
    """Long training at 1536px with heavy augmentation."""
    name = f"yolov8x_v4_seed{seed}"
    print("=" * 60)
    print(f"V4 LONG: {name} — seed={seed}, epochs={epochs}, imgsz=1536")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    model.train(
        data=str(DATA_ALL_YAML),
        imgsz=1536,
        batch=3,                # A100 40GB at 1536px
        epochs=epochs,
        patience=50,            # Early stop if plateau for 50 epochs
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,              # Slightly higher final LR for longer training
        cos_lr=True,
        warmup_epochs=5,        # Longer warmup for 300 epochs
        mosaic=1.0,
        mixup=0.15,             # Mild mixup — adds diversity without ghosts at low ratio
        copy_paste=0.3,         # Aggressive copy-paste for dense shelf scenes
        close_mosaic=20,        # Keep mosaic active til epoch 280
        degrees=0.0,            # NO rotation
        scale=0.7,              # Wider scale range (0.3-1.7x) for multi-scale
        translate=0.15,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.5,
        amp=True,
        workers=8,
        deterministic=True,
        seed=seed,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        save=True,
        save_period=50,
        verbose=True,
    )

    strip_and_save(name)
    print(f"Finished {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


def train_640(seed: int, epochs: int = 300):
    """640px specialist — different receptive field for ensemble diversity."""
    name = f"yolov8x_640_seed{seed}"
    print("=" * 60)
    print(f"V4 640px: {name} — seed={seed}, epochs={epochs}, imgsz=640")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    model.train(
        data=str(DATA_ALL_YAML),
        imgsz=640,
        batch=16,               # 640px fits 16 on A100
        epochs=epochs,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,
        cos_lr=True,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        close_mosaic=20,
        degrees=0.0,
        scale=0.5,
        translate=0.1,
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
        save_period=50,
        verbose=True,
    )

    strip_and_save(name)
    print(f"Finished {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


def train_1024(seed: int, epochs: int = 250):
    """1024px middle-ground — bridges gap between 640 and 1536."""
    name = f"yolov8x_1024_seed{seed}"
    print("=" * 60)
    print(f"V4 1024px: {name} — seed={seed}, epochs={epochs}, imgsz=1024")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    model.train(
        data=str(DATA_ALL_YAML),
        imgsz=1024,
        batch=6,                # A100 at 1024px
        epochs=epochs,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,
        cos_lr=True,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.25,
        close_mosaic=20,
        degrees=0.0,
        scale=0.6,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.45,
        amp=True,
        workers=8,
        deterministic=True,
        seed=seed,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        save=True,
        save_period=50,
        verbose=True,
    )

    strip_and_save(name)
    print(f"Finished {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["long", "multiscale"], required=True,
                        help="long=1536px 300ep, multiscale=640+1024px")
    parser.add_argument("--seeds", type=str, default="42,7")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: 300 for long, 300/250 for multiscale)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.mode == "long":
        epochs = args.epochs or 300
        print(f"V4 LONG: seeds={seeds}, epochs={epochs}, imgsz=1536")
        print(f"Estimated: {len(seeds) * 7:.0f} hours on A100")
        for i, seed in enumerate(seeds):
            print(f"\n### Model {i+1}/{len(seeds)}: seed={seed} ###\n")
            train_long(seed, epochs)

    elif args.mode == "multiscale":
        print(f"V4 MULTISCALE: seeds={seeds}")
        for i, seed in enumerate(seeds):
            # 640px first (fast), then 1024px
            print(f"\n### 640px seed={seed} ###\n")
            train_640(seed, args.epochs or 300)
            print(f"\n### 1024px seed={seed} ###\n")
            train_1024(seed, args.epochs or 250)

    # Summary
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    import glob
    for f in sorted(glob.glob(str(ROOT / "best_yolov8x_*.pt"))):
        p = Path(f)
        print(f"  {p.name} ({p.stat().st_size/1024/1024:.0f} MB)")
    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
