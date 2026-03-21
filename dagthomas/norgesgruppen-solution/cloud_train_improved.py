"""
Improved overnight training — based on winning v1 but with tweaks to push higher.

Changes vs v1:
- imgsz=1536 (match inference resolution, A100 has 40GB VRAM)
- epochs=150 (more training, patience=30 for early stop)
- copy_paste=0.2 (more copy-paste for dense shelves)
- close_mosaic=15 (keep mosaic active longer on tiny dataset)
- erasing=0.5 (more erasing for robustness)

Model weights are resolution-agnostic — training at 1536 doesn't increase model size.
Inference on L4 still works at 1536 (tested in current 0.9002 submission).

Usage:
  python cloud_train_improved.py --seeds 7,99     # ~3.5h each on A100
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

# Merge images (handle both .jpg and .jpeg)
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


def train_one(seed: int, epochs: int = 150):
    """Train one improved YOLOv8x model."""
    name = f"yolov8x_v3_seed{seed}"
    print("=" * 60)
    print(f"Training {name} — seed={seed}, epochs={epochs}, imgsz=1536")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    results = model.train(
        data=str(DATA_ALL_YAML),
        imgsz=1536,             # Match inference resolution
        batch=3,                # 3 fits in A100 40GB at 1536
        epochs=epochs,
        patience=30,            # Early stop if no improvement for 30 epochs
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.2,         # Doubled — more diversity for small dataset
        close_mosaic=15,        # Keep mosaic longer (disable at epoch 135/150)
        degrees=0.0,            # NO rotation!
        scale=0.5,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.5,            # More erasing for robustness
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

    # Strip optimizer
    best_path = ROOT / "runs" / name / "weights" / "best.pt"
    if best_path.exists():
        stripped_path = ROOT / f"best_{name}.pt"
        c = torch.load(str(best_path), map_location="cpu", weights_only=False)
        c["optimizer"] = None
        torch.save(c, str(stripped_path))
        size_mb = stripped_path.stat().st_size / 1024 / 1024
        print(f"Stripped: {stripped_path} ({size_mb:.0f} MB)")

    print(f"Finished {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="7,99",
                        help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Improved training: seeds={seeds}, epochs={args.epochs}, imgsz=1536")

    for i, seed in enumerate(seeds):
        print(f"\n# Model {i+1}/{len(seeds)}: seed={seed}\n")
        train_one(seed, args.epochs)

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    for seed in seeds:
        name = f"yolov8x_v3_seed{seed}"
        stripped = ROOT / f"best_{name}.pt"
        if stripped.exists():
            print(f"  ✓ {stripped.name} ({stripped.stat().st_size/1024/1024:.0f} MB)")
        else:
            print(f"  ✕ {stripped.name}")


if __name__ == "__main__":
    main()
