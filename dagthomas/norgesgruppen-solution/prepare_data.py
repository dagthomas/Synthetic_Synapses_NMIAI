"""
Step 1: Convert COCO annotations to YOLO format and create train/val split.

Creates:
  datasets/train/images/  (symlinks to original images)
  datasets/train/labels/  (YOLO format .txt files)
  datasets/val/images/
  datasets/val/labels/
  datasets/data.yaml
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

# Paths
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train")
ANNO_PATH = DATA_ROOT / "annotations.json"
IMG_DIR = DATA_ROOT / "images"
OUT_DIR = ROOT / "datasets"

VAL_RATIO = 0.10  # 10% for validation


def load_annotations():
    with open(ANNO_PATH, encoding="utf-8") as f:
        return json.load(f)


def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def stratified_split(image_annos, categories, val_ratio):
    """
    Split images into train/val, ensuring rare categories stay in train.
    Strategy: images containing a category with <= 3 annotations go to train.
    Remaining images are split randomly.
    """
    # Count annotations per category
    cat_counts = defaultdict(int)
    for img_id, annos in image_annos.items():
        for anno in annos:
            cat_counts[anno["category_id"]] += 1

    # Find rare categories (<=3 annotations)
    rare_cats = {cid for cid, cnt in cat_counts.items() if cnt <= 3}
    print(f"Rare categories (<=3 annotations): {len(rare_cats)}")

    # Images with rare categories must go to train
    forced_train = set()
    for img_id, annos in image_annos.items():
        for anno in annos:
            if anno["category_id"] in rare_cats:
                forced_train.add(img_id)
                break

    # Remaining images are split randomly
    remaining = [img_id for img_id in image_annos if img_id not in forced_train]
    random.shuffle(remaining)

    n_val = max(1, int(len(image_annos) * val_ratio))
    n_val_from_remaining = min(n_val, len(remaining))

    val_ids = set(remaining[:n_val_from_remaining])
    train_ids = forced_train | set(remaining[n_val_from_remaining:])

    return train_ids, val_ids


def main():
    print("Loading annotations...")
    coco = load_annotations()

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"Images: {len(images)}, Annotations: {len(coco['annotations'])}, Categories: {len(categories)}")

    # Group annotations by image
    image_annos = defaultdict(list)
    for anno in coco["annotations"]:
        image_annos[anno["image_id"]].append(anno)

    # Create YOLO label lines per image
    image_labels = {}
    for img_id, annos in image_annos.items():
        img = images[img_id]
        lines = []
        for anno in annos:
            xc, yc, w, h = convert_bbox_coco_to_yolo(
                anno["bbox"], img["width"], img["height"]
            )
            lines.append(f"{anno['category_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        image_labels[img_id] = lines

    # Split
    train_ids, val_ids = stratified_split(image_annos, categories, VAL_RATIO)
    print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")

    # Create output directories
    for split in ["train", "val"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Write images and labels
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in sorted(ids):
            img = images[img_id]
            src = IMG_DIR / img["file_name"]
            stem = Path(img["file_name"]).stem

            # Copy image (use copy since symlinks may not work on Windows)
            dst_img = OUT_DIR / split / "images" / img["file_name"]
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            # Write label
            dst_lbl = OUT_DIR / split / "labels" / f"{stem}.txt"
            with open(dst_lbl, "w") as f:
                f.write("\n".join(image_labels.get(img_id, [])))

    # Create data.yaml
    names_list = [categories[i] for i in range(len(categories))]
    data_yaml = {
        "path": str(OUT_DIR.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": len(categories),
        "names": names_list,
    }

    yaml_path = OUT_DIR / "data.yaml"
    # Write YAML manually to avoid importing yaml (blocked in sandbox)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for i, name in enumerate(names_list):
            # Escape quotes in names
            escaped = name.replace("'", "''")
            f.write(f"  {i}: '{escaped}'\n")

    print(f"Created data.yaml at {yaml_path}")

    # Print distribution stats
    train_annos = sum(len(image_annos[i]) for i in train_ids)
    val_annos = sum(len(image_annos[i]) for i in val_ids)
    print(f"Train annotations: {train_annos}, Val annotations: {val_annos}")

    # Check category coverage
    train_cats = set()
    for i in train_ids:
        for a in image_annos[i]:
            train_cats.add(a["category_id"])
    val_cats = set()
    for i in val_ids:
        for a in image_annos[i]:
            val_cats.add(a["category_id"])
    print(f"Categories in train: {len(train_cats)}, in val: {len(val_cats)}")
    missing_train = set(categories.keys()) - train_cats
    if missing_train:
        print(f"WARNING: {len(missing_train)} categories missing from train set")


if __name__ == "__main__":
    main()
