"""
Convert COCO annotations to YOLO format and create train/val splits.

Creates:
  datasets/train/images/   (copies of original images)
  datasets/train/labels/   (YOLO format .txt files)
  datasets/val/images/
  datasets/val/labels/
  datasets/data.yaml       (90/10 split)
  datasets/data_all.yaml   (all images in train, for final runs)

Usage:
  python prepare_data.py --coco-dir /path/to/NM_NGD_coco_dataset/train
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)


def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
    y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
    w_norm = max(0.0, min(1.0, w / img_w))
    h_norm = max(0.0, min(1.0, h / img_h))
    return x_center, y_center, w_norm, h_norm


def stratified_split(image_annos, categories, val_ratio):
    """Split images into train/val, keeping rare categories in train."""
    cat_counts = defaultdict(int)
    for img_id, annos in image_annos.items():
        for anno in annos:
            cat_counts[anno["category_id"]] += 1

    rare_cats = {cid for cid, cnt in cat_counts.items() if cnt <= 3}
    print(f"Rare categories (<=3 annotations): {len(rare_cats)}")

    forced_train = set()
    for img_id, annos in image_annos.items():
        for anno in annos:
            if anno["category_id"] in rare_cats:
                forced_train.add(img_id)
                break

    remaining = [img_id for img_id in image_annos if img_id not in forced_train]
    random.shuffle(remaining)

    n_val = max(1, int(len(image_annos) * val_ratio))
    n_val_from_remaining = min(n_val, len(remaining))

    val_ids = set(remaining[:n_val_from_remaining])
    train_ids = forced_train | set(remaining[n_val_from_remaining:])
    return train_ids, val_ids


def write_data_yaml(path, dataset_dir, nc, names_list, train_dir="train/images", val_dir="val/images"):
    """Write YOLO data.yaml without importing yaml (blocked in sandbox)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"path: {str(dataset_dir.resolve())}\n")
        f.write(f"train: {train_dir}\n")
        f.write(f"val: {val_dir}\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for i, name in enumerate(names_list):
            escaped = name.replace("'", "''")
            f.write(f"  {i}: '{escaped}'\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", required=True,
                        help="Path to NM_NGD_coco_dataset/train (contains images/ and annotations.json)")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    anno_path = coco_dir / "annotations.json"
    img_dir = coco_dir / "images"
    out_dir = Path(__file__).resolve().parent / "datasets"

    print("Loading annotations...")
    with open(anno_path, encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    print(f"Images: {len(images)}, Annotations: {len(coco['annotations'])}, Categories: {len(categories)}")

    # Group annotations by image
    image_annos = defaultdict(list)
    for anno in coco["annotations"]:
        image_annos[anno["image_id"]].append(anno)

    # Create YOLO label lines
    image_labels = {}
    for img_id, annos in image_annos.items():
        img = images[img_id]
        lines = []
        for anno in annos:
            xc, yc, w, h = convert_bbox_coco_to_yolo(anno["bbox"], img["width"], img["height"])
            lines.append(f"{anno['category_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        image_labels[img_id] = lines

    # Split
    train_ids, val_ids = stratified_split(image_annos, categories, args.val_ratio)
    print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")

    # Create output directories and write files
    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in sorted(ids):
            img = images[img_id]
            src = img_dir / img["file_name"]
            stem = Path(img["file_name"]).stem
            dst_img = out_dir / split / "images" / img["file_name"]
            if not dst_img.exists():
                shutil.copy2(src, dst_img)
            dst_lbl = out_dir / split / "labels" / f"{stem}.txt"
            with open(dst_lbl, "w") as f:
                f.write("\n".join(image_labels.get(img_id, [])))

    names_list = [categories[i] for i in range(len(categories))]

    # data.yaml — 90/10 split (for validation runs)
    write_data_yaml(out_dir / "data.yaml", out_dir, len(categories), names_list)

    # data_all.yaml — all images in train (for final runs)
    write_data_yaml(out_dir / "data_all.yaml", out_dir, len(categories), names_list,
                    train_dir="train/images", val_dir="train/images")

    print(f"Created data.yaml and data_all.yaml in {out_dir}")

    train_annos = sum(len(image_annos[i]) for i in train_ids)
    val_annos = sum(len(image_annos[i]) for i in val_ids)
    print(f"Train annotations: {train_annos}, Val annotations: {val_annos}")


if __name__ == "__main__":
    main()
