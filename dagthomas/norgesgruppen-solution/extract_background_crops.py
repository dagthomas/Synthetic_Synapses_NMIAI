"""
Extract background (empty shelf) crops from COCO-annotated images.

Creates crops from regions with no product annotations for use as a
"background" class (class 356) in the classifier. This helps the classifier
reject false positive detections on empty shelves, price tags, etc.

Usage:
  python extract_background_crops.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("X:/norgesgruppen")
COCO_DIR = DATA_ROOT / "NM_NGD_coco_dataset" / "train"
OUT_DIR = ROOT / "classifier_data" / "crops" / "356"

CROP_SIZE = 224
TARGET_CROPS = 1000
MIN_CROP_PX = 80       # Minimum crop dimension before resize
GRID_STEP = 150        # Grid sampling step in pixels
MAX_IOU_WITH_ANNO = 0.05  # Maximum overlap with any annotation


def load_coco():
    with open(COCO_DIR / "annotations.json", encoding="utf-8") as f:
        return json.load(f)


def box_iou(box_a, box_b):
    """Compute IoU between box_a [x1,y1,x2,y2] and box_b [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def find_empty_regions(img_w, img_h, anno_boxes, crop_sizes):
    """Find grid cells that don't overlap with any annotation box."""
    regions = []

    for crop_w, crop_h in crop_sizes:
        for y in range(0, img_h - crop_h + 1, GRID_STEP):
            for x in range(0, img_w - crop_w + 1, GRID_STEP):
                cand = [x, y, x + crop_w, y + crop_h]

                # Check overlap with all annotations
                max_iou = 0.0
                for ab in anno_boxes:
                    iou = box_iou(cand, ab)
                    if iou > MAX_IOU_WITH_ANNO:
                        max_iou = iou
                        break
                    max_iou = max(max_iou, iou)

                if max_iou <= MAX_IOU_WITH_ANNO:
                    regions.append((x, y, x + crop_w, y + crop_h))

    return regions


def main():
    print("Loading COCO annotations...")
    coco = load_coco()

    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    annos_by_image = defaultdict(list)
    for anno in coco["annotations"]:
        annos_by_image[anno["image_id"]].append(anno)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_crops = 0
    crop_sizes = [
        (MIN_CROP_PX, MIN_CROP_PX),
        (150, 150),
        (200, 200),
        (120, 200),  # Tall narrow (shelf gap)
        (200, 120),  # Wide short (shelf edge)
    ]

    for img_id, img_info in sorted(images.items()):
        if total_crops >= TARGET_CROPS:
            break

        img_path = COCO_DIR / "images" / img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Convert COCO [x,y,w,h] to [x1,y1,x2,y2]
        anno_boxes = []
        for anno in annos_by_image.get(img_id, []):
            bx, by, bw, bh = anno["bbox"]
            anno_boxes.append([bx, by, bx + bw, by + bh])

        if not anno_boxes:
            continue

        # Find empty regions
        regions = find_empty_regions(img_w, img_h, anno_boxes, crop_sizes)

        if not regions:
            continue

        # Randomly sample from found regions
        rng = np.random.RandomState(img_id)
        n_sample = min(len(regions), max(1, (TARGET_CROPS - total_crops) // max(1, len(images) - img_id)))
        chosen = rng.choice(len(regions), size=min(n_sample, len(regions)), replace=False)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Cannot open {img_path}: {e}")
            continue

        for idx in chosen:
            x1, y1, x2, y2 = regions[idx]
            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

            crop_path = OUT_DIR / f"bg_{img_id}_{total_crops}.jpg"
            crop.save(crop_path, quality=90)
            total_crops += 1

            if total_crops >= TARGET_CROPS:
                break

        if img_id % 50 == 0:
            print(f"  Image {img_id}: {total_crops} background crops so far")

    print(f"\nTotal background crops extracted: {total_crops}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
