"""
Quick evaluation on the VM — test trained weights against training set.
Runs inference on val images and computes detection + classification mAP.

Usage:
  python3 cloud_test.py                          # test all available weights
  python3 cloud_test.py --model runs/rtdetr_x/weights/best.pt  # test specific
  python3 cloud_test.py --list                   # list all trained weights
"""

import argparse
import json
import copy
import time
import numpy as np
from pathlib import Path

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HOME = Path.home()
ANNO_PATH = HOME / "annotations.json"
VAL_IMAGES = HOME / "datasets" / "val" / "images"
TRAIN_IMAGES = HOME / "datasets" / "train" / "images"


def find_all_weights():
    """Find all trained weight files."""
    weights = []
    runs_dir = HOME / "runs"
    if runs_dir.exists():
        for best in runs_dir.glob("*/weights/best.pt"):
            weights.append(best)
    return sorted(weights)


def compute_map(coco_gt, preds, category_agnostic=False):
    """Compute mAP@0.5."""
    if not preds:
        return 0.0

    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]

    if category_agnostic:
        gt_data = {
            "images": list(coco_gt.imgs.values()),
            "annotations": [],
            "categories": [{"id": 0, "name": "product"}],
        }
        for ann_id in coco_gt.getAnnIds():
            ann = copy.deepcopy(coco_gt.anns[ann_id])
            ann["category_id"] = 0
            gt_data["annotations"].append(ann)
        coco_gt_mod = COCO()
        coco_gt_mod.dataset = gt_data
        coco_gt_mod.createIndex()
        preds_mod = [dict(p, category_id=0) for p in preds]
        coco_dt = coco_gt_mod.loadRes(preds_mod)
        coco_eval = COCOeval(coco_gt_mod, coco_dt, "bbox")
    else:
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Use default iouThrs to avoid numpy 0d array bug, then read AP@0.5 (stats[1])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1]  # AP@0.5


def test_model(weight_path, image_dir, imgsz=1280, conf=0.001):
    """Run inference and compute mAP."""
    print(f"\n{'='*60}")
    print(f"Testing: {weight_path}")
    print(f"Images:  {image_dir}")
    print(f"{'='*60}")

    model = YOLO(str(weight_path))
    image_files = sorted(list(image_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} images")

    predictions = []
    start = time.time()

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        results = model.predict(
            str(img_path), imgsz=imgsz, conf=conf, iou=0.6,
            max_det=3000, verbose=False, device="cuda",
        )
        r = results[0] if results else None
        if r is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(classes[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[i]),
                })

    elapsed = time.time() - start
    print(f"Inference: {elapsed:.1f}s, {len(predictions)} detections")

    if not ANNO_PATH.exists():
        print("WARNING: annotations.json not found, skipping mAP computation")
        return

    coco_gt = COCO(str(ANNO_PATH))
    det_map = compute_map(coco_gt, predictions, category_agnostic=True)
    cls_map = compute_map(coco_gt, predictions, category_agnostic=False)
    combined = 0.7 * det_map + 0.3 * cls_map

    print(f"\nResults for {weight_path.parent.parent.name}:")
    print(f"  Detection mAP@0.5:       {det_map:.4f}")
    print(f"  Classification mAP@0.5:  {cls_map:.4f}")
    print(f"  Combined (70/30):        {combined:.4f}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to specific weight file")
    parser.add_argument("--list", action="store_true", help="List all trained weights")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--use-train", action="store_true",
                        help="Use train images instead of val (for all-data trained models)")
    args = parser.parse_args()

    if args.list:
        weights = find_all_weights()
        if weights:
            print("Available trained weights:")
            for w in weights:
                size_mb = w.stat().st_size / 1024 / 1024
                print(f"  {w.relative_to(HOME)} ({size_mb:.0f} MB)")
        else:
            print("No trained weights found yet")
        return

    image_dir = TRAIN_IMAGES if args.use_train else VAL_IMAGES
    if not image_dir.exists():
        print(f"Image dir not found: {image_dir}, trying train/")
        image_dir = TRAIN_IMAGES

    if args.model:
        test_model(Path(args.model), image_dir, args.imgsz)
    else:
        weights = find_all_weights()
        if not weights:
            print("No trained weights found. Wait for training to complete.")
            return
        results = {}
        for w in weights:
            score = test_model(w, image_dir, args.imgsz)
            if score is not None:
                results[str(w.relative_to(HOME))] = score

        if results:
            print(f"\n{'='*60}")
            print("SUMMARY — Combined Score (70% det + 30% cls)")
            print(f"{'='*60}")
            for name, score in sorted(results.items(), key=lambda x: -x[1]):
                print(f"  {score:.4f}  {name}")


if __name__ == "__main__":
    main()
