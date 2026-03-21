"""
Evaluate predictions against COCO ground truth.
Computes detection mAP@0.5 (category-agnostic) and classification mAP@0.5 separately
to match the competition scoring: 70% detection + 30% classification.

Usage:
  python evaluate.py predictions.json
  python evaluate.py predictions.json --detailed
"""

import json
import argparse
import copy
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNO_PATH = "X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json"


def compute_map(coco_gt, preds, category_agnostic=False):
    """Compute mAP@0.5 from predictions.

    If category_agnostic=True, remap all categories to 0 for detection-only mAP.
    """
    if not preds:
        return 0.0

    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]

    if category_agnostic:
        # Create a modified GT and predictions where all categories are 0
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

        preds_mod = []
        for p in preds:
            pm = dict(p)
            pm["category_id"] = 0
            preds_mod.append(pm)

        # Deduplicate: keep highest score per (image_id, bbox)
        seen = {}
        for p in preds_mod:
            key = (p["image_id"], tuple(p["bbox"]))
            if key not in seen or p["score"] > seen[key]["score"]:
                seen[key] = p
        preds_mod = list(seen.values())

        coco_dt = coco_gt_mod.loadRes(preds_mod)
        coco_eval = COCOeval(coco_gt_mod, coco_dt, "bbox")
    else:
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.maxDets = [100, 500, 3000]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # mAP at maxDets=3000
    s = coco_eval.eval["precision"]  # [T, R, K, A, M]
    mAP = np.mean(s[:, :, :, 0, 2][s[:, :, :, 0, 2] > -1])
    return float(mAP)


def evaluate_file(pred_path, coco_gt, detailed=False):
    """Evaluate a prediction file and return scores."""
    with open(pred_path) as f:
        preds = json.load(f)

    cls_map = compute_map(coco_gt, preds, category_agnostic=False)
    det_map = compute_map(coco_gt, preds, category_agnostic=True)
    combined = 0.7 * det_map + 0.3 * cls_map

    return {
        "file": Path(pred_path).name,
        "n_preds": len(preds),
        "det_mAP": det_map,
        "cls_mAP": cls_map,
        "combined": combined,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", help="Prediction JSON file(s)")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    print("Loading ground truth...")
    coco_gt = COCO(ANNO_PATH)

    results = []
    for pred_path in args.predictions:
        print(f"\nEvaluating {pred_path}...")
        r = evaluate_file(pred_path, coco_gt, args.detailed)
        results.append(r)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'File':<40} {'Preds':>8} {'Det mAP':>8} {'Cls mAP':>8} {'Combined':>9}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x["combined"]):
        print(f"{r['file']:<40} {r['n_preds']:>8} {r['det_mAP']:>8.4f} {r['cls_mAP']:>8.4f} {r['combined']:>9.4f}")
    print("=" * 80)
    print("Combined = 0.7 * det_mAP + 0.3 * cls_mAP")


if __name__ == "__main__":
    main()
