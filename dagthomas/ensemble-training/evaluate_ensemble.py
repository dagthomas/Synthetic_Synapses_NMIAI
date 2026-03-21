"""
Evaluate individual models and the WBF ensemble on the training set.

Runs each model's weights through inference, then combines predictions
with Weighted Boxes Fusion and evaluates with pycocotools.

Usage:
  python evaluate_ensemble.py --coco-dir /path/to/NM_NGD_coco_dataset/train
  python evaluate_ensemble.py --coco-dir /data/train --weights weights/yolov8x_seed0.pt weights/yolov8l_seed0.pt
  python evaluate_ensemble.py --coco-dir /data/train --sweep-iou  # sweep WBF IoU threshold
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parent

# Inference config (match submission settings)
IMGSZ = 1536
CONF = 0.001
IOU = 0.6
MAX_DET = 3000

# WBF config
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001


def run_yolo_inference(model_path, image_dir, device="cuda"):
    """Run YOLO inference on all images, return per-image predictions."""
    print(f"  Loading {model_path.name}...")
    model = YOLO(str(model_path))

    image_files = sorted(Path(image_dir).glob("*.jpg"))
    print(f"  Running inference on {len(image_files)} images...")

    all_preds = {}  # image_id -> (boxes_xyxy, scores, labels)
    t0 = time.time()

    for img_path in image_files:
        stem = img_path.stem
        try:
            image_id = int(stem.split("_")[-1])
        except ValueError:
            image_id = hash(stem) % 100000

        results = model.predict(
            str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            augment=False,
            verbose=False,
            device=device,
        )

        if results and len(results[0].boxes) > 0:
            r = results[0]
            all_preds[image_id] = (
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                (r.orig_img.shape[1], r.orig_img.shape[0]),  # (w, h)
            )
        else:
            all_preds[image_id] = (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), (1, 1))

    elapsed = time.time() - t0
    n_dets = sum(len(p[1]) for p in all_preds.values())
    print(f"  Done: {n_dets} detections in {elapsed:.1f}s")
    return all_preds


def ensemble_wbf(all_model_preds, weights=None, iou_thr=WBF_IOU_THR):
    """Combine predictions from multiple models using Weighted Boxes Fusion.

    Args:
        all_model_preds: list of per-model prediction dicts {image_id -> (boxes, scores, labels, (w,h))}
        weights: per-model weights (default: equal)
        iou_thr: WBF IoU threshold

    Returns:
        dict {image_id -> (boxes_xyxy, scores, labels)}
    """
    if weights is None:
        weights = [1.0] * len(all_model_preds)

    # Collect all image IDs across models
    all_image_ids = set()
    for model_preds in all_model_preds:
        all_image_ids.update(model_preds.keys())

    merged = {}
    for image_id in sorted(all_image_ids):
        boxes_list = []
        scores_list = []
        labels_list = []

        # Get image dimensions from first model that has this image
        img_w, img_h = 1, 1
        for model_preds in all_model_preds:
            if image_id in model_preds:
                _, _, _, (w, h) = model_preds[image_id]
                if w > 1:
                    img_w, img_h = w, h
                    break

        for model_preds in all_model_preds:
            if image_id in model_preds:
                boxes, scores, labels, _ = model_preds[image_id]
            else:
                boxes = np.empty((0, 4))
                scores = np.empty(0)
                labels = np.empty(0, dtype=int)

            if len(boxes) > 0:
                # Normalize boxes to [0, 1] for WBF
                norm_boxes = boxes.copy()
                norm_boxes[:, [0, 2]] /= img_w
                norm_boxes[:, [1, 3]] /= img_h
                norm_boxes = np.clip(norm_boxes, 0, 1)
                boxes_list.append(norm_boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(np.empty((0, 4)))
                scores_list.append(np.empty(0))
                labels_list.append(np.empty(0, dtype=int))

        # Run WBF
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=WBF_SKIP_BOX_THR,
        )

        # Denormalize back to pixel coordinates
        if len(fused_boxes) > 0:
            fused_boxes[:, [0, 2]] *= img_w
            fused_boxes[:, [1, 3]] *= img_h

        merged[image_id] = (fused_boxes, fused_scores, fused_labels.astype(int))

    return merged


def preds_to_coco_format(preds):
    """Convert prediction dict to COCO evaluation format."""
    results = []
    for image_id, (boxes, scores, labels) in preds.items():
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            results.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(scores[i]), 4),
            })
    return results


def evaluate_coco(coco_gt, predictions, label=""):
    """Run COCO evaluation and return mAP@0.5."""
    if not predictions:
        print(f"  {label}: No predictions!")
        return 0.0

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # mAP@0.5 is index 1 in COCO eval
    map50 = coco_eval.stats[1]
    print(f"  {label} mAP@0.5 = {map50:.4f}")
    return map50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", required=True,
                        help="Path to NM_NGD_coco_dataset/train")
    parser.add_argument("--weights", nargs="*", default=None,
                        help="Specific weight files to ensemble (default: all in weights/)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep-iou", action="store_true",
                        help="Sweep WBF IoU threshold from 0.3 to 0.7")
    parser.add_argument("--save", default=None,
                        help="Save ensemble predictions to this JSON file")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    image_dir = coco_dir / "images"
    anno_path = coco_dir / "annotations.json"

    print("Loading COCO ground truth...")
    coco_gt = COCO(str(anno_path))

    # Find weight files
    if args.weights:
        weight_files = [Path(w) for w in args.weights]
    else:
        weights_dir = ROOT / "weights"
        weight_files = sorted(weights_dir.glob("*.pt")) if weights_dir.exists() else []

    if not weight_files:
        print("No weight files found! Train models first or specify --weights")
        return

    print(f"\nEvaluating {len(weight_files)} models:")
    for w in weight_files:
        print(f"  - {w.name} ({w.stat().st_size / 1024 / 1024:.1f} MB)")

    # Run inference for each model
    all_model_preds = []
    individual_scores = {}

    for wf in weight_files:
        print(f"\n{'='*60}")
        print(f"Model: {wf.name}")
        print(f"{'='*60}")

        preds = run_yolo_inference(wf, image_dir, device=args.device)

        # Convert to simple format (drop image size tuple)
        simple_preds = {}
        for img_id, (boxes, scores, labels, _) in preds.items():
            simple_preds[img_id] = (boxes, scores, labels)

        # Evaluate individual model
        coco_results = preds_to_coco_format(simple_preds)
        score = evaluate_coco(coco_gt, coco_results, label=wf.stem)
        individual_scores[wf.stem] = score

        all_model_preds.append(preds)

    # Ensemble evaluation
    if len(all_model_preds) >= 2:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE (WBF, {len(all_model_preds)} models)")
        print(f"{'='*60}")

        if args.sweep_iou:
            print("\nSweeping WBF IoU threshold:")
            best_iou, best_score = 0.5, 0.0
            for iou_thr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                merged = ensemble_wbf(all_model_preds, iou_thr=iou_thr)
                coco_results = preds_to_coco_format(merged)
                score = evaluate_coco(coco_gt, coco_results, label=f"IoU={iou_thr}")
                if score > best_score:
                    best_score = score
                    best_iou = iou_thr
            print(f"\nBest IoU threshold: {best_iou} (mAP@0.5 = {best_score:.4f})")
        else:
            merged = ensemble_wbf(all_model_preds, iou_thr=WBF_IOU_THR)
            coco_results = preds_to_coco_format(merged)
            ensemble_score = evaluate_coco(coco_gt, coco_results, label="Ensemble")

            if args.save:
                with open(args.save, "w") as f:
                    json.dump(coco_results, f)
                print(f"\nSaved ensemble predictions to {args.save}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, score in sorted(individual_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}  mAP@0.5 = {score:.4f}")
    if len(all_model_preds) >= 2 and not args.sweep_iou:
        print(f"  {'ENSEMBLE (WBF)':30s}  mAP@0.5 = {ensemble_score:.4f}")

    # Size check
    weights_dir = ROOT / "weights"
    if weights_dir.exists():
        total_mb = sum(f.stat().st_size for f in weights_dir.glob("*.pt")) / 1024 / 1024
        print(f"\n  Total weight size: {total_mb:.1f} MB (limit: 420 MB)")
        if total_mb > 420:
            print("  WARNING: Over 420 MB limit! Need to drop models or use ONNX export.")


if __name__ == "__main__":
    main()
