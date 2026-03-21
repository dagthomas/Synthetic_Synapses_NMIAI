"""
Parameter sweep for NorgesGruppen inference pipeline.

Loads models once, then runs inference with different parameter combinations.
Evaluates each against ground truth with detection/classification mAP split.

Usage:
  python sweep.py
  python sweep.py --quick   # Fewer combinations for faster iteration
"""

import argparse
import json
import time
import copy
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

# Monkeypatch torch.load
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from classifier import load_classifier
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from torchvision.ops import roi_align, nms as tv_nms
    HAS_ROI_ALIGN = True
except ImportError:
    HAS_ROI_ALIGN = False
    from torchvision.ops import nms as tv_nms

# Paths
ROOT = Path(__file__).resolve().parent
ANNO_PATH = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json")
IMG_DIR = ROOT / "datasets" / "train" / "images"
YOLO_WEIGHTS = ROOT / "best.pt"
CLS_WEIGHTS = ROOT / "classifier.safetensors"

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def compute_map(coco_gt, preds, category_agnostic=False):
    """Compute mAP@0.5."""
    if not preds:
        return 0.0
    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]
    if not preds:
        return 0.0

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

        seen = {}
        for p in preds:
            pm = dict(p)
            pm["category_id"] = 0
            key = (pm["image_id"], tuple(pm["bbox"]))
            if key not in seen or pm["score"] > seen[key]["score"]:
                seen[key] = pm
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
    s = coco_eval.eval["precision"]
    return float(np.mean(s[:, :, :, 0, 2][s[:, :, :, 0, 2] > -1]))


def run_yolo_cached(yolo, image_files, device, imgsz, conf, iou, max_det, augment):
    """Run YOLO on all images and cache raw detections."""
    cache_key = f"imgsz{imgsz}_conf{conf}_iou{iou}_aug{augment}"
    detections = {}  # image_path -> (boxes_xyxy, scores, labels, orig_img)

    for img_path in image_files:
        results = yolo.predict(
            str(img_path), imgsz=imgsz, conf=conf, iou=iou,
            max_det=max_det, augment=augment, verbose=False, device=device,
        )
        r = results[0] if results else None
        if r is not None and len(r.boxes) > 0:
            detections[str(img_path)] = (
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.orig_img,
            )
        elif r is not None:
            detections[str(img_path)] = (
                np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int),
                r.orig_img,
            )

    return detections


def classify_detections(classifier, detections, device, mean_gpu, std_gpu,
                        input_size, top_k=2, bg_reject_prob=0.5,
                        blend_mode="default", min_score=0.001,
                        classifier_batch=64, crop_chunk=500):
    """Run classifier on all cached detections with given parameters.

    blend_mode options:
      - "default": agree = conf * (0.5 + 0.5*prob), disagree = conf * prob
      - "geometric": agree = sqrt(conf * prob), disagree = sqrt(conf * prob) * 0.8
      - "classifier_trust": agree = conf * (0.3 + 0.7*prob), disagree = conf * prob
      - "yolo_trust": agree = conf * (0.7 + 0.3*prob), disagree = conf * prob * 0.5
      - "max_conf": always use max(yolo_conf with yolo_cls, blended with cls_cls)
    """
    BG_CLASS_ID = 356
    has_bg_class = classifier.num_classes > BG_CLASS_ID
    predictions = []

    for img_path, (boxes, scores, labels, orig_img) in detections.items():
        if len(boxes) == 0:
            continue

        stem = Path(img_path).stem
        try:
            image_id = int(stem.split("_")[-1])
        except ValueError:
            image_id = 0

        img_h, img_w = orig_img.shape[:2]

        # Filter tiny boxes
        valid = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w = min(img_w, x2) - max(0, x1)
            h = min(img_h, y2) - max(0, y1)
            if w >= 5 and h >= 5:
                valid.append(i)

        if not valid:
            # Keep YOLO-only for tiny boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(scores[i]), 4),
                })
            continue

        # Upload image to GPU
        img_rgb = np.ascontiguousarray(orig_img[:, :, ::-1])
        img_tensor = torch.from_numpy(img_rgb).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        valid_boxes = boxes[valid]
        boxes_gpu = torch.from_numpy(valid_boxes.copy()).float().to(device)
        boxes_gpu[:, 0].clamp_(0, img_w)
        boxes_gpu[:, 1].clamp_(0, img_h)
        boxes_gpu[:, 2].clamp_(0, img_w)
        boxes_gpu[:, 3].clamp_(0, img_h)

        classified_boxes = set()

        for chunk_start in range(0, len(valid), crop_chunk):
            chunk_end = min(chunk_start + crop_chunk, len(valid))
            chunk_boxes = boxes_gpu[chunk_start:chunk_end]

            if HAS_ROI_ALIGN:
                rois = torch.cat([
                    torch.zeros(len(chunk_boxes), 1, device=device),
                    chunk_boxes
                ], dim=1)
                crops = roi_align(
                    img_tensor, rois, output_size=(input_size, input_size),
                    spatial_scale=1.0, sampling_ratio=2, aligned=True,
                )
            else:
                crops_list = []
                for box in chunk_boxes:
                    x1, y1, x2, y2 = box.int().tolist()
                    y1, y2 = max(0, y1), min(img_h, y2)
                    x1, x2 = max(0, x1), min(img_w, x2)
                    crop = img_tensor[:, :, y1:y2, x1:x2]
                    crop = F.interpolate(crop, size=(input_size, input_size),
                                         mode='bilinear', align_corners=False)
                    crops_list.append(crop)
                crops = torch.cat(crops_list, dim=0)

            crops = (crops - mean_gpu) / std_gpu
            crops = crops.half()

            padded = torch.zeros(classifier_batch, 3, input_size, input_size,
                                 device=device, dtype=torch.float16)

            for cls_start in range(0, crops.shape[0], classifier_batch):
                actual = min(classifier_batch, crops.shape[0] - cls_start)
                padded[:actual] = crops[cls_start:cls_start + actual]

                with torch.no_grad():
                    logits = classifier(padded)
                    probs = F.softmax(logits[:actual], dim=1)
                    topk_probs, topk_classes = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

                topk_probs_np = topk_probs.cpu().numpy()
                topk_classes_np = topk_classes.cpu().numpy()

                if has_bg_class:
                    bg_probs = probs[:, BG_CLASS_ID].cpu().numpy()

                for j in range(actual):
                    box_idx = valid[chunk_start + cls_start + j]
                    yolo_conf = float(scores[box_idx])
                    yolo_cls = int(labels[box_idx])

                    # Background rejection
                    if has_bg_class and float(bg_probs[j]) > bg_reject_prob:
                        classified_boxes.add(box_idx)
                        continue

                    classified_boxes.add(box_idx)

                    for k in range(min(top_k, topk_probs_np.shape[1])):
                        cls_id = int(topk_classes_np[j, k])
                        cls_prob = float(topk_probs_np[j, k])

                        if cls_id == BG_CLASS_ID:
                            continue

                        # Compute blended score based on mode
                        if blend_mode == "default":
                            if cls_id == yolo_cls:
                                blended = yolo_conf * (0.5 + 0.5 * cls_prob)
                            else:
                                blended = yolo_conf * cls_prob
                        elif blend_mode == "geometric":
                            blended = (yolo_conf * cls_prob) ** 0.5
                            if cls_id != yolo_cls:
                                blended *= 0.8
                        elif blend_mode == "classifier_trust":
                            if cls_id == yolo_cls:
                                blended = yolo_conf * (0.3 + 0.7 * cls_prob)
                            else:
                                blended = yolo_conf * cls_prob
                        elif blend_mode == "yolo_trust":
                            if cls_id == yolo_cls:
                                blended = yolo_conf * (0.7 + 0.3 * cls_prob)
                            else:
                                blended = yolo_conf * cls_prob * 0.5
                        elif blend_mode == "pure_cls":
                            # Pure classifier probability, weighted by detection confidence
                            blended = yolo_conf * cls_prob
                        elif blend_mode == "boosted":
                            if cls_id == yolo_cls:
                                blended = yolo_conf * (0.4 + 0.6 * cls_prob)
                            else:
                                blended = yolo_conf * cls_prob * 0.8
                        else:
                            blended = yolo_conf * cls_prob

                        if blended >= min_score:
                            x1, y1, x2, y2 = boxes[box_idx]
                            predictions.append({
                                "image_id": image_id,
                                "category_id": cls_id,
                                "bbox": [round(float(x1), 2), round(float(y1), 2),
                                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                "score": round(blended, 4),
                            })

        # Keep YOLO-only for unclassified boxes
        for i in range(len(boxes)):
            if i not in classified_boxes:
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(scores[i]), 4),
                })

        del img_tensor

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer combos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load ground truth
    print("Loading ground truth...")
    coco_gt = COCO(str(ANNO_PATH))

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    print("Loading classifier...")
    classifier = load_classifier(str(CLS_WEIGHTS), device=device)
    input_size = getattr(classifier, 'input_size', 256)
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    # Get images (include both train and val, both .jpg and .jpeg)
    image_files = []
    for split_dir in [IMG_DIR, IMG_DIR.parent.parent / "val" / "images"]:
        if split_dir.exists():
            image_files.extend(split_dir.glob("*.jpg"))
            image_files.extend(split_dir.glob("*.jpeg"))
            image_files.extend(split_dir.glob("*.png"))
    image_files = sorted(set(image_files))
    print(f"Found {len(image_files)} images")

    # Phase 1: Cache YOLO detections for key configurations
    yolo_configs = [
        {"imgsz": 1536, "conf": 0.001, "iou": 0.6, "augment": True, "max_det": 3000},
    ]
    if not args.quick:
        yolo_configs += [
            # IoU sweep
            {"imgsz": 1536, "conf": 0.001, "iou": 0.5, "augment": True, "max_det": 3000},
            {"imgsz": 1536, "conf": 0.001, "iou": 0.7, "augment": True, "max_det": 3000},
            {"imgsz": 1536, "conf": 0.001, "iou": 0.45, "augment": True, "max_det": 3000},
            # Confidence sweep
            {"imgsz": 1536, "conf": 0.005, "iou": 0.6, "augment": True, "max_det": 3000},
            {"imgsz": 1536, "conf": 0.0005, "iou": 0.6, "augment": True, "max_det": 3000},
            {"imgsz": 1536, "conf": 0.01, "iou": 0.6, "augment": True, "max_det": 3000},
            # Resolution sweep
            {"imgsz": 1280, "conf": 0.001, "iou": 0.6, "augment": True, "max_det": 3000},
            {"imgsz": 1920, "conf": 0.001, "iou": 0.6, "augment": True, "max_det": 3000},
            # No TTA baseline (faster, more predictions available for test)
            {"imgsz": 1536, "conf": 0.001, "iou": 0.6, "augment": False, "max_det": 3000},
            # Best IoU + higher resolution combo
            {"imgsz": 1920, "conf": 0.001, "iou": 0.5, "augment": True, "max_det": 3000},
        ]

    yolo_caches = {}
    for cfg in yolo_configs:
        key = f"sz{cfg['imgsz']}_c{cfg['conf']}_i{cfg['iou']}_aug{cfg['augment']}"
        print(f"\nRunning YOLO: {key}...")
        t0 = time.time()
        yolo_caches[key] = run_yolo_cached(
            yolo, image_files, device,
            cfg["imgsz"], cfg["conf"], cfg["iou"], cfg["max_det"], cfg["augment"],
        )
        print(f"  {time.time()-t0:.1f}s, {sum(len(d[0]) for d in yolo_caches[key].values())} detections")

    # Phase 2: Sweep classifier parameters on each YOLO cache
    if args.quick:
        cls_configs = [
            {"top_k": 2, "bg_reject_prob": 0.5, "blend_mode": "default"},
            {"top_k": 2, "bg_reject_prob": 1.0, "blend_mode": "default"},  # no bg reject
            {"top_k": 1, "bg_reject_prob": 0.5, "blend_mode": "default"},
            {"top_k": 2, "bg_reject_prob": 0.5, "blend_mode": "classifier_trust"},
            {"top_k": 2, "bg_reject_prob": 0.5, "blend_mode": "yolo_trust"},
            {"top_k": 2, "bg_reject_prob": 0.3, "blend_mode": "default"},
            {"top_k": 2, "bg_reject_prob": 0.7, "blend_mode": "default"},
        ]
    else:
        # Use only top 3 classifier configs from quick sweep to save time
        cls_configs = [
            {"top_k": 2, "bg_reject_prob": 0.5, "blend_mode": "yolo_trust"},
            {"top_k": 2, "bg_reject_prob": 0.3, "blend_mode": "default"},
            {"top_k": 2, "bg_reject_prob": 0.5, "blend_mode": "default"},
        ]

    results = []
    total_evals = len(yolo_caches) * len(cls_configs)
    eval_idx = 0

    for yolo_key, detections in yolo_caches.items():
        for cls_cfg in cls_configs:
            eval_idx += 1
            cfg_str = f"{yolo_key} | k={cls_cfg['top_k']} bg={cls_cfg['bg_reject_prob']} {cls_cfg['blend_mode']}"
            print(f"\n[{eval_idx}/{total_evals}] {cfg_str}")

            t0 = time.time()
            preds = classify_detections(
                classifier, detections, device, mean_gpu, std_gpu, input_size,
                top_k=cls_cfg["top_k"],
                bg_reject_prob=cls_cfg["bg_reject_prob"],
                blend_mode=cls_cfg["blend_mode"],
            )
            cls_time = time.time() - t0

            t0 = time.time()
            det_map = compute_map(coco_gt, preds, category_agnostic=True)
            cls_map = compute_map(coco_gt, preds, category_agnostic=False)
            combined = 0.7 * det_map + 0.3 * cls_map
            eval_time = time.time() - t0

            results.append({
                "yolo": yolo_key,
                "top_k": cls_cfg["top_k"],
                "bg_reject": cls_cfg["bg_reject_prob"],
                "blend": cls_cfg["blend_mode"],
                "n_preds": len(preds),
                "det_mAP": det_map,
                "cls_mAP": cls_map,
                "combined": combined,
            })

            print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={combined:.4f} "
                  f"({len(preds)} preds, cls={cls_time:.1f}s, eval={eval_time:.1f}s)")

    # Print sorted results
    print("\n" + "=" * 120)
    print(f"{'YOLO Config':<35} {'K':>2} {'BG':>4} {'Blend':<16} {'Preds':>7} {'Det':>7} {'Cls':>7} {'Comb':>7}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: -x["combined"])[:30]:
        print(f"{r['yolo']:<35} {r['top_k']:>2} {r['bg_reject']:>4.1f} {r['blend']:<16} "
              f"{r['n_preds']:>7} {r['det_mAP']:>7.4f} {r['cls_mAP']:>7.4f} {r['combined']:>7.4f}")
    print("=" * 120)

    # Save full results
    out_path = ROOT / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
