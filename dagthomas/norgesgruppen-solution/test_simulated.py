"""
Simulated cross-validation: compare pipeline performance on val-split (24 images)
vs train-split (224 images) to detect overfitting, and perform deep error analysis
to find where we lose the most mAP points.

Analyses:
  1. Train vs Val mAP comparison (detect generalization gap)
  2. Per-image statistics (detection counts, scores, density)
  3. Category distribution shift (classifier disagreements with YOLO)
  4. Per-category AP for worst 20 categories
  5. Error analysis: missed detections, wrong classifications, false positives
"""

import json
import time
import copy
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

# Monkeypatch torch.load for PyTorch 2.6+ compatibility with ultralytics 8.1.0
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
YOLO_WEIGHTS = ROOT / "best.pt"
CLS_WEIGHTS = ROOT / "classifier.safetensors"

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Pipeline params (matching run.py)
TOP_K = 2
BG_REJECT_PROB = 0.5
MIN_SCORE = 0.001
CLASSIFIER_BATCH = 64
CROP_CHUNK = 500
BG_CLASS_ID = 356


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


def compute_per_category_ap(coco_gt, preds):
    """Compute per-category AP@0.5 using COCOeval.

    Returns dict {category_id: AP} for all categories with GT annotations.
    """
    if not preds:
        return {}

    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]
    if not preds:
        return {}

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.maxDets = [100, 500, 3000]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # precision shape: [T, R, K, A, M] where K = num categories
    precision = coco_eval.eval["precision"]  # [1, R, K, A, M]
    cat_ids = coco_eval.params.catIds

    per_cat_ap = {}
    for k_idx, cat_id in enumerate(cat_ids):
        # precision[:, :, k_idx, 0, 2] = IoU=0.5, area=all, maxDets=3000
        p = precision[:, :, k_idx, 0, 2]
        valid = p[p > -1]
        if len(valid) > 0:
            per_cat_ap[cat_id] = float(np.mean(valid))
        else:
            per_cat_ap[cat_id] = -1.0  # no GT for this category

    return per_cat_ap


def run_yolo_inference(yolo, image_files, device):
    """Run YOLO on images and cache raw detections."""
    detections = {}

    for idx, img_path in enumerate(image_files):
        results = yolo.predict(
            str(img_path), imgsz=1536, conf=0.001, iou=0.6,
            max_det=3000, augment=True, verbose=False, device=device,
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

        if (idx + 1) % 25 == 0 or idx == len(image_files) - 1:
            print(f"  YOLO inference: {idx+1}/{len(image_files)}")

    return detections


def classify_detections(classifier, detections, device, mean_gpu, std_gpu,
                        input_size):
    """Run classifier on cached detections with yolo_trust blending.

    Returns (predictions_list, disagreement_info) where disagreement_info
    tracks YOLO vs classifier class disagreements.
    """
    has_bg_class = classifier.num_classes > BG_CLASS_ID
    predictions = []
    # Track disagreements: (image_id, box_idx, yolo_cls, classifier_cls, yolo_conf, cls_prob)
    disagreements = []
    bg_rejections = 0

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

        for chunk_start in range(0, len(valid), CROP_CHUNK):
            chunk_end = min(chunk_start + CROP_CHUNK, len(valid))
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

            padded = torch.zeros(CLASSIFIER_BATCH, 3, input_size, input_size,
                                 device=device, dtype=torch.float16)

            for cls_start in range(0, crops.shape[0], CLASSIFIER_BATCH):
                actual = min(CLASSIFIER_BATCH, crops.shape[0] - cls_start)
                padded[:actual] = crops[cls_start:cls_start + actual]

                with torch.no_grad():
                    logits = classifier(padded)
                    probs = F.softmax(logits[:actual], dim=1)
                    topk_probs, topk_classes = torch.topk(probs, k=min(TOP_K, probs.shape[1]), dim=1)

                topk_probs_np = topk_probs.cpu().numpy()
                topk_classes_np = topk_classes.cpu().numpy()

                if has_bg_class:
                    bg_probs = probs[:, BG_CLASS_ID].cpu().numpy()

                for j in range(actual):
                    box_idx = valid[chunk_start + cls_start + j]
                    yolo_conf = float(scores[box_idx])
                    yolo_cls = int(labels[box_idx])
                    top1_cls = int(topk_classes_np[j, 0])
                    top1_prob = float(topk_probs_np[j, 0])

                    # Track YOLO vs classifier disagreements
                    if top1_cls != yolo_cls and top1_cls != BG_CLASS_ID:
                        disagreements.append({
                            "image_id": image_id,
                            "yolo_cls": yolo_cls,
                            "classifier_cls": top1_cls,
                            "yolo_conf": yolo_conf,
                            "cls_prob": top1_prob,
                        })

                    # Background rejection
                    if has_bg_class and float(bg_probs[j]) > BG_REJECT_PROB:
                        classified_boxes.add(box_idx)
                        bg_rejections += 1
                        continue

                    classified_boxes.add(box_idx)

                    for k in range(min(TOP_K, topk_probs_np.shape[1])):
                        cls_id = int(topk_classes_np[j, k])
                        cls_prob = float(topk_probs_np[j, k])

                        if cls_id == BG_CLASS_ID:
                            continue

                        # yolo_trust blend mode
                        if cls_id == yolo_cls:
                            blended = yolo_conf * (0.7 + 0.3 * cls_prob)
                        else:
                            blended = yolo_conf * cls_prob * 0.5

                        if blended >= MIN_SCORE:
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

    return predictions, disagreements, bg_rejections


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x,y,w,h] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def error_analysis(coco_gt, preds, target_cat_ids, image_ids=None):
    """For each target category, count:
      (a) missed detections (GT not matched by any prediction with IoU >= 0.5)
      (b) wrong classifications (GT matched by prediction but wrong category)
      (c) false positives (predictions with no matching GT)

    Args:
        coco_gt: COCO ground truth
        preds: list of prediction dicts
        target_cat_ids: set of category IDs to analyze
        image_ids: optional set of image IDs to restrict analysis to

    Returns:
        dict {cat_id: {"missed": int, "wrong_cls": int, "false_pos": int,
                       "gt_count": int, "pred_count": int,
                       "confused_with": Counter}}
    """
    from collections import Counter

    # Index predictions by image_id
    pred_by_img = defaultdict(list)
    for p in preds:
        if image_ids and p["image_id"] not in image_ids:
            continue
        pred_by_img[p["image_id"]].append(p)

    results = {}
    for cat_id in target_cat_ids:
        missed = 0
        wrong_cls = 0
        false_pos = 0
        gt_count = 0
        pred_count = 0
        confused_with = Counter()

        # Get all GT annotations for this category
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        gt_anns = coco_gt.loadAnns(ann_ids)

        if image_ids:
            gt_anns = [a for a in gt_anns if a["image_id"] in image_ids]

        gt_count = len(gt_anns)

        # Get all predictions for this category
        cat_preds = [p for p in preds if p["category_id"] == cat_id]
        if image_ids:
            cat_preds = [p for p in cat_preds if p["image_id"] in image_ids]
        pred_count = len(cat_preds)

        # For each GT annotation, check if it's matched
        for ann in gt_anns:
            img_id = ann["image_id"]
            gt_box = ann["bbox"]  # [x, y, w, h]
            img_preds = pred_by_img.get(img_id, [])

            # Find best matching prediction (any category, IoU >= 0.5)
            best_iou = 0
            best_pred = None
            for p in img_preds:
                iou = compute_iou(gt_box, p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = p

            if best_iou < 0.5:
                missed += 1
            elif best_pred["category_id"] != cat_id:
                wrong_cls += 1
                confused_with[best_pred["category_id"]] += 1

        # Count false positives: predictions for this category that don't match any GT
        # (of this category) with IoU >= 0.5
        for p in cat_preds:
            img_id = p["image_id"]
            p_box = p["bbox"]

            # Get GT annotations for this category in this image
            img_gt = [a for a in gt_anns if a["image_id"] == img_id]

            matched = False
            for a in img_gt:
                if compute_iou(p_box, a["bbox"]) >= 0.5:
                    matched = True
                    break

            if not matched:
                false_pos += 1

        # Get top confused categories
        top_confused = confused_with.most_common(3)

        results[cat_id] = {
            "missed": missed,
            "wrong_cls": wrong_cls,
            "false_pos": false_pos,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "top_confused": top_confused,
        }

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_start = time.time()

    # Load ground truth
    print("Loading ground truth...")
    coco_gt = COCO(str(ANNO_PATH))

    # Build category name lookup
    cat_names = {}
    for cat_id, cat_info in coco_gt.cats.items():
        cat_names[cat_id] = cat_info["name"]

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    print("Loading classifier...")
    classifier = load_classifier(str(CLS_WEIGHTS), device=device)
    input_size = getattr(classifier, 'input_size', 256)
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    print(f"Models loaded in {time.time() - t_start:.1f}s")

    # Get image files for each split
    datasets_dir = ROOT / "datasets"
    val_dir = datasets_dir / "val" / "images"
    train_dir = datasets_dir / "train" / "images"

    def get_images(img_dir):
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            files.extend(img_dir.glob(ext))
        return sorted(set(files))

    val_images = get_images(val_dir)
    train_images = get_images(train_dir)
    all_images = sorted(set(val_images + train_images))

    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")
    print(f"Total images: {len(all_images)}")

    # Extract image IDs for each split
    def get_image_ids(image_files):
        ids = set()
        for f in image_files:
            stem = f.stem
            try:
                ids.add(int(stem.split("_")[-1]))
            except ValueError:
                pass
        return ids

    val_ids = get_image_ids(val_images)
    train_ids = get_image_ids(train_images)

    print(f"Val image IDs: {sorted(val_ids)}")

    # ========================================================================
    # Phase 1: Run YOLO inference on ALL images (cache once)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: YOLO Inference (all images)")
    print("=" * 80)
    t0 = time.time()
    detections = run_yolo_inference(yolo, all_images, device)
    n_dets = sum(len(d[0]) for d in detections.values())
    print(f"YOLO done: {n_dets} raw detections in {time.time()-t0:.1f}s")

    # Split detections by train/val
    train_dets = {}
    val_dets = {}
    for img_path, det in detections.items():
        stem = Path(img_path).stem
        try:
            img_id = int(stem.split("_")[-1])
        except ValueError:
            continue
        if img_id in val_ids:
            val_dets[img_path] = det
        elif img_id in train_ids:
            train_dets[img_path] = det

    # ========================================================================
    # Phase 2: Classify detections (separate for train and val)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Classification + Blending")
    print("=" * 80)

    print("\n--- Val split (24 images) ---")
    t0 = time.time()
    val_preds, val_disagree, val_bg_rej = classify_detections(
        classifier, val_dets, device, mean_gpu, std_gpu, input_size
    )
    print(f"  {len(val_preds)} predictions, {len(val_disagree)} disagreements, "
          f"{val_bg_rej} BG rejections in {time.time()-t0:.1f}s")

    print("\n--- Train split (224 images) ---")
    t0 = time.time()
    train_preds, train_disagree, train_bg_rej = classify_detections(
        classifier, train_dets, device, mean_gpu, std_gpu, input_size
    )
    print(f"  {len(train_preds)} predictions, {len(train_disagree)} disagreements, "
          f"{train_bg_rej} BG rejections in {time.time()-t0:.1f}s")

    # Combine for all-images evaluation
    all_preds = val_preds + train_preds
    all_disagree = val_disagree + train_disagree

    # ========================================================================
    # Phase 3: Evaluate mAP for each split
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: mAP Evaluation")
    print("=" * 80)

    # We need to evaluate on subsets. For val, filter preds to val image_ids.
    # The coco_gt has all 248 images, but compute_map filters preds to valid IDs.
    # We need to also restrict GT evaluation to only those images.
    def make_subset_gt(coco_gt, image_ids):
        """Create a COCO GT object restricted to specific image IDs."""
        subset_data = {
            "images": [img for img in coco_gt.imgs.values() if img["id"] in image_ids],
            "annotations": [ann for ann in coco_gt.anns.values() if ann["image_id"] in image_ids],
            "categories": list(coco_gt.cats.values()),
        }
        subset_coco = COCO()
        subset_coco.dataset = subset_data
        subset_coco.createIndex()
        return subset_coco

    val_gt = make_subset_gt(coco_gt, val_ids)
    train_gt = make_subset_gt(coco_gt, train_ids)

    # Val split
    print("\nEvaluating VAL split...")
    val_det_map = compute_map(val_gt, val_preds, category_agnostic=True)
    val_cls_map = compute_map(val_gt, val_preds, category_agnostic=False)
    val_combined = 0.7 * val_det_map + 0.3 * val_cls_map

    # Train split
    print("\nEvaluating TRAIN split...")
    train_det_map = compute_map(train_gt, train_preds, category_agnostic=True)
    train_cls_map = compute_map(train_gt, train_preds, category_agnostic=False)
    train_combined = 0.7 * train_det_map + 0.3 * train_cls_map

    # All images
    print("\nEvaluating ALL images...")
    all_det_map = compute_map(coco_gt, all_preds, category_agnostic=True)
    all_cls_map = compute_map(coco_gt, all_preds, category_agnostic=False)
    all_combined = 0.7 * all_det_map + 0.3 * all_cls_map

    print("\n" + "=" * 80)
    print(f"{'Split':<15} {'Images':>7} {'Preds':>7} {'Det mAP':>9} {'Cls mAP':>9} {'Combined':>9}")
    print("-" * 80)
    print(f"{'Val (24)':<15} {len(val_images):>7} {len(val_preds):>7} {val_det_map:>9.4f} {val_cls_map:>9.4f} {val_combined:>9.4f}")
    print(f"{'Train (224)':<15} {len(train_images):>7} {len(train_preds):>7} {train_det_map:>9.4f} {train_cls_map:>9.4f} {train_combined:>9.4f}")
    print(f"{'All (248)':<15} {len(all_images):>7} {len(all_preds):>7} {all_det_map:>9.4f} {all_cls_map:>9.4f} {all_combined:>9.4f}")
    print("=" * 80)

    gap_det = train_det_map - val_det_map
    gap_cls = train_cls_map - val_cls_map
    gap_comb = train_combined - val_combined
    print(f"\nGeneralization gap (train - val):")
    print(f"  det_mAP:  {gap_det:+.4f}")
    print(f"  cls_mAP:  {gap_cls:+.4f}")
    print(f"  combined: {gap_comb:+.4f}")
    if abs(gap_comb) < 0.02:
        print("  -> Minimal gap: model generalizes well")
    elif gap_comb > 0.05:
        print("  -> SIGNIFICANT GAP: possible overfitting on train images")
    else:
        print("  -> Moderate gap: some overfitting present")

    # ========================================================================
    # Phase 4: Per-image statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: Per-Image Statistics")
    print("=" * 80)

    def compute_per_image_stats(preds, image_ids):
        """Compute per-image detection statistics."""
        # Deduplicate: unique boxes per image
        unique_boxes_per_img = defaultdict(set)
        scores_per_img = defaultdict(list)

        for p in preds:
            if p["image_id"] in image_ids:
                box_key = tuple(p["bbox"])
                unique_boxes_per_img[p["image_id"]].add(box_key)
                scores_per_img[p["image_id"]].append(p["score"])

        counts = [len(v) for v in unique_boxes_per_img.values()]
        all_scores = []
        for v in scores_per_img.values():
            all_scores.extend(v)

        if not counts:
            return {"avg_unique_dets": 0, "avg_score": 0, "avg_preds": 0}

        return {
            "avg_unique_dets": np.mean(counts),
            "std_unique_dets": np.std(counts),
            "min_unique_dets": np.min(counts),
            "max_unique_dets": np.max(counts),
            "avg_score": np.mean(all_scores) if all_scores else 0,
            "median_score": np.median(all_scores) if all_scores else 0,
            "avg_preds_per_img": len(all_scores) / len(image_ids),
        }

    val_stats = compute_per_image_stats(val_preds, val_ids)
    train_stats = compute_per_image_stats(train_preds, train_ids)

    print(f"\n{'Metric':<30} {'Val (24)':>12} {'Train (224)':>12}")
    print("-" * 55)
    for key in ["avg_unique_dets", "std_unique_dets", "min_unique_dets",
                "max_unique_dets", "avg_score", "median_score", "avg_preds_per_img"]:
        v = val_stats.get(key, 0)
        t = train_stats.get(key, 0)
        print(f"{key:<30} {v:>12.2f} {t:>12.2f}")

    # GT stats for comparison
    val_gt_anns = [a for a in coco_gt.anns.values() if a["image_id"] in val_ids]
    train_gt_anns = [a for a in coco_gt.anns.values() if a["image_id"] in train_ids]
    print(f"\n{'GT annotations':<30} {len(val_gt_anns):>12} {len(train_gt_anns):>12}")
    print(f"{'GT per image':<30} {len(val_gt_anns)/len(val_ids):>12.1f} {len(train_gt_anns)/len(train_ids):>12.1f}")

    # ========================================================================
    # Phase 5: Category Distribution Shift
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Category Distribution Shift (YOLO vs Classifier)")
    print("=" * 80)

    # Count YOLO -> Classifier class changes
    yolo_to_cls = defaultdict(lambda: defaultdict(int))
    for d in all_disagree:
        yolo_to_cls[d["yolo_cls"]][d["classifier_cls"]] += 1

    # Sort by total disagreements per YOLO class
    disagree_counts = {k: sum(v.values()) for k, v in yolo_to_cls.items()}
    top_disagree = sorted(disagree_counts.items(), key=lambda x: -x[1])[:25]

    print(f"\nTotal disagreements: {len(all_disagree)} "
          f"(val: {len(val_disagree)}, train: {len(train_disagree)})")
    print(f"\nTop 25 YOLO classes where classifier disagrees most:")
    print(f"{'YOLO Class':<8} {'Name':<35} {'Disagree':>9} {'Top Classifier Override':>40}")
    print("-" * 95)
    for yolo_cls, count in top_disagree:
        name = cat_names.get(yolo_cls, f"class_{yolo_cls}")[:35]
        top_override = sorted(yolo_to_cls[yolo_cls].items(), key=lambda x: -x[1])[:2]
        override_str = ", ".join(
            f"{cat_names.get(c, str(c))[:15]}({n})" for c, n in top_override
        )
        print(f"{yolo_cls:<8} {name:<35} {count:>9} {override_str:>40}")

    # Compare predicted category distribution vs GT
    print("\n--- Category distribution: predictions vs GT ---")
    gt_cat_counts = defaultdict(int)
    for ann in coco_gt.anns.values():
        gt_cat_counts[ann["category_id"]] += 1

    # Count unique (image_id, bbox) per category in predictions (deduplicated)
    # Use the highest-scoring prediction per (image_id, bbox) as the category
    best_pred_per_box = {}
    for p in all_preds:
        key = (p["image_id"], tuple(p["bbox"]))
        if key not in best_pred_per_box or p["score"] > best_pred_per_box[key]["score"]:
            best_pred_per_box[key] = p

    pred_cat_counts = defaultdict(int)
    for p in best_pred_per_box.values():
        pred_cat_counts[p["category_id"]] += 1

    # Find categories with biggest absolute difference
    all_cats = set(gt_cat_counts.keys()) | set(pred_cat_counts.keys())
    cat_diffs = []
    for cat_id in all_cats:
        gt_c = gt_cat_counts.get(cat_id, 0)
        pred_c = pred_cat_counts.get(cat_id, 0)
        diff = pred_c - gt_c
        cat_diffs.append((cat_id, gt_c, pred_c, diff))

    cat_diffs.sort(key=lambda x: -abs(x[3]))
    print(f"\nTop 20 categories with biggest count difference (pred - GT):")
    print(f"{'Cat ID':<8} {'Name':<35} {'GT':>5} {'Pred':>5} {'Diff':>6}")
    print("-" * 62)
    for cat_id, gt_c, pred_c, diff in cat_diffs[:20]:
        name = cat_names.get(cat_id, f"class_{cat_id}")[:35]
        print(f"{cat_id:<8} {name:<35} {gt_c:>5} {pred_c:>5} {diff:>+6}")

    # ========================================================================
    # Phase 6: Per-category AP (worst 20)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: Per-Category AP (Worst 20 Categories)")
    print("=" * 80)

    print("\nComputing per-category AP on all images...")
    per_cat_ap = compute_per_category_ap(coco_gt, all_preds)

    # Filter to categories that have GT (AP != -1) and sort by AP ascending
    valid_cats = [(cat_id, ap) for cat_id, ap in per_cat_ap.items()
                  if ap >= 0 and gt_cat_counts.get(cat_id, 0) > 0]
    valid_cats.sort(key=lambda x: x[1])

    worst_20 = valid_cats[:20]
    print(f"\n{'Rank':<6} {'Cat ID':<8} {'Name':<35} {'AP@0.5':>8} {'GT#':>5} {'Pred#':>6}")
    print("-" * 72)
    for rank, (cat_id, ap) in enumerate(worst_20, 1):
        name = cat_names.get(cat_id, f"class_{cat_id}")[:35]
        gt_c = gt_cat_counts.get(cat_id, 0)
        pred_c = pred_cat_counts.get(cat_id, 0)
        print(f"{rank:<6} {cat_id:<8} {name:<35} {ap:>8.4f} {gt_c:>5} {pred_c:>6}")

    # Also show best 10 for context
    best_10 = valid_cats[-10:]
    print(f"\nBest 10 categories for comparison:")
    print(f"{'Cat ID':<8} {'Name':<35} {'AP@0.5':>8} {'GT#':>5} {'Pred#':>6}")
    print("-" * 65)
    for cat_id, ap in reversed(best_10):
        name = cat_names.get(cat_id, f"class_{cat_id}")[:35]
        gt_c = gt_cat_counts.get(cat_id, 0)
        pred_c = pred_cat_counts.get(cat_id, 0)
        print(f"{cat_id:<8} {name:<35} {ap:>8.4f} {gt_c:>5} {pred_c:>6}")

    # Summary stats
    all_aps = [ap for _, ap in valid_cats]
    print(f"\nAP distribution across {len(valid_cats)} categories with GT:")
    print(f"  Mean AP:   {np.mean(all_aps):.4f}")
    print(f"  Median AP: {np.median(all_aps):.4f}")
    print(f"  Std AP:    {np.std(all_aps):.4f}")
    print(f"  AP=0:      {sum(1 for ap in all_aps if ap == 0)} categories")
    print(f"  AP<0.1:    {sum(1 for ap in all_aps if ap < 0.1)} categories")
    print(f"  AP<0.5:    {sum(1 for ap in all_aps if ap < 0.5)} categories")
    print(f"  AP>=0.9:   {sum(1 for ap in all_aps if ap >= 0.9)} categories")

    # ========================================================================
    # Phase 7: Error Analysis for Worst Categories
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: Error Analysis (Worst 20 Categories)")
    print("=" * 80)

    worst_cat_ids = set(cat_id for cat_id, _ in worst_20)
    print("\nAnalyzing errors for worst-performing categories...")
    errors = error_analysis(coco_gt, all_preds, worst_cat_ids)

    print(f"\n{'Cat ID':<8} {'Name':<30} {'GT#':>5} {'Missed':>7} {'WrongCls':>9} {'FP':>5} {'Confused With':<40}")
    print("-" * 110)
    for cat_id, ap in worst_20:
        err = errors[cat_id]
        name = cat_names.get(cat_id, f"class_{cat_id}")[:30]
        confused_str = ", ".join(
            f"{cat_names.get(c, str(c))[:12]}({n})" for c, n in err["top_confused"]
        ) if err["top_confused"] else "---"
        print(f"{cat_id:<8} {name:<30} {err['gt_count']:>5} {err['missed']:>7} "
              f"{err['wrong_cls']:>9} {err['false_pos']:>5} {confused_str:<40}")

    # Aggregate error types
    total_missed = sum(errors[c]["missed"] for c in worst_cat_ids)
    total_wrong = sum(errors[c]["wrong_cls"] for c in worst_cat_ids)
    total_fp = sum(errors[c]["false_pos"] for c in worst_cat_ids)
    total_gt = sum(errors[c]["gt_count"] for c in worst_cat_ids)

    print(f"\nAggregate for worst 20 categories ({total_gt} GT annotations):")
    print(f"  Missed detections:    {total_missed} ({100*total_missed/total_gt:.1f}%)")
    print(f"  Wrong classifications: {total_wrong} ({100*total_wrong/total_gt:.1f}%)")
    print(f"  False positives:       {total_fp}")
    print(f"  Correctly detected:    {total_gt - total_missed - total_wrong} ({100*(total_gt-total_missed-total_wrong)/total_gt:.1f}%)")

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total runtime: {total_time:.0f}s")
    print(f"\nScores:")
    print(f"  All:   det={all_det_map:.4f} cls={all_cls_map:.4f} combined={all_combined:.4f}")
    print(f"  Train: det={train_det_map:.4f} cls={train_cls_map:.4f} combined={train_combined:.4f}")
    print(f"  Val:   det={val_det_map:.4f} cls={val_cls_map:.4f} combined={val_combined:.4f}")
    print(f"  Gap:   det={gap_det:+.4f} cls={gap_cls:+.4f} combined={gap_comb:+.4f}")
    print(f"\nKey findings:")
    print(f"  - {len(all_disagree)} YOLO-classifier disagreements across {len(all_images)} images")
    print(f"  - {sum(1 for ap in all_aps if ap == 0)} categories with AP=0 (complete failure)")
    print(f"  - {sum(1 for ap in all_aps if ap < 0.1)} categories with AP<0.1")
    print(f"  - Worst category errors: {total_missed} missed, {total_wrong} misclassified, {total_fp} false positive")


if __name__ == "__main__":
    main()
