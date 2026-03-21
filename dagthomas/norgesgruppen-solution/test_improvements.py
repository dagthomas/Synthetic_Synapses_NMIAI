"""
Test inference-time improvements for generalization:
  1. Soft-NMS (Gaussian decay instead of hard suppression)
  2. Score compression (score^alpha to reduce overconfidence)
  3. Combo: Soft-NMS + score compression

Loads models once, runs YOLO inference once, then tests different
post-processing variants and evaluates with detection+classification mAP.
"""

import json
import time
import copy
import numpy as np
from pathlib import Path

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

from torchvision.ops import box_iou

# Paths
ROOT = Path(__file__).resolve().parent
ANNO_PATH = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json")
YOLO_WEIGHTS = ROOT / "best.pt"
CLS_WEIGHTS = ROOT / "classifier.safetensors"

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Fixed pipeline params (best from sweep)
TOP_K = 2
BG_REJECT_PROB = 0.5
BLEND_MODE = "yolo_trust"
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


def soft_nms_gaussian(boxes, scores, sigma=0.5, score_threshold=0.001):
    """Soft-NMS with Gaussian decay.

    For each box (in score-descending order), multiply overlapping boxes'
    scores by exp(-iou^2 / sigma). This preserves correct detections in
    dense shelf regions where products overlap.

    Args:
        boxes: (N, 4) tensor of xyxy boxes
        scores: (N,) tensor of scores
        sigma: Gaussian decay parameter
        score_threshold: minimum score to keep

    Returns:
        keep_indices: indices of kept boxes
        new_scores: updated scores for kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=int), np.array([])

    boxes_t = torch.from_numpy(boxes).float()
    scores_np = scores.copy()
    N = len(scores_np)
    indices = np.arange(N)

    # Sort by score descending
    order = np.argsort(-scores_np)
    keep = []
    keep_scores = []

    # Process boxes in score-descending order
    remaining = list(range(N))
    sorted_remaining = [order[i] for i in range(N)]

    # Work with a mutable scores array
    working_scores = scores_np.copy()

    for iteration in range(N):
        if not sorted_remaining:
            break

        # Find the index with the highest current score among remaining
        best_idx = -1
        best_score = -1
        best_pos = -1
        for pos, idx in enumerate(sorted_remaining):
            if working_scores[idx] > best_score:
                best_score = working_scores[idx]
                best_idx = idx
                best_pos = pos

        if best_score < score_threshold:
            break

        keep.append(best_idx)
        keep_scores.append(best_score)
        sorted_remaining.pop(best_pos)

        if not sorted_remaining:
            break

        # Compute IoU of best box with all remaining
        remaining_indices = np.array(sorted_remaining)
        best_box = boxes_t[best_idx].unsqueeze(0)
        rem_boxes = boxes_t[remaining_indices]

        ious = box_iou(best_box, rem_boxes).squeeze(0).numpy()

        # Gaussian decay: multiply remaining scores by exp(-iou^2 / sigma)
        decay = np.exp(-(ious ** 2) / sigma)
        for i, rem_idx in enumerate(remaining_indices):
            working_scores[rem_idx] *= decay[i]

    return np.array(keep, dtype=int), np.array(keep_scores)


def run_yolo_cached(yolo, image_files, device, imgsz=1536, conf=0.001,
                    iou=0.6, max_det=3000, augment=True):
    """Run YOLO on all images and cache raw detections."""
    detections = {}

    for idx, img_path in enumerate(image_files):
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

        if (idx + 1) % 50 == 0:
            print(f"  YOLO inference: {idx+1}/{len(image_files)}")

    return detections


def apply_soft_nms_to_detections(detections, sigma=0.5):
    """Apply soft-NMS to cached YOLO detections per image.

    Replaces hard NMS scores with soft-NMS Gaussian-decayed scores.
    Returns new detections dict with updated scores and filtered boxes.
    """
    new_detections = {}
    for img_path, (boxes, scores, labels, orig_img) in detections.items():
        if len(boxes) == 0:
            new_detections[img_path] = (boxes, scores, labels, orig_img)
            continue

        # Apply soft-NMS per class (like the original hard NMS)
        all_keep = []
        all_new_scores = []
        for cls_id in np.unique(labels):
            mask = labels == cls_id
            cls_indices = np.where(mask)[0]
            cls_boxes = boxes[mask]
            cls_scores = scores[mask]

            keep_local, new_scores = soft_nms_gaussian(
                cls_boxes, cls_scores, sigma=sigma
            )

            for k, s in zip(keep_local, new_scores):
                all_keep.append(cls_indices[k])
                all_new_scores.append(s)

        if all_keep:
            keep = np.array(all_keep)
            new_scores = np.array(all_new_scores)
            # Sort by original index for consistency
            sort_order = np.argsort(keep)
            keep = keep[sort_order]
            new_scores = new_scores[sort_order]
            new_detections[img_path] = (
                boxes[keep], new_scores, labels[keep], orig_img
            )
        else:
            new_detections[img_path] = (
                np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int),
                orig_img,
            )

    return new_detections


def classify_detections(classifier, detections, device, mean_gpu, std_gpu,
                        input_size, score_alpha=1.0):
    """Run classifier on all cached detections with yolo_trust blending.

    Args:
        score_alpha: power transform exponent for final scores.
                     alpha > 1 compresses scores toward 0.
    """
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
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                score = float(scores[i])
                if score_alpha != 1.0:
                    score = score ** score_alpha
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(score, 4),
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

                    # Background rejection
                    if has_bg_class and float(bg_probs[j]) > BG_REJECT_PROB:
                        classified_boxes.add(box_idx)
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

                        # Apply score compression
                        if score_alpha != 1.0:
                            blended = blended ** score_alpha

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
                score = float(scores[i])
                if score_alpha != 1.0:
                    score = score ** score_alpha
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(score, 4),
                })

        del img_tensor

    return predictions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_start = time.time()

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
    print(f"Models loaded in {time.time() - t_start:.1f}s")

    # Get ALL images (train + val, jpg + jpeg + png)
    image_files = []
    datasets_dir = ROOT / "datasets"
    for split in ["train", "val"]:
        img_dir = datasets_dir / split / "images"
        if img_dir.exists():
            image_files.extend(img_dir.glob("*.jpg"))
            image_files.extend(img_dir.glob("*.jpeg"))
            image_files.extend(img_dir.glob("*.png"))
    image_files = sorted(set(image_files))
    print(f"Found {len(image_files)} images")

    # Phase 1: Cache YOLO detections (run once)
    print("\n=== Phase 1: YOLO Inference (cached) ===")
    t0 = time.time()
    detections = run_yolo_cached(
        yolo, image_files, device,
        imgsz=1536, conf=0.001, iou=0.6, max_det=3000, augment=True,
    )
    n_dets = sum(len(d[0]) for d in detections.values())
    print(f"YOLO done: {n_dets} detections in {time.time()-t0:.1f}s")

    # Phase 2: Test configurations
    print("\n=== Phase 2: Post-processing Variants ===")

    configs = [
        # Baseline
        {"name": "Baseline (yolo_trust)", "soft_nms": False, "sigma": None, "alpha": 1.0},
        # Soft-NMS variants
        {"name": "Soft-NMS sigma=0.3", "soft_nms": True, "sigma": 0.3, "alpha": 1.0},
        {"name": "Soft-NMS sigma=0.5", "soft_nms": True, "sigma": 0.5, "alpha": 1.0},
        {"name": "Soft-NMS sigma=0.7", "soft_nms": True, "sigma": 0.7, "alpha": 1.0},
        # Score compression
        {"name": "Alpha=1.1", "soft_nms": False, "sigma": None, "alpha": 1.1},
        {"name": "Alpha=1.2", "soft_nms": False, "sigma": None, "alpha": 1.2},
        {"name": "Alpha=1.3", "soft_nms": False, "sigma": None, "alpha": 1.3},
        {"name": "Alpha=1.5", "soft_nms": False, "sigma": None, "alpha": 1.5},
    ]

    results = []
    best_alpha = None
    best_alpha_score = 0

    for cfg in configs:
        name = cfg["name"]
        print(f"\n--- {name} ---")
        t0 = time.time()

        # Apply soft-NMS if requested
        if cfg["soft_nms"]:
            working_dets = apply_soft_nms_to_detections(detections, sigma=cfg["sigma"])
            n_after = sum(len(d[0]) for d in working_dets.values())
            print(f"  Soft-NMS: {n_dets} -> {n_after} detections")
        else:
            working_dets = detections

        # Run classifier + blending
        preds = classify_detections(
            classifier, working_dets, device, mean_gpu, std_gpu,
            input_size, score_alpha=cfg["alpha"],
        )

        cls_time = time.time() - t0
        print(f"  {len(preds)} predictions in {cls_time:.1f}s")

        # Evaluate
        t0 = time.time()
        det_map = compute_map(coco_gt, preds, category_agnostic=True)
        cls_map = compute_map(coco_gt, preds, category_agnostic=False)
        combined = 0.7 * det_map + 0.3 * cls_map
        eval_time = time.time() - t0

        results.append({
            "name": name,
            "n_preds": len(preds),
            "det_mAP": det_map,
            "cls_mAP": cls_map,
            "combined": combined,
        })

        print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={combined:.4f} (eval {eval_time:.1f}s)")

        # Track best alpha for combo
        if not cfg["soft_nms"] and cfg["alpha"] != 1.0:
            if combined > best_alpha_score:
                best_alpha_score = combined
                best_alpha = cfg["alpha"]

    # Phase 3: Combo tests (Soft-NMS + best alpha)
    if best_alpha is not None:
        combo_configs = [
            {"name": f"Combo: SoftNMS(0.5) + alpha={best_alpha}",
             "soft_nms": True, "sigma": 0.5, "alpha": best_alpha},
            {"name": f"Combo: SoftNMS(0.3) + alpha={best_alpha}",
             "soft_nms": True, "sigma": 0.3, "alpha": best_alpha},
        ]

        for cfg in combo_configs:
            name = cfg["name"]
            print(f"\n--- {name} ---")
            t0 = time.time()

            working_dets = apply_soft_nms_to_detections(detections, sigma=cfg["sigma"])
            preds = classify_detections(
                classifier, working_dets, device, mean_gpu, std_gpu,
                input_size, score_alpha=cfg["alpha"],
            )

            cls_time = time.time() - t0
            print(f"  {len(preds)} predictions in {cls_time:.1f}s")

            t0 = time.time()
            det_map = compute_map(coco_gt, preds, category_agnostic=True)
            cls_map = compute_map(coco_gt, preds, category_agnostic=False)
            combined = 0.7 * det_map + 0.3 * cls_map
            eval_time = time.time() - t0

            results.append({
                "name": name,
                "n_preds": len(preds),
                "det_mAP": det_map,
                "cls_mAP": cls_map,
                "combined": combined,
            })

            print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={combined:.4f} (eval {eval_time:.1f}s)")

    # Print summary table
    total_time = time.time() - t_start
    baseline_score = results[0]["combined"]

    print("\n" + "=" * 100)
    print(f"{'Config':<40} {'Preds':>7} {'Det mAP':>9} {'Cls mAP':>9} {'Combined':>9} {'Delta':>8}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["combined"]):
        delta = r["combined"] - baseline_score
        delta_str = f"{delta:+.4f}" if delta != 0 else "   ---"
        print(f"{r['name']:<40} {r['n_preds']:>7} {r['det_mAP']:>9.4f} {r['cls_mAP']:>9.4f} "
              f"{r['combined']:>9.4f} {delta_str:>8}")
    print("=" * 100)
    print(f"Combined = 0.7 * det_mAP + 0.3 * cls_mAP")
    print(f"Total time: {total_time:.0f}s")

    # Save results
    out_path = ROOT / "improvement_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
