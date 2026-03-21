"""
Test multi-checkpoint YOLO ensemble for better generalization.

Hypothesis: Different training checkpoints capture different patterns.
Ensembling best.pt (epoch 83) + last.pt (epoch 120) provides model diversity
that should help generalization more than running a single model with TTA.

Configs tested:
  1. Baseline: best.pt + TTA (current best)
  2. Ensemble: best.pt (no TTA) + last.pt (no TTA), merge with NMS
  3. Ensemble + TTA: best.pt (TTA) + last.pt (no TTA), merge with NMS
  4. Ensemble both TTA: best.pt (TTA) + last.pt (TTA), merge with NMS
"""

import json
import time
import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

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

ROOT = Path(__file__).resolve().parent
ANNO_PATH = "X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def compute_map(coco_gt, preds, category_agnostic=False):
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


def merge_nms(det_list, iou_thr=0.6):
    """Merge detections from multiple sources using per-class NMS."""
    all_b, all_s, all_l = [], [], []
    for boxes, scores, labels in det_list:
        if len(boxes) > 0:
            all_b.append(boxes)
            all_s.append(scores)
            all_l.append(labels)
    if not all_b:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes = np.concatenate(all_b)
    scores = np.concatenate(all_s)
    labels = np.concatenate(all_l)

    keep_all = []
    for cls_id in np.unique(labels):
        mask = labels == cls_id
        idx = np.where(mask)[0]
        b_t = torch.from_numpy(boxes[mask]).float()
        s_t = torch.from_numpy(scores[mask]).float()
        keep = tv_nms(b_t, s_t, iou_thr).numpy()
        keep_all.append(idx[keep])
    keep = np.concatenate(keep_all) if keep_all else np.array([], dtype=int)
    keep.sort()
    return boxes[keep], scores[keep], labels[keep]


def run_yolo_on_images(yolo, image_files, device, imgsz=1536, conf=0.001,
                       iou=0.6, max_det=3000, augment=True):
    """Run YOLO and return per-image detections + orig_img."""
    results = {}
    for img_path in image_files:
        r = yolo.predict(
            str(img_path), imgsz=imgsz, conf=conf, iou=iou,
            max_det=max_det, augment=augment, verbose=False, device=device,
        )
        r = r[0] if r else None
        if r is not None and len(r.boxes) > 0:
            results[str(img_path)] = {
                "boxes": r.boxes.xyxy.cpu().numpy(),
                "scores": r.boxes.conf.cpu().numpy(),
                "labels": r.boxes.cls.cpu().numpy().astype(int),
                "orig_img": r.orig_img,
            }
        elif r is not None:
            results[str(img_path)] = {
                "boxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "labels": np.empty(0, dtype=int),
                "orig_img": r.orig_img,
            }
    return results


def classify_and_emit(classifier, detections, device, mean_gpu, std_gpu, input_size):
    """Run classifier on detections, return prediction list."""
    BG_CLASS_ID = 356
    BG_REJECT_PROB = 0.5
    has_bg_class = classifier.num_classes > BG_CLASS_ID
    predictions = []

    for img_path, det in detections.items():
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]
        orig_img = det["orig_img"]

        if len(boxes) == 0:
            continue

        stem = Path(img_path).stem
        try:
            image_id = int(stem.split("_")[-1])
        except ValueError:
            image_id = 0

        img_h, img_w = orig_img.shape[:2]

        valid = [i for i in range(len(boxes))
                 if (min(img_w, boxes[i][2]) - max(0, boxes[i][0])) >= 5
                 and (min(img_h, boxes[i][3]) - max(0, boxes[i][1])) >= 5]

        classified = set()

        if valid:
            img_rgb = np.ascontiguousarray(orig_img[:, :, ::-1])
            img_tensor = torch.from_numpy(img_rgb).float().to(device)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

            valid_boxes = boxes[valid]
            boxes_gpu = torch.from_numpy(valid_boxes.copy()).float().to(device)
            boxes_gpu[:, 0].clamp_(0, img_w)
            boxes_gpu[:, 1].clamp_(0, img_h)
            boxes_gpu[:, 2].clamp_(0, img_w)
            boxes_gpu[:, 3].clamp_(0, img_h)

            for chunk_start in range(0, len(valid), 500):
                chunk_end = min(chunk_start + 500, len(valid))
                chunk_boxes = boxes_gpu[chunk_start:chunk_end]

                if HAS_ROI_ALIGN:
                    rois = torch.cat([torch.zeros(len(chunk_boxes), 1, device=device), chunk_boxes], dim=1)
                    crops = roi_align(img_tensor, rois, output_size=(input_size, input_size),
                                      spatial_scale=1.0, sampling_ratio=2, aligned=True)
                else:
                    crops_list = []
                    for box in chunk_boxes:
                        x1, y1, x2, y2 = box.int().tolist()
                        crop = img_tensor[:, :, max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
                        crop = F.interpolate(crop, size=(input_size, input_size), mode='bilinear', align_corners=False)
                        crops_list.append(crop)
                    crops = torch.cat(crops_list, dim=0)

                crops = (crops - mean_gpu) / std_gpu
                crops = crops.half()

                batch_size = 64
                padded = torch.zeros(batch_size, 3, input_size, input_size, device=device, dtype=torch.float16)

                for cls_start in range(0, crops.shape[0], batch_size):
                    actual = min(batch_size, crops.shape[0] - cls_start)
                    padded[:actual] = crops[cls_start:cls_start + actual]

                    with torch.no_grad():
                        logits = classifier(padded)
                        probs = F.softmax(logits[:actual], dim=1)
                        topk_probs, topk_classes = torch.topk(probs, k=2, dim=1)

                    topk_probs_np = topk_probs.cpu().numpy()
                    topk_classes_np = topk_classes.cpu().numpy()
                    bg_probs = probs[:, BG_CLASS_ID].cpu().numpy() if has_bg_class else None

                    for j in range(actual):
                        box_idx = valid[chunk_start + cls_start + j]
                        yolo_conf = float(scores[box_idx])
                        yolo_cls = int(labels[box_idx])
                        classified.add(box_idx)

                        if has_bg_class and float(bg_probs[j]) > BG_REJECT_PROB:
                            continue

                        for k in range(2):
                            cls_id = int(topk_classes_np[j, k])
                            cls_prob = float(topk_probs_np[j, k])
                            if cls_id == BG_CLASS_ID:
                                continue

                            # yolo_trust blending
                            if cls_id == yolo_cls:
                                blended = yolo_conf * (0.7 + 0.3 * cls_prob)
                            else:
                                blended = yolo_conf * cls_prob * 0.5

                            if blended >= 0.001:
                                x1, y1, x2, y2 = boxes[box_idx]
                                predictions.append({
                                    "image_id": image_id,
                                    "category_id": cls_id,
                                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                                    "score": round(blended, 4),
                                })

            del img_tensor

        # Unclassified tiny boxes
        for i in range(len(boxes)):
            if i not in classified:
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                    "score": round(float(scores[i]), 4),
                })

    return predictions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load GT
    print("Loading ground truth...")
    coco_gt = COCO(ANNO_PATH)

    # Load classifier
    print("Loading classifier...")
    classifier = load_classifier(str(ROOT / "classifier.safetensors"), device=device)
    input_size = classifier.input_size
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    # Get all images
    image_files = []
    for split in ["train", "val"]:
        d = ROOT / "datasets" / split / "images"
        if d.exists():
            image_files.extend(d.glob("*.jpg"))
            image_files.extend(d.glob("*.jpeg"))
            image_files.extend(d.glob("*.png"))
    image_files = sorted(set(image_files))
    print(f"Found {len(image_files)} images")

    # Load YOLO models
    print("Loading YOLO best.pt...")
    yolo_best = YOLO(str(ROOT / "best.pt"))
    print("Loading YOLO last.pt...")
    yolo_last = YOLO(str(ROOT / "runs" / "yolov8x_final" / "weights" / "last.pt"))

    # Run YOLO inference for different configs
    configs = {}

    print("\n--- Running YOLO best.pt with TTA ---")
    t0 = time.time()
    det_best_tta = run_yolo_on_images(yolo_best, image_files, device, augment=True)
    print(f"  {time.time()-t0:.1f}s")

    print("--- Running YOLO best.pt without TTA ---")
    t0 = time.time()
    det_best_notta = run_yolo_on_images(yolo_best, image_files, device, augment=False)
    print(f"  {time.time()-t0:.1f}s")

    print("--- Running YOLO last.pt without TTA ---")
    t0 = time.time()
    det_last_notta = run_yolo_on_images(yolo_last, image_files, device, augment=False)
    print(f"  {time.time()-t0:.1f}s")

    print("--- Running YOLO last.pt with TTA ---")
    t0 = time.time()
    det_last_tta = run_yolo_on_images(yolo_last, image_files, device, augment=True)
    print(f"  {time.time()-t0:.1f}s")

    # Build ensemble detections
    def merge_detections(det_a, det_b, nms_iou=0.6):
        """Merge two detection dicts by per-image NMS."""
        all_paths = set(list(det_a.keys()) + list(det_b.keys()))
        merged = {}
        for path in all_paths:
            det_list = []
            orig_img = None
            for det in [det_a, det_b]:
                if path in det:
                    d = det[path]
                    if orig_img is None:
                        orig_img = d["orig_img"]
                    if len(d["boxes"]) > 0:
                        det_list.append((d["boxes"], d["scores"], d["labels"]))

            if det_list:
                if len(det_list) > 1:
                    boxes, scores, labels = merge_nms(det_list, nms_iou)
                else:
                    boxes, scores, labels = det_list[0]
                merged[path] = {
                    "boxes": boxes, "scores": scores, "labels": labels,
                    "orig_img": orig_img,
                }
        return merged

    # Config 1: Baseline (best.pt + TTA)
    print("\n=== Config 1: Baseline (best.pt TTA) ===")
    t0 = time.time()
    preds = classify_and_emit(classifier, det_best_tta, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["1_baseline"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Config 2: Ensemble (best + last, no TTA)
    print("\n=== Config 2: Ensemble best+last (no TTA) ===")
    t0 = time.time()
    det_ens = merge_detections(det_best_notta, det_last_notta)
    preds = classify_and_emit(classifier, det_ens, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["2_ensemble_notta"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Config 3: Ensemble (best TTA + last no TTA)
    print("\n=== Config 3: Ensemble best(TTA) + last(noTTA) ===")
    t0 = time.time()
    det_ens = merge_detections(det_best_tta, det_last_notta)
    preds = classify_and_emit(classifier, det_ens, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["3_ensemble_best_tta_last_notta"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Config 4: Ensemble both TTA
    print("\n=== Config 4: Ensemble best(TTA) + last(TTA) ===")
    t0 = time.time()
    det_ens = merge_detections(det_best_tta, det_last_tta)
    preds = classify_and_emit(classifier, det_ens, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["4_ensemble_both_tta"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Config 5: Ensemble with looser NMS (IoU=0.5)
    print("\n=== Config 5: Ensemble best(TTA) + last(noTTA), NMS IoU=0.5 ===")
    t0 = time.time()
    det_ens = merge_detections(det_best_tta, det_last_notta, nms_iou=0.5)
    preds = classify_and_emit(classifier, det_ens, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["5_ensemble_nms05"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Config 6: last.pt alone with TTA (is it better or worse than best.pt?)
    print("\n=== Config 6: last.pt TTA (solo) ===")
    t0 = time.time()
    preds = classify_and_emit(classifier, det_last_tta, device, mean_gpu, std_gpu, input_size)
    det_map = compute_map(coco_gt, preds, True)
    cls_map = compute_map(coco_gt, preds, False)
    comb = 0.7 * det_map + 0.3 * cls_map
    print(f"  det={det_map:.4f} cls={cls_map:.4f} comb={comb:.4f} ({len(preds)} preds, {time.time()-t0:.1f}s)")
    configs["6_last_tta_solo"] = {"det": det_map, "cls": cls_map, "comb": comb, "n": len(preds)}

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Config':<45} {'Preds':>7} {'Det':>7} {'Cls':>7} {'Comb':>7} {'Delta':>7}")
    print("-" * 90)
    baseline = configs["1_baseline"]["comb"]
    for name, r in sorted(configs.items(), key=lambda x: -x[1]["comb"]):
        delta = r["comb"] - baseline
        d_str = f"{delta:+.4f}" if name != "1_baseline" else "   ---"
        print(f"{name:<45} {r['n']:>7} {r['det']:>7.4f} {r['cls']:>7.4f} {r['comb']:>7.4f} {d_str:>7}")
    print("=" * 90)

    with open(ROOT / "ensemble_results.json", "w") as f:
        json.dump(configs, f, indent=2)
    print(f"\nResults saved to ensemble_results.json")


if __name__ == "__main__":
    main()
