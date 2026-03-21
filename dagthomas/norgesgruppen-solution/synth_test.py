"""
Synthetic local test — compare single models vs ensemble on val split.
Runs all combos and prints a comparison table.

Usage: python synth_test.py
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parent
ANNO_PATH = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json")
VAL_IMAGES = ROOT / "datasets" / "val" / "images"

# Try to load classifier
try:
    from classifier import load_classifier
    HAS_CLASSIFIER = True
except Exception:
    HAS_CLASSIFIER = False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 256
BG_CLASS_ID = 356
BG_REJECT_PROB = 0.5


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
        preds_mod = []
        seen = {}
        for p in preds:
            pm = dict(p, category_id=0)
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


def run_model(model, img_path, imgsz=1280, conf=0.001, augment=False):
    """Run a single model on an image, return (boxes_xyxy, scores, classes)."""
    results = model.predict(
        str(img_path), imgsz=imgsz, conf=conf, iou=0.6,
        max_det=3000, augment=augment, verbose=False, device="cuda",
    )
    r = results[0] if results else None
    if r is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), r
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, classes, r


def merge_wbf(det_list, img_w, img_h, iou_thr=0.55):
    """Merge detections from multiple models using WBF."""
    boxes_list, scores_list, labels_list, weights = [], [], [], []
    for boxes, scores, labels, weight in det_list:
        if len(boxes) == 0:
            continue
        norm = boxes.copy()
        norm[:, [0, 2]] /= img_w
        norm[:, [1, 3]] /= img_h
        norm = np.clip(norm, 0, 1)
        boxes_list.append(norm.tolist())
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
        weights.append(weight)
    if not boxes_list:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    mb, ms, ml = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=0.001,
    )
    mb = np.array(mb)
    if len(mb) > 0:
        mb[:, [0, 2]] *= img_w
        mb[:, [1, 3]] *= img_h
    return mb, np.array(ms), np.array(ml, dtype=int)


def classify_detections(classifier, orig_img, boxes, scores, labels, device):
    """Re-classify with K=1 yolo_trust blending."""
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    has_bg = classifier.num_classes > BG_CLASS_ID
    img_h, img_w = orig_img.shape[:2]

    img_rgb = np.ascontiguousarray(orig_img[:, :, ::-1])
    img_tensor = torch.from_numpy(img_rgb).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0

    results = []
    classified = set()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w = min(img_w, x2) - max(0, x1)
        h = min(img_h, y2) - max(0, y1)
        if w < 5 or h < 5:
            continue

        bx = torch.tensor([[0, max(0,x1), max(0,y1), min(img_w,x2), min(img_h,y2)]], device=device, dtype=torch.float32)
        try:
            from torchvision.ops import roi_align
            crop = roi_align(img_tensor, bx, output_size=(INPUT_SIZE, INPUT_SIZE), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        except Exception:
            cx1, cy1, cx2, cy2 = int(max(0,x1)), int(max(0,y1)), int(min(img_w,x2)), int(min(img_h,y2))
            crop = F.interpolate(img_tensor[:,:,cy1:cy2,cx1:cx2], size=(INPUT_SIZE,INPUT_SIZE), mode='bilinear', align_corners=False)

        crop = ((crop - mean_gpu) / std_gpu).half()
        with torch.no_grad():
            logits = classifier(crop)
            probs = F.softmax(logits, dim=1)

        if has_bg and probs[0, BG_CLASS_ID].item() > BG_REJECT_PROB:
            continue

        top_prob, top_cls = probs[0].topk(1)
        cls_id = top_cls[0].item()
        cls_prob = top_prob[0].item()

        if cls_id == BG_CLASS_ID:
            continue

        yolo_conf = float(scores[i])
        yolo_cls = int(labels[i])
        if cls_id == yolo_cls:
            blended = yolo_conf * (0.7 + 0.3 * cls_prob)
        else:
            blended = yolo_conf * cls_prob * 0.5

        if blended >= 0.001:
            results.append((i, cls_id, blended))
            classified.add(i)

    return results, classified


def to_preds(image_id, boxes, scores, labels):
    """Convert to COCO prediction format."""
    preds = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        preds.append({
            "image_id": image_id,
            "category_id": int(labels[i]),
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(scores[i]),
        })
    return preds


def to_preds_classified(image_id, boxes, scores, labels, cls_results, classified):
    """Convert with classifier blending."""
    preds = []
    for idx, cls_id, score in cls_results:
        x1, y1, x2, y2 = boxes[idx]
        preds.append({
            "image_id": image_id,
            "category_id": cls_id,
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": round(score, 4),
        })
    for i in range(len(boxes)):
        if i not in classified:
            x1, y1, x2, y2 = boxes[i]
            preds.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": round(float(scores[i]), 4),
            })
    return preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    yolo_v1 = YOLO(str(ROOT / "best.pt"))
    yolo_v2 = YOLO(str(ROOT / "best_v4_seed7.pt"))
    yolo_v3 = YOLO(str(ROOT / "best_v3_seed7.pt"))

    classifier = None
    if HAS_CLASSIFIER and (ROOT / "classifier.safetensors").exists():
        classifier = load_classifier(ROOT / "classifier.safetensors", device=device)
        print("Classifier loaded")

    print("Loading ground truth...")
    coco_gt = COCO(str(ANNO_PATH))

    image_dir = VAL_IMAGES
    if not image_dir.exists():
        print(f"Val images not found at {image_dir}, using train/")
        image_dir = ROOT / "datasets" / "train" / "images"

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))
    print(f"Testing on {len(image_files)} images\n")

    # Collect predictions for each config
    configs = {
        "v1_only": [],
        "v2_only": [],
        "v3_only": [],
        "v1v2_wbf": [],
        "v1v3_wbf": [],
        "v2v3_wbf": [],
        "v1v2v3_wbf": [],
    }
    if classifier:
        configs["v1v2_wbf_cls"] = []
        configs["v1v3_wbf_cls"] = []
        configs["v2v3_wbf_cls"] = []
        configs["v1v2v3_wbf_cls"] = []

    for idx, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])

        # Run models — use 1280 no TTA (T4 safe)
        b1, s1, c1, r1 = run_model(yolo_v1, img_path, imgsz=1280, augment=False)
        b2, s2, c2, r2 = run_model(yolo_v2, img_path, imgsz=1280, augment=False)
        b3, s3, c3, r3 = run_model(yolo_v3, img_path, imgsz=1280, augment=False)

        orig_img = r1.orig_img if r1 is not None else (r2.orig_img if r2 is not None else None)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        # Single model preds
        configs["v1_only"].extend(to_preds(image_id, b1, s1, c1))
        configs["v2_only"].extend(to_preds(image_id, b2, s2, c2))
        configs["v3_only"].extend(to_preds(image_id, b3, s3, c3))

        # 2-model WBF combos
        mb12, ms12, ml12 = merge_wbf([(b1, s1, c1, 1.0), (b2, s2, c2, 1.0)], w, h)
        configs["v1v2_wbf"].extend(to_preds(image_id, mb12, ms12, ml12))

        mb13, ms13, ml13 = merge_wbf([(b1, s1, c1, 1.0), (b3, s3, c3, 1.0)], w, h)
        configs["v1v3_wbf"].extend(to_preds(image_id, mb13, ms13, ml13))

        mb23, ms23, ml23 = merge_wbf([(b2, s2, c2, 1.0), (b3, s3, c3, 1.0)], w, h)
        configs["v2v3_wbf"].extend(to_preds(image_id, mb23, ms23, ml23))

        # 3-model WBF
        mb123, ms123, ml123 = merge_wbf([(b1, s1, c1, 1.0), (b2, s2, c2, 1.0), (b3, s3, c3, 1.0)], w, h)
        configs["v1v2v3_wbf"].extend(to_preds(image_id, mb123, ms123, ml123))

        # Classifier variants
        if classifier and orig_img is not None:
            cr12, cl12 = classify_detections(classifier, orig_img, mb12, ms12, ml12, device)
            configs["v1v2_wbf_cls"].extend(to_preds_classified(image_id, mb12, ms12, ml12, cr12, cl12))

            cr13, cl13 = classify_detections(classifier, orig_img, mb13, ms13, ml13, device)
            configs["v1v3_wbf_cls"].extend(to_preds_classified(image_id, mb13, ms13, ml13, cr13, cl13))

            cr23, cl23 = classify_detections(classifier, orig_img, mb23, ms23, ml23, device)
            configs["v2v3_wbf_cls"].extend(to_preds_classified(image_id, mb23, ms23, ml23, cr23, cl23))

            cr123, cl123 = classify_detections(classifier, orig_img, mb123, ms123, ml123, device)
            configs["v1v2v3_wbf_cls"].extend(to_preds_classified(image_id, mb123, ms123, ml123, cr123, cl123))

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(image_files)}]")

    # Evaluate all configs
    print(f"\n{'='*80}")
    print(f"{'Config':<30} {'Preds':>8} {'Det mAP':>9} {'Cls mAP':>9} {'Combined':>10}")
    print(f"{'-'*80}")

    results = []
    for name, preds in configs.items():
        if not preds:
            continue
        det = compute_map(coco_gt, preds, category_agnostic=True)
        cls = compute_map(coco_gt, preds, category_agnostic=False)
        comb = 0.7 * det + 0.3 * cls
        results.append((name, len(preds), det, cls, comb))

    for name, n, det, cls, comb in sorted(results, key=lambda x: -x[4]):
        print(f"  {name:<28} {n:>8} {det:>9.4f} {cls:>9.4f} {comb:>10.4f}")

    print(f"{'='*80}")
    print("Combined = 0.7 * det_mAP@0.5 + 0.3 * cls_mAP@0.5")

    # Save best predictions
    best = max(results, key=lambda x: x[4])
    print(f"\nBest: {best[0]} ({best[4]:.4f})")
    with open(ROOT / "synth_best_preds.json", "w") as f:
        json.dump(configs[best[0]], f)
    print(f"Saved to synth_best_preds.json")


if __name__ == "__main__":
    main()
