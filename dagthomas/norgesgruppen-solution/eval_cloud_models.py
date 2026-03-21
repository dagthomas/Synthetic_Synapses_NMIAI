"""
Evaluate cloud-trained models: individual and ensemble configurations.
Runs each config against training data and compares mAP scores.
"""
import argparse
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
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from torchvision.ops import roi_align
    HAS_ROI_ALIGN = True
except ImportError:
    HAS_ROI_ALIGN = False

SCRIPT_DIR = Path(__file__).resolve().parent
ANNO_PATH = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json")
IMG_DIR = Path("X:/norgesgruppen/NM_NGD_coco_dataset/train/images")

# Config matching run_ensemble.py
CONF_THR = 0.001
IOU_THR = 0.6
MAX_DET = 3000
TOP_K = 1
MIN_SCORE = 0.001
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001
BG_CLASS_ID = 356
BG_REJECT_PROB = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 256


def merge_wbf(det_list, img_w, img_h):
    if not det_list:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
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
        weights=weights, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR)
    mb = np.array(mb)
    if len(mb) > 0:
        mb[:, [0, 2]] *= img_w
        mb[:, [1, 3]] *= img_h
    return mb, np.array(ms), np.array(ml, dtype=int)


def classify_and_blend(classifier, orig_img, boxes, scores, labels, device, mean_gpu, std_gpu):
    has_bg = classifier.num_classes > BG_CLASS_ID
    img_h, img_w = orig_img.shape[:2]
    results = []
    valid = [i for i in range(len(boxes))
             if min(img_w, boxes[i][2]) - max(0, boxes[i][0]) >= 5
             and min(img_h, boxes[i][3]) - max(0, boxes[i][1]) >= 5]
    if not valid:
        return results

    img_rgb = np.ascontiguousarray(orig_img[:, :, ::-1])
    img_tensor = torch.from_numpy(img_rgb).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
    boxes_gpu = torch.from_numpy(boxes[valid].copy()).float().to(device)
    boxes_gpu[:, 0].clamp_(0, img_w)
    boxes_gpu[:, 1].clamp_(0, img_h)
    boxes_gpu[:, 2].clamp_(0, img_w)
    boxes_gpu[:, 3].clamp_(0, img_h)

    batch_size = 64
    for cs in range(0, len(valid), 500):
        ce = min(cs + 500, len(valid))
        chunk_boxes = boxes_gpu[cs:ce]
        if HAS_ROI_ALIGN:
            rois = torch.cat([torch.zeros(len(chunk_boxes), 1, device=device), chunk_boxes], dim=1)
            crops = roi_align(img_tensor, rois, output_size=(INPUT_SIZE, INPUT_SIZE),
                              spatial_scale=1.0, sampling_ratio=2, aligned=True)
        else:
            crops_list = []
            for box in chunk_boxes:
                x1, y1, x2, y2 = box.int().tolist()
                y1, y2 = max(0, y1), min(img_h, y2)
                x1, x2 = max(0, x1), min(img_w, x2)
                crop = img_tensor[:, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
                crops_list.append(crop)
            crops = torch.cat(crops_list, dim=0)
        crops = ((crops - mean_gpu) / std_gpu).half()
        padded = torch.zeros(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=device, dtype=torch.float16)
        for bs in range(0, crops.shape[0], batch_size):
            actual = min(batch_size, crops.shape[0] - bs)
            padded[:actual] = crops[bs:bs + actual]
            with torch.no_grad():
                logits = classifier(padded)
                probs = F.softmax(logits[:actual], dim=1)
                topk_probs, topk_classes = torch.topk(probs, k=TOP_K, dim=1)
            tp = topk_probs.cpu().numpy()
            tc = topk_classes.cpu().numpy()
            bg_p = probs[:, BG_CLASS_ID].cpu().numpy() if has_bg else None
            for j in range(actual):
                bi = valid[cs + bs + j]
                yc = float(scores[bi])
                yl = int(labels[bi])
                if has_bg and float(bg_p[j]) > BG_REJECT_PROB:
                    continue
                for k in range(TOP_K):
                    cid = int(tc[j, k])
                    cp = float(tp[j, k])
                    if cid == BG_CLASS_ID:
                        continue
                    bl = yc * (0.7 + 0.3 * cp) if cid == yl else yc * cp * 0.5
                    if bl >= MIN_SCORE:
                        results.append((bi, cid, bl))
    del img_tensor
    return results


def run_config(models, classifier, device, mean_gpu, std_gpu, image_files):
    """Run a specific model configuration and return predictions."""
    predictions = []
    for img_idx, img_path in enumerate(image_files):
        stem = img_path.stem
        try:
            image_id = int(stem.split("_")[-1])
        except ValueError:
            image_id = img_idx + 1

        det_list = []
        orig_img = None
        for model, imgsz, weight, use_tta in models:
            res = model.predict(str(img_path), imgsz=imgsz, conf=CONF_THR, iou=IOU_THR,
                                max_det=MAX_DET, augment=use_tta, verbose=False, device=device)
            r = res[0] if res else None
            if r is not None and orig_img is None:
                orig_img = r.orig_img
            if r is not None and len(r.boxes) > 0:
                det_list.append((
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                    weight,
                ))

        if not det_list or orig_img is None:
            continue

        if len(det_list) > 1:
            h, w = orig_img.shape[:2]
            boxes, sc, lb = merge_wbf(det_list, w, h)
        else:
            boxes, sc, lb = det_list[0][:3]

        if len(boxes) == 0:
            continue

        if classifier is not None:
            blended = classify_and_blend(classifier, orig_img, boxes, sc, lb, device, mean_gpu, std_gpu)
            classified = set()
            for bi, cid, score in blended:
                classified.add(bi)
                x1, y1, x2, y2 = boxes[bi]
                predictions.append({"image_id": image_id, "category_id": cid,
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                    "score": round(score, 4)})
            for i in range(len(boxes)):
                if i not in classified:
                    x1, y1, x2, y2 = boxes[i]
                    predictions.append({"image_id": image_id, "category_id": int(lb[i]),
                        "bbox": [round(float(x1), 2), round(float(y1), 2),
                                 round(float(x2-x1), 2), round(float(y2-y1), 2)],
                        "score": round(float(sc[i]), 4)})
        else:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                predictions.append({"image_id": image_id, "category_id": int(lb[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                    "score": round(float(sc[i]), 4)})
    return predictions


def compute_map(coco_gt, preds, category_agnostic=False):
    if not preds:
        return 0.0
    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]
    if category_agnostic:
        gt_data = {"images": list(coco_gt.imgs.values()), "annotations": [],
                   "categories": [{"id": 0, "name": "product"}]}
        for aid in coco_gt.getAnnIds():
            ann = copy.deepcopy(coco_gt.anns[aid])
            ann["category_id"] = 0
            gt_data["annotations"].append(ann)
        coco_gt_mod = COCO()
        coco_gt_mod.dataset = gt_data
        coco_gt_mod.createIndex()
        preds_mod = [dict(p, category_id=0) for p in preds]
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
    s = coco_eval.eval["precision"]
    return float(np.mean(s[:, :, :, 0, 2][s[:, :, :, 0, 2] > -1]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load GT
    print("Loading ground truth...")
    coco_gt = COCO(str(ANNO_PATH))

    # Collect images
    image_files = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.jpeg"))
    print(f"Found {len(image_files)} images")

    # Load all candidate models
    model_files = {
        "v1_local": SCRIPT_DIR / "best.pt",
        "cloud_s42": SCRIPT_DIR / "best_cloud_seed42.pt",
        "cloud_s123": SCRIPT_DIR / "best_cloud_seed123.pt",
        "cloud_v3_s7": SCRIPT_DIR / "best_cloud_v3_seed7.pt",
        "rtdetr_x": SCRIPT_DIR / "best_rtdetr_x.pt",
        "rtdetr_x_fixed": SCRIPT_DIR / "best_rtdetr_x_fixed.pt",
        "hires": SCRIPT_DIR / "best_yolov8x_hires.pt",
    }

    loaded = {}
    for name, path in model_files.items():
        if path.exists():
            print(f"Loading {name} from {path.name}...")
            loaded[name] = YOLO(str(path))
        else:
            print(f"SKIP {name} — {path.name} not found")

    # Load classifier
    cls_path = SCRIPT_DIR / "classifier.safetensors"
    classifier = load_classifier(cls_path, device=device) if cls_path.exists() else None
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) if classifier else None
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1) if classifier else None

    # Define configs to test: list of (model, imgsz, wbf_weight, use_tta)
    configs = {}

    # Individual models (no TTA for speed)
    for name in ["v1_local", "cloud_s42", "cloud_s123", "cloud_v3_s7"]:
        if name in loaded:
            configs[f"{name}_solo"] = [(loaded[name], 1280, 1.0, False)]

    # RT-DETR variants
    for name in ["rtdetr_x", "rtdetr_x_fixed"]:
        if name in loaded:
            configs[f"{name}_solo"] = [(loaded[name], 1280, 1.0, False)]

    # Hires model
    if "hires" in loaded:
        configs["hires_solo"] = [(loaded["hires"], 1536, 1.0, False)]

    # Current best (v1 + rtdetr_x) — baseline
    if "v1_local" in loaded and "rtdetr_x" in loaded:
        configs["BASELINE_v1+rtdetr"] = [
            (loaded["v1_local"], 1536, 1.0, True),
            (loaded["rtdetr_x"], 1280, 0.8, False),
        ]

    # New ensemble: cloud_s42 + rtdetr_x_fixed
    if "cloud_s42" in loaded and "rtdetr_x_fixed" in loaded:
        configs["cloud42+rtdetr_fixed"] = [
            (loaded["cloud_s42"], 1536, 1.0, True),
            (loaded["rtdetr_x_fixed"], 1280, 0.8, False),
        ]

    # cloud_s42 + rtdetr_x (original)
    if "cloud_s42" in loaded and "rtdetr_x" in loaded:
        configs["cloud42+rtdetr_orig"] = [
            (loaded["cloud_s42"], 1536, 1.0, True),
            (loaded["rtdetr_x"], 1280, 0.8, False),
        ]

    # cloud_v3_s7 (1536 trained) + rtdetr_x_fixed
    if "cloud_v3_s7" in loaded and "rtdetr_x_fixed" in loaded:
        configs["cloudv3_s7+rtdetr_fixed"] = [
            (loaded["cloud_v3_s7"], 1536, 1.0, True),
            (loaded["rtdetr_x_fixed"], 1280, 0.8, False),
        ]

    # Best cloud + v1 + rtdetr_x_fixed (3-model)
    if "cloud_s42" in loaded and "v1_local" in loaded and "rtdetr_x_fixed" in loaded:
        configs["3model_v1+cloud42+rtdetr_fixed"] = [
            (loaded["v1_local"], 1280, 1.0, False),
            (loaded["cloud_s42"], 1536, 1.0, True),
            (loaded["rtdetr_x_fixed"], 1280, 0.8, False),
        ]

    # hires + rtdetr_x_fixed
    if "hires" in loaded and "rtdetr_x_fixed" in loaded:
        configs["hires+rtdetr_fixed"] = [
            (loaded["hires"], 1536, 1.0, True),
            (loaded["rtdetr_x_fixed"], 1280, 0.8, False),
        ]

    print(f"\n{'='*90}")
    print(f"Testing {len(configs)} configurations...")
    print(f"{'='*90}")

    results = []
    for cfg_name, model_list in configs.items():
        t0 = time.time()
        model_names = "+".join([f"{m[0].ckpt_path.split('/')[-1].split('.')[0]}@{m[1]}{'TTA' if m[3] else ''}"
                                if hasattr(m[0], 'ckpt_path') else f"model@{m[1]}"
                                for m in model_list])
        print(f"\n>>> {cfg_name}")
        preds = run_config(model_list, classifier, device, mean_gpu, std_gpu, image_files)
        dt = time.time() - t0

        det_map = compute_map(coco_gt, preds, category_agnostic=True)
        cls_map = compute_map(coco_gt, preds, category_agnostic=False)
        combined = 0.7 * det_map + 0.3 * cls_map

        results.append({
            "config": cfg_name,
            "n_preds": len(preds),
            "det_mAP": det_map,
            "cls_mAP": cls_map,
            "combined": combined,
            "time_s": dt,
        })
        print(f"    det={det_map:.4f}  cls={cls_map:.4f}  combined={combined:.4f}  preds={len(preds)}  time={dt:.1f}s")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Config':<40} {'Preds':>7} {'Det mAP':>8} {'Cls mAP':>8} {'Combined':>9} {'Time':>6}")
    print(f"{'-'*100}")
    for r in sorted(results, key=lambda x: -x["combined"]):
        print(f"{r['config']:<40} {r['n_preds']:>7} {r['det_mAP']:>8.4f} {r['cls_mAP']:>8.4f} {r['combined']:>9.4f} {r['time_s']:>5.0f}s")
    print(f"{'='*100}")

    # Save results
    out_path = SCRIPT_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
