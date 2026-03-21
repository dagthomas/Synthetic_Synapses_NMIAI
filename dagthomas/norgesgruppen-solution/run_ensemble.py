"""
NorgesGruppen Object Detection — Ensemble Pipeline

Ensemble: YOLOv8x + RT-DETR-x merged with Weighted Boxes Fusion (WBF).
Two diverse architectures (CNN + Transformer) provide complementary strengths.

Usage: python run_ensemble.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
import time
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

try:
    from torchvision.ops import roi_align
    HAS_ROI_ALIGN = True
except ImportError:
    HAS_ROI_ALIGN = False

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_WEIGHTS = SCRIPT_DIR / "best.pt"
RTDETR_WEIGHTS = SCRIPT_DIR / "best_rtdetr_x.pt"
CLASSIFIER_WEIGHTS = SCRIPT_DIR / "classifier.safetensors"

# Config
YOLO_IMGSZ = 1536
RTDETR_IMGSZ = 1280
CONF_THR = 0.001
IOU_THR = 0.6
MAX_DET = 3000
TOP_K = 1
MIN_SCORE = 0.001
TIMEOUT_SECONDS = 285

# WBF config
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001

# Background rejection
BG_CLASS_ID = 356
BG_REJECT_PROB = 0.5

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 256


def merge_wbf(det_list, img_w, img_h, iou_thr=WBF_IOU_THR):
    """Merge detections from multiple models using Weighted Boxes Fusion."""
    if not det_list:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    for boxes, scores, labels, weight in det_list:
        if len(boxes) == 0:
            continue
        # WBF requires normalized coordinates [0, 1]
        norm_boxes = boxes.copy()
        norm_boxes[:, [0, 2]] /= img_w
        norm_boxes[:, [1, 3]] /= img_h
        # Clip to [0, 1]
        norm_boxes = np.clip(norm_boxes, 0, 1)
        boxes_list.append(norm_boxes.tolist())
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
        weights.append(weight)

    if not boxes_list:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    # Denormalize back to pixel coordinates
    merged_boxes = np.array(merged_boxes)
    if len(merged_boxes) > 0:
        merged_boxes[:, [0, 2]] *= img_w
        merged_boxes[:, [1, 3]] *= img_h

    return merged_boxes, np.array(merged_scores), np.array(merged_labels, dtype=int)


def classify_and_blend(classifier, orig_img, boxes, scores, labels,
                       device, mean_gpu, std_gpu):
    """Re-classify detections and return blended predictions (K=1)."""
    has_bg_class = classifier.num_classes > BG_CLASS_ID
    img_h, img_w = orig_img.shape[:2]
    n = len(boxes)
    results = []

    valid = []
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        w = min(img_w, x2) - max(0, x1)
        h = min(img_h, y2) - max(0, y1)
        if w >= 5 and h >= 5:
            valid.append(i)

    if not valid:
        return results

    img_rgb = np.ascontiguousarray(orig_img[:, :, ::-1])
    img_tensor = torch.from_numpy(img_rgb).float().to(device)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    valid_boxes = boxes[valid]
    boxes_gpu = torch.from_numpy(valid_boxes.copy()).float().to(device)
    boxes_gpu[:, 0].clamp_(0, img_w)
    boxes_gpu[:, 1].clamp_(0, img_h)
    boxes_gpu[:, 2].clamp_(0, img_w)
    boxes_gpu[:, 3].clamp_(0, img_h)

    batch_size = 64
    crop_chunk = 500

    for chunk_start in range(0, len(valid), crop_chunk):
        chunk_end = min(chunk_start + crop_chunk, len(valid))
        chunk_boxes = boxes_gpu[chunk_start:chunk_end]

        if HAS_ROI_ALIGN:
            rois = torch.cat([
                torch.zeros(len(chunk_boxes), 1, device=device),
                chunk_boxes
            ], dim=1)
            crops = roi_align(
                img_tensor, rois, output_size=(INPUT_SIZE, INPUT_SIZE),
                spatial_scale=1.0, sampling_ratio=2, aligned=True,
            )
        else:
            crops_list = []
            for box in chunk_boxes:
                x1, y1, x2, y2 = box.int().tolist()
                y1, y2 = max(0, y1), min(img_h, y2)
                x1, x2 = max(0, x1), min(img_w, x2)
                crop = img_tensor[:, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(INPUT_SIZE, INPUT_SIZE),
                                     mode='bilinear', align_corners=False)
                crops_list.append(crop)
            crops = torch.cat(crops_list, dim=0)

        crops = (crops - mean_gpu) / std_gpu
        crops = crops.half()

        padded = torch.zeros(batch_size, 3, INPUT_SIZE, INPUT_SIZE,
                             device=device, dtype=torch.float16)

        for cls_start in range(0, crops.shape[0], batch_size):
            actual = min(batch_size, crops.shape[0] - cls_start)
            padded[:actual] = crops[cls_start:cls_start + actual]

            with torch.no_grad():
                logits = classifier(padded)
                probs = F.softmax(logits[:actual], dim=1)
                topk_probs, topk_classes = torch.topk(probs, k=TOP_K, dim=1)

            topk_probs_np = topk_probs.cpu().numpy()
            topk_classes_np = topk_classes.cpu().numpy()

            if has_bg_class:
                bg_probs = probs[:, BG_CLASS_ID].cpu().numpy()

            for j in range(actual):
                box_idx = valid[chunk_start + cls_start + j]
                yolo_conf = float(scores[box_idx])
                yolo_cls = int(labels[box_idx])

                if has_bg_class and float(bg_probs[j]) > BG_REJECT_PROB:
                    continue

                for k in range(TOP_K):
                    cls_id = int(topk_classes_np[j, k])
                    cls_prob = float(topk_probs_np[j, k])

                    if cls_id == BG_CLASS_ID:
                        continue

                    # yolo_trust blending
                    if cls_id == yolo_cls:
                        blended = yolo_conf * (0.7 + 0.3 * cls_prob)
                    else:
                        blended = yolo_conf * cls_prob * 0.5

                    if blended >= MIN_SCORE:
                        results.append((box_idx, cls_id, blended))

    del img_tensor
    return results


def run_inference(input_dir, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Load models
    print("Loading YOLOv8x...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    has_rtdetr = RTDETR_WEIGHTS.exists()
    rtdetr = None
    if has_rtdetr:
        print("Loading RT-DETR-x...")
        rtdetr = YOLO(str(RTDETR_WEIGHTS))
    else:
        print("WARNING: No RT-DETR weights, using YOLO only")

    has_classifier = CLASSIFIER_WEIGHTS.exists()
    classifier = None
    mean_gpu = std_gpu = None
    if has_classifier:
        print("Loading classifier...")
        classifier = load_classifier(CLASSIFIER_WEIGHTS, device=device)
        global INPUT_SIZE
        INPUT_SIZE = getattr(classifier, 'input_size', INPUT_SIZE)
        mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    print(f"Models loaded in {time.time() - start_time:.1f}s")

    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} test images")

    predictions = []

    for img_idx, img_path in enumerate(image_files):
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            print(f"WARNING: Timeout at {img_idx}/{len(image_files)}")
            break

        stem = img_path.stem
        try:
            image_id = int(stem.split("_")[-1])
        except ValueError:
            image_id = img_idx + 1

        # Run YOLO with TTA
        yolo_results = yolo.predict(
            str(img_path), imgsz=YOLO_IMGSZ, conf=CONF_THR, iou=IOU_THR,
            max_det=MAX_DET, augment=True, verbose=False, device=device,
        )
        yr = yolo_results[0] if yolo_results else None
        orig_img = yr.orig_img if yr is not None else None

        det_list = []
        if yr is not None and len(yr.boxes) > 0:
            det_list.append((
                yr.boxes.xyxy.cpu().numpy(),
                yr.boxes.conf.cpu().numpy(),
                yr.boxes.cls.cpu().numpy().astype(int),
                1.0,  # weight for WBF
            ))

        # Run RT-DETR (no TTA — transformers don't need it as much)
        if rtdetr is not None and orig_img is not None:
            rtdetr_results = rtdetr.predict(
                orig_img, imgsz=RTDETR_IMGSZ, conf=CONF_THR, iou=IOU_THR,
                max_det=MAX_DET, augment=False, verbose=False, device=device,
            )
            rr = rtdetr_results[0] if rtdetr_results else None
            if rr is not None and len(rr.boxes) > 0:
                det_list.append((
                    rr.boxes.xyxy.cpu().numpy(),
                    rr.boxes.conf.cpu().numpy(),
                    rr.boxes.cls.cpu().numpy().astype(int),
                    0.8,  # slightly lower weight for RT-DETR
                ))

        if not det_list or orig_img is None:
            continue

        # Merge with WBF if multiple models
        if len(det_list) > 1:
            h, w = orig_img.shape[:2]
            boxes, scores_arr, labels_arr = merge_wbf(det_list, w, h)
        else:
            boxes, scores_arr, labels_arr = det_list[0][:3]

        if len(boxes) == 0:
            continue

        if classifier is not None:
            blended = classify_and_blend(
                classifier, orig_img, boxes, scores_arr, labels_arr,
                device, mean_gpu, std_gpu
            )

            classified_boxes = set()
            for box_idx, cls_id, score in blended:
                classified_boxes.add(box_idx)
                x1, y1, x2, y2 = boxes[box_idx]
                predictions.append({
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(score, 4),
                })

            for i in range(len(boxes)):
                if i not in classified_boxes:
                    x1, y1, x2, y2 = boxes[i]
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(labels_arr[i]),
                        "bbox": [round(float(x1), 2), round(float(y1), 2),
                                 round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                        "score": round(float(scores_arr[i]), 4),
                    })
        else:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels_arr[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(scores_arr[i]), 4),
                })

        if (img_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{img_idx+1}/{len(image_files)}] {len(predictions)} dets, {elapsed:.0f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    total_time = time.time() - start_time
    print(f"\nDone! {len(predictions)} predictions in {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_inference(args.input, args.output)
