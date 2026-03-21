# NorgesGruppen Data — Object Detection Summary

## Problem
Detect and classify grocery products on store shelf images.
Scoring: **70% detection mAP + 30% classification mAP** (mAP@0.5).

## Training Data
- **248 shelf images** with ~22,700 COCO-format bounding box annotations
- **357 categories** (0–355 named products, 356 = `unknown_product`)
- **327 product reference images** with multi-angle photos (front, back, sides, top, bottom)

## Architecture: Two-Stage Pipeline

```
Image → YOLOv8x Detection (TTA) → Crop Extraction → EfficientNet-B2 Re-classification → JSON
```

### Stage 1: YOLOv8x Detection (targets the 70% detection score)

**Training — Two-phase strategy:**

| Phase | Data | Epochs | Purpose |
|-------|------|--------|---------|
| Validation run | 90/10 stratified split | Up to 300 (patience=50) | Find optimal epoch count |
| Final run | 100% training data | ~120 (1.15× best val epoch) | Maximize detection quality |

**Hyperparameters:**
- Input size: **1280×1280**
- Batch size: **4** (fits L4 24 GB VRAM)
- Optimizer: AdamW, lr=0.001, cosine annealing to lr×0.01
- Augmentation: mosaic=1.0, mixup=0.15, copy_paste=0.1, close_mosaic=30
- Rare categories (≤3 annotations) forced into training set to prevent data loss

**Inference:**
- TTA enabled (augment=True) — ~4 passes per image for better recall
- Confidence threshold: 0.01 (very low to maximize recall)
- NMS IoU: 0.6, max detections: 1000

### Stage 2: EfficientNet-B2 Classifier (targets the 30% classification score)

**Training — Two-phase fine-tuning:**

| Phase | Layers | Epochs | Learning Rate |
|-------|--------|--------|---------------|
| 1. Head-only | Classifier head, frozen backbone | 5 | 3e-4 |
| 2. Full fine-tune | All layers unfrozen | 55 | Backbone 3e-5, Head 3e-4 |

**Training data (with oversampling):**
- Shelf crops from COCO annotations (1× each) — padded 10%
- Product reference images (3× oversampled) — boosts per-product recognition
- Input size: 224×224, ImageNet normalization

**Class balancing:**
- `WeightedRandomSampler` with inverse frequency weights (capped at 50×)
- Label smoothing: 0.1
- Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, Affine, RandomErasing

**Inference:**
- Batch size 128, FP16 inference
- Confidence threshold: **0.3** — only overrides YOLO's class if the classifier is confident enough
- numpy-based preprocessing for speed (no PIL overhead)

## Model Weights

| File | Size | Description |
|------|------|-------------|
| `best.pt` | 137 MB | YOLOv8x detection model |
| `classifier.safetensors` | ~17 MB | EfficientNet-B2 (FP16, safetensors format) |

## Submission Constraints
- **Sandbox:** Python 3.11, L4 GPU (24 GB VRAM), 300s timeout
- **Pinned versions:** ultralytics==8.1.0, timm==0.9.12, PyTorch 2.6.0+cu124
- **Blocked imports:** `os`, `sys`, `subprocess`, `pickle`, `yaml`, `requests`, etc. — use `pathlib` and `json` instead
- **Critical fix:** `torch.load` monkeypatch required (PyTorch 2.6 defaults to `weights_only=True`, breaking ultralytics 8.1.0)
- **Limits:** 420 MB max zip, 3 weight files max, 10 Python files max, 300s timeout, 3 submissions/day

## Files

```
norgesgruppen-solution/
├── run.py                      # Inference entry point (submitted in zip)
├── classifier.py               # EfficientNet-B2 model loader
├── train_yolo.py               # YOLOv8x two-phase training script
├── train_classifier.py         # EfficientNet-B2 fine-tuning script
├── prepare_data.py             # COCO → YOLO format conversion + stratified split
├── prepare_classifier_data.py  # Crop extraction from shelf images + reference image organization
├── best.pt                     # YOLOv8x weights
├── classifier.safetensors      # EfficientNet-B2 weights (FP16)
├── datasets/
│   ├── data.yaml               # YOLO dataset config (90/10 split)
│   └── data_all.yaml           # YOLO dataset config (100% training)
└── classifier_data/
    ├── category_map.json       # Product code → category ID mapping
    ├── crops/                  # Shelf crops by category
    └── refs/                   # Reference product images by category
```
