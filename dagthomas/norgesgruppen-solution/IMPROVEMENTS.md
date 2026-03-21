# NorgesGruppen Object Detection -- Improvement Roadmap

> Score = 70% detection mAP + 30% classification mAP (mAP@0.5)
>
> Sandbox: Python 3.11, NVIDIA L4 (24 GB VRAM), 300s timeout, no network access.
> Limits: 420 MB zip, 3 weight files, 10 Python files.

---

## Already Implemented

- [x] All-data YOLO weights (120-epoch training, best checkpoint at epoch 83)
- [x] Inference at imgsz=1536, conf=0.001, max_det=3000
- [x] EfficientNet-B2 classifier with top-K soft ensembling
- [x] GPU-accelerated `roi_align` crop pipeline with padded batch fix
- [x] ConvNeXt-V2 nano classifier (15M params, ImageNet-22k pretrained) — 99.7% train acc, mAP@0.5 0.932→0.938
- [x] 256x256 classifier input resolution (+0.035 small object recall)
- [x] Disable HorizontalFlip in classifier training (product text must stay readable)
- [x] Heavier RandomErasing + augmentation on reference images

## In Progress (2026-03-20)

- [x] SAHI image tiling in run.py — 2×2 tiles at 1280 + full image at 1536, per-class NMS merge
- [x] Background reject class — 1000 crops extracted, train_classifier.py updated to 357 classes
- [ ] YOLO retraining with mixup=0.0 (running in background, 120 epochs)
- [ ] Classifier retraining with 357 classes (after YOLO finishes)

---

## Future Improvements

### Priority 1 -- High Impact

#### 1. Background Reject Class
**Impact: High | Effort: Medium** — **IMPLEMENTED** (pending classifier retrain)

- [x] Add a 357th "background / not-a-product" class to the classifier
- [x] Extract ~1000 random empty-shelf crops from training images (regions not overlapping any annotation)
- [ ] Train classifier with the extra class
- [x] At inference, discard any detection where the classifier's top prediction is "background"
- [x] Serves as an AI-driven false-positive filter, directly pushes precision up

#### 2. Objects365 Pretrained YOLO Weights
**Impact: High | Effort: High**

- [ ] Download `yolov8x-o365.pt` from Ultralytics hub
- [ ] Objects365 has 365 categories including food, packaging, and retail items -- much better transfer-learning base than COCO for grocery shelves
- [ ] Full retraining required (expect several hours on L4)
- [ ] Verify exported weights load correctly under ultralytics==8.1.0

#### 3. Image Tiling / SAHI
**Impact: High | Effort: Medium** — **IMPLEMENTED**

- [x] Slice each image into a 2x2 grid with 15% overlap
- [x] Run YOLO at 1280 on each tile (preserves native resolution of tiny products)
- [x] Merge tile detections using per-class NMS (torchvision.ops.nms)
- [x] Combined with full-image pass at 1536 for multi-scale coverage
- [x] Adaptive timing safety — auto-disables tiling if projected to exceed timeout

#### 4. ArcFace Metric Learning
**Impact: High | Effort: High**

- [ ] Replace Cross-Entropy loss with ArcFace margin-based softmax
- [ ] Better suited for extreme class imbalance (357 classes, some with <= 3 training examples)
- [ ] Maps crops into a well-separated embedding space -- solves few-shot learning for rare categories
- [ ] Common winning strategy in Kaggle retail recognition competitions
- [ ] Validate that the approach works with the existing top-K soft ensembling pipeline

---

### Priority 2 -- Medium Impact

#### 5. Larger Classifier -- ConvNeXt-V2 Tiny
**Impact: Medium | Effort: Low**

- [ ] 27.9M params vs nano's 15M -- richer feature extraction
- [ ] FP16 safetensors: ~56 MB (well within the 420 MB zip limit)
- [ ] Only pursue if nano's inference speed leaves enough headroom in the 300s budget
- [ ] Benchmark: time per crop at 256x256 on L4

#### 6. YOLO Hyperparameter Tuning
**Impact: Medium | Effort: High**

- [ ] Experiment with `copy_paste` augmentation ratio
- [ ] Tune `close_mosaic` timing (currently default)
- [ ] Adjust learning rate schedule (cosine vs step)
- [ ] Sweep IoU threshold for NMS at inference
- [ ] Each experiment = full retraining cycle

#### 7. Model Ensembling
**Impact: Medium | Effort: High**

- [ ] Train multiple YOLO variants (different model sizes, different random seeds)
- [ ] Merge predictions with Weighted Boxes Fusion (WBF)
- [ ] Significant mAP boost expected, but requires multiple training runs
- [ ] All weight files must fit within the 420 MB / 3-file limit -- may need ONNX export to combine

---

### Priority 3 -- Lower Impact / Polish

#### 8. Higher Classifier Resolution (288x288)
**Impact: Low-Medium | Effort: Low**

- [ ] Extra resolution helps read small nutritional text and brand logos
- [ ] Better differentiation of visually similar products (e.g., Diet Coke vs Zero Sugar)
- [ ] Trade-off: slower inference per crop -- profile before committing

#### 9. Classifier Test-Time Augmentation
**Impact: Low | Effort: Low**

- [ ] Multi-crop at inference: original + horizontally flipped + slightly zoomed
- [ ] Average softmax probabilities across augmentations
- [ ] Small mAP boost at ~3x classifier cost per crop
- [ ] Only worthwhile if total inference time is well under 300s

---

## Sandbox Constraints (Quick Reference)

| Constraint | Value |
|---|---|
| GPU | NVIDIA L4, 24 GB VRAM |
| Timeout | 300 seconds |
| Max zip size | 420 MB |
| Max weight files | 3 |
| Max Python files | 10 |
| Python | 3.11 |
| ultralytics | 8.1.0 (8.2+ weights will fail) |
| timm | 0.9.12 (1.0+ weights will fail) |
| PyTorch | 2.6.0+cu124 |
| ONNX opset | <= 20 |
| Network | None (offline sandbox) |
| Blocked imports | `os`, `sys`, `subprocess`, `pickle`, `yaml`, `requests`, `urllib`, `shutil`, `socket`, `ctypes`, `builtins`, `importlib`, `marshal`, `shelve`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`, `http.client` |
| Required patch | `torch.load` monkeypatch (`weights_only=False`) before importing ultralytics |
