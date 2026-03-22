# NorgesGruppen Object Detection

Detect and classify grocery products on store shelf images using a two-stage ensemble pipeline: multi-model YOLO detection fused with Weighted Boxes Fusion, followed by ConvNeXt-V2 re-classification.

---

## Features

- Dual YOLOv8x detection with test-time augmentation (TTA)
- Weighted Boxes Fusion (WBF) for multi-model ensemble merging
- ConvNeXt-V2 nano re-classification with smart score blending
- Background rejection via trained class 356
- Rare category protection through stratified data splitting
- Sandbox-compatible inference (blocked imports, torch.load patch)
- 285s safety timeout (300s sandbox limit)

---

## User Flows

### Inference Pipeline

```mermaid
flowchart LR
    Input[Shelf Images] --> A[YOLOv8x A<br/>1536px + TTA]
    Input --> B[YOLOv8x B<br/>1280px + TTA]
    A --> WBF[WBF Merge<br/>IoU=0.55]
    B --> WBF
    WBF --> Crop[Extract Crops<br/>256x256]
    Crop --> Cls[ConvNeXt-V2<br/>nano]
    Cls --> Blend[Score Blend<br/>+ BG Reject]
    Blend --> JSON[predictions.json]
```

### Training Pipeline

```mermaid
flowchart TB
    COCO[COCO Annotations] --> Prep[prepare_data.py<br/>COCO to YOLO]
    Prep --> Split[Stratified Split<br/>90/10 val]
    Split --> Phase1[Phase 1: Val Run<br/>Find best epoch]
    Phase1 --> Phase2[Phase 2: Final<br/>100% data, 1.15x epochs]

    COCO --> Crops[prepare_classifier_data.py<br/>Extract crops + refs]
    Crops --> Head[Phase 1: Head-Only<br/>5 epochs]
    Head --> Finetune[Phase 2: Full Fine-tune<br/>55 epochs]
```

---

## Acceptance Criteria

- [x] Detection mAP@0.5 > 0.95 on train set
- [x] Classification mAP@0.5 > 0.95 on train set
- [x] Combined score > 0.97 on train set
- [x] Runs within 300s on L4 GPU
- [x] Zip under 420 MB
- [x] No blocked imports
- [x] torch.load monkeypatch applied before ultralytics import
- [ ] Test set combined score > 0.95

---

## Edge Cases

- Products with identical packaging but different sizes (e.g., 500g vs 1kg)
- Occluded products where only partial label is visible
- Empty shelf regions triggering false positive detections (mitigated by BG class)
- Rare categories with 1-3 training examples (protected by stratified split)
- TTA timeout risk on large images (mitigated by 285s safety cutoff)
