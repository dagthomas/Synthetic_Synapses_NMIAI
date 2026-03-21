# YOLO Ensemble Training

Train multiple YOLOv8 variants with different seeds/sizes, merge with Weighted Boxes Fusion.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Upload your dataset — you need the NM_NGD_coco_dataset/train folder
#    containing images/ and annotations.json

# 3. Prepare YOLO-format data
python prepare_data.py --coco-dir /path/to/NM_NGD_coco_dataset/train

# 4. Train all variants (runs sequentially, ~6-8h total on A100)
python train_ensemble.py

# 5. Evaluate individual models + ensemble
python evaluate_ensemble.py --coco-dir /path/to/NM_NGD_coco_dataset/train

# 6. Export best combination for submission
python export_for_submission.py
```

## Variants

| # | Name | Model | Seed | Epochs | Batch | Est. time (A100) |
|---|------|-------|------|--------|-------|-------------------|
| 0 | yolov8x_seed0 | YOLOv8x | 0 | 120 | 4 | ~90 min |
| 1 | yolov8x_seed42 | YOLOv8x | 42 | 120 | 4 | ~90 min |
| 2 | yolov8l_seed0 | YOLOv8l | 0 | 150 | 6 | ~60 min |
| 3 | yolov8x_seed7 | YOLOv8x | 7 | 120 | 4 | ~90 min |

### Why these variants?

- **Different seeds** produce diverse detectors (different augmentation order, weight init noise). WBF benefits most from diversity — if all models make the same mistakes, ensembling doesn't help.
- **Different model sizes** (x vs l) see different feature scales. The lighter model often catches patterns the heavy model overfits past.
- **mixup=0.0** across all variants — mixup creates ghost products on dense grocery shelves.

## Training Options

```bash
# Train specific variants only
python train_ensemble.py --variants 0 1

# Override epoch count
python train_ensemble.py --epochs 200

# Use 90/10 validation split (for experimentation)
python train_ensemble.py --val-split

# Resume a crashed run
python train_ensemble.py --resume yolov8x_seed42

# List available variants
python train_ensemble.py --list
```

## Evaluation

```bash
# Evaluate all trained models + WBF ensemble
python evaluate_ensemble.py --coco-dir /path/to/NM_NGD_coco_dataset/train

# Evaluate specific weights only
python evaluate_ensemble.py --coco-dir /data/train \
  --weights weights/yolov8x_seed0.pt weights/yolov8l_seed0.pt

# Sweep WBF IoU threshold to find optimal value
python evaluate_ensemble.py --coco-dir /data/train --sweep-iou

# Save ensemble predictions
python evaluate_ensemble.py --coco-dir /data/train --save ensemble_preds.json
```

## Export for Submission

```bash
# Auto-select best models that fit in 420 MB
python export_for_submission.py

# Manually pick models
python export_for_submission.py --pick yolov8x_seed0 yolov8x_seed42

# Custom WBF IoU
python export_for_submission.py --wbf-iou 0.50
```

This generates an `export/` folder containing:
- Weight files (stripped of optimizer state)
- `run.py` with built-in WBF (no external dependencies needed in sandbox)

### Submission Constraints

| Constraint | Limit | Notes |
|---|---|---|
| Max zip size | 420 MB | Budget ~380 MB for YOLO weights (30 MB for classifier) |
| Max weight files | 3 | Pick best 2-3 models |
| Max Python files | 10 | Generated run.py is self-contained |
| Timeout | 300s on L4 GPU | N models × 248 images must finish in time |
| ultralytics | 8.1.0 | 8.2+ weights won't load |

### Timing Budget

Each YOLO forward pass at 1536 takes ~0.4s on L4. With 248 images:

| Models | Total passes | Est. time | Fits? |
|---|---|---|---|
| 2 | 496 | ~200s | Yes |
| 3 | 744 | ~300s | Tight |
| 4 | 992 | ~400s | No |

**Recommendation:** Use 2 models for safe timing. 3 models only if you've measured < 0.35s/pass on L4.

## File Structure

```
ensemble-training/
├── README.md
├── requirements.txt
├── prepare_data.py          # COCO → YOLO format conversion
├── train_ensemble.py        # Train multiple YOLO variants
├── evaluate_ensemble.py     # Evaluate models + WBF ensemble
├── export_for_submission.py # Package for submission
├── datasets/                # Created by prepare_data.py
│   ├── data.yaml
│   ├── data_all.yaml
│   ├── train/
│   └── val/
├── runs/                    # Training output (ultralytics format)
├── weights/                 # Stripped best.pt per variant
└── export/                  # Submission-ready package
```

## Downloading Results

After training, download these files from the cloud instance:
```bash
# Just the stripped weights (small)
scp -r instance:ensemble-training/weights/ ./

# Or the full export package
scp -r instance:ensemble-training/export/ ./
```

Then integrate into your main solution:
1. Copy the selected `.pt` files into `norgesgruppen-solution/`
2. Use the generated `run.py` or integrate WBF into your existing pipeline
