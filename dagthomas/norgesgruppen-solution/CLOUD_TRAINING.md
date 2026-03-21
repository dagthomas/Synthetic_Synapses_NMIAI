# GCE GPU Cloud Training Guide

## Overview

We use Google Compute Engine A100 40GB VMs to train models in parallel, then download weights for local submission building.

**GCP Project:** `ai-nm26osl-1721`
**Cost:** ~$3.67/hr per A100 VM — always delete when done!

## Prerequisites

- gcloud CLI installed: `winget install Google.CloudSDK`
- Path: `C:\Users\dagth\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin`
- Authenticated: `gcloud auth login`

For all commands below, ensure gcloud is on PATH:
```bash
export PATH="/c/Users/dagth/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin:$PATH"
```

## 1. Create GPU VM

```bash
# A100 40GB — best available for training
gcloud compute instances create nmiai-train \
  --zone=us-central1-f \
  --machine-type=a2-highgpu-1g \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE

# If stockout, try other zones:
# us-central1-a/b/c, us-east1-b, us-west1-b, us-west4-b
# europe-west4-a/b, asia-southeast1-a/b/c
```

**GPU Quotas (europe-west4):** A100 40GB: 16, L4: 16, V100: 8, T4: 8

## 2. Install Dependencies

```bash
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1 &&
  pip install ultralytics==8.1.0 timm==0.9.12 safetensors pycocotools 2>&1 | tail -3'
```

**Critical versions:** ultralytics==8.1.0 (newer weights incompatible), timm==0.9.12

## 3. Upload Data

### Option A: From local machine (tarball — avoids pscp issues)
```bash
# Create tarballs locally
cd X:/KODE/AINMNO/norgesgruppen-solution
tar czf /tmp/datasets.tar.gz datasets
tar czf /tmp/classifier_data.tar.gz classifier_data

# Upload (use explicit /home/dagth/ path, NOT ~/)
gcloud compute scp /tmp/datasets.tar.gz nmiai-train:/home/dagth/ --zone=us-central1-f
gcloud compute scp /tmp/classifier_data.tar.gz nmiai-train:/home/dagth/ --zone=us-central1-f

# Extract on VM
gcloud compute ssh nmiai-train --zone=us-central1-f --command='cd ~ && tar xzf datasets.tar.gz && tar xzf classifier_data.tar.gz'
```

### Fix data.yaml paths (REQUIRED!)
The local data.yaml has Windows paths. Fix them:
```bash
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  sed -i "s|path:.*|path: /home/dagth/datasets|" ~/datasets/data.yaml'
```

### Upload scripts and weights
```bash
# Scripts
for f in train_rtdetr.py train_yolo_v2.py train_yolo.py train_classifier.py \
         train_extra.py prepare_data.py classifier.py classifier_config.json; do
  gcloud compute scp "X:/KODE/AINMNO/norgesgruppen-solution/$f" nmiai-train:/home/dagth/ --zone=us-central1-f
done

# Pretrained weights
for f in yolov8x.pt rtdetr-x.pt rtdetr-l.pt; do
  gcloud compute scp "X:/KODE/AINMNO/norgesgruppen-solution/$f" nmiai-train:/home/dagth/ --zone=us-central1-f
done
```

## 4. Training

### Important: Use `python3` (not `python`) and `nohup` for background
```bash
# Single job
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  cd $HOME && nohup python3 train_rtdetr.py --final --epochs 150 --model rtdetr-x > rtdetr_x_train.log 2>&1 &'

# Chained jobs (sequential — one GPU, one job at a time!)
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  cd $HOME && nohup bash -c "
    python3 train_rtdetr.py --final --epochs 150 --model rtdetr-x > rtdetr_x_train.log 2>&1 &&
    python3 train_rtdetr.py --final --epochs 80 --model rtdetr-l > rtdetr_l_train.log 2>&1
  " > /dev/null 2>&1 &'
```

### Available training jobs

| Script | Command | ~Time (A100) | Output |
|--------|---------|-------------|--------|
| RT-DETR-x | `python3 train_rtdetr.py --final --epochs 150 --model rtdetr-x` | ~1.5h | `runs/rtdetr_x/weights/best.pt` |
| RT-DETR-l | `python3 train_rtdetr.py --final --epochs 80 --model rtdetr-l` | ~45m | `runs/rtdetr_l/weights/best.pt` |
| YOLO v2 | `python3 train_yolo_v2.py --final --epochs 200` | ~1h | `runs/yolov8x_v2/weights/best.pt` |
| YOLO v1 | `python3 train_yolo.py --final --epochs 150` | ~45m | `runs/yolov8x_final/weights/best.pt` |
| YOLO seed42 | `python3 train_extra.py --job yolo_seed42 --epochs 150` | ~45m | `runs/yolov8x_seed42/weights/best.pt` |
| YOLO 640px | `python3 train_extra.py --job yolo_640 --epochs 200` | ~20m | `runs/yolov8x_640/weights/best.pt` |
| Classifier nano | `python3 train_classifier.py --epochs 60 --batch-size 128` | ~20m | `classifier.safetensors` |
| Classifier tiny | `python3 train_extra.py --job cls_tiny --epochs 80` | ~30m | `classifier_tiny.safetensors` |

### Batch sizes on A100 40GB
- YOLO 1280px: batch=4 (same as L4 — GPU bottleneck is VRAM per image)
- YOLO 640px: batch=16
- RT-DETR-x 1280px: batch=4
- RT-DETR-l 1280px: batch=8
- Classifier: batch=128

## 5. Monitor Progress

```bash
# Check epoch progress
gcloud compute ssh nmiai-train --zone=us-central1-f --command="
  grep -oP '\d+/150' ~/rtdetr_x_train.log | tail -1 &&
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# Tail the log
gcloud compute ssh nmiai-train --zone=us-central1-f --command="tail -5 ~/rtdetr_x_train.log"

# Check all processes
gcloud compute ssh nmiai-train --zone=us-central1-f --command="ps aux | grep python3 | grep -v grep"

# Check extra jobs queue status
gcloud compute ssh nmiai-train --zone=us-central1-f --command="cat ~/extra_jobs.log"
```

## 6. Download Weights

```bash
# RT-DETR-x (strips optimizer automatically via export in script)
gcloud compute scp nmiai-train:/home/dagth/best_rtdetr_x.pt \
  "X:/KODE/AINMNO/norgesgruppen-solution/best_rtdetr_x.pt" --zone=us-central1-f

# YOLO v2
gcloud compute scp nmiai-train2:/home/dagth/best_v2.pt \
  "X:/KODE/AINMNO/norgesgruppen-solution/best_v2.pt" --zone=us-east1-b

# Or download raw weights and strip locally:
gcloud compute scp nmiai-train:/home/dagth/runs/rtdetr_x/weights/best.pt \
  "X:/KODE/AINMNO/norgesgruppen-solution/best_rtdetr_x_raw.pt" --zone=us-central1-f

# Strip optimizer locally
python -c "import torch; c=torch.load('best_rtdetr_x_raw.pt',map_location='cpu',weights_only=False); c['optimizer']=None; torch.save(c,'best_rtdetr_x.pt')"
```

### Expected weight sizes (after optimizer stripping)
- YOLOv8x: ~131 MB
- RT-DETR-x: ~131 MB
- RT-DETR-l: ~64 MB
- Classifier nano: ~30 MB (safetensors FP16)
- Classifier tiny: ~50 MB (safetensors FP16)

## 7. Build Submission

The ensemble uses `run_ensemble.py` with:
- `best.pt` — YOLO detector
- `best_rtdetr_x.pt` — RT-DETR detector
- `classifier.safetensors` — ConvNeXt classifier

```bash
# Verify weights work locally
python evaluate.py

# Build submission zip
cd norgesgruppen-solution
zip -r ../submissions/submission_ensemble.zip \
  run_ensemble.py classifier.py classifier_config.json \
  best.pt best_rtdetr_x.pt classifier.safetensors \
  -x ".*" "__MACOSX/*"
```

**Submission limits:** 420 MB max zip, 3 weight files max, 10 Python files max

## 8. Cleanup (IMPORTANT!)

```bash
# Delete VMs when done — A100 costs ~$3.67/hr each!
gcloud compute instances delete nmiai-train --zone=us-central1-f --quiet
gcloud compute instances delete nmiai-train2 --zone=us-east1-b --quiet

# Verify
gcloud compute instances list
```

## Current Active VMs (2026-03-20)

| VM | Zone | GPU | Current Job | Queued |
|---|---|---|---|---|
| `nmiai-train` | us-central1-f | A100 40GB | RT-DETR-x (150ep) → RT-DETR-l (80ep) | YOLO seed42, YOLO 640 |
| `nmiai-train2` | us-east1-b | A100 40GB | YOLO v2 (200ep) → Classifier nano (80ep) | Classifier tiny, YOLO seed42 |

## What We Train and Why

### Detection Models (for WBF ensemble)
- **YOLOv8x** (CNN): Strong on dense shelf scenes, good with augmentation
- **RT-DETR-x** (Transformer): Global attention catches things YOLO misses, complementary errors
- **YOLOv8x seed42**: Same architecture, different initialization — pure ensemble diversity
- **YOLOv8x 640px**: Different receptive field scale — catches different sized products

### Classification Models (re-classify detections)
- **ConvNeXt-V2 nano** (15M params): Fast, good baseline, 256x256 input
- **ConvNeXt-V2 tiny** (28M params): Larger capacity for 357 fine-grained classes

### Key Training Decisions
- **imgsz=1280**: Matches competition sandbox L4 inference size
- **All-data training** (no val split): Maximize training signal on small dataset (248 images)
- **patience=0**: No early stopping — train full epochs with cosine LR
- **No HorizontalFlip on classifier**: Preserves text/logo features on grocery products
- **copy_paste augmentation**: Critical for dense shelf scenes with many similar objects

## Troubleshooting

### SSH fails with plink error
Try reconnecting. If persistent, clear the host key cache:
```bash
# Windows: delete key from registry
reg delete "HKCU\Software\SimonTatham\PuTTY\SshHostKeys" /v "rsa2@22:34.63.149.250" /f
```

### pscp "unable to open ~/" error
Always use explicit paths (`/home/dagth/`) instead of `~/` for scp targets.

### "python: command not found" on VM
Use `python3` — the Deep Learning VM has Python 3.10 at `/usr/bin/python3`.

### "libGL.so.1 not found"
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Windows paths in data.yaml
```bash
sed -i "s|path:.*|path: /home/dagth/datasets|" ~/datasets/data.yaml
```
