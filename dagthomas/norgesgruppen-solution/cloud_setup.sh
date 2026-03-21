#!/bin/bash
# Quick setup for Google Compute Engine GPU VM
# Run this after SSH-ing into the VM

# === Option 1: Create VM with gcloud ===
# gcloud compute instances create nmiai-train \
#   --zone=europe-west4-a \
#   --machine-type=a2-highgpu-1g \
#   --accelerator=type=nvidia-tesla-a100,count=1 \
#   --image-family=pytorch-latest-gpu \
#   --image-project=deeplearning-platform-release \
#   --boot-disk-size=200GB \
#   --maintenance-policy=TERMINATE

# === Option 2: Use Deep Learning VM (pre-installed PyTorch) ===
# gcloud compute instances create nmiai-train \
#   --zone=europe-west4-a \
#   --machine-type=g2-standard-12 \
#   --accelerator=type=nvidia-l4,count=1 \
#   --image-family=pytorch-2-4-cu124-debian-12 \
#   --image-project=deeplearning-platform-release \
#   --boot-disk-size=200GB

# === After SSH into VM ===

# 1. Install exact versions matching sandbox
pip install ultralytics==8.1.0 timm==0.9.12 safetensors pycocotools

# 2. Upload training data (from local machine)
# gcloud compute scp --recurse norgesgruppen-solution/datasets nmiai-train:~/datasets
# gcloud compute scp --recurse norgesgruppen-solution/*.py nmiai-train:~/
# gcloud compute scp norgesgruppen-solution/yolov8x.pt nmiai-train:~/
# gcloud compute scp norgesgruppen-solution/rtdetr-x.pt nmiai-train:~/
# gcloud compute scp norgesgruppen-solution/rtdetr-l.pt nmiai-train:~/

# 3. Run training jobs
# RT-DETR-x (highest priority — ~2h on A100)
# python train_rtdetr.py --final --epochs 150 --model rtdetr-x

# YOLO with different seed (for ensemble diversity)
# python train_yolo.py --final --epochs 120  # with seed modification

# 4. Download weights when done
# gcloud compute scp nmiai-train:~/runs/rtdetr_x/weights/best.pt ./best_rtdetr_x.pt
