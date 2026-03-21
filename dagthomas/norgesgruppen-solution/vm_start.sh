#!/bin/bash
set -e
cd ~

SEEDS="${1:-42,123,256}"
echo "=== VM Start Script ==="
echo "Seeds: $SEEDS"

# Kill old training
pkill -f cloud_train 2>/dev/null || true
pkill -f "python3.*yolo" 2>/dev/null || true
sleep 1

# Clean old runs
rm -rf ~/runs/yolov8x_seed* ~/best_yolov8x_*

# Merge all images into one dir
rm -rf ~/datasets/merged
mkdir -p ~/datasets/merged/images ~/datasets/merged/labels
find ~/datasets/train/images/ -name "*.jpg" -o -name "*.jpeg" | xargs -I{} cp -n {} ~/datasets/merged/images/
find ~/datasets/val/images/ -name "*.jpg" -o -name "*.jpeg" | xargs -I{} cp -n {} ~/datasets/merged/images/
find ~/datasets/train/labels/ -name "*.txt" | xargs -I{} cp -n {} ~/datasets/merged/labels/
find ~/datasets/val/labels/ -name "*.txt" | xargs -I{} cp -n {} ~/datasets/merged/labels/

IMG_COUNT=$(ls ~/datasets/merged/images/*.jpg 2>/dev/null | wc -l)
LBL_COUNT=$(ls ~/datasets/merged/labels/*.txt 2>/dev/null | wc -l)
echo "Merged: $IMG_COUNT images, $LBL_COUNT labels"

# Update data_all.yaml
sed "s|train: .*|train: merged/images|; s|val: .*|val: merged/images|" ~/datasets/data.yaml > ~/datasets/data_all.yaml
echo "data_all.yaml updated"

# Start training
echo "Starting training with seeds=$SEEDS epochs=120"
nohup python3 ~/cloud_train_overnight.py --seeds "$SEEDS" --epochs 120 > ~/overnight_train.log 2>&1 &
echo "PID: $!"
echo "=== Done ==="
