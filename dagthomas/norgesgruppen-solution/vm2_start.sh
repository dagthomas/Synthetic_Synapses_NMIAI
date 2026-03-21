#!/bin/bash
cd ~
pkill -f cloud_train 2>/dev/null || true
pkill -f "python3.*yolo" 2>/dev/null || true
sleep 2
rm -rf ~/runs/yolov8x_* ~/best_*
rm -f ~/datasets/merged/labels.cache
echo "Starting improved training (1536, 150ep, seeds 7,99)"
nohup python3 ~/cloud_train_improved.py --seeds 7,99 --epochs 150 > ~/overnight_train.log 2>&1 &
echo "PID: $!"
