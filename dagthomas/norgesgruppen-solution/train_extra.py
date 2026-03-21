"""
Extra training jobs for ensemble diversity.

Job 1: YOLOv8x with different seed (seed=42) — different initialization for WBF
Job 2: YOLOv8x at imgsz=640 — fast model, different scale features
Job 3: ConvNeXt-V2 tiny classifier (larger, better than nano)

Usage:
  python train_extra.py --job yolo_seed42
  python train_extra.py --job yolo_640
  python train_extra.py --job cls_tiny
  python train_extra.py --job all  # run all sequentially
"""

import argparse
from pathlib import Path

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"


def get_all_data_yaml():
    """Create/return data_all.yaml pointing val at train (for final training)."""
    all_data_yaml = DATASET_DIR / "data_all.yaml"
    if not all_data_yaml.exists():
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(all_data_yaml, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("val:"):
                    f.write("val: train/images\n")
                else:
                    f.write(line)
    return str(all_data_yaml)


def train_yolo_seed42(epochs=150):
    """YOLOv8x with seed=42 — different initialization for ensemble diversity."""
    print("=" * 60)
    print(f"YOLOv8x seed=42: {epochs} epochs, all data")
    print("=" * 60)

    model = YOLO("yolov8x.pt")
    results = model.train(
        data=get_all_data_yaml(),
        imgsz=1280,
        batch=4,
        epochs=epochs,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,        # slight mixup for diversity
        copy_paste=0.15,
        close_mosaic=30,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        erasing=0.2,
        seed=42,           # KEY: different seed
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name="yolov8x_seed42",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )
    export_weights("yolov8x_seed42", "best_seed42.pt")
    return results


def train_yolo_640(epochs=200):
    """YOLOv8x at 640px — faster, different scale detection strengths."""
    print("=" * 60)
    print(f"YOLOv8x 640px: {epochs} epochs, all data")
    print("=" * 60)

    model = YOLO("yolov8x.pt")
    results = model.train(
        data=get_all_data_yaml(),
        imgsz=640,
        batch=16,          # 640px allows larger batch
        epochs=epochs,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        close_mosaic=30,
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name="yolov8x_640",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )
    export_weights("yolov8x_640", "best_640.pt")
    return results


def train_classifier_tiny(epochs=80, batch_size=64):
    """ConvNeXt-V2 tiny classifier — larger model for better classification."""
    print("=" * 60)
    print(f"ConvNeXt-V2 tiny classifier: {epochs} epochs")
    print("=" * 60)

    # Import classifier training (reuse existing code)
    import json
    from collections import Counter
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    import timm
    from PIL import Image
    from safetensors.torch import save_file

    CLASSIFIER_DATA = ROOT / "classifier_data"
    NUM_CLASSES = 357
    model_name = "convnextv2_tiny"
    input_size = 256

    # Import dataset class from train_classifier
    from train_classifier import CropDataset, get_class_weights, build_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}, Input: {input_size}x{input_size}")

    crop_transform = build_transforms(input_size, train=True, is_ref=False)
    ref_transform = build_transforms(input_size, train=True, is_ref=True)

    train_dataset = CropDataset(
        roots=[
            (CLASSIFIER_DATA / "crops", 1, False),
            (CLASSIFIER_DATA / "refs", 3, True),
        ],
        transform=crop_transform,
        ref_transform=ref_transform,
    )
    print(f"Training samples: {len(train_dataset)}")

    sample_weights = get_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    print(f"Loading {model_name} pretrained weights...")
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.1f}M")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    head_names = ["head", "classifier", "fc"]
    def is_head_param(name):
        return any(h in name for h in head_names)

    freeze_epochs = 5
    print(f"\n--- Phase 1: Frozen backbone ({freeze_epochs} epochs) ---")
    for name, param in model.named_parameters():
        if not is_head_param(name):
            param.requires_grad = False

    best_loss = float("inf")

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            print(f"\n--- Phase 2: Unfreezing all layers ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if not is_head_param(n)],
                 "lr": 3e-4 * 0.1},
                {"params": [p for n, p in model.named_parameters() if is_head_param(n)],
                 "lr": 3e-4},
            ], weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6
            )

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.1f}%")

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.1f}%")

        if epoch >= freeze_epochs:
            scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ROOT / "classifier_tiny_best.pth")
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

    # Export to safetensors FP16
    print("\nExporting to safetensors FP16...")
    model.load_state_dict(torch.load(ROOT / "classifier_tiny_best.pth", map_location="cpu"))
    model = model.half().cpu()
    state_dict = {k: v for k, v in model.state_dict().items()}
    save_file(state_dict, ROOT / "classifier_tiny.safetensors")
    size_mb = (ROOT / "classifier_tiny.safetensors").stat().st_size / 1024 / 1024
    print(f"Saved classifier_tiny.safetensors ({size_mb:.1f} MB)")

    config = {"model_name": model_name, "input_size": input_size, "num_classes": NUM_CLASSES}
    with open(ROOT / "classifier_tiny_config.json", "w") as f:
        json.dump(config, f)


def export_weights(run_name, output_name):
    """Strip optimizer state and save best weights."""
    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    if best.exists():
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        ckpt["optimizer"] = None
        dst = ROOT / output_name
        torch.save(ckpt, str(dst))
        print(f"Exported {best} -> {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
    else:
        print(f"WARNING: {best} not found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True,
                        choices=["yolo_seed42", "yolo_640", "cls_tiny", "all"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.job == "yolo_seed42" or args.job == "all":
        train_yolo_seed42(args.epochs or 150)
    if args.job == "yolo_640" or args.job == "all":
        train_yolo_640(args.epochs or 200)
    if args.job == "cls_tiny" or args.job == "all":
        train_classifier_tiny(args.epochs or 80)


if __name__ == "__main__":
    main()
