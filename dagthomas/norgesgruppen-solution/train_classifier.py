"""
Train ConvNeXt-V2 nano classifier on shelf crops + product reference images.

Key improvements over EfficientNet-B2 baseline:
- ConvNeXt-V2 nano (15M params, ImageNet-22k pretrained) — much better fine-grained classification
- 256x256 resolution — reads small text and logos
- No HorizontalFlip — preserves brand name text features
- Heavier RandomErasing (p=0.5) — bridges background domain gap for reference images
- Separate transforms for crops vs refs — refs get extra augmentation

Usage:
  python train_classifier.py
  python train_classifier.py --epochs 80 --batch-size 64
  python train_classifier.py --model efficientnet_b2 --input-size 224  # fallback
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
CLASSIFIER_DATA = ROOT / "classifier_data"
NUM_CLASSES = 357  # 356 product classes (0-355) + 1 background class (356)


class CropDataset(Dataset):
    """Dataset loading crops and reference images organized by category_id."""

    def __init__(self, roots, transform=None, ref_transform=None):
        """
        Args:
            roots: list of (dir_path, weight, is_ref) tuples.
            transform: transform for shelf crops
            ref_transform: transform for reference images (heavier augmentation)
        """
        self.samples = []  # (path, label, is_ref)
        self.transform = transform
        self.ref_transform = ref_transform

        for root_dir, weight, is_ref in roots:
            root_dir = Path(root_dir)
            if not root_dir.exists():
                print(f"WARNING: {root_dir} not found, skipping")
                continue
            for cat_dir in sorted(root_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                try:
                    cat_id = int(cat_dir.name)
                except ValueError:
                    continue
                for img_file in cat_dir.iterdir():
                    if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        for _ in range(weight):
                            self.samples.append((str(img_file), cat_id, is_ref))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_ref = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if is_ref and self.ref_transform:
            img = self.ref_transform(img)
        elif self.transform:
            img = self.transform(img)
        return img, label


def get_class_weights(dataset):
    """Compute inverse-frequency weights for WeightedRandomSampler, capped at 50x."""
    counts = Counter()
    for _, label, _ in dataset.samples:
        counts[label] += 1

    max_count = max(counts.values())
    weights = []
    for _, label, _ in dataset.samples:
        w = min(max_count / counts[label], 50.0)
        weights.append(w)

    return weights


def build_transforms(input_size, train=True, is_ref=False):
    """Build transforms.

    Key design decisions:
    - NO HorizontalFlip: grocery products have text/logos that flipping destroys
    - Heavier RandomErasing for refs: bridges white-background → shelf-background domain gap
    - RandomResizedCrop with wider scale range for refs: simulates varied crop quality
    """
    if not train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if is_ref:
        # Reference images: heavier augmentation to bridge domain gap
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
            # NO HorizontalFlip — preserves text features
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.1, 0.4)),  # Heavy erasing — simulates occlusion
        ])
    else:
        # Shelf crops: moderate augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            # NO HorizontalFlip — preserves text features
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])


def train_model(model_name="convnextv2_nano", input_size=256,
                epochs=60, batch_size=64, lr=3e-4, freeze_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}, Input: {input_size}x{input_size}, Classes: {NUM_CLASSES}")

    # Build datasets with separate transforms for crops vs refs
    crop_transform = build_transforms(input_size, train=True, is_ref=False)
    ref_transform = build_transforms(input_size, train=True, is_ref=True)

    train_dataset = CropDataset(
        roots=[
            (CLASSIFIER_DATA / "crops", 1, False),
            (CLASSIFIER_DATA / "refs", 3, True),  # Oversample refs 3x
        ],
        transform=crop_transform,
        ref_transform=ref_transform,
    )

    print(f"Training samples: {len(train_dataset)}")

    # Count crop vs ref
    n_crops = sum(1 for _, _, is_ref in train_dataset.samples if not is_ref)
    n_refs = sum(1 for _, _, is_ref in train_dataset.samples if is_ref)
    print(f"  Shelf crops: {n_crops}, Reference images: {n_refs}")

    # Weighted sampler for class balance
    sample_weights = get_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model — ConvNeXt-V2 nano pretrained on ImageNet-22k
    print(f"Loading {model_name} pretrained weights...")
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.1f}M")
    model = model.to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Identify head layer name (varies by model architecture)
    head_names = ["head", "classifier", "fc"]
    def is_head_param(name):
        return any(h in name for h in head_names)

    # Phase 1: Freeze backbone, train head only
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
            # Reset optimizer with lower LR for backbone
            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if not is_head_param(n)],
                 "lr": lr * 0.1},
                {"params": [p for n, p in model.named_parameters() if is_head_param(n)],
                 "lr": lr},
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

            # Gradient clipping for stability
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
        lr_current = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.1f}% LR: {lr_current:.6f}")

        if epoch >= freeze_epochs:
            scheduler.step()

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ROOT / "classifier_best.pth")
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

    # Export to safetensors FP16
    print("\nExporting to safetensors FP16...")
    model.load_state_dict(torch.load(ROOT / "classifier_best.pth", map_location="cpu"))
    model = model.half().cpu()

    state_dict = {k: v for k, v in model.state_dict().items()}
    save_file(state_dict, ROOT / "classifier.safetensors")

    size_mb = (ROOT / "classifier.safetensors").stat().st_size / 1024 / 1024
    print(f"Saved classifier.safetensors ({size_mb:.1f} MB)")

    # Save model config for inference
    config = {"model_name": model_name, "input_size": input_size, "num_classes": NUM_CLASSES}
    with open(ROOT / "classifier_config.json", "w") as f:
        json.dump(config, f)
    print(f"Saved classifier_config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="convnextv2_nano",
                        help="timm model name (default: convnextv2_nano)")
    parser.add_argument("--input-size", type=int, default=256,
                        help="Input resolution (default: 256)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
    )


if __name__ == "__main__":
    main()
