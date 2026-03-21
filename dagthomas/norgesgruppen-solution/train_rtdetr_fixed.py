"""
RT-DETR training — fixed for ultralytics 8.1.0 + PyTorch 2.7.

The bug: RTDETR("rtdetr-x.pt") loads as DetectionModel (wrong head).
Fix: Load from YAML config, then transfer backbone weights manually.

Usage:
  python3 train_rtdetr_fixed.py --model rtdetr-x --epochs 150
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
from ultralytics.models.rtdetr import RTDETR

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"


def get_all_data_yaml():
    all_data_yaml = DATASET_DIR / "data_all.yaml"
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(all_data_yaml, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("val:"):
                f.write("val: train/images\n")
            else:
                f.write(line)
    return str(all_data_yaml)


def train(model_name="rtdetr-x", epochs=150):
    print("=" * 60)
    print(f"RT-DETR FIXED: {model_name}, {epochs} epochs")
    print("=" * 60)

    # Method 1: Try loading from YAML to get correct RTDETRDetectionModel
    yaml_name = f"{model_name}.yaml"
    pt_name = f"{model_name}.pt"

    # Load pretrained checkpoint to get backbone weights
    pretrained_path = ROOT / pt_name
    if pretrained_path.exists():
        print(f"Loading pretrained weights from {pt_name}...")
        # Load model from YAML config (correct architecture)
        model = RTDETR(yaml_name)
        # Now load pretrained and transfer matching weights
        ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        if "model" in ckpt:
            pretrained_sd = ckpt["model"].float().state_dict()
        else:
            pretrained_sd = ckpt
        model_sd = model.model.state_dict()
        transferred = 0
        for k, v in pretrained_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                model_sd[k] = v
                transferred += 1
        model.model.load_state_dict(model_sd)
        print(f"Transferred {transferred}/{len(model_sd)} layers from pretrained")
    else:
        print(f"No pretrained weights, training from scratch with {yaml_name}")
        model = RTDETR(yaml_name)

    # Verify model type
    print(f"Model type: {type(model.model).__name__}")
    print(f"Head type: {type(model.model.model[-1]).__name__}")

    batch = 8 if "l" in model_name else 4
    results = model.train(
        data=get_all_data_yaml(),
        imgsz=1280,
        batch=batch,
        epochs=epochs,
        patience=0,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name=model_name.replace("-", "_") + "_fixed",
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )

    # Export
    run_name = model_name.replace("-", "_") + "_fixed"
    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    if best.exists():
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        ckpt["optimizer"] = None
        dst = ROOT / f"best_{run_name}.pt"
        torch.save(ckpt, str(dst))
        print(f"Exported {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rtdetr-x")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    train(args.model, args.epochs)
