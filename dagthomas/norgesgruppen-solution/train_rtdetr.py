"""
Train RT-DETR-x (transformer detector) for better generalization.
Transformers have global attention → less spatial overfitting on small datasets.

Usage:
  python train_rtdetr.py --final --epochs 150
  python train_rtdetr.py --export
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


def train_final(model_name="rtdetr-x", epochs=150):
    print("=" * 60)
    print(f"RT-DETR: {model_name}, {epochs} epochs, all data")
    print("=" * 60)

    all_data_yaml = DATASET_DIR / "data_all.yaml"
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(all_data_yaml, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("val:"):
                f.write("val: train/images\n")
            else:
                f.write(line)

    model = RTDETR(f"{model_name}.pt")

    # Fix ultralytics 8.1.0 bug: pretrained RT-DETR loads as DetectionModel
    # which uses v8DetectionLoss (crashes on RTDETRDecoder missing .stride)
    # Monkey-patch to use the correct RTDETRDetectionLoss
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    nc = model.model.model[-1].nc  # Get num classes from decoder
    model.model.init_criterion = lambda: RTDETRDetectionLoss(nc=nc, use_vfl=True)

    batch = 8 if "l" in model_name else 4
    results = model.train(
        data=str(all_data_yaml),
        imgsz=1280,
        batch=batch,
        epochs=epochs,
        patience=0,
        optimizer="AdamW",
        lr0=0.0001,       # Lower LR for transformers
        lrf=0.01,
        cos_lr=True,
        # RT-DETR doesn't use mosaic/mixup — use its built-in augmentation
        amp=True,
        workers=8,
        project=str(ROOT / "runs"),
        name=model_name.replace("-", "_"),
        exist_ok=True,
        save=True,
        save_period=25,
        verbose=True,
    )
    return results


def export_best_weights(model_name="rtdetr-x"):
    run_name = model_name.replace("-", "_")
    best = ROOT / "runs" / run_name / "weights" / "best.pt"
    if best.exists():
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        ckpt["optimizer"] = None
        dst = ROOT / f"best_{run_name}.pt"
        torch.save(ckpt, str(dst))
        print(f"Exported {best} -> {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
    else:
        print(f"WARNING: {best} not found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rtdetr-x")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.export:
        export_best_weights(args.model)
    elif args.final:
        train_final(args.model, args.epochs)
        export_best_weights(args.model)


if __name__ == "__main__":
    main()
