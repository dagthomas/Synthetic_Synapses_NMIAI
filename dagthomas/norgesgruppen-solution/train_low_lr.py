"""YOLO v3: Lower LR, more epochs for max performance."""
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
    all_data_yaml = DATASET_DIR / "data_all.yaml"
    if not all_data_yaml.exists():
        with open(DATA_YAML, "r") as f:
            lines = f.readlines()
        with open(all_data_yaml, "w") as f:
            for line in lines:
                f.write("val: train/images\n" if line.startswith("val:") else line)
    return str(all_data_yaml)

model = YOLO("yolov8x.pt")
model.train(
    data=get_all_data_yaml(),
    imgsz=1280,
    batch=4,
    epochs=300,
    patience=0,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.005,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.05,
    copy_paste=0.2,
    close_mosaic=50,
    degrees=10.0,
    translate=0.15,
    scale=0.6,
    fliplr=0.5,
    erasing=0.3,
    amp=True,
    workers=8,
    project=str(ROOT / "runs"),
    name="yolov8x_lowlr",
    exist_ok=True,
    save=True,
    save_period=50,
    verbose=True,
)

best = ROOT / "runs/yolov8x_lowlr/weights/best.pt"
if best.exists():
    c = torch.load(str(best), map_location="cpu", weights_only=False)
    c["optimizer"] = None
    torch.save(c, ROOT / "best_lowlr.pt")
    print("Exported best_lowlr.pt")
