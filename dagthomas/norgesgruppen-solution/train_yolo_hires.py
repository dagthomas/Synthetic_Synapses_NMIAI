"""YOLOv8x trained at 1920px — learns finer product features.
Infer at 1280-1536 on L4, but benefits from high-res training on A100."""
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
    imgsz=1920,
    batch=2,              # 1920px needs batch=2 on A100 40GB
    epochs=150,
    patience=0,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.15,
    close_mosaic=30,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    erasing=0.2,
    amp=True,
    workers=8,
    project=str(ROOT / "runs"),
    name="yolov8x_hires",
    exist_ok=True,
    save=True,
    save_period=25,
    verbose=True,
)

best = ROOT / "runs/yolov8x_hires/weights/best.pt"
if best.exists():
    c = torch.load(str(best), map_location="cpu", weights_only=False)
    c["optimizer"] = None
    torch.save(c, ROOT / "best_hires.pt")
    print("Exported best_hires.pt")
