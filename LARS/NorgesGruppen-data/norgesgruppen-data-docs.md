# NorgesGruppen Data: Object Detection

Detect grocery products on store shelves. Upload your model code as a `.zip` file — it runs in a sandboxed Docker container on our servers.

---

## How It Works

1. Download the training data from the competition website (requires login)
2. Train your object detection model locally
3. Write a `run.py` that takes shelf images as input and outputs predictions
4. Zip your code + model weights
5. Upload at the submit page
6. Our server runs your code in a sandbox with GPU (NVIDIA L4, 24 GB VRAM) — no network access
7. Your predictions are scored: **70% detection** (did you find products?) + **30% classification** (did you identify the right product?)
8. Score appears on the leaderboard

> Download training data and product reference images from the **Submit** page on the competition website (login required).

---

## Training Data

### COCO Dataset (`NM_NGD_coco_dataset.zip`, ~864 MB)

- 248 shelf images from Norwegian grocery stores
- ~22,700 COCO-format bounding box annotations
- 356 product categories (`category_id` 0–355)
- Images from 4 store sections: Egg, Frokost, Knekkebrød, Varmedrikker

### Product Reference Images (`NM_NGD_product_images.zip`, ~60 MB)

- 327 individual products with multi-angle photos (main, front, back, left, right, top, bottom)
- Organized by barcode: `{product_code}/main.jpg`, `{product_code}/front.jpg`, etc.
- Includes `metadata.json` with product names and annotation counts

### Annotation Format

```json
{
  "images": [
    {"id": 1, "file_name": "img_00001.jpg", "width": 2000, "height": 1500}
  ],
  "categories": [
    {"id": 0, "name": "VESTLANDSLEFSA TØRRE 10STK 360G", "supercategory": "product"},
    {"id": 1, "name": "COFFEE MATE 180G NESTLE", "supercategory": "product"},
    ...
    {"id": 356, "name": "unknown_product", "supercategory": "product"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 42,
      "bbox": [141, 49, 169, 152],
      "area": 25688,
      "iscrowd": 0,
      "product_code": "8445291513365",
      "product_name": "NESCAFE VANILLA LATTE 136G NESTLE",
      "corrected": true
    }
  ]
}
```

- `bbox` is `[x, y, width, height]` in pixels (COCO format)
- `product_code` is the barcode
- `corrected` indicates manually verified annotations

---

## Submission Format

### Zip Structure

`run.py` **must be at the root** of the zip — not inside a subfolder. This is the most common submission error.

```
submission.zip
├── run.py          # Required: entry point
├── model.onnx      # Optional: model weights (.pt, .onnx, .safetensors, .npy)
└── utils.py        # Optional: helper code
```

| Limit | Value |
|-------|-------|
| Max zip size (uncompressed) | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (.pt, .pth, .onnx, .safetensors, .npy) | 3 |
| Max weight size total | 420 MB |
| Allowed file types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

### `run.py` Contract

Your script is executed as:

```bash
python run.py --input /data/images --output /output/predictions.json
```

**Input:** `/data/images/` contains JPEG shelf images named `img_XXXXX.jpg` (e.g. `img_00042.jpg`)

**Output:** Write a JSON array to the `--output` path:

```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | int | Numeric ID from filename (`img_00042.jpg` → `42`) |
| `category_id` | int | Product category ID (0–355). See `categories` in `annotations.json` |
| `bbox` | [x, y, w, h] | Bounding box in COCO format |
| `score` | float | Confidence score (0–1) |

### Creating Your Zip

**Linux / macOS:**
```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```

**Windows (PowerShell):**
```powershell
cd my_submission
Compress-Archive -Path .\* -DestinationPath ..\submission.zip
```

> ⚠️ Do **not** right-click → Compress/Send to ZIP — both nest files inside a subfolder.

**Verify:**
```bash
unzip -l submission.zip | head -10
# You should see run.py directly — not my_submission/run.py
```

---

## Scoring

```
Score = 0.7 × detection_mAP + 0.3 × classification_mAP
```

Both use **mAP@0.5** (Mean Average Precision at IoU threshold 0.5).

### Detection mAP — 70%

- Each prediction is matched to the closest ground truth box
- True positive if **IoU ≥ 0.5** (category ignored)
- Rewards accurate bounding box localization

### Classification mAP — 30%

- True positive if **IoU ≥ 0.5 AND `category_id` matches** ground truth
- 356 product categories (IDs 0–355)

### Detection-Only Strategy

Set `category_id: 0` for all predictions → score up to **0.70**. Correct product identification unlocks the remaining 30%.

Score range: **0.0** (worst) → **1.0** (perfect)

### Submission Limits

| Limit | Value |
|-------|-------|
| Submissions in-flight | 2 per team |
| Submissions per day | 3 per team |
| Infrastructure failure freebies | 2 per day (don't count against your 3) |

Limits reset at **midnight UTC**. Infrastructure errors (our fault) don't count against your daily limit — up to 2 per day.

### Leaderboard

The public leaderboard uses the **public test set**. Final ranking uses the **private test set** (never revealed to participants).

---

## Sandbox Environment

| Resource | Limit |
|----------|-------|
| Python | 3.11 |
| CPU | 4 vCPU |
| Memory | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Network | None (fully offline) |
| Timeout | 300 seconds |

### GPU

NVIDIA L4 is **always available** — no opt-in needed:
- `torch.cuda.is_available()` returns `True`
- For ONNX: use `["CUDAExecutionProvider", "CPUExecutionProvider"]`

### Pre-installed Packages

| Package | Version |
|---------|---------|
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ultralytics | 8.1.0 |
| onnxruntime-gpu | 1.20.0 |
| opencv-python-headless | 4.9.0.80 |
| albumentations | 1.3.1 |
| Pillow | 10.2.0 |
| numpy | 1.26.4 |
| scipy | 1.12.0 |
| scikit-learn | 1.4.0 |
| pycocotools | 2.0.7 |
| ensemble-boxes | 1.0.9 |
| timm | 0.9.12 |
| supervision | 0.18.0 |
| safetensors | 0.4.2 |

> You **cannot** `pip install` at runtime.

### Models in the Sandbox

| Framework | Models | Pin this version |
|-----------|--------|-----------------|
| ultralytics 8.1.0 | YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x | `ultralytics==8.1.0` |
| torchvision 0.21.0 | Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN | `torchvision==0.21.0` |
| timm 0.9.12 | ResNet, EfficientNet, ViT, Swin, ConvNeXt, etc. | `timm==0.9.12` |

### Models NOT in the Sandbox

YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers are **not installed**. Two options:

1. **Export to ONNX** — load with `onnxruntime`, opset ≤ 20, use `CUDAExecutionProvider`
2. **Include model code** — `.py` files + `.pt` state_dict weights (standard PyTorch ops only)

> **HuggingFace `.bin` files:** Rename `.bin` → `.pt` (identical format), or convert: `safetensors.torch.save_file(state_dict, "model.safetensors")`

> **Models > 420 MB:** Quantize to FP16 or INT8. FP16 is recommended for L4 — smaller and faster.

### Training Setup

```bash
# YOLOv8
pip install ultralytics==8.1.0

# torchvision detector
pip install torch==2.6.0 torchvision==0.21.0

# Custom model with timm backbone
pip install torch==2.6.0 timm==0.9.12

# GPU training (add CUDA index)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Train anywhere (laptop, Colab, GCP, etc.) — use `state_dict` saves or ONNX export for maximum compatibility.

### Version Compatibility

| Risk | What happens | Fix |
|------|-------------|-----|
| ultralytics 8.2+ weights on 8.1.0 | Model class changed, load fails | Pin `ultralytics==8.1.0` or export to ONNX |
| torch 2.7+ full model save on 2.6.0 | May reference newer operators | Use `torch.save(model.state_dict())`, not `torch.save(model)` |
| timm 1.0+ weights on 0.9.12 | Layer names changed, load fails | Pin `timm==0.9.12` or export to ONNX |
| ONNX opset > 20 | onnxruntime 1.20.0 can't load it | Export with `opset_version=17` |

### Recommended Weight Formats

| Approach | Format | When to use |
|----------|--------|-------------|
| ONNX export | `.onnx` | Universal — any framework, 2–3x faster on CPU |
| ultralytics .pt (pinned 8.1.0) | `.pt` | Simple YOLOv8/RT-DETR workflow |
| state_dict + model class | `.pt` | Custom architectures with standard PyTorch ops |
| safetensors | `.safetensors` | Safe loading, no pickle, fast |

### Security Restrictions

Blocked by the security scanner:
- `import os`, `import subprocess`, `import socket`, `import ctypes`, `import builtins`
- `eval()`, `exec()`, `compile()`, `__import__()`
- ELF binaries, symlinks, path traversal

Use `pathlib` instead of `os` for file operations.

---

## Examples

### Random Baseline

Minimal `run.py` to verify your setup works:

```python
import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = []
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        for _ in range(random.randint(5, 20)):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 356),
                "bbox": [
                    round(random.uniform(0, 1500), 1),
                    round(random.uniform(0, 800), 1),
                    round(random.uniform(20, 200), 1),
                    round(random.uniform(20, 200), 1),
                ],
                "score": round(random.uniform(0.01, 1.0), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

### YOLOv8 Example

> ⚠️ The pretrained COCO model outputs COCO class IDs (0–79), not product IDs (0–355). Fine-tune on competition training data with `nc=356` for correct classification. Detection-only submissions still score up to 70%.

```python
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

Include `yolov8n.pt` in your zip. With GPU available, larger models (YOLOv8m/l/x) are feasible within the 300s timeout.

### ONNX Inference Example

**Export (on your training machine):**

```python
# From ultralytics:
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=17)

# From any PyTorch model:
import torch
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17)
```

**Inference (`run.py`):**

```python
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    session = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path).convert("RGB").resize((640, 640))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

        outputs = session.run(None, {input_name: arr})
        # Process outputs based on your model's output format
        # ...

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `run.py not found at zip root` | Zip the **contents**, not the folder |
| `Disallowed file type: __MACOSX/...` | Use terminal: `zip -r ../sub.zip . -x ".*" "__MACOSX/*"` |
| `Disallowed file type: .bin` | Rename `.bin` → `.pt` or convert to `.safetensors` |
| `Security scan found violations` | Remove `os`, `subprocess`, `socket` imports — use `pathlib` |
| `No predictions.json in output` | Ensure `run.py` writes to the `--output` path |
| `Timed out after 300s` | Use GPU (`model.to("cuda")`) or a smaller model |
| `Exit code 137` | OOM (8 GB limit) — reduce batch size or use FP16 |
| `Exit code 139` | Segfault — version mismatch. Re-export with matching versions or use ONNX |
| `ModuleNotFoundError` | Package not in sandbox — export to ONNX or include model code in `.py` files |
| `KeyError` / `RuntimeError` on model load | Version mismatch — pin exact sandbox versions or export to ONNX |

---

## Tips

- Start with the **random baseline** to verify your setup works end-to-end
- **GPU is always on** — YOLOv8m/l/x and custom transformers are feasible within 300s
- Use `torch.cuda.is_available()` so your code works both locally (CPU) and on the server (GPU)
- **FP16 quantization** is recommended — smaller weights, faster GPU inference
- **ONNX + `CUDAExecutionProvider`** gives good GPU performance for any framework
- Process images **one at a time** to stay within the 8 GB memory limit
- Use `torch.no_grad()` during inference
- Always test locally with `test_locally.py` from the starter kit before uploading

---

## MCP Setup

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
