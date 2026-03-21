# NM i AI — Norwegian AI Championship

Competition platform: [app.ainm.no](https://app.ainm.no)

## Project Structure

```
├── docs/                              Documentation & training data
│   ├── grocery-bot/                   Warm-up WebSocket game (not scored)
│   ├── norgesgruppen-data/            Object detection challenge
│   │   ├── NM_NGD_coco_dataset/       248 shelf images + COCO annotations (356 categories)
│   │   └── NM_NGD_product_images/     327 products with multi-angle reference photos
│   ├── tripletex/                     AI accounting agent challenge
│   └── astar-island/                  Viking civilisation prediction challenge
└── norgesgruppen-solution/            Solution workspace for object detection
```

## Active Challenges

### 1. NorgesGruppen Data — Object Detection (Main)
- Detect grocery products on store shelves
- Score: 70% detection mAP + 30% classification mAP (mAP@0.5)
- Submit `run.py` in a `.zip` — runs in Docker sandbox (Python 3.11, NVIDIA L4 GPU, 24 GB VRAM)
- Sandbox has: ultralytics 8.1.0, PyTorch 2.6.0+cu124, onnxruntime-gpu 1.20.0, timm 0.9.12
- Security: `os`, `sys`, `subprocess`, `pickle`, `yaml` imports are blocked — use `pathlib` and `json`
- Training data: `docs/norgesgruppen-data/NM_NGD_coco_dataset/train/`
- Annotations: COCO format `[x, y, width, height]`, 357 classes (0-356, where 356 = unknown_product)
- Limits: 3 submissions/day, 420 MB max zip, 300s timeout

### 2. Tripletex — AI Accounting Agent
- Build HTTPS `/solve` endpoint that executes accounting tasks via Tripletex API
- Prompts in 7 languages (nb, en, es, pt, nn, de, fr), 30 task types, 56 variants each
- Auth: Basic Auth with username `0` and session token as password
- Score: correctness × tier multiplier + efficiency bonus (up to 6.0 per task)

### 3. Astar Island — Viking Civilisation Prediction
- Observe Norse simulator through 15×15 viewport, predict 40×40×6 probability tensor
- 50 queries per round shared across 5 seeds
- Score: entropy-weighted KL divergence (0-100)
- Always use probability floor of 0.01 to avoid infinite KL divergence
- API base: `https://api.ainm.no/astar-island`

### 4. Grocery Bot — Warm-up (Not Scored)
- WebSocket game at `wss://game.ainm.no/ws?token=<jwt>`
- 21 maps, 5 difficulty levels, 2s response timeout

## Training Rules

- **Never run concurrent GPU training jobs** — always run sequentially to avoid CUDA OOM errors
- **Pin package versions** to match sandbox: `ultralytics==8.1.0`, `timm==0.9.12`
- Train with batch sizes that fit in 24 GB VRAM (L4 GPU target)

## NorgesGruppen Submission Rules — NEVER BREAK THESE

### Blocked Imports (security scanner rejects the entire submission)
`os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http.client`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`

**Use instead:** `pathlib` (not `os`), `json` (not `yaml`), `safetensors` (not `pickle`)

### Blocked Calls
`eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names

### Critical: torch.load Monkeypatch
PyTorch 2.6.0 in sandbox defaults `weights_only=True`, breaking ultralytics 8.1.0. **MUST** add this before `from ultralytics import YOLO`:
```python
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

### run.py Contract
- Invoked as: `python run.py --input /data/images --output /output/predictions.json`
- Output: JSON array `[{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]`
- `image_id` from filename: `img_00042.jpg` → `42`

### Submission Limits
- Max zip: 420 MB, max weight files: 3, max Python files: 10
- Allowed types: `.py`, `.json`, `.yaml`, `.yml`, `.cfg`, `.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy`
- Timeout: 300 seconds on L4 GPU — keep TTA to ≤4 passes per image

### Version Compatibility
- ultralytics 8.2+ weights **fail** on 8.1.0 — always pin 8.1.0
- timm 1.0+ weights **fail** on 0.9.12 — always pin 0.9.12
- ONNX opset > 20 **fails** on onnxruntime 1.20.0

## Key Commands

```bash
# MCP docs server
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp

# NorgesGruppen submission
cd norgesgruppen-solution && zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```
