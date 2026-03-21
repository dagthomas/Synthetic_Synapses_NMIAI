# NM i AI — Project Summary

> Norwegian AI Championship — Competition at [app.ainm.no](https://app.ainm.no)
> Last updated: 2026-03-19

---

## Challenges Overview

| Challenge | Type | Status | Approach |
|-----------|------|--------|----------|
| **NorgesGruppen Data** | Object Detection | Active, submitted | YOLOv8x + EfficientNet-B2 two-stage pipeline |
| **Astar Island** | Simulation Prediction | Active, Round 2 | Bayesian Dirichlet updating with adaptive priors |
| **Tripletex** | AI Accounting Agent | Not started | Requires HTTPS `/solve` endpoint |
| **Grocery Bot** | Warm-up (unscored) | Skipped | WebSocket game, not scored |

---

## 1. NorgesGruppen Data — Object Detection

### Problem
Detect and classify grocery products on store shelf images. Scoring is **70% detection mAP + 30% classification mAP** (mAP@0.5), which motivated a two-stage architecture that optimizes each component separately.

### Training Data
- **248 shelf images** with ~22,700 COCO-format bounding box annotations
- **357 categories** (0–355 named products, 356 = `unknown_product`)
- **327 product reference images** with multi-angle photos (front, back, sides, top, bottom)

### Architecture: Two-Stage Pipeline

```
Image → YOLOv8x Detection (TTA) → Crop Extraction → EfficientNet-B2 Re-classification → JSON
```

#### Stage 1: YOLOv8x Detection (targets the 70% detection score)

**Training — Two-phase strategy:**

| Phase | Data | Epochs | Purpose |
|-------|------|--------|---------|
| Validation run | 90/10 stratified split | Up to 300 (patience=50) | Find optimal epoch count |
| Final run | 100% training data | ~120 (1.15× best val epoch) | Maximize detection quality |

**Hyperparameters:**
- Input size: **1280×1280**
- Batch size: **4** (fits L4 24 GB VRAM)
- Optimizer: AdamW, lr=0.001, cosine annealing to lr×0.01
- Augmentation: mosaic=1.0, mixup=0.15, copy_paste=0.1, close_mosaic=30
- Rare categories (≤3 annotations) forced into training set to prevent data loss

**Inference:**
- TTA enabled (augment=True) — ~4 passes per image for better recall
- Confidence threshold: 0.01 (very low to maximize recall)
- NMS IoU: 0.6, max detections: 1000

#### Stage 2: EfficientNet-B2 Classifier (targets the 30% classification score)

**Training — Two-phase fine-tuning:**

| Phase | Layers | Epochs | Learning Rate |
|-------|--------|--------|---------------|
| 1. Head-only | Classifier head frozen backbone | 5 | 3e-4 |
| 2. Full fine-tune | All layers unfrozen | 55 | Backbone 3e-5, Head 3e-4 |

**Training data (with oversampling):**
- Shelf crops from COCO annotations (1× each) — padded 10%
- Product reference images (3× oversampled) — boosts per-product recognition
- Input size: 224×224, ImageNet normalization

**Class balancing:**
- `WeightedRandomSampler` with inverse frequency weights (capped at 50×)
- Label smoothing: 0.1
- Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, Affine, RandomErasing

**Inference:**
- Batch size 128, FP16 inference
- Confidence threshold: **0.3** — only overrides YOLO's class if the classifier is confident enough
- numpy-based preprocessing for speed (no PIL overhead)

### Model Weights
| File | Size | Description |
|------|------|-------------|
| `best.pt` | 137 MB | YOLOv8x detection model |
| `classifier.safetensors` | ~17 MB | EfficientNet-B2 (FP16, safetensors format) |

### Submission Constraints
- **Sandbox:** Python 3.11, L4 GPU (24 GB VRAM), 300s timeout
- **Pinned versions:** ultralytics==8.1.0, timm==0.9.12, PyTorch 2.6.0+cu124
- **Blocked imports:** `os`, `sys`, `subprocess`, `pickle`, `yaml`, `requests`, etc.
- **Critical fix:** `torch.load` monkeypatch required (PyTorch 2.6 defaults to `weights_only=True`, breaking ultralytics 8.1.0)
- **Max zip size:** 420 MB, max 3 weight files, max 10 Python files, 300s timeout

---

## 2. Astar Island — Viking Civilisation Prediction

### Problem
Observe a Norse civilisation simulator through a 15×15 viewport window on a 40×40 grid. Predict the final state as a 40×40×6 probability tensor (6 terrain classes). Budget: **50 queries per round** shared across 5 seeds. Scored by entropy-weighted KL divergence (0–100).

### Architecture: Explore → Estimate → Predict → Submit

#### Phase 1: Exploration Strategy (explore.py)

**4-phase budget-conscious approach (50 queries total):**

| Phase | Queries | Strategy |
|-------|---------|----------|
| 0. Analyze | 0 (free) | Extract terrain, settlements, ports from initial grids; build dynamism heatmap |
| 1. Coverage | ~30 | Round-robin 6 greedy-placed viewports across all 5 seeds |
| 2. Depth | ~10 | Re-query highest-entropy viewports for multiple observations |
| 3. Adaptive | ~5 | Target unobserved dynamic cells near settlements |

**Viewport placement:** Greedy algorithm maximizing dynamism heatmap score (cells near settlements scored higher). After each placement, heatmap reduced by 80% in that region to avoid overlap.

**Per-seed vs shared planning:** If base terrain (ocean/mountain) is identical across seeds, uses a shared viewport plan. Otherwise, creates per-seed plans.

#### Phase 2: Parameter Estimation (estimator.py)

Analyzes observations to detect round-specific hidden parameters:

- **Settlement survival rate** — what % of settlements survive
- **Expansion rate** — how much do settlements spread to adjacent plains
- **Forest loss rate** — how much forest gets converted near settlements
- **Population/food/wealth/defense** averages across alive settlements
- **Faction structure** — count, sizes, dominance patterns

**Regime detection:** Classifies the round as harsh/mild winters, high/low trade, high/low expansion, dominant/fragmented/balanced conflict.

**Adaptive prior strength:** Ranges from 0.5 (trust observations) to 3.0 (trust R1 calibrated priors), computed from divergence between observed stats and R1 baseline.

#### Phase 3: Prediction (predict.py)

**Two-tier strategy:**

**Tier 1 — Static Prior (no observations needed):**
- Per-integer-distance calibrated priors from Round 1 ground truth
- Distance computed via EDT to nearest settlement
- Incorporates: terrain type, coastal awareness, settlement cluster density, forest adjacency
- Parameter adjustments from estimator (survival/expansion multipliers)

**Tier 2 — Bayesian Observation-Informed:**
```
posterior = (prior_strength × static_prior + observation_counts) / (prior_strength + n_obs)
```
- Dirichlet posterior update: prior_strength acts as pseudo-counts
- Spatial smoothing for unobserved cells (weighted neighbor average, radius 2)
- Probability floor enforced at 0.01 (avoids infinite KL divergence)

#### Submission History (Round 2)
| Submission | Key Change | Approach |
|------------|------------|----------|
| Sub 1 | Baseline | Basic pipeline, 3-bucket distance priors |
| Sub 2 | Bayesian update | Switched to Dirichlet posterior updating |
| Sub 3 | Fine-grained priors | Per-integer-distance interpolation, coastal/cluster features, per-seed viewports |
| Sub 4 | Adaptive parameters | ParameterEstimator for regime detection and dynamic prior strength |

#### Round 2 Findings
- Low average population (1.16 vs R1 baseline 2.5) → harsh survival conditions
- High food (0.4+) → mild winters
- 54 factions (very fragmented)
- Prior strength computed: 2.53 (slightly more trust in observations than R1 default of 3.0)

---

## 3. Tripletex — AI Accounting Agent

### Problem
Build an HTTPS `/solve` endpoint that receives accounting task prompts (in 7 languages) and executes them via the Tripletex API. 30 task types × 56 variants each. Scored on correctness × tier multiplier + efficiency bonus (up to 6.0 per task).

### Status
**Not started.** Requires deploying an HTTPS server, not a zip submission. No local solution directory exists yet.

---

## Key Files

```
norgesgruppen-solution/
├── run.py                    # Inference entry point (submitted)
├── train_yolo.py             # YOLOv8x two-phase training
├── train_classifier.py       # EfficientNet-B2 fine-tuning
├── prepare_data.py           # COCO → YOLO format conversion
├── prepare_classifier_data.py # Crop extraction for classifier
├── classifier.py             # Classifier model loader
├── best.pt                   # YOLOv8x weights
└── classifier.safetensors    # EfficientNet-B2 weights (FP16)

astar-island-solution/
├── submit.py                 # Orchestrator: explore → estimate → predict → submit
├── explore.py                # 4-phase viewport exploration
├── predict.py                # Two-tier Bayesian prediction
├── estimator.py              # Round parameter estimation
├── client.py                 # API client with rate limiting
├── utils.py                  # ObservationAccumulator, heatmaps
├── config.py                 # Constants
├── IMPROVEMENTS.md           # Iteration log
└── data/r1_features.json     # Round 1 ground truth analysis
```

---

## Training Rules
- **Never run concurrent GPU training jobs** — sequential only to avoid CUDA OOM
- **Pin package versions** to match sandbox (ultralytics==8.1.0, timm==0.9.12)
- **Batch sizes** must fit in 24 GB VRAM (L4 GPU target)
