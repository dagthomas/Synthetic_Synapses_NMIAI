# Product Requirements

Requirements derived from competition rules and scoring criteria.

---

## Astar Island

### Objective
Predict 40x40x6 probability distributions for a Norse civilization simulator across 5 seeds per round.

### Scoring
- Entropy-weighted KL divergence, scored 0-100
- Only dynamic cells (entropy > 0.01) count
- Probability floor of 0.01 required (avoids infinite KL)
- Target: >90 average score

### Constraints
- 50 simulation queries per round (shared across 5 seeds)
- 15x15 viewport per query
- ~165 minute round window
- Rate limit: 5 sim calls/sec, 2 submit calls/sec

### Requirements
- [x] Adaptive exploration strategy (entropy-targeted viewports)
- [x] Hierarchical calibration model (fine/coarse/base/global)
- [x] Regime detection (collapse/moderate/boom)
- [x] Parametric simulator with CMA-ES fitting
- [x] Autonomous parameter optimization (autoloop)
- [x] Real-time monitoring dashboard
- [x] Automatic round detection and submission (daemon)
- [ ] Score consistently >92 on boom rounds

---

## NorgesGruppen Data

### Objective
Detect and classify grocery products on store shelf images.

### Scoring
- `Final = 0.70 * detection_mAP@0.5 + 0.30 * classification_mAP@0.5`
- Detection: category-agnostic box localization
- Classification: correct product class + box

### Constraints
- Docker sandbox: Python 3.11, L4 GPU (24 GB), 300s timeout, offline
- Blocked imports: `os`, `sys`, `subprocess`, `pickle`, `yaml`, etc.
- Max zip: 420 MB, max 3 weight files, max 10 Python files
- 3 submissions per day

### Requirements
- [x] YOLOv8x detection with TTA
- [x] Multi-model ensemble with WBF fusion
- [x] ConvNeXt-V2 nano re-classification
- [x] Background rejection (class 356)
- [x] torch.load monkeypatch for PyTorch 2.6.0
- [x] Submission under 420 MB and 300s timeout
- [ ] Test set score > 0.95 combined

---

## Tripletex

### Objective
Build HTTPS `/solve` endpoint that executes accounting tasks via Tripletex API.

### Scoring
- `correctness * tier_multiplier + efficiency_bonus` (up to 6.0 per task)
- 30 task types, 56 variants each
- 7 languages: nb, en, es, pt, nn, de, fr

### Requirements
- [ ] HTTPS endpoint deployment
- [ ] LLM agent for prompt interpretation
- [ ] Tripletex API integration (employees, customers, products, invoices, etc.)
- [ ] Multi-language prompt handling

---

## Grocery Bot

**Status:** Skipped (not scored, warm-up only)
