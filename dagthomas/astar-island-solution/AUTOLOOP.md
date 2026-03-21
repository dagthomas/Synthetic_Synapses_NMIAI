# Autoloop — Autonomous Experiment System

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): instead of manually editing code, we program a loop that proposes changes, evaluates them, and keeps what works.

## Quick Start

```bash
cd astar-island-solution

# Run the fast vectorized version (recommended)
python autoloop_fast.py --seeds 5

# Check progress (in another terminal)
python autoloop_fast.py --summary

# Stop: Ctrl+C (results are saved continuously)
```

## How It Works

### The Core Loop

```
┌──────────────────────────────────────────────────┐
│  1. Load best parameter set from log             │
│  2. Randomly perturb 1-3 parameters              │
│  3. Build prediction with new params             │
│  4. Score against R2-R11 ground truth (LOO)      │
│  5. If better → keep + write best_params.json    │
│     If close → keep with 20% probability         │
│     If worse → discard                           │
│  6. Log result → go to step 2                    │
└──────────────────────────────────────────────────┘
```

### Architecture

```
daemon.py                 ← Autonomous orchestrator (runs everything)
├── autoloop_fast.py      ← Main loop + FastHarness (vectorized, 9 rounds LOO)
│   └── writes best_params.json on new best
├── predict_gemini.py     ← Production prediction (reads best_params.json)
├── fast_predict.py       ← Vectorized numpy prediction (~20ms/eval)
├── autoloop.py           ← Parameter space definition + ExperimentLog
├── calibration.py        ← CalibrationModel (historical ground truth)
├── eval_production.py    ← Direct production code evaluation
└── data/
    ├── calibration/      ← R1-R11 ground truth (training data)
    │   ├── round1/ ... round11/
    ├── rounds/           ← Saved observations per round
    └── autoloop_fast_log.jsonl  ← Experiment results
```

### Three Key Components

**1. Fixed Evaluation Harness (`FastHarness`)**

Pre-loads and caches everything that doesn't depend on parameters:
- Ground truth tensors for R1-R4 (20 seeds × 40×40×6)
- Feature key indices per cell
- Coastal masks, static/dynamic masks
- Observation data (FK buckets, global multipliers)

Leave-one-out cross-validation: when testing on round N, the calibration model is trained on all OTHER rounds. This prevents data leakage and tests generalization.

**2. Vectorized Prediction (`fast_predict.py`)**

Replaces Python for-loops over 1600 cells with numpy array operations:

```python
# Old (51ms): loop over every cell
for y in range(40):
    for x in range(40):
        prior = cal.prior_for(feature_keys[y][x])
        ...

# New (10ms): numpy fancy indexing
cal_priors = lookup_table[idx_grid]  # (40, 40, 6) in one operation
pred = cal_priors * pw + empiricals * strength  # vectorized blend
pred *= multipliers[None, None, :]  # broadcast multiply
```

Result: **~160,000 experiments per hour** at full 5-seed evaluation.

**3. Parameter Proposer (`perturb_params`)**

Hill-climbing with Metropolis exploration:
- Pick 1-3 random parameters
- Perturb by Gaussian noise (scaled to parameter's step size)
- After 500 iterations without improvement, perturb 2-4 parameters (wider search)
- Accept near-ties with 20% probability to escape local optima

## Parameter Space

### FK Bucket Blending
Controls how current-round observations blend with the calibration prior.

| Parameter | Default | Range | What it does |
|-----------|---------|-------|-------------|
| `fk_prior_weight` | 5.0 | 0.5-12.0 | Weight of calibration prior in blend |
| `fk_max_strength` | 8.0 | 2.0-25.0 | Max weight of FK empirical |
| `fk_min_count` | 5 | 2-25 | Min observations to use FK data |
| `fk_strength_fn` | sqrt | sqrt/log/linear | How FK weight scales with count |

### Global Multipliers
Regime adaptation — adjusts predictions based on observed vs expected class distribution.

| Parameter | Default | Range | What it does |
|-----------|---------|-------|-------------|
| `mult_power` | 0.4 | 0.1-1.0 | Dampening (lower = more reactive) |
| `mult_smooth` | 5.0 | 1.0-20.0 | Smoothing constant |
| `mult_sett_lo/hi` | 0.15/2.0 | 0.02-0.5/1.5-4.0 | Settlement clamp range |
| `mult_forest_lo/hi` | 0.5/1.8 | 0.2-0.8/1.2-2.5 | Forest clamp range |
| `mult_empty_lo/hi` | 0.75/1.25 | 0.5-0.95/1.05-1.5 | Empty clamp range |

### Floor
Minimum probability for nonzero classes.

| Parameter | Default | Range | What it does |
|-----------|---------|-------|-------------|
| `floor_nonzero` | 0.005 | 0.001-0.015 | Floor for classes that can appear |

Mountain is always zeroed on non-mountain cells. Port is always zeroed on non-coastal cells. These structural zeros are not parameterized — they're hard rules from the GT analysis.

### Calibration Model Weights (biggest tuning knob!)
Controls the hierarchical blend of historical ground truth data.

```
Prediction = weighted blend of:
  Fine:   exact (terrain, dist, coastal, forest_neighbors, has_port) match
  Coarse: (terrain, dist, coastal, has_port) match
  Base:   terrain-only match
  Global: overall class distribution
```

| Parameter | Default | Range | What it does |
|-----------|---------|-------|-------------|
| `cal_fine_base` | 1.0 | 0.3-3.0 | Fine level base weight |
| `cal_fine_divisor` | 120.0 | 30-500 | Fine weight scaling (count / divisor) |
| `cal_fine_max` | 4.0 | 1.0-10.0 | Fine weight cap |
| `cal_coarse_base` | 0.75 | 0.2-2.0 | Coarse level base weight |
| `cal_coarse_divisor` | 200.0 | 50-500 | Coarse weight scaling |
| `cal_coarse_max` | 3.0 | 1.0-8.0 | Coarse weight cap |
| `cal_base_base` | 0.5 | 0.1-2.0 | Base level base weight |
| `cal_base_divisor` | 1000.0 | 200-3000 | Base weight scaling |
| `cal_base_max` | 1.5 | 0.5-5.0 | Base weight cap |
| `cal_global_weight` | 0.4 | 0.05-2.0 | Global regularizer weight |
| `cal_heuristic_blend` | 0.0 | 0.0-0.5 | Blend with hardcoded R1 heuristic |

## Reading the Output

```
[  308] ***BEST 90.718*** R2=87.8 R3=92.0 R4=92.3 (21ms, 140175/hr) | mult_power=0.1534
  │              │              │                    │      │              │
  │              │              │                    │      │              └─ Parameter change
  │              │              │                    │      └─ Throughput
  │              │              │                    └─ Time per experiment
  │              │              └─ Per-round scores (leave-one-out)
  │              └─ New best average score
  └─ Experiment number
```

Periodic summaries every 1000 iterations show:
- Total experiments and acceptance rate
- Current best parameters (diff from default)
- Recent accepted experiments

## Interpreting Results

### What the autoloop found (4700+ experiments)

**Best: 91.05 average** (R2=88.2, R3=92.8, R4=92.2)

Key shifts from defaults:
- **FK empirical trusted MORE** (prior_weight 5→0.85, strength 8→16.6, fn sqrt→linear)
- **Multipliers barely dampened** (power 0.4→0.1) — trust observations directly
- **Calibration: coarse level up, fine level down** — fewer exact-match cells means fine level overfits; coarse generalizes better
- **Global regularizer up** (0.4→1.3) — stronger smoothing toward overall distribution
- **Floor slightly up** (0.005→0.007) — more safety margin

### Warning: Overfitting

With only 3 test rounds (R2, R3, R4), there's overfitting risk. Watch for:
- One round score much higher than others (unbalanced optimization)
- Parameters at extreme edges of their ranges
- Improvements < 0.05 (likely noise)

The best protection: when R5 ground truth becomes available, add it to calibration and re-run the autoloop. More training data → more reliable optimization.

## Extending the System

### Adding a new round's data

After a round completes and scores are available:

```python
# Save ground truth
python -c "
from client import AstarIslandClient
import json
from pathlib import Path

client = AstarIslandClient()
round_id = 'xxx'  # the round's UUID
d = Path('data/calibration/round5')
d.mkdir(parents=True, exist_ok=True)
detail = client.get_round_detail(round_id)
with open(d / 'round_detail.json', 'w') as f:
    json.dump(detail, f)
for seed in range(5):
    analysis = client.get_analysis(round_id, seed)
    with open(d / f'analysis_seed_{seed}.json', 'w') as f:
        json.dump(analysis, f)
"
```

Then update `ROUND_NAMES` and `ROUND_IDS` in `autoloop_fast.py` and restart.

### Adding new parameters

1. Add to `PARAM_SPACE` in `autoloop.py` with type, range, and step
2. Add default to `DEFAULT_PARAMS`
3. Use the parameter in `FastHarness.evaluate()` or `make_pred_fn()`

### Adding structural experiments

For changes that can't be expressed as numeric parameters (e.g., "add spatial smoothing"), create a new prediction function variant and test it manually first:

```python
from autoexperiment import BacktestHarness, compute_score
harness = BacktestHarness(seeds_per_round=5)
scores = harness.evaluate(your_new_pred_fn)
```

If it improves scores, integrate it into the parameterized prediction function.

## Files

| File | Purpose |
|------|---------|
| `autoloop_fast.py` | Fast experiment loop (vectorized, ~160k/hr) |
| `autoloop.py` | Original loop + param space definitions |
| `autoexperiment.py` | Predefined experiment suite |
| `fast_predict.py` | Vectorized prediction functions |
| `data/autoloop_fast_log.jsonl` | Experiment log (JSONL, append-only) |
| `data/autoexperiment_results.json` | Predefined experiment results |
