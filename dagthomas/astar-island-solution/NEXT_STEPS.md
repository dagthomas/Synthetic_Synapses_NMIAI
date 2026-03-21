# Astar Island — Next Steps Plan

## Current State
- **Production score**: 90.9 avg (R2-R6 LOO), actual R5=86.3, R6=87.6, R7=pending
- **Ceiling reached**: 20 Opus iterations found no single-function improvement
- **Theoretical max**: ~98 (7 point gap remaining)
- **Calibration data**: 6 rounds (48k cells, 125 fine keys)

## The Gap: Where Are We Losing 7 Points?

| Source | Est. Loss | Fixable? |
|--------|----------|----------|
| Observation noise (1-2 samples/cell) | 3-4 pts | Viewport optimization |
| Calibration averaging across regimes | 1-2 pts | Per-regime calibration |
| Empty/Forest boundary imprecision | 1-2 pts | Better spatial models |
| Settlement probability calibration | 0.5-1 pt | Near-optimal already |

## Phase 1: Viewport Selection Optimization (est. +1-2 pts)

### Problem
We use a fixed 3x3 grid (9 viewports × 5 seeds = 45 queries) giving 100% coverage
but only 1 observation per cell. The remaining 5 queries target "most dynamic" areas.
Ocean-heavy viewports waste observations on static cells.

### Plan
Create `experiments/viewport_optimization/` with:

1. **`analyze_viewport_value.py`** — Score each possible viewport position by how
   much information it provides (entropy of prior predictions in that area).
   Compare: fixed grid vs interest-scored vs settlement-centered.

2. **`simulate_viewports.py`** — Using saved observation data, simulate what would
   happen if we had queried different viewports. For cells we didn't observe,
   use the FK bucket empirical as a proxy. Compare prediction quality with
   different viewport strategies.

3. **`smart_explore.py`** — New exploration module that replaces the fixed 3x3 grid:
   - Phase 1 (35 queries): Score all possible 15×15 viewports by
     `interest = Σ(prior_entropy × dynamism)` per cell. Select top viewports
     with <50% overlap, round-robin across seeds.
   - Phase 2 (15 queries): After Phase 1, recompute interest based on
     what we've observed. Target highest-residual-uncertainty areas.

4. **`test_viewport.py`** — Backtest framework that simulates different viewport
   strategies using R2-R6 ground truth + saved observations. Can't perfectly
   simulate new viewports (we don't have that data), but can measure coverage
   efficiency and estimate improvement.

### Expected Impact
- More observations on dynamic cells (near settlements) → better FK empirical
- Fewer wasted observations on ocean/mountain → more queries for 2nd samples
- 2nd observations on high-entropy cells dramatically reduce prediction error

### How to Test
```bash
cd experiments/viewport_optimization
python analyze_viewport_value.py   # Shows current waste
python simulate_viewports.py       # Estimates improvement
```

## Phase 2: Per-Regime Calibration (est. +1-2 pts)

### Problem
The CalibrationModel averages ground truth across ALL rounds. But R3 (collapse, 0%
survival) and R6 (boom, 244% survival) have completely different distributions.
Averaging them gives a poor prior for extreme rounds.

### Plan
Create `experiments/regime_calibration/` with:

1. **`cluster_rounds.py`** — Classify rounds into regimes (collapse/moderate/thriving)
   based on observed settlement statistics. Use K-means or simple thresholds on
   observed settlement % and observed survival indicators.

2. **`regime_cal_model.py`** — Build separate CalibrationModels per regime.
   At prediction time, detect the current regime from observations, then use
   the matching calibration model (or weighted blend of nearby regimes).

3. **`test_regime_cal.py`** — Backtest comparing:
   - Current: single CalibrationModel from all rounds
   - New: regime-matched CalibrationModel
   - Blend: weighted combination based on regime similarity

### Expected Impact
- Collapse rounds get priors from other collapse rounds (R3, R4)
- Thriving rounds get priors from other thriving rounds (R1, R2, R6)
- Should significantly improve extreme-regime predictions

### How to Test
```bash
cd experiments/regime_calibration
python cluster_rounds.py           # Shows regime clusters
python test_regime_cal.py          # Backtests regime-matched vs single cal
```

## Phase 3: Ensemble Prediction (est. +0.5-1 pt)

### Problem
Single prediction function has a fixed tradeoff between prior trust and empirical
trust. Some cells benefit from more prior, others from more empirical.

### Plan
Create `experiments/ensemble/` with:

1. **`build_ensemble.py`** — Create 3-5 prediction variants with different
   strategies:
   - Variant A: High prior trust (current best)
   - Variant B: High empirical trust (FK weight >> prior weight)
   - Variant C: Regime-adaptive (per-regime calibration from Phase 2)
   - Variant D: Spatial-smoothed (heavy neighbor averaging)
   - Variant E: Conservative (high floor, low variance)

2. **`optimize_weights.py`** — Find optimal per-cell or per-feature-key weights
   for combining variants. Use LOO on R2-R6 to learn which variant works best
   for which cell type.

3. **`test_ensemble.py`** — Backtest the ensemble vs individual variants.

### Expected Impact
- Different cell types benefit from different strategies
- Settlement cells: trust empirical (variant B)
- Forest cells: trust prior (variant A)
- High-entropy cells: trust smoothing (variant D)

### How to Test
```bash
cd experiments/ensemble
python build_ensemble.py           # Creates variant prediction functions
python optimize_weights.py         # Learns optimal blend weights
python test_ensemble.py            # Backtests ensemble vs individuals
```

## Phase 4: Observation Data Utilization (est. +0.5 pt)

### Problem
We observe settlement stats (population, food, wealth, defense) during exploration
but only use alive/dead as a binary signal. Low food → higher collapse probability.

### Plan
Create `experiments/settlement_stats/` with:

1. **`analyze_stats.py`** — Correlate observed settlement stats with GT outcomes
   across R5-R6 (where we have both). Questions:
   - Does low food predict collapse?
   - Does high population predict expansion?
   - Do port settlements behave differently?

2. **`stats_features.py`** — Add settlement stats as features to the prediction:
   - Compute mean population/food per seed from observations
   - Use as an additional regime signal (alongside sett%)
   - Per-settlement survival probability based on stats

3. **`test_stats.py`** — Backtest stats-enhanced predictions.

### How to Test
```bash
cd experiments/settlement_stats
python analyze_stats.py            # Shows stat correlations
python test_stats.py               # Backtests stats-enhanced predictions
```

## Phase 5: Spatial Correlation Model (est. +0.5-1 pt)

### Problem
Current pipeline treats each cell independently (via feature key). But neighboring
cells are correlated — a settlement cluster's fate is shared. Our 3×3 smoothing
is crude.

### Plan
Create `experiments/spatial/` with:

1. **`neighbor_correlation.py`** — Measure actual GT correlation between adjacent
   cells across all rounds. How much does knowing cell (y,x) help predict (y+1,x)?

2. **`spatial_model.py`** — Implement a spatial consistency model:
   - After initial per-cell prediction, propagate information from observed cells
     to unobserved neighbors
   - Weight by distance and terrain similarity
   - Iterative belief propagation (2-3 rounds)

3. **`test_spatial.py`** — Backtest spatial model vs independent predictions.

### How to Test
```bash
cd experiments/spatial
python neighbor_correlation.py     # Quantifies spatial correlation
python test_spatial.py             # Backtests spatial model
```

## Execution Order

Priority order (highest impact, lowest risk first):

1. **Phase 1 (Viewport)** — Highest potential, requires `explore.py` changes
2. **Phase 2 (Regime Cal)** — High potential, self-contained
3. **Phase 4 (Stats)** — Quick to analyze, may reveal useful signals
4. **Phase 3 (Ensemble)** — Moderate potential, builds on Phase 2
5. **Phase 5 (Spatial)** — Research-heavy, uncertain payoff

## Quick Start

```bash
# Create experiment directories
mkdir -p experiments/{viewport_optimization,regime_calibration,ensemble,settlement_stats,spatial}

# Run Phase 2 first (self-contained, high impact):
cd experiments/regime_calibration
python cluster_rounds.py
python test_regime_cal.py

# Then Phase 1 (viewport optimization):
cd ../viewport_optimization
python analyze_viewport_value.py
python simulate_viewports.py
```

## Validation Protocol

Every experiment MUST be validated on the full production harness (R2-R6 LOO, 5 seeds)
before deployment. The test script pattern:

```python
# test_<experiment>.py
from test_ideas import eval_fn, build_context  # reuse existing harness
from predict_gemini import gemini_predict       # baseline

baseline_avg = eval_fn(gemini_predict, "BASELINE")
new_avg = eval_fn(my_new_fn, "NEW APPROACH")
print(f"Delta: {new_avg - baseline_avg:+.2f}")
# ONLY deploy if delta > +0.1 consistently across all rounds
```
