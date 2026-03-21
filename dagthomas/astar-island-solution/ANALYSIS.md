# Astar Island — Round Analysis & Improvement Plan

## Score History

| Round | Score | Rank | Regime | Dynamic Cells | Key Finding |
|-------|-------|------|--------|---------------|-------------|
| 1 | - | - | - | - | No submission |
| 2 | 66.4 | 53/153 | Thriving | ~1350/seed | Settlements survive (24.5%), ports form |
| 3 | 68.0 | 11/100 | Total collapse | ~400/seed | 0 settlements/ports/ruins in GT |
| 4 | 53.3 | 65/100 | Near-collapse | ~1340/seed | 0 sett in GT, floor=0.015 wasted mass |

**Leaderboard top**: ~103. Our best: 68.0. Gap: ~35 points.

## Round 4 Post-Mortem

**Score: 53.3 (REGRESSION from 68.0)**. Root causes:
1. **Floor 0.015 too high**: mountain/port/ruin each got 1.5% on all dynamic cells.
   GT had 0% mountain on dynamic cells → 3.7% probability mass wasted per cell.
2. **Collapse not detected**: Exploration saw 1015 alive settlements (mid-simulation)
   but GT shows 0 settlements survived to end. Our collapse detection only checks
   observed alive count, which is unreliable since we observe mid-simulation.
3. **Global multipliers not wired into posterior**: Computed (sett=0.77, port=0.55)
   but not applied to predictions.
4. **CalibrationModel not yet active**: Would have scored 88.5 on R4 (backtest).

### CalibrationModel Cross-Validation Results

| Test Round | Train Rounds | Calibrated Score | Actual Score | Delta |
|------------|-------------|-----------------|--------------|-------|
| R2 | R1,R3,R4 | 76.9 | 66.4 | +10.5 |
| R3 | R1,R2,R4 | 44.1 | 68.0 | -23.9 |
| R4 | R1,R2,R3 | 88.5 | 53.3 | +35.2 |

**Key lesson**: CalibrationModel alone is dangerous for extreme rounds (R3 collapse).
It gives the AVERAGE of training data, which hurts when test round is an outlier.
Global multipliers are ESSENTIAL to correct the calibrated prior toward current-round regime.

## Scoring Formula

```
score = 100 × exp(-3 × weighted_kl)
weighted_kl = Σ(entropy(cell) × KL(gt, pred)) / Σ entropy(cell)
```

Only dynamic cells (entropy > 0.01) count. Ocean/mountain are always static.

**Theoretical maximum with floor=0.015: ~88-92 per round.**
**Theoretical maximum with floor=0.01: ~92-94 per round.**

## Round Regimes

### Round 2 — Thriving
- GT has settlements (1.9%), ports (0.1%), forest (20.1%)
- Dynamic cells: ~1350/seed (84% of map)
- Settlement survival: 24.5% (some R1 settlements persist)
- **Error decomposition (seed 0)**:
  - Settlement class: 10.8% of wKL (dominant!)
  - Empty: 0.3%, Forest: 1.2%, Port: 1.1%, Ruin: 1.3%
- Key error: predicting settlement probability wrong

### Round 3 — Total Collapse
- GT has 0 settlements, 0 ports, 0 ruins
- Dynamic cells: ~400/seed (25% of map)
- All settlements collapsed — only empty/forest/mountain remain
- **Error decomposition (seed 0)**:
  - Empty: 9.5% of wKL (dominant!)
  - Forest: 3.1%
  - Settlement/Port/Ruin: ~1.2% combined
- Key error: getting the empty/forest RATIO wrong
- If empty/forest ratio were perfect: score → 85.1 (from 66.1)

## Key Insight: Error Source Changes By Regime

| Regime | Primary Error | Secondary Error |
|--------|---------------|-----------------|
| Collapse | Empty/Forest ratio (98.6%) | Settlement residual (1.2%) |
| Thriving | Settlement prediction (73.8%) | Empty/Forest/Port (26.2%) |

## What We Need To Fix (Priority Order)

### 1. Wire Global Multipliers into Prediction (HIGH — ~5-10 pts)
We compute `observed/expected` ratios per class during exploration but don't
use them in the posterior yet. This would:
- Automatically adjust for regime (collapse vs thriving)
- Scale settlement/port/ruin priors based on what we actually observe
- Improve the empty/forest ratio by detecting forest expansion/contraction

### 2. Feature-Key Calibration from Historical Rounds (HIGH — ~5-8 pts)
Build a CalibrationModel from R1+R2+R3 ground truth data (all saved in
`data/calibration/`). The reference solution uses hierarchical fallback:
- Fine: exact (terrain, dist_bucket, coastal, forest_neighbors, has_port) → ground truth
- Coarse: (terrain, dist_bucket, coastal, has_port) → ground truth
- Base: (terrain) → ground truth
This is far superior to our hardcoded R1 average tables.

### 3. Reduce Floor to 0.01 (MEDIUM — ~4 pts theoretical)
Floor 0.01 vs 0.015 gives ~4 points theoretical max difference.
Both are safe (GT has exact zeros, floor only matters for nonzero GT classes).
The 0.005 "safety margin" of 0.015 is unnecessary with our robust
`_enforce_floor()` function.

### 4. Better Viewport Selection (MEDIUM — ~2-3 pts)
Current fixed 3x3 grid wastes queries on ocean-heavy viewports.
Interest-scored candidates (like reference) focus queries where they matter.

### 5. Pool Observations Across Seeds via Feature Keys (MEDIUM — ~3-5 pts)
Both we and the reference pool observations. But the reference pools by
feature key, which effectively increases sample size per feature type.
With 50 queries × 225 cells = 11,250 cell observations, bucketed into
~20-30 feature groups → ~400 observations per group. Much more statistical
power than per-cell (1-2 observations).

### 6. Cell-Level Posterior with Strong Observation Weight (LOW — ~1-2 pts)
Reference uses `strength = 2.5 × count` (capped at 18) for direct cell
observations. We use Bayesian with prior_strength=0.8. The reference trusts
observations more aggressively for cells with direct data.

## Terrain Transition Patterns (from R1-R3 ground truth)

### Static (never change)
- Ocean (10) → Empty (class 0): 100%
- Mountain (5) → Mountain (class 5): 100%
- Map border: always ocean

### Plains (11) — always near settlements
- Thriving: mostly empty (75%), some settlement (15%), port (2%), forest (5%)
- Collapse: almost all empty (95%), some forest (5%)
- Distance to settlement is the key predictor

### Forest (4)
- Near settlements: can become empty or settlement (20-30%)
- Far from settlements: stays forest (95%+)
- Forest neighbors count helps predict: more forest neighbors → stays forest

### Settlement (1)
- Thriving: ~25% survive, ~5% become ruins, rest become empty
- Collapse: 100% become empty
- Population and food stats predict survival

### Port (2)
- Very rare outcome. Coastal + near settlement + thriving required
- Most initial ports don't survive as ports

## Calibration Data Available

```
data/calibration/
├── round1/   # 5 seeds × (initial_grid + ground_truth)
├── round2/   # 5 seeds × (initial_grid + ground_truth)
└── round3/   # 5 seeds × (initial_grid + ground_truth)
```

15 complete (initial_state → ground_truth) mappings available for building
the CalibrationModel. This is enough for reliable feature-key statistics.

## Current Architecture (post-R4 fixes)

1. **CalibrationModel** from R1-R4 ground truth (123 fine keys, 35 coarse keys)
   - Hierarchical: fine → coarse → base → global fallback
   - Replaces hardcoded R1 average tables
2. **Global multipliers** (observed/expected ratios) adjust calibrated priors
   - Settlement/port/ruin clamp: [0.15, 2.0] — handles collapse to boom
   - Applied before floor enforcement
3. **Floor = 0.01** (hard minimum per class)
4. **Tier 2 cell-level Bayesian DISABLED** — overfits to 1-2 samples/cell
   - TODO: Replace with feature-key bucketed posterior (reference approach)

### Backtest Results (leave-one-out, static prior + multipliers)

| Round | Actual | Cal+Mult | Delta | Notes |
|-------|--------|----------|-------|-------|
| R2 | 66.4 | 76.7 | +10.3 | Thriving, mult boosts settlement |
| R3 | 68.0 | 81.8 | +13.8 | Collapse, mult crushes settlement to 0.15 |
| R4 | 53.4 | 88.7 | +35.3 | Near-collapse, cal model nails it |

### Remaining Improvement Opportunities

1. **Feature-key bucketed posterior** (+5-10 pts estimated)
   - Pool observations by (terrain, dist, coastal, forest_neighbors, has_port)
   - ~100 observations per bucket (vs 1-2 per cell) → reliable empirical priors
   - Reference solution's approach: `strength = sqrt(count)`, max 12
2. **Better viewport selection** (+2-3 pts)
   - Interest-scored candidates instead of fixed 3x3 grid
3. **Upper multiplier clamp tuning** — currently 2.0 for settlement
4. **Collapse detection from settlement stats** — pop/food trends
