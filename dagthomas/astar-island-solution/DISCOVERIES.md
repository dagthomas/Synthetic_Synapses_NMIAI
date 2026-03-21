# Astar Island — Key Discoveries

## 1. Ground Truth is Based on Exactly 200 Simulations

Every GT probability across all 4 rounds and 20 seeds is an exact multiple of 0.005 (1/200). This means the organizers run the simulator 200 times per seed with the hidden parameters, then count outcomes per cell.

**Implications:**
- Minimum nonzero GT probability is 0.005 (1/200 sims showed that class)
- A floor of 0.01 wastes probability mass — 0.005 is the true minimum
- GT has many exact zeros (class never appeared in 200 runs)
- Predicting 0 for a class where GT is 0 is safe (KL term = 0·log(0/q) = 0)

## 2. Structural Zeros — Classes That Never Appear

Analysis across R1-R4 (20 seeds, 32,000 cells) reveals hard structural rules:

### Mountain (class 5) is ALWAYS 0 on dynamic cells
- 100% of dynamic cells across all rounds have GT mountain = 0.000
- Mountains are geological — they don't appear/disappear over 50 simulated years
- Mountain cells are static (entropy = 0) and excluded from scoring
- **Saving: 1% probability mass per dynamic cell** (from removing floor)

### Port (class 2) is ALWAYS 0 on non-coastal cells
- Ports require ocean adjacency — they physically cannot form inland
- 85-96% of dynamic cells are non-coastal → port is impossible there
- On the ~15% coastal dynamic cells, port is possible and gets a real floor
- **Saving: ~0.8% probability mass on most dynamic cells**

### Static cells are deterministic
- Ocean: always [1, 0, 0, 0, 0, 0] — no floor needed, predict exact 1.0
- Mountain: always [0, 0, 0, 0, 0, 1] — same
- Border cells: always ocean (entire map border row/column)

## 3. The Scoring Formula Favors Precision on High-Entropy Cells

```
score = 100 × exp(-3 × weighted_kl)
weighted_kl = Σ(entropy(cell) × KL(gt, pred)) / Σ(entropy(cell))
```

The exponential decay means:
- wKL = 0.00 → score 100
- wKL = 0.05 → score 86.1
- wKL = 0.10 → score 74.1
- wKL = 0.15 → score 63.8
- wKL = 0.20 → score 54.9

**Small improvements in wKL yield big score gains.** Going from 0.15 → 0.05 gains +22 points. Every 0.01 reduction in wKL is worth ~2-3 score points.

## 4. The Dominant Error Source Changes By Regime

| Regime | Primary Error Source | % of wKL |
|--------|---------------------|----------|
| Collapse (R3) | Empty/Forest ratio | 98.6% |
| Thriving (R2) | Settlement prediction | 73.8% |
| Moderate (R4) | Settlement + Empty/Forest | ~50/50 |

**You must detect the regime from observations and adapt.** A single prediction strategy cannot handle both collapse and thriving scenarios.

## 5. Mid-Simulation Observations ≠ End-State Ground Truth

Our 50 queries observe the simulation at one point in time. But GT represents the final state after 50 simulated years. Settlements can be alive when we observe them but collapse before the end.

- R4: We observed 1015 alive settlements → but GT had 0 surviving
- This makes binary collapse detection unreliable
- **Solution:** Global multipliers (observed/expected ratios) provide continuous regime detection rather than binary thresholds

## 6. Feature-Key Pooling Beats Per-Cell Observation

With 50 queries × 225 cells/viewport = 11,250 cell observations, but only 1-2 observations per cell.

- **Per-cell Bayesian update HURTS** — overfits to single stochastic samples
- **Feature-key pooling HELPS** — groups cells by (terrain, distance_bucket, coastal, forest_neighbors, has_port), giving ~100 observations per group
- The empirical distribution from 100 observations is reliable; from 1-2 it's noise

## 7. Calibration From Historical Ground Truth

Building a prior from R1-R4 ground truth data (hierarchical: fine key → coarse → base → global) is far superior to hardcoded average tables.

**But calibration alone is dangerous** — it gives the average across rounds, which fails on extreme rounds (R3 collapse scored 44.1 with calibration-only).

**Calibration + multipliers + feature-key buckets** handles all regimes:
- Calibration: historical baseline
- Multipliers: current-round regime adaptation
- Feature-key buckets: current-round empirical data

## 8. Floor Optimization — 0.005 Not 0.01

The scoring docs recommend floor=0.01, but this is overly conservative.

- GT minimum nonzero is 0.005 (1/200)
- Using floor=0.005 on classes that CAN appear saves probability mass
- Using floor=0 on classes that CANNOT appear (mountain inland, port inland) saves even more
- Combined saving: ~1.5-2% probability mass per cell → **+5 score points**

### Smart Floor Rules
```
For each non-static cell:
  mountain (class 5) → 0.0    (never appears on non-mountain cells)
  port (class 2)     → 0.0    if not coastal (never appears inland)
  all other classes   → max(prediction, 0.005)  then renormalize

For static cells:
  ocean    → [1, 0, 0, 0, 0, 0]  (no floor needed)
  mountain → [0, 0, 0, 0, 0, 1]  (no floor needed)
```

## 9. Round Weights Compound — Later Rounds Are Worth More

| Round | Weight | Score 90 = Weighted |
|-------|--------|---------------------|
| R1 | 1.050 | 94.5 |
| R2 | 1.103 | 99.2 |
| R3 | 1.158 | 104.2 |
| R4 | 1.216 | 109.4 |
| R5 | 1.276 | 114.9 |
| R6 | ~1.340 | ~120.6 |

Leaderboard score = max(round_score × round_weight). A score of 90 on R5 would give 114.9 — enough to challenge the leader (113.9).

## 10. Backtested Performance

Full pipeline (CalibrationModel + GlobalMultipliers + FeatureKeyBuckets + SmartFloor):

| Round | Old Score | Backtested | Delta | Regime |
|-------|-----------|------------|-------|--------|
| R2 | 66.4 | 89.2 | +22.8 | Thriving |
| R3 | 68.0 | 92.8 | +24.8 | Total collapse |
| R4 | 53.3 | 92.8 | +39.5 | Near-collapse |

Consistent ~90-93 across all regimes. At R5 weight (1.276), a score of 92 = **117.4 weighted**.
