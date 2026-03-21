# Astar Island — Complete Knowledge Base

All findings from R1-R5 analysis, 10,000+ automated experiments, and cross-round pattern mining.

## Simulation Rules (Confirmed from Ground Truth)

### Hard Rules (100% reliable across all rounds)
1. **Mountain (code 5) never changes** — GT is always [0,0,0,0,0,1]
2. **Ocean (code 10) never changes** — GT is always [1,0,0,0,0,0]
3. **Border cells are always ocean** — entire map border is static
4. **Mountain probability is always 0 on non-mountain dynamic cells** — zero it for free
5. **Port probability is always 0 on non-coastal cells** — zero it for free
6. **GT is computed from exactly 200 simulations** — all probabilities are k/200 (multiples of 0.005)
7. **Minimum nonzero GT probability is 0.005** (1 occurrence in 200 sims)

### Settlement Dynamics (Pattern varies by round)

**Survival depends on:**
- **Coastal vs inland**: Coastal settlements die MORE (0-36% survival) than inland (30-64%). Reason: ports attract raiding, coastal cells are more exposed.
- **Cluster density**: Sparse settlements (≤1 neighbor within Manhattan distance 5) survive 47-77%. Dense clusters (2+ neighbors) only survive 11-49%. Competition and raiding between neighbors kills dense clusters.
- **Round regime**: Settlement survival ranged from 0% (R3 total collapse) to 62% (R2 thriving) across 5 rounds.

**Expansion:**
- Only happens at distance 1-2 from existing settlements. dist=1: 1.8%, dist=2: 1.1%, dist=3: 0.2%, dist=4+: ~0%.
- Very rare overall — even in thriving rounds, only ~1% of nearby plains become settlements.

**Port formation:**
- Requires coastal location + proximity to settlement (avg dist = 3.3).
- Port probability >5% only within distance ~4 of a settlement.
- Initial ports almost never survive as ports — they collapse to empty.

**Ruin formation:**
- In thriving rounds, ruins form AWAY from initial settlements. R5: only 8/48 ruin cells from initial settlements, 40 from expanded-then-collapsed settlements.
- In collapse rounds, ruins are rare (few cells reach ruin state; they go straight to empty).
- Ruin probability correlates with intermediate settlement survival — need settlements to exist first before they can become ruins.

### Forest Dynamics
- Forest is extremely stable — 97-100% retention rate even near settlements.
- Forest loss (to settlement/empty) only happens at distance 1-2 in thriving rounds.
- In collapse rounds, forest actually GROWS (reclaims abandoned land).
- Forest neighbors count (in feature key) helps predict stability.

## Round Regime Classification

| Round | Survival | Expansion | Dynamic | Regime | Key Characteristic |
|-------|----------|-----------|---------|--------|-------------------|
| R1 | 57.2% | 0.8% | 1318 | THRIVING | High survival, moderate expansion |
| R2 | 61.7% | 0.2% | 1355 | THRIVING | Highest survival, low expansion |
| R3 | 0.0% | 0.0% | 419 | COLLAPSE | Total death, very few dynamic cells |
| R4 | 0.8% | 0.0% | 1341 | COLLAPSE | Near-total death but many dynamic cells |
| R5 | 28.3% | 0.9% | 1259 | MODERATE | Partial survival, some expansion |

**Key insight**: R3 and R4 are both collapse rounds but very different. R3 has few dynamic cells (settlements died without creating uncertainty). R4 has many dynamic cells (settlements struggled, creating high uncertainty). This means the *amount of uncertainty* is a separate parameter from *settlement survival*.

## Scoring Formula

```
score = 100 × exp(-3 × weighted_kl)
weighted_kl = Σ(entropy(cell) × KL(gt, pred)) / Σ(entropy(cell))
```

Where `entropy(cell) = -Σ p_i log(p_i)` using ground truth probabilities. Only cells with entropy > 0.01 contribute.

**Score sensitivity**: wKL=0 → 100, wKL=0.05 → 86.1, wKL=0.10 → 74.1, wKL=0.15 → 63.8

## Error Analysis Across Rounds

### R2 (actual score: 66.4 → backtested 91.3)
- **Dominant error**: Settlement class (10.8% of wKL)
- **Regime**: Thriving (61.7% survival)
- **Our fix**: CalibrationModel + FK buckets fixed settlement prediction

### R3 (actual score: 68.0 → backtested 93.0)
- **Dominant error**: Empty/Forest ratio (98.6% of wKL)
- **Regime**: Total collapse (0% survival)
- **Our fix**: Smart floor (zero mountain/port) + multipliers

### R4 (actual score: 53.3 → backtested 93.8)
- **Dominant error**: Settlement + Empty/Forest (50/50)
- **Regime**: Near-collapse (0.8% survival)
- **Root cause of bad score**: Floor at 0.015 wasted mass; no calibration model
- **Our fix**: CalibrationModel + smart floor + multipliers

### R5 (actual score: 86.3, RANK 1)
- **Dominant error**: Settlement (4.5% of wKL)
- **Regime**: Moderate (28.3% survival)
- **Remaining gap**: Port underpredicted on coastal cells; settlement underpredicted on near-settlement cells
- **Theoretical max**: 98.5

## Architecture

### Pipeline
```
1. CalibrationModel (5 rounds, 40k cells, 125 fine keys)
   → Hierarchical prior: fine → coarse → base → global
2. Feature-key bucketed empirical (~100 obs/key from current round)
   → Blended with calibration: prior * 5.0 + empirical * sqrt(count)
3. Global multipliers (observed/expected ratio, power=0.4)
   → Per-class dampening: settlement/port/ruin use power=0.6
4. Smart floor (mountain=0, inland port=0, rest=0.005)
5. Static cell locking (ocean=[1,0,0,0,0,0], mountain=[0,0,0,0,0,1])
```

### Key Parameters (production values)
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| fk_prior_weight | 5.0 | Trust calibration (5 rounds of data) |
| fk_max_strength | 8.0 | Cap empirical weight (sqrt scaling) |
| fk_min_count | 5 | Need ≥5 observations to use FK data |
| mult_power | 0.4 | Base dampening on observed/expected |
| mult_power (sett/port) | 0.6 | More dampening for volatile classes |
| mult_sett_clamp | [0.15, 2.5] | Wide range for regime adaptation |
| floor_nonzero | 0.005 | = 1/200, minimum GT granularity |

### Backtested Scores (leave-one-out, production pipeline, emp_scale=1.5)
| Round | Backtest | Actual | Regime |
|-------|----------|--------|--------|
| R2 | 92.0 | 66.4 | Thriving (old code) |
| R3 | 93.9 | 68.0 | Collapse (old code) |
| R4 | 93.5 | 53.3 | Near-collapse (broken floor) |
| R5 | 87.7 | 86.3 | Moderate |
| R6 | 87.5 | 87.6 | Booming |
| R7 | 75.3 | 74.0 | EXTREME boom (384% survival) |
| **Avg** | **88.3** | | |

### R7 Post-Mortem
R7 was the most extreme round: 384% settlement survival, 25.6% settlement probability.
Our observations showed 14% settlement (mid-simulation), but GT ended at 25.6%.
The multiplier barely activated (ratio=1.09) because expansion happened AFTER observation.
**Key lesson: extreme boom rounds can't be detected from mid-sim observations.**
Per-regime calibration was tested and didn't help (detected MODERATE, not BOOM).

### Score vs Weighted Leaderboard
| Round | Weight | Score=86 | Score=90 | Score=93 |
|-------|--------|----------|----------|----------|
| R5 | 1.276 | 109.7 | 114.9 | 118.7 |
| R6 | 1.340 | 115.2 | 120.6 | 124.6 |
| R7 | 1.407 | 121.0 | 126.6 | 130.9 |

## What We're Still Missing (Improvement Opportunities)

### 1. Coastal settlement handling (est. +1-2 pts)
Coastal settlements die at 2× the rate of inland ones. Our calibration model averages
them but our feature key DOES distinguish coastal (fk[2]=True). The issue is the FK
empirical from current round may not have enough coastal settlement observations.
**Fix**: Add a separate coastal_settlement prior that predicts higher collapse rates.

### 2. Settlement density signal (est. +0.5-1 pt)
Dense clusters die much faster. Our feature key has distance bucket but not explicit
density count. Adding "number of settlements within radius 3" to the feature key
could help. But this increases key cardinality and reduces observations per key.
**Fix**: Add density as a boolean (dense vs sparse) to keep cardinality manageable.

### 3. Ruin location prediction (est. +0.5 pt)
Ruins form where EXPANDED settlements later collapse — not on initial settlement
positions. Our model doesn't know which cells had expansion followed by collapse.
**Fix**: Use observation data to detect expansion (cell was plains, now settlement)
and predict higher ruin probability on those cells.

### 4. Observation-based cell-level updates (est. +2-5 pts)
Currently DISABLED because per-cell Bayesian update with 1-2 samples overfits.
The reference solution uses feature-key bucketed updates (which we do) but ALSO
cell-level updates with high weight (strength=2.5×count, cap 18).
**Fix**: Re-enable cell-level updates but with MUCH higher prior strength (10-20)
to prevent overfitting to single samples.

### 5. Viewport selection optimization (est. +1-2 pts)
Currently fixed 3×3 grid wastes queries on ocean-heavy viewports.
Interest-scored selection would focus queries on settlement-dense areas.
**Fix**: Implement the reference solution's viewport scoring approach.

## Autoloop (Autonomous Experiment System)

See `AUTOLOOP.md` for full documentation. Key results:
- 6,500+ experiments in 3 minutes (160k/hr)
- Found: per-class multiplier dampening, FK linear scaling, heuristic blend
- Converged at 89.6 avg (autoloop harness) / 91.3 avg (production pipeline)
- CalibrationModel weight tuning was the biggest previously-untapped knob
