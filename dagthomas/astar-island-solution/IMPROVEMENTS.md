# Astar Island — Improvements Log

## Round 2 — Submissions (2026-03-19)

### Submission 1: Initial pipeline
- Built full pipeline: config, client, utils, explore, predict, submit, analyze, train
- Static prior (Tier 1) then observation-informed (Tier 2) with 50 queries
- 3-bucket distance priors (near/mid/far), hardcoded blend weights
- ~81% cell coverage per seed

### Submission 2: Calibrated priors
- Pulled Round 1 ground truth, calibrated priors from real data
- Bayesian (Dirichlet) updating replaced hardcoded blend weights
- Still using 3-bucket distance priors

### Submission 3: Feature-rich priors
- **Per-integer-distance interpolation** from R1 ground truth (d=1..9+ for plains and forest)
- **Coastal awareness**: coastal plains get port probability, coastal settlements too
- **Cluster density**: settlement survival varies by sett_r5 (neighbors within radius 5)
- **Per-seed viewport planning**: each seed gets its own viewport plan since terrain isn't shared
- **Wider spatial smoothing**: radius-2 kernel weighted by observation count and distance

### Submission 4: Parameter-adaptive predictions (current)
- **ParameterEstimator** (`estimator.py`): analyzes settlement stats from observations
  - Tracks population, food, wealth, defense, faction sizes
  - Compares observed terrain transitions to R1 baseline
  - Detects regime: winter harshness, trade activity, expansion rate, conflict structure
  - Recommends adaptive Dirichlet prior strength
  - Computes prior adjustment multipliers for settlement survival, expansion, forest loss
- **Adaptive prior strength**: 2.53 for R2 (vs default 3.0) — slightly more trust in observations
- **Prior adjustments**: survival=0.99x, expansion=0.98x, forest_loss=1.01x (R2 nearly matches R1)
- **Forest adjacency bonus**: settlements with 3+ adjacent forests get boosted survival
- R2 findings: low population (1.16 vs R1 2.5), high food (mild winters), very fragmented (54 factions)

### Score
- **Round 2: 66.4, rank 53/153** (seed scores: 64.5, 68.0, 66.2, 67.6, 65.7)
- Main error: underestimating settlement expansion at d=4-8 (R1 priors too conservative)

### Architecture v2 (deployed 2026-03-20)
- **Fixed 3x3 grid coverage**: 9 viewports at (0,13,25)x(0,13,25) = 100% map coverage per seed
  - 45 queries for full coverage, 5 for adaptive on highest-uncertainty seeds
  - Previous greedy approach only covered ~81%
- **Cross-seed pooling via GlobalTransitionMatrix**: pools terrain transitions across all 5 seeds
  - Since hidden parameters are identical across seeds, this gives 5x more data
  - GTM overrides R1 lookup tables where it has sufficient data (>=5 observations per bucket)
  - Blend: 60% global (this round) + 40% R1 fallback
  - Coastal variants tracked separately
- **Full pipeline**: grid coverage -> cross-seed pooling -> parameter estimation -> adaptive Bayesian prediction

---

## Key Findings from Round 1 Ground Truth (8000 cells, 5 seeds)

### Distance is the dominant feature
| Terrain | d=1 | d=3 | d=5 | d=7 | d=9+ |
|---------|-----|-----|-----|-----|------|
| Plains->Settlement | 22.6% | 15.1% | 5.2% | 1.0% | 0% |
| Forest->Settlement | 23.1% | 15.5% | 5.1% | 0.9% | 0.1% |
| Forest->Forest | 64.3% | 73.9% | 92.6% | 99.0% | 99.9% |

### Coastal effect is huge for ports
- Coastal plains d<=5: P(port) = 11.6% vs inland: P(port) = 0%
- Coastal settlements: P(port) = 34.6% vs inland: P(port) = 0%

### Settlement cluster density affects survival
- sett_r5=1 (isolated): P(survive) = 38.8%
- sett_r5=2: P(survive) = 44.6% (best!)
- sett_r5=5+: P(survive) = 31.6% (overcrowded -> more conflict?)

### Mountains never change (100% static across all 8000 cells)

---

## Critical Insight: Hidden Parameters Change Per Round

From `overview.md`: *"the admin creates a round with a fixed map, **many hidden parameters**"*
and *"Hidden parameters: Values controlling the world's behavior (same for all seeds in a round)"*

**This means:** winter severity, expansion rate, raid aggression, trade range, etc. differ
between rounds. Our R1 priors may be wrong for R2 if parameters shifted significantly.

**Implications:**
- Cannot blindly use R1 lookup tables — they reflect R1's specific parameter set
- Observations from the current round are MORE valuable than historical priors
- Need to detect parameter shifts: compare observation patterns vs R1 expectations
- Ideal: use R1 priors as starting point, then let Bayesian updating correct quickly

---

## Unused Data: Settlement Internal Stats

Our simulate queries return full settlement stats that we collect but NEVER use:
```json
{"population": 3.161, "food": 0.375, "wealth": 0.073, "defense": 1.0,
 "has_port": false, "alive": true, "owner_id": 28}
```

**How to use this (HIGH IMPACT for next round):**
- **Population/food** predict survival: low food = collapse imminent
- **owner_id** reveals faction structure: dominant factions survive, small ones get raided
- **Aggregate stats** reveal hidden parameters: if avg population is low, winter is harsh
  this round; if many ports, trade range is high
- **Dead settlements in observed viewport** directly tell us ruin probability
- Can build a "parameter fingerprint" from a few observations to adjust priors

---

## Remaining Improvements (Priority Order)

### 1. Use settlement stats from observations (HIGH IMPACT)
**Status:** Data collected but unused
- Aggregate population/food/wealth across observations to estimate hidden parameters
- Low avg food -> harsh winters -> more ruins, fewer settlements
- Many ports -> high trade -> settlements near coast survive better
- Use to adjust Dirichlet prior strength: if observations suggest different regime
  from R1, lower PRIOR_STRENGTH to let observations dominate faster

### 2. Parameter-adaptive priors (HIGH IMPACT)
**Status:** Not started
- After first ~10 queries, compare observed terrain distributions vs R1 expectations
- If significantly different, reduce PRIOR_STRENGTH from 3.0 to 1.0
- If similar, keep priors strong
- Could also interpolate between multiple round priors once we have R2 ground truth

### 3. Cross-round prior refinement (HIGH, ongoing)
**Status:** Round 1 collected. Need Round 2 after it completes.
- After R2 scores: compare R1 vs R2 ground truth to measure parameter drift
- Build "envelope" priors that cover the range of observed parameters
- `results.json` now saved per round for easy comparison

### 4. Gradient boosted model (MEDIUM, needs 3+ rounds)
**Status:** `train.py` skeleton ready
- Features: terrain, distance, coastal, adj_forest, cluster density,
  PLUS aggregated settlement stats as global features
- Need enough data to avoid overfitting

### 5. Smarter query allocation (MEDIUM)
**Status:** Per-seed planning done. Could be smarter.
- Allocate more queries to seeds with more dynamic area
- Consider information gain per query
- First few queries: broad coverage. Later: target high-entropy cells

### 6. Forest adjacency for settlement survival (MEDIUM)
**Status:** Not used
- From mechanics: "Forests provide food to adjacent settlements"
- Settlements with more adjacent forest should survive better
- R1 data shows weak signal (adj_forest doesn't strongly predict forest->settlement)
  but may predict settlement survival better

### 7. Faction/owner_id analysis (LOW-MEDIUM)
**Status:** Data collected, not analyzed
- Large factions (many settlements with same owner_id) may dominate
- Could predict which settlements survive based on faction size

---

## Client Robustness Fixes
- [x] Retry on 429 for POST (with backoff)
- [x] Retry on 429 for GET (with backoff)
- [x] Init retry with generous backoff
- [x] Per-round results.json saved automatically
- [ ] Save/load accumulators (avoid rebuilding from JSON files)
- [ ] Add timeout handling for slow API responses
