# Astar Island — Ideas & Findings Log

## Proven Improvements (Implemented)

### 1. Smart Floor — Zero Impossible Classes (+5 pts)
**Source:** Manual GT analysis (R1-R4)
**Score:** 87→92 avg
Mountain is ALWAYS 0 on dynamic cells. Port is ALWAYS 0 on non-coastal cells.
Zeroing these saves 1.5-2% probability mass per cell for correct classes.
Floor remaining at 0.005 (1/200 = min GT granularity).

### 2. CalibrationModel from Historical Ground Truth (+10-15 pts)
**Source:** Reference solution analysis
**Score:** 68→84 avg
Hierarchical prior: fine feature key → coarse → base → global.
Each round of ground truth data improves it. Currently 6 rounds (48k cells, 125 fine keys).

### 3. Feature-Key Bucketed Empirical (+5-8 pts)
**Source:** Reference solution analysis
**Score:** 84→90 avg
Pool current-round observations by feature key (~100 obs/bucket vs 1-2 per cell).
Blend: prior * 5.0 + empirical * sqrt(count), capped at 8.0.

### 4. Global Multipliers with Wide Clamps (+5-10 pts)
**Source:** Reference solution + manual analysis
**Score:** Integrated with above
Observed/expected ratio per class. Settlement clamp [0.15, 2.5].
Per-class dampening power (settlement/port=0.6, rest=0.4).

### 5. Distance-Aware Multiplier Power (+0.5 pts)
**Source:** ADK Gemini Research Agent (iteration 3-4)
**Score:** 91.3→91.8 avg
Settlement cells (dist=0): power=0.75 (highly reactive to survival).
Expansion cells (dist≥1): power=0.50 (dampened — survival ≠ expansion).

### 6. Selective Spatial Smoothing (+0.3 pts)
**Source:** ADK Gemini Research Agent (iteration 3-4)
**Score:** Included in above
Smooth settlement and ruin classes with 3x3 uniform filter (α=0.75).
Do NOT smooth port — smoothing leaks port mass to inland cells where it gets zeroed.

### 7. Regime-Adaptive Empirical Trust (+0.2 pts)
**Source:** ADK Gemini Research Agent
In collapse (settlement ratio ~0): trust observations up to weight 12.
In thriving (ratio ~1.5): trust observations max weight 6 (trust prior more).

### 8. Dynamic Temperature Softening (+0.1 pts)
**Source:** ADK Gemini Research Agent
Near settlements: apply temperature T > 1.0 to soften predictions (spread probability).
Radius and T scale with settlement survival ratio.

## Proven Harmful (Do NOT Implement)

### Dirichlet Full Matrix Calibration (-0.1 avg) ❌
6x6 weight matrix in log-prob space. LOO test: Diag -0.08 avg, Full ~-0.1 avg.
All methods (Histogram, Isotonic, Dirichlet) hurt or are neutral.
The prediction pipeline already captures cross-class interactions via feature keys,
global multipliers, and temperature scaling. Post-hoc recalibration double-corrects.
Files kept for reference: `dirichlet_cal.py`, `fit_dirichlet.py`.

### Zero Ruin on Non-Settlement Cells (-34 pts!)
Ruins CAN form on plains/forest (from expanded-then-collapsed settlements).
Zeroing ruin on non-settlement initial cells is catastrophically wrong.

### Zero Settlement/Port/Ruin at Distance > 8 (-0.8 pts)
Expansion can reach beyond distance 8 in thriving rounds.

### Manhattan Expansion Caps at Distance > 6 (-2.3 pts)
ADK agent tested this — ruins exist far from settlements due to expand→collapse.

### Per-Cell Bayesian Update with 1-2 Observations (-30 pts!)
Overfits to single stochastic samples. Feature-key pooling is the correct approach.

### Floor = 0.015 (-2 pts vs 0.005)
Wastes probability mass. 0.005 is the minimum GT granularity (1/200 sims).

### Smart Threshold Floor (zero predictions < 0.002) (-0.3 pts)
ADK agent tested — slight regression. The Bayesian prior handles this naturally.

### Per-Regime Calibration (0 pts)
Weight historical rounds by regime similarity to current round.
Problem: can't detect BOOM from mid-sim observations (14% obs → MODERATE detected,
but GT was 25.6% BOOM). Backtested: 88.1 vs 88.2 standard — no improvement.

### 45+ Claude Opus Code Modifications (0 pts)
Multi-model researcher ran 45+ experiments with Opus code gen.
Best: 91.8 (exact tie with baseline). No single-function modification beats it.

### Alive-Count Settlement Boost (HARMFUL)
Boost settlement multiplier when observed alive count > 1400.
R7 improves +0.4-1.2 but R2 drops -0.5-6.7 and R6 drops -2.9-14.8.
The boost can't distinguish R6 (already correct) from R7 (needs boost).
**Any boost factor makes average WORSE.** The pipeline is at its ceiling.

### 9. Active Learning Observation Queries — SHIPPED
**Source:** Entropy-targeted viewport selection
Replace fixed 3-viewport diagonal scout with entropy-targeted viewports.
Phase 1: top-5 entropy viewports × 5 seeds = 25 queries for maximum info.
Phase 2: remaining budget on entropy-targeted coverage (all regimes).
BOOM regimes reserve 2 batches for repeat queries (multi-sampling).
Expected: reduce wasted queries on low-entropy (ocean/mountain) cells.

## Untested Ideas (Worth Exploring)

### A. Per-Settlement Population/Food Prediction
We observe settlement stats (population, food, wealth) during exploration.
Low food → higher collapse probability. High population → higher survival.
Currently we use this only for binary alive/dead, not as continuous features.
**Estimated impact:** +0.5-1 pt

### B. Settlement Cluster Survival Model
Dense clusters (2+ neighbors in r=5) die at 2x rate of sparse ones.
Add cluster density to the feature key or as a post-hoc multiplier.
**Estimated impact:** +0.3-0.5 pt (autoloop found this direction improves slightly)

### C. Coastal Settlement Penalty
Coastal settlements die at 2x rate of inland. Our feature key has coastal=True
but the calibration averages over all rounds. A per-round coastal penalty
based on observed coastal vs inland survival could help.
**Estimated impact:** +0.3-0.5 pt

### D. Forest Suppression Near Thriving Settlements
In thriving rounds, forests near settlements get cleared for expansion.
A distance-decay forest reduction (γ = e^{-λd}) near settlements could
improve empty/forest ratio.
**Source:** predict_gemini.py experimental_pred_fn (tested, marginal improvement +0.1)
**Estimated impact:** +0.1-0.3 pt (already partially implemented)

### E. Port Distance-Decay Boost on Coastal Cells
Port formation happens within distance ~3.3 of settlements on coastal cells.
A regime-adaptive coastal port boost (α * e^{-βd}) could improve port prediction.
**Source:** predict_gemini.py experimental_pred_fn (tested, marginal improvement +0.1)
**Estimated impact:** +0.1-0.3 pt (already partially implemented)

### F. Viewport Selection Optimization -- PARTIALLY IMPLEMENTED
Fixed 3x3 grid kept for 100% coverage (only 14% waste, not worth dropping).
Improved adaptive phase: ranks ALL seed-viewport combinations by dynamic score,
allowing most dynamic seeds to get 2-3 extra queries instead of 1 each.
Full viewport overhaul (57% coverage for 2.5x obs) too risky — calibration-only
cells perform much worse than observed cells.
**Actual impact:** Marginal (+0.1-0.3 from better adaptive allocation)

### F2. Critical Bug Fix: Data Loss Prevention
Moved save_exploration_data BEFORE print statements in explore.py.
Added graceful None handling in predict_gemini.py.
Fixed unicode arrow crash. **These prevent 10+ point drops from lost observations.**

### G. Cross-Seed Observation Sharing via Feature Keys
All 5 seeds share the same hidden parameters. Currently we pool observations
across seeds in FK buckets. Could we also share settlement-level statistics?
**Estimated impact:** Already done via FK bucketing

### H. Observed Settlement % as Regime Predictor ❌ REVERTED
CRITICAL FINDING: Observed sett% ≈ GT sett% (strong signal).
BUT scaling empirical trust by deviation HURTS on full R2-R6 harness.
Head-to-head: WITHOUT Idea H = 90.9, WITH Idea H = 90.3 (-0.6).
The regime-adaptive blending already captures this signal via ratio[1].
Adding a second scaling layer causes double-counting and overfitting.
**Lesson: Always validate against ALL rounds before deploying.** (more direct signal use)

### I. Per-Class Blending Weights ⚠️ TESTED — MIXED
Instead of one fk_prior_weight for all classes, use different weights:
- Forest: high prior weight (very stable, calibration is reliable)
- Settlement: high empirical weight (varies hugely between rounds)
**Backtested: +0.41 pts alone, but -0.21 when combined with H.**
Better to use H alone. Per-class FK may overfit on R6 (booming round).
Do NOT combine with H unless further evidence.

## Key Numbers (Reference)

| Metric | Value |
|--------|-------|
| GT simulations | 200 per seed |
| Min GT probability | 0.005 (1/200) |
| Scoring formula | 100 × exp(-3 × wKL) |
| Theoretical max (smart floor) | 98-100 depending on round |
| Current production backtest | 88.60 avg (R2-R11 LOO, 9 rounds) |
| Current actual scores | R9=93.65, R10=93.0, R11=87.6 |
| Autoloop harness baseline | 88.29 (9 rounds, new defaults) |
| Settlement survival range | 0% (R3) to 244% (R6) |
| Observed sett% ≈ GT sett% | Within ~2% |

## Session Findings (2026-03-20)

### R6 Analysis (score=87.6, rank=2)
- **BOOMING round** — 244% settlement survival (settlements more than doubled)
- Highest settlement activity of any round: 24.4% GT probability on dynamic cells
- Balanced KL decomposition: Empty 2.4%, Ruin 0.8%, Forest 0.8%, Sett 0.2%, Port 0.2%
- **Theoretical max = 100.0** — floor doesn't limit this round at all
- Our prediction was very close: pred sett=26.5% vs GT sett=24.4% (slightly over)

### Key Discovery: Observed Sett% = GT Sett%
The settlement percentage in mid-simulation observations directly predicts the GT settlement probability:
| Round | Observed Sett% | GT Sett% | Ratio |
|-------|---------------|----------|-------|
| R5 | 12.1% | 14.6% | 1.21 |
| R6 | 22.7% | 24.4% | 1.07 |
| R7 | 14.0% | ~15.4% est | ~1.10 |

This means the observed percentage is the strongest regime signal we have.

### Idea H Validation (+0.61 pts)
Using observed sett% deviation to scale empirical trust:
`confidence = 1.0 + 2.0 * abs(obs_sett_pct - 0.15)`
- Extreme rounds (collapse/boom): confidence >> 1, trust observations more
- Average rounds: confidence ≈ 1, no change
- Backtested: R2=+1.4, R3=+0.1, R4=+0.7, R5=+0.9, R6=-0.2

### ADK Agent Findings (Gemini Pro)
- Ran 11 successful experiments, best at 91.1 avg
- Independently discovered distance-aware multiplier power and selective smoothing
- Confirmed that hardcoded distance limits HURT (manhattan caps dropped score by 2.3 pts)
- Concluded the Bayesian framework is near-optimal for numeric parameters

### Autoloop Findings (6 rounds of data)
- 2000 experiments with R6 included, converged at 88.84 (simplified harness)
- Confirms parameter space is exhausted
- Best gains come from structural changes (per-class weights, regime scaling)

### Multi-Model Agent FIXED ✅
Pipeline: Haiku (direction) → Opus (code, "code printer" prompt) → Flash-Lite (extraction) → Backtest
~50s per iteration. ~55% success rate. Opus generates real prediction code that compiles.
Key: use "code printer" prompt framing, JSON output format, Gemini for code extraction.

**20 iterations completed. No improvement found over baseline (91.8).**
Tried: log-odds blending (62.4, catastrophic), observation-count weighting (91.4),
per-class dynamic weighting (crash), settlement food conditioning (crash),
coastal penalty (91.8 tie), proximity multipliers (90.3), regional density (crash).

**Conclusion: The current pipeline is near-optimal for single-function structural changes.
Further gains likely need architectural changes (e.g., viewport selection, observation
strategy, or ensemble of multiple prediction functions).**

## Regime Summary (All Rounds)

| Round | Survival | Observed Sett% | GT Sett% | Score | Regime |
|-------|----------|----------------|----------|-------|--------|
| R1 | 93% | - | 16.1% | - | Thriving |
| R2 | 106% | - | 20.5% | 66.4 | Thriving |
| R3 | 0% | - | 0.9% | 68.0 | Collapse |
| R4 | 0.8% | - | 9.6% | 53.3 | Near-collapse |
| R5 | 28% | 12.1% | 14.6% | 86.3 | Moderate |
| R6 | 244% | 22.7% | 24.4% | 87.6 | Booming |
| R7 | 384% | 14.0% | 25.6% | 74.0 | EXTREME BOOM |
| R8 | 0% | 9.4%* | 3.4% | 66.9 | COLLAPSE (no obs!) |
| R9 | - | - | - | 93.65 | Moderate |
| R10 | - | - | - | 93.0 | Decline |
| R11 | - | ~22% | 25-36% | 87.6 | Booming |

*R8 observations were collected but lost to unicode crash. Score would have been ~93 with observations.

## Session Findings (2026-03-21) — Boom Round Optimization

### R11 Analysis (score=87.6, rank=#24)
- **BOOMING round** — 25-36% settlement across seeds (seed 2 highest at 35.8%)
- 0 initial settlements (pure expansion from hidden settlements)
- Per-seed scores: 85.7, 86.5, 90.8, 87.2, 87.7
- EXTREME_BOOM detection helped (+0.5 to +1.6 per seed vs normal)
- Pattern clear: #3-4 on moderate/decline, #24 on boom

### Key Breakthrough: Vectorized Prediction (+10.06 avg!)
Rewrote predict_gemini.py from per-FK-key loop to vectorized numpy.
Old: 78.54 avg LOO. New: 88.60 avg LOO.

**Three core improvements:**
1. **Lower prior weight** (5.0 → 1.5): Trust FK empirical observations much more.
   Boom rounds: calibration prior is biased toward moderate, FK observations
   capture the actual boom signal. Single biggest lever.
2. **Cell-level distance dampening** (exp_damp=0.4): Apply full global multiplier
   only at settlement cells (dist=0), dampen to 40% at expansion cells (dist≥1).
   Old code used per-FK power exponents (0.75 vs 0.50) — much less effective.
3. **Entropy-weighted global temperature** (T_high=1.15): Soften ALL uncertain cells
   based on calibration entropy, not just cells near settlements. Huge on boom
   AND non-boom.

Additional: base_power 0.4→0.3, floor 0.005→0.008, smooth_alpha 25%→15%,
emp_max_weight 8→20.

### Per-Round Impact (LOO backtest)
| Round | Before | After | Delta | Type |
|-------|--------|-------|-------|------|
| R2 | 78.79 | 89.47 | +10.68 | Boom |
| R3 | 90.93 | 92.22 | +1.29 | Collapse |
| R4 | 85.42 | 92.66 | +7.24 | Moderate |
| R5 | 73.81 | 86.32 | +12.51 | Moderate |
| R6 | 72.36 | 87.06 | +14.70 | Boom |
| R7 | 56.56 | 73.78 | +17.22 | Extreme Boom |
| R9 | 83.00 | 92.73 | +9.73 | Moderate |
| R10 | 86.24 | 93.25 | +7.01 | Decline |
| R11 | 79.73 | 89.93 | +10.20 | Boom |
| **Avg** | **78.54** | **88.60** | **+10.06** | |
| **Boom** | **69.55** | **83.59** | **+14.04** | |
| **Non-boom** | **83.03** | **91.11** | **+8.08** | |

### Autoloop Updated
- R11 added to harness (9 rounds total, 5 seeds each)
- New DEFAULT_PARAMS matching production: prior_w=1.5, emp_max=20, T_high=1.15,
  base_power=0.3, floor=0.008, smooth_alpha=0.15, dist_aware_mult=True, exp_damp=0.4
- New parameters added: dist_aware_mult, dist_exp_damp, regime_prior_scale
- Old log archived as autoloop_fast_log_old_defaults.jsonl
- New harness baseline: 88.29 avg (boom=83.59, nonboom=90.65)

## Session Findings (2026-03-21) — Ceiling Analysis & Structural Investigation

### CMA-ES Optimizer: Parameter Space Exhausted
Replaced Metropolis-Hastings with CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
CMA-ES learns parameter correlations and navigates narrow valleys better.

**Result:** +0.015 pts (87.663 -> 87.678). Converged after 12,362 evaluations.
Sigma collapsed from 0.15 -> ~0.001, confirming the MH autoloop already found
the global optimum. **The 28-parameter continuous space is fully exhausted.**

File: `autoloop_cmaes.py` (drop-in replacement with `--resume` flag).

### Spatial Clustering in FK: Fragmentation Kills It
Added settlement cluster density (settlements within Manhattan d=5) to the feature key.
Two approaches tested:

1. **In FK (6-tuple):** -0.28 avg. Keys increased 103->137, diluting per-key calibration.
2. **Post-hoc multiplier:** +0.06 at boost=0.05, harmful at any higher value.

**Conclusion:** The pipeline already captures spatial info through distance bucketing.
Adding cluster density fragments the FK without sufficient data to fill the new buckets.

### Stochastic Cellular Automaton Analysis
**The Norse simulator is NOT Conway's Game of Life.** It's a multi-agent, multi-class,
stochastic territorial expansion simulator with:
- **Multiple factions (owners)** competing for territory
- **Per-settlement stats:** population, food, wealth, defense
- **Two hidden per-round regime parameters:** vigor and spread rate (lambda)

#### Key Finding: P(settlement | distance) = vigor * exp(-lambda * d)

Fitted exponential distance-decay kernel to ground truth:

| Round | Vigor  | Lambda | R^2   | Regime          |
|-------|--------|--------|-------|-----------------|
| R1    | 0.318  | 0.205  | 0.874 | Thriving        |
| R2    | 0.255  | 0.088  | 0.702 | Thriving (wide) |
| R3    | FAIL   | -      | -     | Collapse        |
| R4    | 0.164  | 0.191  | 0.860 | Moderate        |
| R5    | 0.400  | 0.417  | 0.928 | Moderate (local)|
| R6    | 0.355  | 0.113  | 0.847 | Boom (wide)     |
| R7    | 0.711  | 0.623  | 0.939 | Boom (local)    |
| R9    | 0.224  | 0.145  | 0.850 | Moderate        |
| R10   | 0.105  | 1.025  | 0.969 | Decline (sharp) |
| R11   | 0.533  | 0.184  | 0.876 | Boom (wide)     |
| R12   | 0.847  | 0.640  | 0.730 | Extreme boom    |

**Lambda varies 11.6x** between rounds (0.088-1.025). This is the biggest
per-round signal the pipeline doesn't explicitly model.

**Lambda IS estimable from 50 mid-sim observations** with <12% error:
R2=0.6%, R5=2.5%, R7=4.0%, R10=6.6%, R11=11.9%

**BUT: lambda kernel blending gives only +0.025 pts** because the existing
FK empirical + global multiplier already captures the per-round distance signal
implicitly. The lambda is NOT new information to the pipeline.

### Anti-Conway Effect Confirmed
- More neighboring settlements = LOWER survival (competition)
- No "birth threshold" like Conway's exactly-3 rule
- Settlement probability decays monotonically with distance to NEAREST settlement
- Gravity model (sum from ALL settlements) is WORSE than nearest-distance (r=0.265 vs r=0.377)
- Dense clusters suppress survival, not enhance it

### Diffusion Model: Captures Terrain Barriers (+0.02 within-bucket)
Reaction-diffusion solver routes settlement probability around mountains/oceans.
- Overall correlation: r=0.349 (vs 1/distance r=0.377) — slightly worse
- BUT within distance buckets: +0.02 discrimination at d=4-5 and d=6-8
- Mountains reduce P(settlement) by ~25% when blocking the nearest path
- With tuned hyperparams on subset: r=0.478 vs r=0.420 (+0.058)
- **Actionable for cells behind terrain barriers**, but overall impact is small

File: `test_diffusion.py`

### Regime-Conditional Calibration: +0.50 avg -- SHIPPED TO PRODUCTION
Weight historical rounds by Gaussian proximity to estimated vigor of current round.
Each FK key gets a regime-specific prior instead of a round-averaged prior.

**Best sigma=0.04:** avg +0.504, nonboom +0.640, boom +0.185
- R3 (collapse): +2.59 pts — huge win (weighted toward collapse rounds)
- R10 (decline): +1.42 pts
- R6 (boom): +0.57 pts
- Robust across sigma 0.02-0.06 (all give +0.48-0.50)

Implementation: `CalibrationModel.add_round(weight=...)` with
`weight = max(0.05, exp(-(vigor_diff^2) / (2*sigma^2)))`.
Vigor estimated from observations: `sett_count / total_dynamic_cells_observed`.

Files modified: `calibration.py`, `predict.py`, `predict_gemini.py`, `daemon.py`,
`submit.py`, `autoloop_fast.py` (FastHarness regime_conditional flag).

### Summary: What Works and What Doesn't

| Approach | Delta | Verdict |
|----------|-------|---------|
| CMA-ES optimizer | +0.015 | Confirms MH was optimal |
| Cluster in FK | -0.28 | FK fragmentation |
| Cluster multiplier | +0.06 | Marginal at best |
| Lambda kernel | +0.025 | Redundant with FK empirical |
| Gravity model | negative | Worse than nearest distance |
| Diffusion field | +0.02/bucket | Small, terrain barrier signal |
| Regime-conditional cal | +0.12 | Promising but needs proper implementation |
| Variance-weighted loss | untested standalone | Available in CMA-ES `--robust` |

### Why the Pipeline is Near Its Ceiling

The current pipeline captures:
1. **Per-distance class distribution** (FK calibration + empirical)
2. **Per-round regime** (global multiplier from observations)
3. **Spatial smoothing** (settlement/ruin smoothing)
4. **Uncertainty handling** (entropy-weighted temperature)

What it CANNOT capture:
1. **Terrain barrier routing** — mountains/oceans blocking expansion paths
2. **Multi-settlement geometry** — corridors, clusters, competition
3. **Per-faction dynamics** — different owners compete for territory
4. **Timestep-level dynamics** — expansion/collapse sequence within simulation

### Remaining Paths to Break the Ceiling

#### A. Regime-Conditional Calibration (Easiest, +0.1-0.5 est)
Build per-vigor calibration priors. For each FK key, learn how the distribution
changes with the round's vigor level. Use estimated vigor from observations
to interpolate. Clean implementation needed.

#### B. Diffusion-Enhanced Distance Feature (+0.2-0.5 est)
Use diffusion field as an ADDITIONAL feature (not replacement) for cells behind
terrain barriers. Compute diffusion_field / expected_diffusion_at_distance as a
ratio. High ratio = unobstructed path. Low ratio = behind a mountain.
Add as a 2-level bucket (blocked/unblocked) to FK or as a post-hoc multiplier.

#### C. Parametric Simulator + Ensemble ✅ IMPLEMENTED v2 — +3.43 avg

**v2 improvements:**
- Gaussian-power distance decay: exp(-(d/scale)^power) with hard max_reach cutoff
- Multi-start CMA-ES with regime-specific warm starts (collapse/moderate/boom)
- 16 hidden parameters (added decay_power, max_reach, cluster_optimal, cluster_quad)
- Pre-computed simulator cache (sim_precompute.py) for fast alpha sweeps
- Adaptive blend alpha per regime (0.25 collapse, 0.35 moderate, 0.45 boom)

**Backtest results (GT-fitted, 10 rounds, LOO, with pre-computed alpha sweep):**
- Statistical baseline: 80.19 avg
- Simulator alone: 71.55 avg
- **Ensemble at alpha=0.40: 83.62 avg (+3.43)**
- Improves on EVERY round. Best: round9 +6.4, round7 +4.1, round11 +3.8
- Best ensemble rounds: round4=90.0, round11=90.1, round9=89.5

**Key finding:** Gaussian decay (power=2-3) creates sharp cutoffs that match the
actual simulator's expansion pattern. Boom rounds use lower power (1.2-1.8) for
wider expansion, collapse rounds use higher power (3.0+) for tight clusters.

Files: sim_data.py, sim_model.py, sim_inference.py, sim_backtest.py, sim_precompute.py
Integration: predict_gemini.py (sim_pred + sim_alpha), daemon.py (auto-fits + adaptive alpha)

#### D. Small CNN on GT Data (+0.5-2 est)
Train a tiny convolutional neural network:
- Input: terrain (40x40), settlement positions, estimated (vigor, lambda)
- Output: 40x40x6 probability tensor
- Data: 55 training pairs (augmented with rotations/flips = 440)
- Risk: overfitting with so few samples. Need strong regularization.

#### E. Multi-Step Observation Strategy (+0.5-1 est)
Instead of allocating all 50 queries to maximize coverage, use a 3-phase approach:
- Phase 1 (15 queries): Estimate vigor and lambda
- Phase 2 (20 queries): Target cells where regime-conditional prior differs most
  from standard prior (information-gain sampling)
- Phase 3 (15 queries): Fill coverage gaps
