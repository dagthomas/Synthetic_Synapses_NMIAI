# Session Log — 2026-03-20

## Timeline

### Phase 1: Round 3 Analysis → Floor Fix (R4 submitted)
- Analyzed R3 (68.0, rank 11): total collapse, 0 settlements survived
- **Discovery**: Empty/Forest ratio was 98.6% of error in collapse rounds
- **Discovery**: Our old floor (0.001-0.005) was below the scoring minimum (0.01)
- Changed floor to 0.015 (later found too high), then 0.01
- Submitted R4 with floor fix → **53.3 (rank 65)** — REGRESSION
- **Lesson**: Floor 0.015 wastes mass on mountain/port/ruin (3.7% per cell)

### Phase 2: CalibrationModel + Smart Floor (R5 submitted)
- Built CalibrationModel from R1-R4 ground truth (hierarchical: fine→coarse→base→global)
- **Discovery**: GT is exactly N=200 simulations (probabilities = k/200)
- **Discovery**: Mountain ALWAYS 0 on dynamic cells, Port ALWAYS 0 on non-coastal
- Implemented smart floor: zero impossible classes, floor remaining at 0.005
- Smart floor alone: +5 points across all rounds
- CalibrationModel + FK buckets + multipliers + smart floor = 92.5 avg backtest
- Submitted R5 → **86.3 (RANK 1!)** — first place

### Phase 3: Autoloop + ADK Agent (R6 submitted)
- Built autoloop_fast.py: 160k experiments/hour, vectorized predictions
- 5000+ experiments, converged at ~91 (simplified harness)
- Built Google ADK research agent with Gemini Pro
- ADK agent found: distance-aware multiplier power + selective smoothing (+0.5 pts)
- **Discovery**: Port smoothing leaks mass to inland (must exclude port from smoothing)
- **Discovery**: Settlement cells (dist=0) need higher multiplier power (0.75) than expansion (0.50)
- Submitted R6 with Gemini code → **87.6 (rank 2)**

### Phase 4: Deep Analysis + Idea Testing (R7 submitted)
- **Discovery**: R6 was booming (244% survival, highest ever)
- **Discovery**: Observed sett% ≈ GT sett% (strongest regime signal)
- **Discovery**: Coastal settlements die 2x more than inland
- **Discovery**: Sparse settlements survive 2x better than dense clusters
- **Discovery**: Expansion only happens at distance 1-2
- **Discovery**: Ruins form on expanded-then-collapsed cells, not initial settlements
- Tested Idea H (direct sett% regime scaling): seemed +0.61 but was ACTUALLY -0.6
- **Critical Lesson**: Always validate against CORRECT baseline on ALL rounds
- Reverted Idea H, verified production = winner experiment at 90.9 avg
- Submitted R7 with clean winner code

### Phase 5: Multi-Model Researcher
- Built multi_researcher.py: Haiku (analysis) → Opus (code) → Flash-Lite (extraction) → Backtest
- **Key fix**: "code printer" prompt framing + JSON output + Gemini extraction
- 20 iterations of Opus code generation, 55% success rate
- No improvement found over baseline (91.8) — pipeline is near-optimal for single-function changes
- **Conclusion**: Need architectural changes, not code tweaks

## What We Learned About the Simulation

### Hard Rules (100% reliable)
1. Mountain never changes (GT = [0,0,0,0,0,1])
2. Ocean never changes (GT = [1,0,0,0,0,0])
3. Border is always ocean
4. Mountain prob = 0 on ALL non-mountain dynamic cells
5. Port prob = 0 on ALL non-coastal cells
6. GT computed from exactly 200 simulations (probs = k/200)

### Settlement Dynamics
7. Coastal settlements die at 2x rate of inland
8. Dense clusters (2+ neighbors in r=5) die at 2x rate of sparse
9. Expansion only at distance 1-2 from settlements (never dist≥4)
10. Ruins form where expanded settlements later collapse
11. Settlement survival ranges 0% (R3) to 244% (R6) across rounds
12. Observed mid-sim settlement % directly predicts GT settlement %

### Scoring
13. `score = 100 × exp(-3 × weighted_kl)` — exponential decay
14. Only entropy>0.01 cells contribute (ocean/mountain excluded)
15. 0.005 is the minimum nonzero GT probability
16. Theoretical max with smart floor: 98-100 depending on round

### What Works
17. CalibrationModel (historical ground truth) is the strongest prior
18. Feature-key bucketed empirical (~100 obs/bucket) beats per-cell Bayesian
19. Global multipliers with per-class dampening detect regime
20. Smart floor (zero impossible + floor at 0.005) saves ~2% mass/cell
21. Distance-aware multiplier power (dist=0: 0.75, dist≥1: 0.50)
22. Selective smoothing (settlement/ruin only, NOT port)
23. Regime-adaptive empirical trust (collapse→trust obs more)

### What Doesn't Work
24. Per-cell Bayesian update with 1-2 observations (overfits, -30 pts)
25. Zeroing ruin on non-settlement cells (-34 pts, ruins CAN form anywhere)
26. Manhattan distance caps for expansion (-2.3 pts, ruins exist far away)
27. Floor 0.015 (-2 pts vs 0.005, wastes mass)
28. Log-odds space blending (catastrophic, -29 pts)
29. Idea H (direct sett% scaling) looked good in isolation, hurt in production
30. 20 Opus iterations of single-function code changes: none beat baseline

## Architecture Decisions

### Production Pipeline
```
explore.py (50 queries, fixed 3x3 grid + 5 adaptive)
  → GlobalMultipliers + FeatureKeyBuckets from observations
  → predict_gemini.py:gemini_predict()
    1. CalibrationModel (6 rounds, 48k cells)
    2. FK empirical blend (prior*5 + emp*sqrt(count), max 8)
    3. Distance-aware multipliers (dist=0: power=0.75, dist≥1: 0.50)
    4. Temperature softening near settlements
    5. Selective spatial smoothing (sett/ruin, not port)
    6. Smart floor (mountain=0, port=0 inland, rest=0.005)
    7. Lock static + borders
```

### Research Infrastructure
- `autoloop_fast.py`: 160k experiments/hr, numeric parameter search (converged)
- `multi_researcher.py`: Haiku→Opus→Flash-Lite→Backtest, ~50s/iteration
- `research_agent/`: Google ADK agent with Gemini Pro (7 tools)
- `test_ideas.py`: Quick idea validation against full harness
- `autoexperiment.py`: Predefined experiment suite

### Key Files
- `predict_gemini.py`: Production prediction (THE function that runs)
- `calibration.py`: CalibrationModel from historical ground truth
- `fast_predict.py`: Vectorized prediction for autoloop
- `submit.py`: Full pipeline orchestrator (calls gemini_predict)
- `.env`: API keys (GOOGLE_API_KEY, ASTAR_TOKEN)

## Score History
| Round | Score | Rank | Code Version | Key Change |
|-------|-------|------|-------------|-----------|
| R1 | - | - | No submission | - |
| R2 | 66.4 | 53 | Old code | Hardcoded R1 priors |
| R3 | 68.0 | 11 | Old code + observations | GTM + Bayesian |
| R4 | 53.3 | 65 | Floor 0.015 (broken) | Floor too high |
| R5 | 86.3 | **1** | Cal + FK + smart floor | Full new pipeline |
| R6 | 87.6 | 2 | + Gemini improvements | Distance-aware mult |
| R7 | pending | - | Same as R6 (clean) | Reverted Idea H |

## Phase 6: R7 Analysis + R8 Recovery

### R7 Post-Mortem (score=74.0, rank=2)
- **EXTREME BOOM**: 384% settlement survival, 25.6% sett probability
- Observations showed 14% sett (mid-sim) → detected as MODERATE, not BOOM
- Calibration (averaging 7 rounds incl. collapse) underpredicted settlement
- **Per-regime calibration tested**: didn't help (88.1 vs 88.2 standard)
- **Blending sweep found**: emp_scale=1.5 gives +0.1 avg, +0.7 on R7
- **Key lesson**: Extreme booms can't be detected from mid-sim observations

### R8 Crisis: Lost Observations
- Unicode arrow crash in explore.py killed the process mid-exploration
- All 50 queries used but observation data NOT saved (save was after print)
- R8 submitted with calibration-only (no observations) → will score lower
- **Three fixes applied**: save-first, unicode removed, None FK handling

### Research Conclusion (75+ experiments)
- 45+ Claude Opus code modifications: none beat baseline
- 30-iteration big run: best = 91.8 (exact tie)
- Per-regime calibration: no improvement
- **Pipeline is at architectural ceiling** for single-function changes

## Score History (Updated)
| Round | Score | Rank | Key Factor |
|-------|-------|------|-----------|
| R2 | 66.4 | 53 | Old code |
| R3 | 68.0 | 11 | Old code |
| R4 | 53.3 | 65 | Broken floor |
| R5 | 86.3 | **1** | New pipeline |
| R6 | 87.6 | 2 | + Gemini improvements |
| R7 | 74.0 | 2 | Extreme boom (undetectable) |
| R8 | pending | - | Cal-only (observations lost) |

## Phase 7: Boom Optimization + Daemon (2026-03-21)

### Boom Round Analysis
- R11 = 87.6 (#24) — boom round similar to R6 (22% settlements)
- Pattern: #3-4 on moderate/decline, #24 on boom
- EXTREME_BOOM detection helped (+0.5-1.6/seed) but not enough

### Major Rewrite: predict_gemini.py Vectorized (+10 avg!)
Rewrote from per-FK-key loop to vectorized numpy. Three core breakthroughs:

1. **Lower prior weight** (5.0 → 1.5): FK observations capture boom signal, calibration prior is biased toward moderate
2. **Cell-level distance dampening** (exp_damp=0.4): Full multiplier at settlement cells, 40% at expansion. Old per-FK power exponents were much less effective.
3. **Entropy-weighted global temperature** (T_high=1.15): Soften ALL uncertain cells globally, not just near settlements.

Additional: base_power 0.4→0.3, floor 0.005→0.008, smoothing 25%→15%, emp_max 8→20.

LOO backtest: 78.54 → 88.60 avg (+10.06), boom 69.55 → 83.59 (+14.04).

### Autonomous Daemon
Built `daemon.py` — runs unattended:
- Starts autoloop_fast.py for continuous parameter optimization
- Monitors for new rounds, pauses autoloop, explores + submits, resumes
- Downloads completed round data, restarts autoloop with fresh calibration
- Syncs best_params.json from autoloop to production automatically
- Health-checks autoloop, restarts if crashed

### Infrastructure Changes
- `predict_gemini.py` now reads from `best_params.json` (auto-updated by autoloop)
- `autoloop_fast.py` writes `best_params.json` on new best
- R11 added to harness (9 rounds total)
- `eval_production.py` + `sweep_production.py`: direct production evaluation
- DEFAULT_PARAMS updated to match new production baseline

### R12 Submitted
- R12 detected as MODERATE (13.7% settlement, ratio 0.822)
- All 5 seeds accepted with improved code

## Score History (Updated)
| Round | Score | Rank | Key Factor |
|-------|-------|------|-----------|
| R2 | 66.4 | 53 | Old code |
| R3 | 68.0 | 11 | Old code |
| R4 | 53.3 | 65 | Broken floor |
| R5 | 86.3 | **1** | New pipeline |
| R6 | 87.6 | 2 | + Gemini improvements |
| R7 | 74.0 | 2 | Extreme boom (undetectable) |
| R8 | 66.9 | - | Cal-only (observations lost) |
| R9 | 93.65 | **4** | + Adaptive exploration |
| R10 | 93.0 | **3** | Solid moderate performance |
| R11 | 87.6 | 24 | Boom round weakness |
| R12 | pending | - | New vectorized prediction |

## Production Pipeline (Current)
```
daemon.py (autonomous orchestrator)
  ├── autoloop_fast.py → best_params.json (continuous optimization)
  └── Round detection → explore.py → predict_gemini.py → submit
                                       ↑ reads best_params.json
```

### predict_gemini.py (Current)
```
1. CalibrationModel (11 rounds, 88k cells)
2. FK empirical blend (prior_w=1.5, emp_max=20, sqrt strength)
3. Global multiplier (base_power=0.3, cell-level distance dampening 0.4)
4. Entropy-weighted global temperature (T_high=1.15)
5. Selective spatial smoothing (15%, settlement/ruin only)
6. Proportional redistribution of structural zeros
7. Floor at 0.008, lock static + borders
```

## Config Consistency (All Updated to R2-R11)
- autoloop_fast.py: R2-R11 ✓
- eval_production.py: R2-R11 ✓
- daemon.py: auto-detects all rounds ✓
