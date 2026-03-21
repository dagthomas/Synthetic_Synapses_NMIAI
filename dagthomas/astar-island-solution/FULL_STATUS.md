# Astar Island — Full Status Report (2026-03-20)

## Team: Synthetic Synapses

## Scores

| Round | Score | Rank | Regime | Weighted | Notes |
|-------|-------|------|--------|----------|-------|
| R1 | - | - | - | - | No submission |
| R2 | 66.4 | 53 | Thriving (62%) | 73.2 | Old code, no calibration |
| R3 | 68.0 | 11 | Collapse (0%) | 78.7 | Old code |
| R4 | 53.3 | 65 | Collapse (1%) | 64.8 | Floor bug (0.015) |
| R5 | 86.3 | **1** | Moderate (28%) | 110.1 | New pipeline deployed |
| R6 | 87.6 | 2 | Booming (244%) | 117.4 | + Gemini improvements |
| R7 | 74.0 | 2 | EXTREME boom (384%) | 104.2 | Undetectable regime |
| R8 | 66.9 | 103 | Collapse (0%) | - | **Observations lost** (unicode crash) |
| R9 | pending | - | - | - | Crash-proof pipeline |

**Best weighted: R6 = 117.4** (87.6 x 1.34)
**Leader: ~130+** (estimated)

## Production Pipeline

```
submit.py
  -> explore.py (50 queries: 45 coverage + 5 smart adaptive)
     -> Saves each observation INCREMENTALLY to disk (crash-proof)
     -> Builds: GlobalMultipliers, FeatureKeyBuckets, GlobalTransitionMatrix
  -> predict_gemini.py:gemini_predict(state, global_mult, fk_buckets)
     1. CalibrationModel (8 rounds, 64k cells, 125 fine keys)
        - Hierarchical: fine -> coarse -> base -> global
        - Params: fine_div=100, coarse_div=100, base_div=100, global_weight=0.01
     2. FK empirical blend: prior*5.0 + empirical*sqrt(count) / (5.0 + sqrt(count))
        - emp_max_weight = clip(12 - 4*ratio[1], 6, 12) * 1.5
        - min_count = 5
     3. Distance-aware multipliers:
        - dist=0 (settlements): power = [0.4, 0.75, 0.75, 0.75, 0.4, 0.4]
        - dist>=1 (expansion): power = [0.4, 0.50, 0.60, 0.50, 0.4, 0.4]
        - Settlement clamp: [0.15, 2.5]
     4. Temperature softening near settlements:
        - radius = 2 + int(3 * min(ratio[1], 1.2))
        - T_max = 1.0 + 0.10 * sqrt(min(ratio[1], 1.0))
     5. Selective spatial smoothing (settlement + ruin only, NOT port):
        - alpha=0.75, kernel=3x3 uniform
     6. Structural zeros: mountain=0 on non-mountain, port=0 on non-coastal
     7. Floor: 0.005 (1/200 = min GT granularity)
     8. Lock static: ocean=[1,0,0,0,0,0], mountain=[0,0,0,0,0,1], borders=ocean
```

## Backtested Performance (R2-R7 leave-one-out)

| Round | Score | Theoretical Max | Gap |
|-------|-------|----------------|-----|
| R2 | 92.0 | 100 | 8.0 |
| R3 | 93.9 | 99 | 5.1 |
| R4 | 94.1 | 100 | 5.9 |
| R5 | 87.7 | 100 | 12.3 |
| R6 | 87.5 | 100 | 12.5 |
| R7 | 75.3 | 100 | 24.7 |
| **AVG** | **88.4** | **100** | **11.4** |

## Where We Lose Points (KL Decomposition)

### R5/R6 (moderate/boom, gap 12 pts each)
- Settlement class: 4% of weighted KL (slightly underpredicted)
- Empty/Forest ratio: 5% (imprecise boundary)
- These are observation noise — we only get 1 sample per cell

### R7 (extreme boom, gap 25 pts)
- **Settlement: 12% of weighted KL** (massive underprediction)
- Our calibration predicts ~15% settlement, GT has 25.6%
- Multiplier only reaches 1.06 (mid-sim observations show 14%, not 25.6%)
- **Root cause: expansion happens AFTER we observe. Fundamentally undetectable.**

## Key Simulation Rules (Confirmed from GT Analysis)

### Hard Rules
1. Mountain/Ocean never change. Borders always ocean.
2. GT from exactly 200 simulations (probs = k/200, min nonzero = 0.005)
3. Mountain probability = 0 on ALL non-mountain dynamic cells
4. Port probability = 0 on ALL non-coastal cells

### Settlement Dynamics
5. Coastal settlements die 2x more than inland
6. Sparse settlements (<=1 neighbor in r=5) survive 2x better than dense (2+)
7. Expansion only at distance 1-2 (never dist>=4)
8. Ruins form on expanded-then-collapsed cells, not initial settlement positions
9. Settlement survival ranges 0% (R3) to 384% (R7)

### Regime Detection
10. Observed settlement % approximates GT settlement % (ratio 1.07-1.21)
11. BUT extreme booms show moderate observations (14% obs -> 25.6% GT on R7)
12. No reliable early-warning signal for extreme booms found
13. Food/pop/wealth stats don't distinguish boom from moderate

## What We Tried and Failed

| Idea | Score Impact | Why It Failed |
|------|-------------|---------------|
| Per-cell Bayesian update (1-2 obs) | **-30 pts** | Overfits to single stochastic samples |
| Zero ruin on non-settlement cells | **-34 pts** | Ruins CAN form on expanded-then-collapsed plains |
| Log-odds space blending | **-29 pts** | Log transform doesn't help discrete distributions |
| Floor 0.015 (vs 0.005) | **-2 pts** | Wastes mass on mountain/port/ruin |
| Manhattan distance caps (dist>6) | **-2.3 pts** | Ruins exist far from settlements |
| Idea H (sett% regime scaling) | **-0.6 pts** | Double-counts with existing multiplier |
| Per-regime calibration | **0 pts** | Can't detect boom from mid-sim observations |
| Per-class FK weights | **-0.2 pts** | Hurts R6 when combined |
| Alive-count settlement boost | **-0.5 to -3 pts** | Boosts R6 (already correct) along with R7 |
| 75+ Opus code modifications | **0 pts** | None beat baseline |
| 5000+ autoloop parameter tweaks | **0 pts** | Converged at ceiling |

## What Works

| Feature | Score Impact | Source |
|---------|-------------|--------|
| Smart floor (zero impossible classes) | **+5 pts** | Manual GT analysis |
| CalibrationModel (historical GT) | **+10-15 pts** | Reference solution |
| FK bucketed empirical | **+5-8 pts** | Reference solution |
| Global multipliers (wide clamps) | **+5-10 pts** | Manual + reference |
| Distance-aware multiplier power | **+0.5 pts** | ADK Gemini agent |
| Selective smoothing (no port) | **+0.3 pts** | ADK Gemini agent |
| Empirical trust scale 1.5x | **+0.1 pts** | Manual sweep |
| Incremental obs saves (crash-proof) | **prevents -27 pts** | R8 post-mortem |

## Research Infrastructure

| Tool | What | Speed | Status |
|------|------|-------|--------|
| `predict_gemini.py` | Production prediction | - | **OPTIMAL (confirmed)** |
| `submit.py` | Full pipeline | 2 min/round | Crash-proof |
| `autoloop_fast.py` | Numeric parameter search | 160k/hr | Converged |
| `multi_researcher.py` | Haiku->Opus->Flash-Lite->Backtest | ~50s/iter | Working |
| `research_agent/` | Google ADK agent (Gemini Pro) | ~3 exp/min | Working |
| `autoexperiment.py` | Predefined experiment suite | 3s/exp | Reference |
| `test_ideas.py` | Quick idea validation | 30s/exp | Reference |

## Calibration Data

```
data/calibration/
  round1/ (5 seeds, thriving 57%)
  round2/ (5 seeds, thriving 62%)
  round3/ (5 seeds, collapse 0%)
  round4/ (5 seeds, collapse 1%)
  round5/ (5 seeds, moderate 28%)
  round6/ (5 seeds, booming 244%)
  round7/ (5 seeds, extreme boom 384%)
  round8/ (5 seeds, collapse 0%) ← no observations
```

## Observation Data

```
data/rounds/
  <round2_id>/ 50 obs files + results.json
  <round3_id>/ 50 obs files + results.json
  <round4_id>/ 50 obs files + results.json
  <round5_id>/ 50 obs files + results.json
  <round6_id>/ 50 obs files + results.json
  <round7_id>/ 50 obs files + results.json
  <round8_id>/ 0 obs files (lost to crash)
  <round9_id>/ 50 obs files (incremental saves, crash-proof)
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `predict_gemini.py` | **THE prediction function** (deployed) | ~130 |
| `calibration.py` | CalibrationModel (historical GT) | ~240 |
| `explore.py` | Query strategy + observation collection | ~330 |
| `submit.py` | Orchestrator (explore -> predict -> submit) | ~180 |
| `utils.py` | GlobalMultipliers, FK buckets, ObsAccumulator | ~380 |
| `fast_predict.py` | Vectorized prediction (for autoloop) | ~200 |
| `config.py` | Constants + .env loader | ~35 |
| `client.py` | API client with rate limiting | ~180 |
| `analyze.py` | Post-round analysis + GT comparison | ~300 |
| `multi_researcher.py` | Haiku+Opus+Flash code gen loop | ~680 |
| `autoloop_fast.py` | Fast numeric parameter search | ~280 |
| `research_agent/` | Google ADK agent (5 files) | ~1000 |

## Environment

- `.env`: GOOGLE_API_KEY, ASTAR_TOKEN (expires March 26)
- Python 3.12, numpy, scipy, requests
- Claude Code CLI (for multi_researcher.py)
- Google ADK + google-generativeai (for research agents)

## Remaining Improvement Opportunities

### Realistic (with current API constraints)
1. **More calibration data** — each round improves the model (+0.1-0.3/round)
2. **Better adaptive queries** — we give 3 extra to most dynamic seed (implemented)
3. **Consistency** — avoid crashes, always submit with observations

### Speculative (would need different approach)
4. **Viewport selection overhaul** — tested, only marginal (+0.1-0.3)
5. **Ensemble of prediction functions** — untested, moderate effort
6. **Detect boom from settlement growth RATE** — would need 2 observations of same cell at different times
7. **Use settlement food/pop as continuous features** — tested correlations, no signal found

### Impossible (API limitation)
8. **Observe later in simulation** — would directly see expansion, +10-25 pts on boom rounds
9. **More than 50 queries** — more observations = less noise
10. **Know the hidden parameters** — would make prediction trivial

## Scoring Formula

```
score = 100 * exp(-3 * weighted_kl)
weighted_kl = sum(entropy[cell] * KL(gt[cell], pred[cell])) / sum(entropy[cell])
```

Only cells with entropy > 0.01 contribute. Lower wKL = higher score.

| wKL | Score |
|-----|-------|
| 0.00 | 100.0 |
| 0.02 | 94.2 |
| 0.05 | 86.1 |
| 0.10 | 74.1 |
| 0.15 | 63.8 |
| 0.20 | 54.9 |
| 0.30 | 40.7 |

## Leaderboard

Best weighted score = max(round_score * round_weight) across all rounds.
Weight = 1.05^round_number (later rounds worth more).

Our best: R6 = 87.6 * 1.34 = 117.4
Leader: estimated ~130+ (scoring 95+ on weighted rounds)
