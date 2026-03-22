# Product Requirements — Astar Island

## Challenge Overview

Astar Island is a prediction challenge in NM i AI (Norwegian AI Championship). Predict the final state of a Norse civilization simulator from limited viewport observations.

---

## Rules

- **Map**: 40x40 grid with terrain (land, forest, mountains, ocean) and Norse settlements
- **Viewport**: 15x15 observable window, positioned per query
- **Budget**: 50 observation queries shared across 5 random seeds
- **Seeds**: Same terrain, different random outcomes
- **Output**: 40x40x6 probability tensor per seed (empty, settlement, port, ruin, forest, farmland)
- **Ground truth**: Generated from exactly 200 Monte Carlo simulations (all probabilities are k/200)

---

## Scoring

```
score = 100 * exp(-3 * weighted_kl)
weighted_kl = sum(entropy(cell) * KL(gt, pred)) / sum(entropy(cell))
```

- Entropy-weighted: high-uncertainty cells matter more
- KL divergence: rewards calibrated probabilities, punishes overconfidence
- Only cells with entropy > 0.01 contribute
- Score sensitivity: wKL=0.05 -> 86.1, wKL=0.10 -> 74.1, wKL=0.15 -> 63.8

---

## Simulation Dynamics

### Hard Rules (confirmed from ground truth)
1. Mountains and ocean never change
2. Border cells are always ocean
3. Port requires ocean adjacency (impossible inland)
4. Minimum nonzero GT probability is 0.005 (1/200)

### Settlement Dynamics (vary by round)
- **Survival**: 0% (total collapse) to 62% (thriving) across rounds
- **Expansion**: Distance 1-2 from existing settlements only, ~1% probability
- **Ports**: Coastal + proximity to settlement (avg dist 3.3)
- **Ruins**: From collapsed settlements, not from direct ruin formation

### Regime Classification

| Regime | Survival | Expansion | Strategy |
|--------|----------|-----------|----------|
| Collapse | 0-2% | 0% | Trust statistical model |
| Moderate | 2-15% | <1% | Balanced ensemble |
| Boom | >15% | >1% | Trust GPU simulator |

---

## Constraints

- Rounds last ~165 minutes, new round every ~3 hours
- Predictions can be overwritten while round is active
- API rate limits apply to observation queries
- 5 seeds per round, 50 total queries to distribute across them
- Must use probability floor >= 0.005 to avoid infinite KL divergence

---

## Targets

- Backtested ceiling: 88.4 average (R2-R7)
- Best single round: 93.0 (R17)
- Leader estimate: ~130+ weighted
- Key gap: extreme boom rounds (R7 type) where expansion happens after observation window
