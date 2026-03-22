# System State

Current state of the Astar Island competition system as of 2026-03-20.

---

## Scores

| Round | Score | Rank | Regime | Notes |
|-------|-------|------|--------|-------|
| R2 | 66.4 | 53 | Thriving | Old code, no calibration |
| R3 | 68.0 | 11 | Collapse | Old code |
| R4 | 53.3 | 65 | Collapse | Floor bug |
| R5 | 86.3 | **1st** | Moderate | New pipeline deployed |
| R6 | 87.6 | 2nd | Boom | + Gemini improvements |
| R7 | 74.0 | 2nd | Extreme boom | Undetectable regime |
| R17 | 93.0 | - | Boom | Best ever |
| R20 | 89.4 | - | Moderate | Fixed obs correction bug |

Backtested average (R2-R7 leave-one-out): **88.4**

---

## Research Scale

| Metric | Value |
|--------|-------|
| Autoloop experiments | 1,028,171 |
| Parameters optimized | 44 continuous |
| Research iterations | 1,864 |
| Code variants generated | 497 |
| Breakthrough ideas | 32 |
| Calibration rounds | 20 |
| Ground truth cells | 160,000 |
| GPU sim speed | 124,000 sims/sec |

---

## Pipeline Status

- Daemon: autonomous 24/7 operation
- Autoloop: running (160K experiments/hr)
- Multi-researcher: 617 iterations completed
- Gemini researcher: 1,247 iterations completed
- GPU simulator: RTX 5090, CMA-ES fitting in ~8s
- Re-submission: up to 10 iterations per round (165-min window)
