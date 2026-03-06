# Next Steps — GPU Optimal Solver (2026-03-06)

## Current State

| Difficulty | Score | Leader | Orders | Bottleneck |
|-----------|-------|--------|--------|-----------|
| Easy | 142 | ~142 | 17 | Done (optimal) |
| Medium | 184 | 214 | 22 | Sequential DP ceiling + need more orders |
| Hard | 180 | 252 | 21 | Sequential DP ceiling + need more orders |
| Expert | 139 | 303 | 15 | Need fresh capture + better coordination |

## What's Blocking Progress

### Sequential DP Ceiling (Primary)
Each bot is planned independently with others locked. This can't find globally optimal multi-bot plans. Hard is stuck at 180 regardless of:
- Training time (tested up to 600s)
- State count (tested 50K-100K)
- Refinement iterations (tested up to 20)
- Perturbation strategies (single/pair reset, type reshuffling)
- Escape attempts (up to 6)

After perturbation resets, all bots converge to the same local optimum.

### Insufficient Order Discovery
More orders require beating current score first. The virtuous cycle (better score → more orders → better training → better score) is stalled because training can't improve.

## Priority Actions

### 1. Break the Sequential DP Ceiling
Ideas (by feasibility):

**A. Order assignment pre-optimization**
Before DP, decide which bot handles which orders. Currently, type specialization hints at this, but each bot still sees all orders in DP. Explicit order assignment would reduce each bot's search space and improve coordination.

**B. Multi-phase coordination**
Split the 300-round game into phases (e.g., 3x100 rounds). In each phase, assign explicit goals per bot. Optimize the phase-goal assignment globally, then run DP per bot within each phase.

**C. Post-DP collision resolution**
Plan all bots independently (ignoring collisions), then resolve conflicts with MAPF post-processing. This lets each bot find its globally best path before worrying about coordination.

**D. Genetic algorithm over DP seeds**
Run many fast DP passes with different random orderings, type assignments, and order caps. Use a GA to evolve the meta-parameters that produce the best combined score.

**E. Joint 2-bot DP with larger state budget**
The 2-bot DP failed at 50K states (49x expansion). With 500K+ states and better pruning, it might become viable for the 2 most congested bots while others use single-bot DP.

### 2. Fresh Captures for Competition Day
- Orders change daily — all captures must be redone on March 19
- Plan: capture early morning, train all day, final replay in evening
- Have automated pipeline ready: `production_run.py` handles the full loop

### 3. More Order Discovery Cycles
Even at current quality, more replay cycles discover more orders:
- Medium: 22 orders (probably 30+ exist)
- Hard: 21 orders (probably 30+ exist)
- Expert: 15 orders (probably 20+ exist, need fresh capture first)

## What Works (Don't Change)
- Pipeline architecture (zig capture → GPU → replay → iterate)
- 50K states for pipeline, 100K for deep training
- 3 pass1 orderings (forward, reverse, random)
- Type specialization and zone assignments
- Contribution-based weakest-first refinement
- `--no-filler` flag (mandatory)
- Zig FFI for fast verification

## Comprehensive Strategy Doc
See `TRAINING_STRATEGY.md` for full documentation of all processes, tuning, and findings.
