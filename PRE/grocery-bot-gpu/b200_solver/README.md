# B200-Scale GPU Solver

Deep multi-bot solver designed for long training sessions (hours, not seconds).
Works on any CUDA GPU — the 5090 runs the same math, just slower than a B200.

## Architecture

### Phase 1: Exhaustive Orderings
Instead of trying 3 random bot orderings, try ALL N! permutations.
- Hard (5 bots): 120 orderings × ~35s = **70 minutes on 5090**, 12 min on B200
- Medium (3 bots): 6 orderings × ~15s = **1.5 minutes**
- Expert (10 bots): Sample top 500 from 3.6M, **~4 hours** on 5090

### Phase 2: Deep Refinement
50+ iterations at 200K states (vs 10 iterations at 50K in pipeline).
- Hard: 50 × 5 bots × 14s = **58 minutes on 5090**, 10 min on B200
- Expert: 50 × 7 bots × 14s = **82 minutes on 5090**

### Phase 3: Joint 3-Bot DP (The Ceiling Breaker)
Plans 3 bots simultaneously in a shared state space, exploring the full
N^3 cross-product of actions (343 combos for 7 actions/bot).

This is what breaks the sequential DP ceiling:
- Sequential: each bot planned independently, can't find globally optimal coordination
- Joint 3-bot: perfect coordination between 3 bots guaranteed by exhaustive search

Hardware scaling:
- **5090 at 50K states**: 17M expansions/round, ~2.5 min per triple
- **5090 at 200K states**: 68M expansions/round, ~10 min per triple
- **B200 at 1M states**: 343M expansions/round, ~1 min per triple
- **B200 at 2M states**: 686M expansions/round, ~2 min per triple

## Usage

```bash
# 3-hour deep training session on Hard
python -m b200_solver.deep_solve hard --max-hours 3

# Expert with custom settings
python -m b200_solver.deep_solve expert --max-hours 6 \
    --states-1bot 200000 --states-3bot 100000 --max-orderings 500

# Skip orderings, just do deep refine + 3-bot
python -m b200_solver.deep_solve hard --skip-orderings --max-hours 1

# Just 3-bot joint refinement
python -m b200_solver.deep_solve hard --skip-orderings --refine-iters 5 --max-hours 0.5
```

## Time Estimates

Run `python -m b200_solver.config` to see estimates for your hardware.

### RTX 5090 (32GB) — Hard (5 bots)
| Phase | Time |
|-------|------|
| All 120 orderings (100K states) | ~70 min |
| 50-iter refinement (200K states) | ~58 min |
| 3-bot joint (10 triples × 50K) | ~25 min |
| **Total deep treatment** | **~2.5 hours** |

### NVIDIA B200 (192GB) — Hard (5 bots)
| Phase | Time |
|-------|------|
| All 120 orderings (500K states) | ~12 min |
| 50-iter refinement (2M states) | ~10 min |
| 3-bot joint (10 triples × 1M) | ~10 min |
| **Total deep treatment** | **~32 min** |

## Why This Breaks the 180 Ceiling

The sequential DP ceiling exists because:
1. Bot 0 plans optimal path for itself
2. Bot 1 plans around Bot 0's locked trajectory
3. But Bot 0's "optimal" path may block Bot 1's even better path
4. Refinement can't escape: each bot's greedy-best locks the others

Joint 3-bot DP solves this by searching the combined state space of all 3 bots.
The solver can find configurations where Bot 0 takes a slightly suboptimal path
that enables Bots 1 and 2 to be much more efficient — a tradeoff that sequential
DP can never discover.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Hardware profiles and time estimates |
| `joint_3bot_dp.py` | 3-bot joint GPU DP solver |
| `deep_solve.py` | Main orchestrator (orderings + refine + joint) |
