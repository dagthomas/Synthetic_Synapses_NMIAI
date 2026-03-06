# grocery-bot-b200 — B200-Optimized Solver + Stepladder Pipeline

GPU-optimized grocery bot solver targeting NVIDIA B200 (192GB HBM3e).
Falls back gracefully on RTX 5090 (32GB) with auto-tuned parameters.

## Architecture

```
grocery-bot-b200/
  _shared.py              # sys.path setup to import from grocery-bot-gpu/
  b200_config.py          # GPU detection + parameter scaling
  b200_beam_search.py     # Enhanced single-bot DP (chunked, CPU history)
  joint_beam_search.py    # N-bot joint DP (2-5 bots, core breakthrough)
  squad_solver.py         # Squad orchestrator + LNS + all-perm pass-1
  deep_optimize.py        # Hours-long offline training CLI
  stepladder.py           # Day-long automated pipeline
  competition_day.py      # March 19 orchestrator
```

All shared utilities (game_engine, precompute, solution_store, etc.) are
imported from `grocery-bot-gpu/` — no code duplication.

## Key Innovation: N-Bot Joint DP

Instead of planning bots one at a time (sequential DP, ceiling at 180 on Hard),
`joint_beam_search.py` searches over the JOINT state space of 2-5 bots.

- **Distance-adaptive expansion**: Bots far apart get 1 action (greedy), nearby get 5-10
- **Effective branching ~20-40x** (not naive 7^N) via proxy-score action pruning
- **True coordination**: unique coverage bonus, redundancy penalty, spacing

## Usage

```bash
# Single difficulty solve
python squad_solver.py hard --gpu auto

# Deep training (hours)
python deep_optimize.py hard --budget 7200 --max-states 500000

# Day-long pipeline
python stepladder.py hard --hours 6

# Competition day
python competition_day.py --gpu b200
```

## GPU Scaling

| Config | 5090 (32GB) | B200 (192GB) |
|--------|-------------|--------------|
| 1-bot beam | 200K | 50M |
| Joint 3-bot | 50K | 5-10M |
| Joint 5-bot | — | 1M |
| Pass-1 orderings | 3 | 120 (all 5!) |
| Refine iters | 10 | 100 |
