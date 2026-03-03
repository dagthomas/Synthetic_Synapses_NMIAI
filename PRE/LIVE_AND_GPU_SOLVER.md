# Live Testing & Sequential GPU Solver

## Overview

The live testing system (`live_gpu_stream.py`) and the sequential GPU solver (`gpu_sequential_solver.py`) together form the core of the production pipeline for multi-bot difficulties (Medium, Hard, Expert). The architecture is **anytime**: it always has some plan ready to execute, continuously upgrading to a better plan as background computation completes.

---

## `live_gpu_stream.py` — Anytime Online Solver

### Architecture: 3-Tier Plan Hierarchy

```
Tier 0: Greedy BFS          — always available, <1ms per round
Tier 1: MAPF planner        — 1–10s startup, multi-strategy (mab sweep)
Tier 2+: GPU DP passes      — progressive state budgets, keep upgrading
```

The bot always uses the **best available plan**. Plans are upgraded atomically when a background thread finds a better score.

### Background Threads

Four concurrent threads run from game start:

| Thread | Difficulty | Purpose |
|--------|-----------|---------|
| `_mapf_worker` | Multi-bot only | MAPF planner sweep (mab=1..5), fast first plan |
| `_gpu_worker` | Single-bot only | Full exhaustive DP passes (provably optimal for Easy) |
| `_gpu_refine_worker` | Multi-bot only | Sequential DP with progressive budgets + warm refinement |
| `_per_round_gpu_worker` | All | Per-round DP from actual live state each round |

### Plan State & Upgrade Logic

```python
@dataclass
class PlanState:
    score: int               # CPU-verified score
    actions: list            # 300 × [(act, item)] × num_bots
    expected_pos: list       # positions[rnd][bid] = (x, y) before action
    goals: dict              # bid → [(rnd, pos, act, item_idx), ...]
    source: str              # 'none', 'greedy', 'mapf', 'gpu_pass1', 'gpu_seq_p0', ...
```

`_update_plan(score, actions, ...)` upgrades only if `score > current_plan.score`. Thread-safe (locked). On upgrade, `_round_offset` resets so plan replay starts fresh.

### Initialization (Round 0)

1. WebSocket data → `ws_to_capture(data)` → capture dict
2. Build `MapState`, `walkable` set, `PrecomputedTables`
3. `_data_ready.set()` → unblocks background threads
4. Rebuild per-round GPU searcher
5. Start `_mapf_worker(gen)` and appropriate GPU thread

### Order Tracking

Every round, `_check_new_orders(data)` scans incoming WS data for order IDs not yet seen. New orders:
- Appended to `capture['orders']`
- Increment `_solve_gen` (signals background threads to restart)
- Trigger new MAPF worker and per-round searcher rebuild

### GPU State Budget Schedule

```python
PASSES = {
    'easy':   [50_000, 500_000, 2_000_000],   # single-bot: _gpu_worker
    'medium': [20_000, 200_000, 1_000_000],   # multi-bot: _gpu_refine_worker
    'hard':   [10_000, 100_000, 500_000],
    'expert': [5_000,  50_000,  200_000],
}
```

### Per-Round GPU Worker (`_per_round_gpu_worker`)

Triggered every game round with actual live bot positions from WS data. Used as fallback when the main plan desyncs.

**Single-bot**: Standard DP from current position, `horizon` rounds lookahead.

**Multi-bot**: Sequential locked-bot DP:
1. Sort bots by priority: **delivery bots first** (carrying items, closer to dropoff)
2. Plan bot 0 solo
3. Plan bot 1 with bot 0's trajectory locked
4. Plan bot N with bots 0..N-1 locked
5. Conflict resolution safety net (higher-priority bots win cell conflicts)

Per-round parameters:
```python
PR_PARAMS = {
    'easy':   {'max_states': 50_000, 'horizon': 80},
    'medium': {'max_states': 15_000, 'horizon': 50},
    'hard':   {'max_states': 25_000, 'horizon': 55},
    'expert': {'max_states': 12_000, 'horizon': 45},
}
```

### `_gpu_refine_worker` — Multi-Bot Progressive Solver

Runs for Hard/Expert. Budget schedule:

```python
_budgets = {
    'hard':   [20_000, 100_000, 500_000, 2_000_000],
    'expert': [10_000,  50_000, 200_000, 1_000_000],
}
```

**Pass 0 (cold start)**: `solve_multi_restart` — screens 60 bot orderings via cheap GPU greedy rollout, runs top 3 in full DP, 1 refinement pass.

**Pass 1+ (warm start)**: `refine_from_solution` — re-plans each bot with all others locked, growing budget. Before each pass, checks if MAPF or external plan improved and adopts it if better.

**Extended refinement loop**: After all budget passes, keeps refining with the largest budget until new orders arrive.

### Action Selection Each Round

```
1. If plan available and not desynced:
     → Follow plan goals (BFS toward next goal position)
     → Fall back to greedy when goals exhausted
2. If per-round GPU result available for this round:
     → Use per-round GPU action
3. Otherwise:
     → Greedy BFS fallback
```

### Usage

```bash
python live_gpu_stream.py "wss://..."
python live_gpu_stream.py "wss://..." --save          # save capture + solution
python live_gpu_stream.py "wss://..." --max-states 50000
python live_gpu_stream.py "wss://..." --no-refine     # skip refinement passes
python live_gpu_stream.py "wss://..." --cpu           # force CPU (no CUDA)
```

---

## `gpu_sequential_solver.py` — Sequential GPU DP Solver

### Core Idea

Multi-bot games are too large for a joint DP (state space = product of per-bot states). The sequential approach decomposes the problem:

> **Plan each bot individually, with all previously-planned bots' trajectories locked.**

Each bot's DP stays single-bot sized (~200K states), making it GPU-tractable.

### Two-Pass Algorithm

#### Pass 1: Sequential Planning

```
Bot 0:   plan solo                         → bot_actions[0]
Bot 1:   plan with bot 0 locked            → bot_actions[1]
Bot 2:   plan with bots 0,1 locked         → bot_actions[2]
...
Bot N-1: plan with bots 0..N-2 locked     → bot_actions[N-1]
```

Multiple orderings can be tried (`num_pass1_orderings`):
- Ordering 0: forward `[0, 1, ..., N-1]`
- Ordering 1: reverse `[N-1, ..., 1, 0]`
- Ordering 2+: random shuffles (seeded)

Best Pass 1 result is kept as the starting point for refinement.

#### Pass 2+: Iterative Refinement

Re-plan each bot with **all other bots locked** (using current best actions). Immediately CPU-verify the new combined actions. Keep only improvements.

```
For each refinement iteration:
  For each bot in refine_order:
    1. Lock all OTHER bots' current best actions
    2. Run GPU DP for this bot
    3. CPU verify new combined score
    4. If improved: keep new actions, update best
    5. If not improved: revert to old actions

  Early stop: N consecutive bots with no improvement
  Alternate order: iter0=forward, iter1=reverse, iter2+=random
```

### Locked Trajectories (`pre_simulate_locked`)

To lock bots, their exact positions for every round must be computed. This is done by **simulating all bots together** with the CPU game engine (or Zig FFI for speed).

```python
def pre_simulate_locked(gs_template, all_orders, bot_actions, locked_bot_ids, _zig_ctx=None):
    # Returns:
    # locked_actions:      [num_locked, 300] int8
    # locked_action_items: [num_locked, 300] int16
    # locked_pos_x:        [num_locked, 300] int16
    # locked_pos_y:        [num_locked, 300] int16
```

The Zig FFI path (`zig_presim_locked`) is ~2.7x faster than the Python path and used automatically when a seed is available.

### Type Specialization

For 3+ bots, item types are assigned to bots round-robin by frequency:
- Most frequent type → Bot 0, Bot N, Bot 2N, ...
- Second most frequent → Bot 1, Bot N+1, ...

This reduces inter-bot competition. The GPU `_eval` function rewards bots for targeting their assigned types and penalizes collecting types already covered by locked bots.

### Pipeline Mode

The last `pipeline_fraction` (default 40%) of bots in the planning order are set to `pipeline_mode=True`. A pipeline bot targets items for the **next** order (+1, +2 depth) while the primary bots work on the current active order. This enables pre-fetching so items are ready the instant the current order completes.

```
Example (5 bots, pipeline_fraction=0.4 → 2 pipeline bots):
  plan_order[-2]: pipeline_depth=1  (targets order+1)
  plan_order[-1]: pipeline_depth=2  (targets order+2)
```

### `solve_multi_restart` — Cold Start with Ordering Screening

Used for the first plan when no existing solution is available.

1. Generate K=60 random bot orderings
2. Run N=20 step greedy GPU rollout for all K orderings simultaneously
3. Pick top `num_restarts` (default 3) orderings by estimated score
4. Run full sequential DP for each, keep best

This finds a better initial bot ordering than the default forward ordering, especially for Hard/Expert where bot order significantly affects score.

### `refine_from_solution` — Warm Start Refinement

Takes an existing combined_actions plan and runs refinement passes on it. Used by `_gpu_refine_worker` to progressively improve plans with growing budgets.

### `cpu_verify`

After every DP step, the full combined action sequence is verified by replaying all bots on the CPU game engine. This catches plan/simulation discrepancies and ensures only genuinely better plans are kept.

Zig FFI path (`zig_verify`) is ~4.5x faster than Python and used automatically when a seed is available.

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_states` | Difficulty-dependent | State budget per bot DP search |
| `max_refine_iters` | 2 | Max refinement passes (0 = skip) |
| `num_pass1_orderings` | 1 | Number of Pass 1 orderings to try |
| `pass1_states` | = max_states | Budget for Pass 1 (can be lower for multi-start) |
| `pipeline_fraction` | 0.4 | Fraction of bots in pipeline mode |
| `max_pipeline_depth` | 3 | Max order lookahead for pipeline bots |
| `no_filler` | False | Only use real captured orders (critical for live) |
| `use_type_specialization` | True | Assign types to bots |

### Default State Budgets

```python
DEFAULT_MAX_STATES = {
    'easy':   500_000,
    'medium': 1_000_000,
    'hard':   2_000_000,
    'expert': 5_000_000,
}
```

### Standalone Usage

```python
from gpu_sequential_solver import solve_sequential, refine_from_solution, solve_multi_restart

# From capture data (live game):
score, actions = solve_sequential(
    capture_data=capture,
    device='cuda',
    max_states=200_000,
    max_refine_iters=2,
    no_filler=True,
)

# From seed (offline):
score, actions = solve_sequential(
    seed=42,
    difficulty='hard',
    device='cuda',
)

# Improve existing plan:
score, actions = refine_from_solution(
    combined_actions=existing_actions,
    capture_data=capture,
    device='cuda',
    max_states=500_000,
)

# Cold start with ordering search:
score, actions = solve_multi_restart(
    capture_data=capture,
    device='cuda',
    max_states=50_000,
    num_restarts=3,
    num_screen=60,
)
```

---

## Data Flow Diagram

```
WebSocket Round Data
        │
        ▼
  _check_new_orders()
        │ (new orders)
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
  bump _solve_gen                                      _rebuild_pr_searcher()
        │                                                          │
        ├── _start_mapf(gen) ──► MAPF sweep ──────────────────────┤
        │                          (mab=1..5)                      │
        │                             │                            │
        │                      _update_plan()                      │
        │                             │                            │
        └── _gpu_refine_worker ◄──────┘                            │
              Cold: solve_multi_restart()                          │
              Warm: refine_from_solution()                         │
                    (growing budgets)                              │
                    _update_plan()                                 │
                                                                   │
Round arrives ──► _per_round_gpu_worker() ◄────────────────────────┘
                    Sequential locked-bot DP
                    from ACTUAL live positions
                    _pr_actions[rnd] = ws_actions
                         │
                         ▼
               _get_actions(rnd, ws_data)
                         │
                  ┌──────┴──────┐
                  │             │
              Plan goals   Per-round GPU   Greedy BFS
              (if synced)  (if available)  (fallback)
```

---

## Critical Notes

- **`--no-filler` is essential for live games.** Without it, the DP plans for fake filler orders with random items. Bots collect those items, get dead inventory (items can't be dropped), and score ~50% less. Always use `no_filler=True` in capture-based solves.
- **Stale-gen plans are still accepted.** If a GPU pass finishes after new orders arrived, it's still published if it scores better — it covers known orders correctly and the greedy fallback handles new orders.
- **`_round_offset` tracks plan drift.** If the bot's actual position diverges from the plan's expected position (desync), the offset adjusts which plan round to use or triggers fallback.
- **Zig FFI requires a seed.** Capture-only games (no seed in capture dict) fall back to Python simulation. The Zig path is ~3-4.5x faster for both `cpu_verify` and `pre_simulate_locked`.
- **GPU table caching** (`PrecomputedTables.to_gpu_tensors(device)`) is cached per device — safe to create many `GPUBeamSearcher` instances per round without re-uploading BFS distance tables.
