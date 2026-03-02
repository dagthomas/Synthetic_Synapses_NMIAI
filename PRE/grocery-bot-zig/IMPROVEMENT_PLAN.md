# Zig Grocery Bot: Improvement Plan

## Current State
The Zig bot uses reactive, per-round greedy heuristics. While well-tuned per difficulty, fundamental architectural limits cap its performance.

| Difficulty | Current Max | Leader | Gap |
|---|---|---|---|
| Easy | 131 | 141 | -10 |
| Medium | 140 | 160 | -20 |
| Hard | 128 | 157 | -29 |
| Expert | 96 | 195 | -99 |

## What the Python Solver Proves

The Python solver (`grocery-bot-gpu/`) achieves **dramatically higher scores** using two key techniques the Zig bot lacks:

| Difficulty | Zig Max | Python Planner | Python Multi+Optimizer | Leader | vs Leader |
|---|---|---|---|---|---|
| Easy | 131 | 153 | **150** | 141 | **+9** |
| Medium | 140 | 152 | **166** | 160 | **+6** |
| Hard | 128 | 145 | **161** | 157 | **+4** |
| Expert | 96 | 130 | **135** | 195 | **-60** |

## Key Techniques to Implement

### 1. Multi-Strategy Selection (Eliminates Deadlocks)

**Impact: Medium +++, Hard +++, Expert +++**

The single biggest improvement. Instead of one fixed `max_pickers` value per difficulty, **try multiple values per game and keep the best**.

The Python solver found that:
- Some seeds deadlock with `max_active=4` but work perfectly with `max_active=3`
- Other seeds need `max_active=5` to score well
- **No single value works for all seeds**

**Implementation:**
Since games are deterministic within a day, the approach is:
1. Play the game once with default settings
2. Play again with `max_pickers=2`, `max_pickers=3`, etc.
3. Keep the highest score

Results from Python multi-strategy sweep (40 seeds):
- Hard: min score went from 11 → 90 (eliminated ALL deadlocks)
- Expert: min score went from 3 → 73

**Zig implementation:** Add a command-line parameter for `max_pickers` override. Run the game 5-8 times with different values. The 10s cooldown means testing 8 configs takes only 80 seconds.

### 2. Iterative Optimization (Action-Space Local Search)

**Impact: ALL difficulties +++**

This is the technique that pushes scores 20-50 points above the planner/heuristic baseline.

**Concept:**
1. Generate initial solution using current heuristics (300 actions per bot)
2. Save game state checkpoints every round
3. Pick random round R, random bot B
4. Try a different action for bot B at round R
5. Re-simulate from round R using the same heuristics
6. If score improved, keep the change and update all checkpoints
7. Repeat thousands of times

**Why it works:**
The heuristic makes locally optimal decisions, but a single early mistake cascades through the entire game. By trying random perturbations and keeping improvements, we find "lucky breaks" — cases where a slightly suboptimal move at round 50 leads to much better outcomes by round 200.

**Example:** On Medium seed 7001, one perturbation at round 57 improved score from 112 → 139 (+27 points!). The initial heuristic was making a wrong assignment that bottlenecked delivery.

**Zig implementation:**
1. Implement an in-process game simulator (no WebSocket needed) — ~200 lines
2. Generate all orders from seed (match server's RNG)
3. Run initial game to get action log + state checkpoints
4. Loop: perturb one action, re-simulate, keep improvements
5. Replay best action sequence via WebSocket

The key requirement is a **local game simulator** that exactly matches the server's behavior. The Python `game_engine.py` (~460 lines) is a reference implementation.

### 3. Full Order Foresight

**Impact: Medium ++, Hard +, Expert +**

Currently the Zig bot only sees active + preview orders (2 orders). But ALL orders are generated from the game seed using a deterministic RNG. If we can reproduce the RNG, we know every future order.

**Benefits:**
- **Zero dead inventory**: Never pick an item that won't be needed
- **Better trip planning**: Pick items needed for order N+2 during order N
- **Smarter pre-positioning**: Move idle bots toward future item locations

**Implementation:**
1. Reverse-engineer the server's order generation RNG (seed → order sequence)
2. Pre-generate all orders at game start
3. In trip scoring, add a "future utility" factor — items needed in many future orders are more valuable
4. In pre-positioning, move idle bots toward items for the next 3-5 orders

### 4. Better Assignment with Round-Trip Cost

**Impact: Medium ++**

Currently the orchestrator uses distance-to-item for assignment. For small bot counts (1-3), using `distance(bot→item) + distance(item→dropoff)` as the assignment cost gives significantly better results because it prioritizes items that create SHORT round-trips.

**Caveat:** This ONLY helps for ≤3 bots. For 5+ bots, it causes all bots to cluster near the dropoff. Keep current distance-only metric for Hard/Expert.

### 5. BFS-Based Collision Avoidance

**Impact: Hard +, Expert +**

When the simple directional approach fails (can't move in optimal direction due to collision), use full BFS pathfinding that treats occupied cells as temporary walls. This finds detour routes around blocked corridors instead of waiting.

**Implementation:**
In `pathfinding.zig`, add a `bfs_first_step_avoiding(start, goal, occupied)` function that runs BFS excluding occupied cells. Use as fallback when normal `getFirstStep` would result in collision.

### 6. Trip Order TSP Optimization

**Impact: All difficulties +**

Currently trips evaluate all permutations (up to 3! = 6). This is already done in `trip.zig`. But the Python planner also evaluates trip order including the **return to dropoff** — which can change the optimal pickup sequence.

Ensure the TSP includes the final leg to dropoff:
```
trip_cost = dist(bot, item1) + dist(item1, item2) + dist(item2, item3) + dist(item3, dropoff)
```

## Recommended Implementation Priority

1. **Multi-strategy selection** (1 hour) — Biggest bang for buck. Just add CLI param + run script.
2. **Local game simulator** (4 hours) — Required for optimization. Port game_engine.py to Zig.
3. **Iterative optimization** (2 hours) — Once simulator exists, implement the optimization loop.
4. **Full order foresight** (2 hours) — Reverse-engineer order RNG, pre-generate orders.
5. **Round-trip cost for ≤3 bots** (30 min) — Simple orchestrator change.
6. **BFS collision avoidance** (1 hour) — Fallback pathfinding.

## Architecture: Offline Pre-Computation + Replay

The optimal competition architecture:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Play once with   │────▶│ Run offline       │────▶│ Replay best │
│ default heuristic│     │ optimizer (60-300s)│     │ via WebSocket│
│ to get seed/state│     │ multiple configs   │     │             │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

Since games are deterministic within a day:
1. Play once → observe seed, items, orders
2. Offline: run 8-10 different max_pickers configs
3. Offline: run optimizer on each for 30-60 seconds
4. Pick best action sequence
5. Replay via WebSocket → guaranteed same result

Total time: ~5 minutes per difficulty, all automated.

## Proven Results (5 seeds each, 60s/seed)

| Difficulty | Zig Max | Multi+Optimizer Max | Multi+Optimizer Mean | Leader |
|---|---|---|---|---|
| Easy | 131 | **150** | 143.8 | 141 |
| Medium | 140 | **166** | 155.2 | 160 |
| Hard | 128 | **161** | 142.2 | 157 |
| Expert | 96 | **135** | 119.0 | 195 |
| **Total** | **495** | **612** | **560** | **653** |

Beats leader on Easy, Medium, and Hard. Expert remains the biggest challenge — the 10-bot coordination problem likely requires fundamentally better multi-agent coordination (temporal scheduling, corridor reservation, or full MAPF solver).
