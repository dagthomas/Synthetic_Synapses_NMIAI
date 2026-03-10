# Technologies & Algorithms Reference

Comprehensive reference for the AINM Grocery Bot competition system. Covers all solvers, algorithms, GPU pipeline, pathfinding, and infrastructure across the Zig bot, GPU DP solver, and nightmare LMAPF solver.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Game Engine](#game-engine)
3. [Zig Bot (Reactive Solver)](#zig-bot-reactive-solver)
4. [GPU DP Solver (Offline/Online)](#gpu-dp-solver-offlineonline)
5. [Nightmare Solver (LMAPF)](#nightmare-solver-lmapf)
6. [Precomputation & Data Structures](#precomputation--data-structures)
7. [Pipeline & Orchestration](#pipeline--orchestration)
8. [Infrastructure](#infrastructure)
9. [Key Algorithms Summary](#key-algorithms-summary)

---

## System Architecture

```
                         +-----------------------+
                         |   Live Game Server    |
                         |   (WebSocket, wss://) |
                         +-----------+-----------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
     +--------v--------+   +--------v--------+   +---------v--------+
     |    Zig Bot       |   |  GPU Live Stream |   | Nightmare LMAPF  |
     | (reactive, fast) |   | (anytime tiers)  |   | (20-bot coord)   |
     +---------+--------+   +--------+---------+   +---------+--------+
              |                      |                      |
              +----------+-----------+----------+-----------+
                         |                      |
                +--------v--------+    +--------v--------+
                | Order Capture   |    | Solution Store  |
                | (PostgreSQL)    |    | (PostgreSQL)    |
                +--------+--------+    +--------+--------+
                         |                      |
                +--------v---------------------v--------+
                |         GPU Offline Optimizer          |
                |  (Sequential DP, Beam Search, CUDA)   |
                +--------+-----------------------------+
                         |
                +--------v--------+
                |  Replay Engine  |
                | (desync-aware)  |
                +-----------------+
```

**Three solver implementations** target different difficulty tiers:
- **Zig Bot**: Fast reactive solver for initial captures and Easy/Medium
- **GPU DP Solver**: CUDA-accelerated dynamic programming for Hard/Expert
- **Nightmare Solver**: LMAPF-based 20-bot coordination for Nightmare mode

---

## Game Engine

### Pure Python Simulator (`game_engine.py`)

Deterministic game simulator replicating server logic exactly. Used for offline testing, order discovery, and solution verification.

**State Representation**:
- `MapState`: Static map data (grid as numpy int8 array, items, dropoff zones, spawn)
- `Order`: Item type IDs (np.int8), delivery bitmask, completion tracking
- Cell types: FLOOR=0, WALL=1, SHELF=2, DROPOFF=3
- 7 actions: WAIT=0, UP=1, DOWN=2, LEFT=3, RIGHT=4, PICKUP=5, DROPOFF=6

**Key Mechanics Modeled**:
- Bot collision (lower ID wins, invalid action = silent wait)
- Spawn stacking (bots share spawn cell)
- Chain reaction delivery (order completion cascades)
- Infinite item shelves (same item_id pickable repeatedly)
- Dead inventory (no drop/discard action)
- 3-item inventory cap

### Zig Simulator (`ffi.zig`)

Native Zig reimplementation of game logic, exposed as a shared library (DLL/SO) for Python ctypes integration.

**Exports**:
- `ffi_verify()` — Replay full action sequence, return final score (4.5x faster than Python)
- `ffi_presim_locked()` — Simulate with locked bots, record positions per round (2.7x faster)
- `ffi_verify_live()` / `ffi_presim_locked_live()` — Same for live capture data

---

## Zig Bot (Reactive Solver)

### Overview

Native Zig WebSocket client providing fast reactive decisions. Primary use: initial game capture (order discovery) and Easy/Medium baseline scores.

**Build**: `zig build -Doptimize=ReleaseFast -Ddifficulty=expert`
**Binary**: `grocery-bot-{difficulty}.exe`

### Decision Cascade (`strategy.zig`)

Priority-ordered per-bot decision each round:

| Priority | Action | Condition |
|----------|--------|-----------|
| 1 | Drop off | At dropoff with active items |
| 2 | Evacuate | At dropoff without items (multi-bot) |
| 3 | Escape | Oscillation detected (4+ revisits in 24 rounds) |
| 4 | Pick up | Adjacent to needed item |
| 5 | Deliver | Has active items → navigate to dropoff |
| 6 | Follow trip | Existing trip plan → navigate to next pickup |
| 7 | Plan trip | No trip → evaluate 1/2/3-item candidates |
| 8 | Deliver fallback | Has items → go to dropoff |
| 9 | Pre-position | Idle → move near likely-needed items |
| 10 | Dead inventory | Non-matching items → camp near dropoff |
| 11 | Wait | Nothing to do |

### Trip Planner (`trip.zig`) — Mini-TSP

Exhaustive evaluation of 1/2/3-item trip candidates:

1. **Candidate enumeration**: Scan all items, filter by active/preview need, compute distances
2. **Permutation evaluation**: All orderings (1! / 2! / 3! = 1/2/6 permutations)
3. **Scoring**: `value * 10000 / cost`
   - Active items: +20, Preview: +3 (or +18 if completing)
   - Order completion: +80, +150 per preview item (chain reaction exploit)
   - Endgame: halve value if cost > 50% remaining rounds

### BFS Pathfinding (`pathfinding.zig`)

- **All-pairs precomputed**: BFS from every walkable cell at round 0
- **O(1) lookup**: `getPrecomputedDm(pos)` → distance map for any start position
- **Collision-aware BFS**: First-move collision check against other bots
- **Best adjacency**: `findBestAdj()` — closest accessible side of shelf item

### Space-Time A* MAPF (`spacetime.zig`)

Multi-agent path finding with temporal collision avoidance:

- **Reservation table**: `[MAX_TIME_HORIZON][H][W]` — bot claims (x, y, t) tuples
- **Multi-step commitment**: Store ST-A* path, follow for 12+ rounds instead of replanning
- **Swap conflict detection**: Prevents (A→B, B→A) in same timestep
- **Aisle one-way flow**: Alternating up/down per narrow column
- **Occupancy penalty**: +2-3 cost for entering occupied aisle columns
- **Configuration**: Enabled for Hard (12-step commits), disabled for Expert (over-constraining with 10 bots)

### Persistent Bot State

```
PersistentBot:
  trip_ids[3][32]      — Item IDs for current trip
  trip_count, trip_pos  — Trip progress
  stall_count           — Oscillation detection counter
  pos_hist[24]          — 24-round position history for oscillation
  escape_rounds         — Remaining escape mode rounds
  delivering            — Currently navigating to dropoff
```

### WebSocket Client (`main.zig`)

- Parses game state JSON per round
- 25ms delay before each action (prevents 1-round offset desync)
- DP replay plan loading (`--dp-plan` flag)
- 1-round offset detection: position mismatch threshold + 2 consecutive rounds
- Game log output to stdout (captured for PostgreSQL)

---

## GPU DP Solver (Offline/Online)

### Core Algorithm: GPU Beam Search (`gpu_beam_search.py`)

CUDA-accelerated single-bot dynamic programming via exhaustive BFS with deduplication.

**State Representation** (packed int64):
```
[bot_x:8][bot_y:8][inv0:8][inv1:8][inv2:8][active_idx:8][del_bitmask:16] = 64 bits
```

**Batch Tensors** (on CUDA):
- `bot_x[B], bot_y[B]` — int16 positions
- `bot_inv[B,3]` — int8 inventory slots
- `active_idx[B]` — int32 order index
- `active_del[B,MAX_ORDER_SIZE]` — int8 delivery bitmask (128 columns for 7-item orders)
- `score[B], orders_comp[B]` — int32 tracking

**GPU Acceleration**:
| Technique | Speedup | Details |
|-----------|---------|---------|
| TF32 precision | ~1.5x | `allow_tf32=True` (10-bit mantissa sufficient for integer values) |
| torch.compile | 3.5x | Triton kernel fusion, mode='default' (not reduce-overhead) |
| Vectorized delivery | — | Out-of-place ops via `torch.stack()` for CUDA graph compatibility |
| Precomputed lookups | — | All BFS/trip tables uploaded to GPU at init |

**Action Expansion**:
- `_dp_expand()`: Position-aware — only valid moves + adjacent pickups (B × (6+MAX_ADJ))
- `_smart_expand()`: BFS-guided candidates (move to active, preview, +2 orders, preferred types)
- Deduplication by int64 hash after each expansion

**Evaluation Heuristic** (`_eval()`):
- Active order completion value (5 + remaining items)
- Trip table lookup: O(1) multi-item cost estimation via `trip_cost_gpu[cell_idx, combo_idx]`
- Coordination penalty: -500 to -2000 per locked-bot type overlap
- Aisle congestion penalty: -4000 per locked bot in same narrow aisle
- Speed bonus: `100 * decay^iteration` (anytime preference for faster solutions)
- Zone bonuses for preferred geographic regions

**Typical Performance**:
- ~50M state evaluations per 5s on RTX 5090 (32 GB VRAM)
- 50K state budget: 6-10s per bot
- 200K state budget: 20-40s per bot

### Multi-Bot Sequential DP (`gpu_sequential_solver.py`)

Orchestrates per-bot GPU DP with progressive refinement:

**Pass 1 (Cold Start)**:
1. Bot 0 solo (no locked bots)
2. Bot 1 with bot 0's actions locked
3. Bot 2 with bots 0,1 locked
4. ... up to num_bots

**Pass 2+ (Refinement)**:
- Re-plan each bot with ALL other bots locked
- Fixes collision displacement from earlier passes
- Typically 2-10 refinement iterations

**Locked Trajectory Management**:
- `pre_simulate_locked()`: CPU simulation of locked bots → per-round position arrays
- Optional Zig FFI fast-path (2.7x speedup)
- Locked bot data flows into `_eval()` as coordination signals

**Bot Ordering Strategies**:
- Forward (ID 0, 1, 2, ...)
- Reverse (ID N-1, N-2, ...)
- Random (shuffled IDs)
- 3 orderings tried per solve (configurable)

**Configuration** (`SolveConfig`):
- `max_states`: 50K-500K per bot (50K optimal for pipeline throughput)
- `max_refine_iters`: 2-10 (more iterations = diminishing returns)
- `num_pass1_orderings`: 2-3 (forward, reverse, random)
- `max_dp_bots`: GPU DP top N bots, rest get CPU greedy
- `speed_bonus`: Anytime order completion bonus
- `no_filler`: Critical — ignore filler orders

**Greedy Fallback** (for non-DP bots):
- Fetch preview + high-frequency non-active types
- NEVER touch active order types (catastrophic DP interference)
- Contribute ~+18 points over DP-only on Expert

### Anytime Online Solver (`live_gpu_stream.py`)

Multi-tier live game solver with continuous background optimization:

| Tier | Latency | Method | Details |
|------|---------|--------|---------|
| Tier 0 | <1ms | Greedy BFS | Precomputed distance tables, always available |
| Tier 1 | 1-10s | MAPF | Background planner, multiple `max_active_bots` |
| Tier 2+ | 20-60s | GPU DP | Growing state budgets (5K→50K→200K→1M) |

**State Budgets Per Difficulty**:
| Difficulty | Pass 1 | Pass 2 | Pass 3 |
|-----------|--------|--------|--------|
| Easy | 50K | 500K | 2M |
| Medium | 20K | 200K | 1M |
| Hard | 10K | 100K | 500K |
| Expert | 5K | 50K | 200K |
| Nightmare | 5K | 25K | 100K |

**Background Threads**:
- `_gpu_worker()`: All-pass loop with growing budgets
- `_gpu_refine_worker()`: Sequential warm-start refinement
- `_pr_worker()`: Per-round low-budget GPU DP (15K-50K states, 40-80 round horizon)

**Plan Upgrade**: Accept only score improvements; reset round offset on upgrade.

### Offline Training (`nightmare_offline.py`)

Multi-restart + checkpoint-based local search for nightmare mode:

**Multi-Restart** (15% budget):
- V3 and V4 baseline solvers
- Stochastic perturbations: randomized stall escape, stall count jitter
- Perturbation rate: 0.5%-6% per restart

**Checkpoint Search** (70% budget):
1. Store game state every 25 rounds during baseline simulation
2. Pick random checkpoint (bias toward early rounds)
3. Force one random bot to take different action
4. Re-run solver from checkpoint to end
5. Keep if improved (+10-20 points typical per search)

**Results**: Mean 273.7 (+20.6% over V3 baseline 227), max 284.

---

## Nightmare Solver (LMAPF)

### Overview

Large-scale Multi-Agent Path Finding for 20 bots on 30x18 grid with 3 dropoff zones, 500 rounds, 21 item types.

### V3 Chain Reaction Pipeline (`nightmare_solver_v2.py`)

Exploits chain reaction mechanic: completing active order → preview becomes active → auto-deliver matching inventory at all dropoff zones → cascade.

**Architecture**:
1. **Future Order Lookahead**: Extract orders from capture, combine preview + next 2 orders
2. **Task Allocation**: Classify bots and assign roles
3. **Pathfinding**: PIBT with corridor awareness
4. **Opportunistic Pickup**: All bots grab adjacent needed items
5. **Stall Detection**: Force escape after 3+ rounds at same position

### Task Allocation (`nightmare_task_alloc.py`) — MRTA

Multi-Robot Task Allocation with zone partitioning:

**Bot Classification**:
| Type | Description | Action |
|------|-------------|--------|
| Active carrier | Has items matching active order | Deliver to nearest dropoff |
| Preview carrier | Has preview-order items only | Stage 1-3 cells from dropoff |
| Dead bot | Has non-active, non-preview items | Flee to corridor parking |
| Empty bot | No inventory | Fetch needed items or park |

**Zone Strategy** (3 dropoffs):
- Bots 0-6: LEFT zone (dropoff 0)
- Bots 7-13: MID zone (dropoff 1)
- Bots 14-19: RIGHT zone (dropoff 2)
- Cost = dist_to_item + 0.4 * dist_item_to_dropoff

**Allocation Flow**:
1. DELIVER: Active carriers → balanced dropoff (least loaded)
2. FILL-UP: Active carriers with space → detour for nearby items (<6 rounds)
3. PREVIEW STAGE: Preview carriers → near-dropoff parking
4. DEAD FLEE: Corridor parking away from dropoff
5. EMPTY FETCH: Active pickup first, then preview (max 4 pickers), then park

### PIBT Pathfinder (`nightmare_pathfinder.py`)

Priority Inheritance with Backtracking — recursive push chains:

**Algorithm**:
1. Rank bots by task urgency (deliver > pickup > stage > flee > park)
2. Per bot in priority order:
   - Rank candidate moves (BFS distance + traffic + congestion penalties)
   - Claim first unclaimed destination
   - If destination claimed: try recursive push (depth limit 4)
3. Swap detection: prevent (A→B, B→A) cycles

**Move Ranking**:
- Distance to goal (BFS-optimal first step)
- Traffic penalty (one-way aisle rules)
- Congestion penalty: 2.0 * heatmap[dest]
- Optimal bonus: -0.5 for BFS-optimal direction
- Corridor penalty: +0.1 for non-corridor cells in narrow aisles

**Narrow Aisle Detection**: Columns where >90% non-corridor cells are walkable with shelf neighbors on both sides.

### Traffic Management (`nightmare_traffic.py`)

**One-Way Aisles**: Alternating up/down per narrow column (segment 0→down, 1→up, 2→down).

**Congestion Heatmap**: Decaying heat (0.7 decay), +1.0 per bot position per round. Penalty flows into pathfinder scoring.

### Universal Opportunistic Pickup

All bot types (not just pickup-assigned) grab adjacent items matching active shortfall or preview orders:

- **Active shortfall**: Any bot adjacent to needed active-order item picks it up
- **Preview**: Bots with <2 items can pick preview items (guards against dead inventory)
- **Impact**: +44 mean score improvement (biggest single optimization)

---

## Precomputation & Data Structures

### All-Pairs Shortest Paths (`precompute.py`)

**GPU BFS via Matrix Multiply**:
- Parallel frontier expansion using `torch.mm()` on adjacency matrix
- Output: `dist_matrix[N,N]` int16, `next_step_matrix[N,N]` int8
- Disk cache: `.npz` files keyed by grid MD5 hash

**Item Type Distance Tables**:
- `dist_to_type[num_types, H, W]` — nearest item of each type from each cell
- `step_to_type[num_types, H, W]` — first action toward nearest item
- Multi-dropoff: `np.minimum()` across all dropoff zone distances

### Trip Tables (`precompute.py: TripTable`)

Backward-chaining DP for multi-item trip cost estimation:

1. Enumerate all sorted type tuples of size 1..3 (~40-100 combos)
2. For each (start_cell, combo): try all type orderings (up to 6 permutations)
3. Chain: cost[type_i] = pickup + min(dist[adj[i]→adj[i+1]] + cost[i+1])
4. Output: `trip_cost[N_cells, N_combos]` int16

**GPU Upload**: `to_gpu_tensors()` for O(1) lookup in `_eval()`.

### Adjacency Tables

For each walkable cell, precompute which items are adjacent (Chebyshev distance 1):
- `adj_items[cell, MAX_ADJ]` — item indices
- `adj_count[cell]` — number of adjacent items
- Enables O(1) valid pickup enumeration in action expansion

---

## Pipeline & Orchestration

### Production Pipeline (`production_run.py`)

Fast-iterate pipeline within 288s token window:

```
Zig Capture (30s) → GPU Optimize (25s) → Replay (30s) → Discover Orders
                          ↑                     |
                          +---------------------+
                          (iterate 5-8 times)
```

**Iteration Schedule**:
| Iteration | States | Refine | Time | Speed Bonus |
|-----------|--------|--------|------|-------------|
| 0 (cold) | 50K | 2 | 25s | 100 |
| 1-2 (warm) | 50K | 2 | 18s | 50/25 |
| 3 (deep) | 100K | 3 | 35s | 12.5 |
| 4+ (warm) | 50K | 2 | 20s | 6.25 |

**Key Insight**: Iteration throughput > per-iteration quality. 8+ iterate loops with order discovery compounds better than 2-3 deep loops.

### Competition Day (`competition_day.py`)

"Time Millionaire" strategy for unlimited offline time between token windows:

```
Bootstrap (5min) → Deep Train (unlimited) → Replay & Discover → Retrain
                        ↑                           |
                        +---------------------------+
                        (until stall: 3 cycles no new orders)
```

**Deep Training Parameters**: 100K states, 30 refine iterations, 3 orderings, speed_bonus=150.

### Local Benchmarking (`iterate_local.py`)

**Full Foresight Mode**: Solve → Simulate → Discover orders → Re-solve with tighter order set.

**Key Finding**: Fewer orders = less state fragmentation. 25 orders + 50K states > 40 orders + 100K states.

### Replay Engine (`replay_solution.py`)

Adaptive replay with desync correction:

| Mode | Condition | Action |
|------|-----------|--------|
| SYNCED | Position matches DP plan | Send cached DP action |
| DESYNCED | Position diverges | Goal-following BFS toward next DP target |
| EXHAUSTED | DP plan complete | Greedy fallback strategy |

**Goal Following**: Extract pickup/dropoff sequence from DP plan, navigate from actual position via BFS. Preserves multi-bot coordination while tolerating positional drift.

### Solution Store (`solution_store.py`)

PostgreSQL-backed persistence:

```sql
captures (difficulty, date)     → map + orders (JSONB)
gpu_solutions (difficulty, date) → score + actions + metadata
dp_plans (difficulty, date)     → Zig-format DP plan
order_sequences (difficulty, map_seed) → full order history
```

**Score Safety**: Never overwrites better score (unless force=True).
**Capture Merging**: Positional comparison — always keeps longer order list.
**Daily Reset**: UTC-based date key; old data preserved for analysis.

---

## Infrastructure

### Hardware

- **GPU**: NVIDIA RTX 5090, 32 GB VRAM
- **CUDA**: PyTorch 2.10 + CUDA 12.8 + Triton 3.6.0
- **CPU**: Used for BFS precompute, simulation, and Zig FFI

### Software Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| GPU Solver | Python + PyTorch CUDA | DP beam search, state evaluation |
| Reactive Bot | Zig 0.15.2 | Fast WebSocket client, BFS, ST-A* |
| FFI Bridge | Zig DLL + Python ctypes | Fast verification and pre-simulation |
| Game Simulator | Python (numpy) | Offline testing and order discovery |
| Database | PostgreSQL 5433 | Captures, solutions, replay logs |
| Dashboard | SvelteKit + Node.js | Live game view, pipeline monitoring |
| Replay DB | PostgreSQL | Game history and analysis |

### Build System

**Zig**: `build.zig` with per-difficulty compilation:
```bash
zig build -Doptimize=ReleaseFast -Ddifficulty=expert     # executable
zig build -Doptimize=ReleaseFast                          # shared library (DLL)
```

**Python**: No build step; GPU kernels JIT-compiled via torch.compile.

### Sweep Testing

Per-difficulty sweep scripts on dedicated ports:
| Script | Port | Difficulty |
|--------|------|-----------|
| `sweep_easy.py` | 9850 | Easy (1 bot, 12x10) |
| `sweep_medium.py` | 9860 | Medium (3 bots, 16x12) |
| `sweep_hard.py` | 9870 | Hard (5 bots, 22x14) |
| `sweep_expert.py` | 9880 | Expert (10 bots, 28x18) |
| `sweep_nightmare.py` | 9890 | Nightmare (20 bots, 30x18) |

---

## Key Algorithms Summary

| Algorithm | Where Used | Complexity | Purpose |
|-----------|-----------|------------|---------|
| BFS (all-pairs) | precompute.py, pathfinding.zig | O(V+E) per source | Shortest paths, distance maps |
| GPU BFS (matrix multiply) | precompute.py | O(N^3) vectorized | Parallel shortest paths on CUDA |
| Beam Search DP | gpu_beam_search.py | O(B*A*R) | Single-bot optimal planning |
| Sequential Multi-Bot DP | gpu_sequential_solver.py | O(N*B*A*R) | Multi-bot with locked trajectories |
| Mini-TSP (exhaustive) | trip.zig | O(3!) = O(6) | 1-3 item trip optimization |
| Space-Time A* | spacetime.zig | O(V*T*log(V*T)) | Temporal collision avoidance |
| PIBT (recursive) | nightmare_pathfinder.py | O(N*D) | 20-bot push-chain coordination |
| MRTA (greedy) | nightmare_task_alloc.py | O(N*M) | Bot-to-task assignment |
| Checkpoint Local Search | nightmare_offline.py | Anytime | Solution improvement via perturbation |
| Hungarian Assignment | (tested, rejected) | O(N^3) | Globally optimal matching (hurts in practice) |
| Chain Reaction Exploit | nightmare_solver_v2.py | — | Multi-order cascading delivery |

**Legend**: B=beam width (states), A=actions, R=rounds, N=bots, D=push depth, M=items, V=vertices, T=time horizon.
