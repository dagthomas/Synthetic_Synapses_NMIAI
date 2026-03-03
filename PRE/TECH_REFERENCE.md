# Technology Reference

All technologies, algorithms, and design decisions used in the AINM Grocery Bot project.
Covers both the **Zig bot** (reactive live play) and the **GPU DP solver** (offline optimization).

---

## Table of Contents

1. [Project Layout](#project-layout)
2. [Zig Bot](#zig-bot)
   - [Language & Toolchain](#language--toolchain)
   - [Build System](#build-system)
   - [Source File Map](#source-file-map)
   - [Core Algorithms](#core-algorithms)
   - [Decision Pipeline](#decision-pipeline)
   - [WebSocket Client](#websocket-client)
   - [Anti-Oscillation & Desync](#anti-oscillation--desync)
   - [Per-Difficulty Tuning](#per-difficulty-tuning)
3. [GPU DP Solver](#gpu-dp-solver)
   - [Language & Libraries](#language--libraries)
   - [Source File Map](#gpu-source-file-map)
   - [Game Engine (Pure Python Simulator)](#game-engine-pure-python-simulator)
   - [GPU Beam Search / DP — Single Bot](#gpu-beam-search--dp--single-bot)
   - [Sequential GPU DP — Multi-Bot](#sequential-gpu-dp--multi-bot)
   - [Iterative Refinement](#iterative-refinement)
   - [Precomputed Tables](#precomputed-tables)
   - [Python MAPF Planner](#python-mapf-planner)
   - [Replay & Desync Correction](#replay--desync-correction)
   - [Solution Store](#solution-store)
   - [Capture System](#capture-system)
4. [Infrastructure](#infrastructure)
   - [PostgreSQL Replay DB](#postgresql-replay-db)
   - [SvelteKit Dashboard](#sveltekit-dashboard)
5. [Algorithms Index](#algorithms-index)
6. [Key Constants & Limits](#key-constants--limits)

---

## Project Layout

```
grocery-bot-zig/       Zig reactive bot (live play & capture)
  src/
    main.zig           WebSocket client, game loop, desync detection
    strategy.zig       Core decision engine (~1360 lines)
    trip.zig           Mini-TSP trip planner
    pathfinding.zig    BFS distance maps, collision-aware routing
    spacetime.zig      Space-Time A* (MAPF stub)
    sim.zig            Pure-Zig game simulator
    sim_main.zig       Simulator entry point
    parser.zig         JSON game-state parser
    ws.zig             WebSocket client with TLS
    types.zig          Core structs (Pos, Bot, GameState, …)
  build.zig            Zig build script (per-difficulty compile options)
  build_all.py         Python helper: builds all 5 executables

grocery-bot-gpu/       Python + PyTorch GPU solver
  game_engine.py       Pure-Python game simulator (authoritative CPU reference)
  gpu_beam_search.py   GPU DP / beam search (GPUBeamSearcher class)
  gpu_sequential_solver.py  Sequential per-bot DP + refinement
  replay_solution.py   Live replay with desync correction
  zig_capture.py       Run Zig bot, parse game log → capture.json
  solution_store.py    Load/save best.json, capture.json, meta.json
  planner.py           MAPF coordinated planner (Python)
  live_solver.py       Live game solver (precompute or reactive modes)
  precompute.py        PrecomputedTables (BFS distances, item adjacency)
  pathfinding.py       Python BFS utilities
  configs.py           Difficulty configs (grid size, bot count, …)
  ws_client.py         WebSocket replay client helper
  solutions/           Per-difficulty solution store
    easy/              capture.json, best.json, meta.json
    medium/
    hard/
    expert/
```

---

## Zig Bot

### Language & Toolchain

| Item | Value |
|------|-------|
| Language | Zig 0.15.2 |
| Executable path | `C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe` |
| Optimization | `ReleaseFast` (required — Debug is 10× slower) |
| TLS | Zig `std.crypto.tls` — built-in, no external dep |
| JSON | Hand-rolled parser in `parser.zig` (no external dep) |
| Memory | `std.heap.GeneralPurposeAllocator` for dynamic allocs |
| Concurrency | Single-threaded; `std.Thread.sleep()` for the 15ms delay |

Zig 0.15 API notes:
- `std.Thread.sleep(ns)` — NOT `std.time.sleep`.
- `std.debug.print` writes to **stderr**.
- `std.io.getStdOut()` does not exist in this version.
- Build options use `@import("config")` → `config.difficulty` (enum).

### Build System

`build.zig` uses Zig's native build API. A `Difficulty` enum build option controls
compile-time constants, letting the same source produce 5 specialized executables:

```
grocery-bot.exe         (auto = runtime detection from bot count)
grocery-bot-easy.exe
grocery-bot-medium.exe
grocery-bot-hard.exe
grocery-bot-expert.exe
```

Plus matching simulator executables (`grocery-sim-*.exe`).

```bash
# Build all via Python helper
python build_all.py

# Build one
zig.exe build -Doptimize=ReleaseFast -Ddifficulty=expert
```

The `config` module is a `b.addOptions()` anonymous import. `strategy.zig` reads
`config.difficulty` at compile time to eliminate dead branches — e.g., MAPF code is
compiled out for Easy/Medium.

### Source File Map

| File | Lines | Role |
|------|-------|------|
| `main.zig` | 258 | Entry point: live mode & replay mode, desync detection |
| `strategy.zig` | ~1360 | **All decision logic** — orchestrator, trip execution, anti-osc |
| `trip.zig` | 359 | Mini-TSP: 1/2/3-item route scoring |
| `pathfinding.zig` | 88 | BFS distance maps, collision-aware first-step |
| `spacetime.zig` | — | Space-Time A* (reservation table, currently disabled) |
| `sim.zig` | — | Pure-Zig game simulator (no WebSocket) |
| `sim_main.zig` | — | Simulator CLI entry |
| `parser.zig` | 120 | Zero-copy JSON parser for game state |
| `ws.zig` | 376 | WebSocket client (RFC 6455 + TLS via `std.crypto`) |
| `types.zig` | 108 | `Pos`, `Dir`, `Bot`, `GameState`, `NeedList`, `DistMap` |

### Core Algorithms

#### BFS Distance Maps (`pathfinding.zig`)

For each target cell, compute the shortest-path distance and the first direction to take
from any starting cell. Uses a standard BFS on the grid, respecting walls and shelves.

Key function:
```zig
pub fn bfs(state, start, target, bot_id, bot_positions) BfsResult
```

Collision-aware: the first step avoids cells occupied by other bots (except the spawn
tile and the target itself). Subsequent steps ignore occupied cells (bots move
simultaneously).

#### Trip Planner — Mini-TSP (`trip.zig`)

Evaluates all 1-, 2-, and 3-item pickup combinations and scores them:

```
score = (value × 10000) / cost
```

Where:
- `value` = number of items in the combo that are needed by the active order
- `cost` = total walking distance (pickup → pickup → dropoff), using BFS distances
- Massive bonus for **order-completing trips** (picks up the last needed items)
- Large bonus for combos that include preview-order items (future value)

A "trip" is a cached plan of up to 3 item positions to visit in order, ending at dropoff.

#### Orchestrator (`strategy.zig` ~614–780)

Centralized multi-bot assignment. Runs every round before per-bot decisions:

1. Count how many of each item type are already en-route (assigned to bots).
2. Assign active-order items to bots greedily by distance. Respects `max_pickers`
   (3 for 5–7 bots, 4 for 8+ bots) to concentrate items on fewer bots.
3. Optional: type blocking (Medium only, `bot_count < 5`) prevents assigning duplicate
   types beyond what the order needs.
4. Phase 2: assign preview-order items to idle bots (up to `max_preview_carriers`).
5. Concentration bonus (3-bot Medium only): +5 score weight if a bot already has a
   matching item, to batch items together.

#### Anti-Oscillation System (`strategy.zig`)

Multi-layer stall detection:
1. **Position history** (24 rounds): detect if bot visits the same cells repeatedly.
2. **Stall counter**: if bot hasn't moved in 6 rounds → trigger `escape_mode`.
3. **Escape mode** (4 rounds): pick a random direction and force moves, ignoring normal
   routing.
4. **Near-dropoff patience**: bots within distance ≤ 2 of dropoff wait up to 3 rounds
   before escalating (avoids false escapes for bots legitimately at dropoff).
5. **Stuck-order escape**: if the same active order has been in progress for 25 rounds,
   force a physical escape every 12 rounds.

#### `PersistentBot` State (`strategy.zig`)

Each bot keeps cross-round state in a `PersistentBot` struct (array, not heap-allocated):

```zig
pub const PersistentBot = struct {
    trip_ids: [INV_CAP][32]u8,     // cached trip item IDs
    trip_adjs: [INV_CAP]Pos,       // adjacency target positions
    trip_count, trip_pos: u8,      // trip length and progress
    has_trip, delivering: bool,
    stall_count: u16,
    pos_hist: [HIST_LEN]Pos,       // 24-round position history
    osc_count: u16,
    escape_rounds: u8,
    pickup_fail_count: u8,
    ...
};
```

### Decision Pipeline

Per-bot, each round, `decideActions()` in `strategy.zig` runs this priority cascade:

```
1. DROP OFF        — at dropoff with active-order items → deliver
2. EVACUATE        — at dropoff without active items → flee (if multi-bot)
3. ESCAPE          — oscillation/stall detected → escape movement
4. PICK UP         — adjacent needed item → pick (active pass first, preview second)
5. DELIVER         — has active items, not near pickup → navigate to dropoff
   └─ endgame buffer: dist + inv_len + 1 (dynamic, scales with inventory)
   └─ far_with_few: dist>8, inv<2, 5+ bots → skip delivery, pick more first
6. FOLLOW TRIP     — trip plan exists → navigate to next pickup position
7. PLAN TRIP       — no trip → score all 1/2/3-item combos, pick best
8. DELIVER FALLBACK — has any items → go to dropoff
9. PRE-POSITION    — idle → move toward likely-needed item clusters
10. DEAD INVENTORY — items that can't be delivered → camp near dropoff
11. WAIT           — nothing to do
```

The cascade is re-evaluated every round — no planning horizon, fully reactive.

### WebSocket Client (`ws.zig`)

Hand-rolled RFC 6455 WebSocket client (376 lines):
- TCP connect via `std.net.tcpConnectToHost`.
- TLS via `std.crypto.tls.Client` (Zig built-in, no OpenSSL/libssl).
- HTTP upgrade handshake with Sec-WebSocket-Key / Accept header validation.
- Frame parsing: handles FIN bit, text/binary frames, server sends unmasked frames.
- Client sends masked frames (required by RFC 6455).
- Recv loop buffers partial frames.

### Anti-Oscillation & Desync

**15ms send delay** (`main.zig:224`):
```zig
std.Thread.sleep(15 * std.time.ns_per_ms);
```
Without this, on fast connections the response arrives while the server is still
processing the previous round → 1-round action offset → all actions are shifted → bot
crashes into walls and misses pickups.

**Desync detection** (`main.zig:186–210`):
- After each round, compare actual bot positions to `expected_next_pos[]` stored by
  `strategy.zig`.
- Threshold: **1 mismatch** (aggressive).
- Trigger: **2 consecutive** mismatched rounds.
- Recovery: skip a decision cycle to re-sync round numbering.

### Per-Difficulty Tuning

| Parameter | Easy | Medium | Hard | Expert |
|-----------|------|--------|------|--------|
| `max_pickers` | 1 | 3 | 3 | 4 |
| `max_preview_carriers` | 1 | 1 | 2 | 3 |
| `concentration_bonus` | — | +5 (3-bot) | — | — |
| `base_detour` | 5 | 1 | 1 | 1 |
| `completing_detour` | 8 | 4 | 4 | 4 |
| `type_blocking` | off | on | off | off |
| MAPF (ST-A*) | off | off | off | off |

---

## GPU DP Solver

### Language & Libraries

| Item | Value |
|------|-------|
| Language | Python 3.11+ |
| GPU framework | PyTorch 2.x (CUDA tensors) |
| Hardware target | NVIDIA RTX 5090, 34.2 GB VRAM |
| CUDA version | 12.4 (`cu124` wheel) |
| NumPy | Used for locked-trajectory arrays and precomputed tables |
| WebSocket | `websockets` (async, Python stdlib `asyncio`) |
| Data format | JSON (game capture, solutions, metadata) |

The solver falls back to CPU (`device='cpu'`) when CUDA is unavailable, but is
significantly slower (minutes vs seconds).

### GPU Source File Map

| File | Role |
|------|------|
| `gpu_beam_search.py` | `GPUBeamSearcher` — core GPU DP engine |
| `gpu_sequential_solver.py` | Sequential per-bot DP + iterative refinement |
| `game_engine.py` | Authoritative CPU game simulator |
| `precompute.py` | `PrecomputedTables` (BFS dist, item adjacency, cached to `.npz`) |
| `pathfinding.py` | Python BFS utilities for map preprocessing |
| `configs.py` | Difficulty configs (grid dims, bot counts, item types) |
| `solution_store.py` | Load/save `best.json`, `capture.json`, `meta.json` |
| `replay_solution.py` | Live replay with adaptive desync correction |
| `zig_capture.py` | Run Zig bot, parse game log, merge into capture |
| `live_solver.py` | Live solver (precompute mode & reactive capture mode) |
| `planner.py` | MAPF coordinated Python planner |
| `ws_client.py` | WebSocket replay helper |

### Game Engine (Pure Python Simulator)

`game_engine.py` is the authoritative reference simulator. It is used:
1. As the CPU verifier after every GPU DP pass (to confirm the action sequence
   actually achieves the DP score).
2. To pre-simulate locked bots for multi-bot DP.
3. To replay and verify solutions before submission.

Key exports:
```python
init_game(seed, difficulty) → (GameState, all_orders)
init_game_from_capture(capture_data, num_orders) → (GameState, all_orders)
step(gs, round_actions, all_orders)             # mutates gs in-place
build_map_from_capture(capture_data) → MapState
actions_to_ws_format(round_actions, map_state) → list[dict]
```

Action constants: `ACT_WAIT=0, ACT_MOVE_UP=1, ..., ACT_PICKUP=5, ACT_DROPOFF=6`

`GameState` holds: `bot_positions [N,2]`, `bot_inventories [N,3]`, `score`, `round`,
`map_state`, `active_order_idx`, `active_delivered [6]`.

`MapState` holds: `grid [H,W]`, `items []`, `drop_off (x,y)`, `spawn (x,y)`,
`item_adjacencies [][]`, `num_items`, `width`, `height`.

### GPU Beam Search / DP — Single Bot

`GPUBeamSearcher` in `gpu_beam_search.py`.

**State representation** (all tensors, batch dimension B):

```python
bot_x, bot_y:   [B] int16   — position
bot_inv:        [B, 3] int8 — inventory (-1=empty, 0–15=type_id)
active_idx:     [B] int32   — current active order index
active_del:     [B, 6] int8 — per-item delivery flags
score:          [B] int32   — cumulative score
orders_comp:    [B] int32   — orders completed (for dedup)
```

**State deduplication** via int64 hash packed from all state fields:

```python
hash = x | (y << 8) | (inv0 << 16) | (inv1 << 24) | (inv2 << 32) |
       (active_idx << 40) | (active_del_packed << 48)
```

`torch.unique()` deduplicates states, keeping the highest score per hash.

**Per-round expansion:**

```
For each state in B:
  Try all 7 actions (wait, move ×4, pickup, dropoff)
  Apply game rules (wall collision, pickup validity, delivery scoring)
  → expanded states: up to B×7
Dedup by hash, keep best score
Truncate to max_states if needed (keep top scores)
```

**Pickup validity** (GPU tensor logic):
- Bot must be adjacent to the shelf containing the item.
- Item must not already be in inventory.
- Inventory must not be full.
- Item must be of a type needed by the active or preview order.

**Score update** (GPU):
- Delivery: if all `active_del` flags for the active order are set → +5 bonus,
  advance to next order.
- Per item delivered: +1.

**Multi-bot extension**: locked trajectories are uploaded as `locked_pos_x[L, 300]`,
`locked_pos_y[L, 300]` tensors. During each round's expansion, the bot being planned
cannot move to any cell occupied by a locked bot in that round.

### Sequential GPU DP — Multi-Bot

`solve_sequential()` in `gpu_sequential_solver.py`.

```
Pass 1 (sequential):
  bot_actions = {}
  for bot_id in plan_order:          # default [0, 1, ..., N-1]
    locked = pre_simulate_locked(all planned bots so far)
    searcher = GPUBeamSearcher(locked_trajectories=locked)
    score, acts = searcher.dp_search(...)
    bot_actions[bot_id] = acts

CPU verify Pass 1 combined actions → best_score

Pass 2+ (refinement, up to max_refine_iters):
  for bot_id in range(num_bots):
    locked = pre_simulate_locked(ALL other bots)
    searcher = GPUBeamSearcher(locked_trajectories=locked)
    score, acts = searcher.dp_search(...)
    new_score = cpu_verify(replace bot_id's actions)
    if new_score > best_score: keep; else: revert
  if no improvement: stop early
```

**`pre_simulate_locked()`**: CPU-simulates all locked bots through all 300 rounds to
produce exact position arrays. This is necessary because locked bots interact with each
other (their collision outcomes depend on all bots moving simultaneously).

**`cpu_verify()`**: runs `game_engine.step()` for all 300 rounds with the combined
actions, returns the final score. This is the ground truth — GPU DP score is an
approximation (single-bot assumes other bots don't exist; CPU verify accounts for all
interactions).

**Multi-restart** (`solve_multi_restart()`): Tries `num_restarts` different bot planning
orders (default, reversed, then random). Keeps the best CPU-verified score.

### Iterative Refinement

Key insight: after Pass 1, bot 0 was planned solo but bot 1 may now block it.
Refinement re-plans each bot with all others locked, picking up improvements.

Each refinement step immediately CPU-verifies before committing → safe to revert if the
new plan hurts the combined score (due to interactions not visible to single-bot DP).

Typical gains:
- Pass 1 → Refinement iter 1: +5–15 points
- Refinement iter 2: +0–5 points
- Rarely improves after iter 2

### Precomputed Tables

`PrecomputedTables` in `precompute.py`:
- Runs BFS from every cell to every cell on the map grid.
- Stores `dist_map[target_y, target_x, src_y, src_x]` = shortest path length.
- Stores `item_adjacencies[item_idx]` = list of valid adjacent floor cells for pickup.
- Cached to `cache/tables_<hash>.npz` keyed by capture hash.
- Load time: ~0.1s from cache, ~2s to recompute.

### Python MAPF Planner

`planner.py` — multi-agent pathfinding coordinated planner. Used by `live_solver.py`
for live play (not GPU DP offline).

Key techniques:
- **Hungarian assignment** (scipy): optimal bot-to-task assignment by distance.
- **Reservation table**: tracks which cells are reserved each round to avoid collisions.
- **Space-Time A\***: finds collision-free paths through reserved cells.
- **Iterative local search** (`planner_optimizer.py`): perturb one assignment, re-simulate,
  keep improvements.
- **Multi-strategy**: try different `max_active_bots` values (1–N), keep best score.

The live solver (`live_solver.py`) imports `planner.solve` — any improvements to
`planner.py` automatically propagate to the live solver.

### Replay & Desync Correction

`replay_solution.py` implements a two-mode replay strategy:

**SYNCED mode**: actual position matches expected → send DP action as-is.

**Offset mode**: positions match `round - N` rather than `round` (missed N rounds) →
increment `round_offset`, shift DP index. Handles connection lag causing the server to
advance without our response.

**Goal extraction** (`extract_goals()`): pre-parse the DP action sequence to find all
PICKUP and DROPOFF actions. Between consecutive goals, the bot is just navigating.
On desync, use BFS toward the next goal position as a recovery action.

**BFS recovery** (`bfs_next_action()`): when a bot is desynced, BFS toward the goal
avoiding currently occupied cells. Falls back to greedy Manhattan-distance move.

### Solution Store

`solution_store.py` manages the on-disk solution state:

```
solutions/<difficulty>/
  capture.json   — map structure + all observed orders (grows over time)
  best.json      — 300-round action sequence [[(act, item), ...], ...]
  meta.json      — score, date, capture_hash, optimizations_run
```

Safety invariants:
- `save_solution()` never overwrites a better score.
- `save_capture()` merges (caller's responsibility) — appends new orders, never
  replaces the grid/items (first capture is authoritative for the map).
- `meta.json` tracks the MD5 hash of `capture.json` to detect stale solutions
  (different capture = different map).

### Capture System

`zig_capture.py` orchestrates the capture workflow:

1. Run `grocery-bot-<diff>.exe <ws_url>` as a subprocess.
2. Watch for new `game_log_*.jsonl` files in `grocery-bot-zig/`.
3. Parse the JSONL: extract `grid`, `items`, `drop_off` from round 0; collect every
   unique order ID seen in any round.
4. Merge with existing `capture.json`: add new order IDs, keep existing grid/items
   (grid doesn't change between games of the same difficulty).

Why iterative capture matters: the first Zig run sees ~15 orders (300 rounds / ~20
rounds per order). Each replay run sees additional orders that happen to appear for the
first time in that seed's schedule. After 3–5 iterations: 30+ orders. More orders →
DP can plan further ahead → higher score.

---

## Infrastructure

### PostgreSQL Replay DB

- Docker Compose: `grocery-bot-zig/replay/docker-compose.yml`
- Port: 5433 (local), mapped from container port 5432
- Schema: `runs` table (difficulty, seed, score, timestamp) + `rounds` table (per-round
  state snapshots for visualization)
- Sweeps auto-insert via `replay/recorder.py`

```bash
cd grocery-bot-zig/replay
docker compose up -d db
```

### SvelteKit Dashboard

- Directory: `grocery-bot-zig/replay/app/`
- Dev server: `npm run dev` → http://localhost:5173
- Pages:
  - `/` — historical game viewer (SVG grid)
  - `/live` — real-time live game
  - `/gpu` — GPU solve monitor (Matrix-style terminal, score graph, state exploration)

The `/live` page connects to `live_solver.py` (MAPF planner, not GPU DP). The `/gpu`
page connects to `gpu_solve_stream.py` or `gpu_multi_solve_stream.py` via WebSocket
and renders solve progress in real time.

---

## Algorithms Index

| Algorithm | Where | Used For |
|-----------|-------|----------|
| BFS (grid pathfinding) | `pathfinding.zig`, `pathfinding.py` | Navigation, distance maps |
| Space-Time A\* (ST-A\*) | `spacetime.zig` (stub), `planner.py` | MAPF collision-free routing |
| Mini-TSP (1/2/3-item) | `trip.zig` | Optimal pickup order evaluation |
| GPU DP (exact BFS + dedup) | `gpu_beam_search.py` | Single-bot optimal planning |
| Sequential per-bot DP | `gpu_sequential_solver.py` | Multi-bot optimization |
| Iterative refinement | `gpu_sequential_solver.py` | Improving multi-bot solutions |
| Hungarian assignment | `planner.py` (scipy) | Bot-to-task optimal assignment |
| BFS goal-seeking | `replay_solution.py` | Desync recovery |
| Simulated annealing (SA) | `planner_optimizer.py` | Local search optimization |
| Multi-strategy (mab) | `multi_solve.py` | Trying different `max_active_bots` |
| Anti-oscillation (hist) | `strategy.zig` | Preventing Zig bot stalls |
| Reservation table | `spacetime.zig`, `planner.py` | Collision avoidance |

---

## Key Constants & Limits

| Constant | Value | Source |
|----------|-------|--------|
| `MAX_ROUNDS` | 300 | `configs.py`, `types.zig` |
| `INV_CAP` | 3 | Items per bot |
| `MAX_BOTS` | 10 | Expert difficulty |
| `MAX_ITEMS` | 200 | Shelf items per map |
| `MAX_ORDERS` | 100 | Orders per game |
| `MAX_ORDER_SIZE` | 6 | Items per order |
| `UNREACHABLE` | 9999 | BFS sentinel |
| `HIST_LEN` | 24 | Anti-osc position history |
| `stall_threshold` | 6 | Rounds before escape mode |
| `escape_rounds` | 4 | Escape mode duration |
| WS send delay | 15ms | Prevents 1-round offset |
| Desync threshold | 1 mismatch | Aggressive detection |
| Desync trigger | 2 consecutive | Avoids false positives |

### Difficulty Configs

| | Easy | Medium | Hard | Expert |
|--|------|--------|------|--------|
| Grid | 12×10 | 16×12 | 22×14 | 28×18 |
| Bots | 1 | 3 | 5 | 10 |
| Aisles | 2 | 3 | 4 | 5 |
| Item types | 4 | 8 | 12 | 16 |
| Order size | 3–4 | 3–5 | 3–5 | 4–6 |
| Sweep port | 9850 | 9860 | 9870 | 9880 |

### GPU DP State Budgets (default)

| Difficulty | `max_states` | Notes |
|------------|-------------|-------|
| Easy | 500,000 | Single-bot: provably optimal |
| Medium | 1,000,000 | |
| Hard | 2,000,000 | |
| Expert | 5,000,000 | Reduce if OOM |

### Action Encoding

| Constant | Integer | WebSocket string |
|----------|---------|-----------------|
| `ACT_WAIT` | 0 | `"wait"` |
| `ACT_MOVE_UP` | 1 | `"move_up"` |
| `ACT_MOVE_DOWN` | 2 | `"move_down"` |
| `ACT_MOVE_LEFT` | 3 | `"move_left"` |
| `ACT_MOVE_RIGHT` | 4 | `"move_right"` |
| `ACT_PICKUP` | 5 | `"pick_up"` |
| `ACT_DROPOFF` | 6 | `"drop_off"` |

### Cell Types (grid encoding)

| Constant | Value | Meaning |
|----------|-------|---------|
| `CELL_FLOOR` | 0 | Walkable |
| `CELL_WALL` | 1 | Blocked |
| `CELL_SHELF` | 2 | Item shelf (not walkable) |
| `CELL_DROPOFF` | 3 | Delivery point (walkable) |
