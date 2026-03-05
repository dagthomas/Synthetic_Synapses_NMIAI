# GPU Pipeline — Complete Technical Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Token & Session Model](#token--session-model)
4. [Pipeline Modes](#pipeline-modes)
5. [Iterate Pipeline — Full Walkthrough](#iterate-pipeline--full-walkthrough)
6. [Single Pipeline — Full Walkthrough](#single-pipeline--full-walkthrough)
7. [SvelteKit Dashboard UI](#sveltekit-dashboard-ui)
8. [Server Endpoints](#server-endpoints)
9. [Python Components](#python-components)
10. [GPU DP Solver Deep Dive](#gpu-dp-solver-deep-dive)
11. [Game Engine](#game-engine)
12. [Precomputation](#precomputation)
13. [Solution Storage](#solution-storage)
14. [Replay System](#replay-system)
15. [Order Capture & Discovery](#order-capture--discovery)
16. [Live GPU Stream Solver](#live-gpu-stream-solver)
17. [Local Iterate Script](#local-iterate-script)
18. [SSE Event Reference](#sse-event-reference)
19. [Configuration & Tuning](#configuration--tuning)
20. [File Reference](#file-reference)

---

## Overview

The pipeline is an automated system for maximizing scores in a competitive grocery bot game. It exploits a key property: **game tokens are reusable within a ~288-second window, and the same token always produces the same deterministic game**. This enables an iterative strategy:

1. **Play** a game to discover orders
2. **Optimize** offline with GPU DP using discovered orders
3. **Replay** the optimized plan on the same token to score higher and discover more orders
4. **Repeat** until the token expires

Each optimize+replay cycle takes ~25 seconds and typically discovers 1-3 additional orders, creating a snowball effect where each iteration feeds better data to the next.

The system spans three languages:
- **Zig** — Fast real-time bot for initial game play
- **Python/CUDA** — GPU dynamic programming solver (RTX 5090, 32GB VRAM)
- **JavaScript/SvelteKit** — Web dashboard for orchestration and visualization

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SvelteKit Dashboard (port 5173)                  │
│  /pipeline page — Two modes: Single | Iterate                       │
│  Grid visualization, score chart, iteration tabs, log console       │
├───────────────┬─────────────────────────┬───────────────────────────┤
│  POST         │  POST                   │  POST                     │
│  /api/pipeline│  /api/pipeline           │  /api/optimize            │
│  /run         │  /iterate               │  /replay                  │
└───────┬───────┴────────────┬────────────┴──────────┬────────────────┘
        │                    │                       │
        │ spawns             │ spawns                │ spawns
        v                    v                       v
 ┌──────────────┐  ┌─────────────────┐     ┌─────────────────┐
 │live_gpu_     │  │ grocery-bot.exe │     │replay_solution  │
 │stream.py     │  │ (Zig bot)       │     │.py              │
 │              │  └────────┬────────┘     └────────┬────────┘
 │ WebSocket ◄──┤           │ WebSocket              │ WebSocket
 │ game server  │           │ game server            │ game server
 └──────┬───────┘           │                        │
        │                   v                        │
        │           ┌───────────────┐                │
        │           │capture_from_  │                │
        │           │game_log.py    │                │
        │           └───────┬───────┘                │
        │                   │                        │
        v                   v                        v
 ┌──────────────────────────────────────────────────────────────┐
 │                    solution_store.py                          │
 │  solutions/<difficulty>/                                     │
 │    best.json     — action sequences (never overwrite better) │
 │    capture.json  — grid + items + accumulated orders         │
 │    meta.json     — score, seed, timestamps, opt count        │
 └──────────────────────────┬───────────────────────────────────┘
                            │
                            v
 ┌──────────────────────────────────────────────────────────────┐
 │                   optimize_and_save.py                        │
 │  Loads capture → GPU DP solve → saves solution               │
 │  Modes: cold-start | warm-only | normal+warm-refine          │
 └──────────────────────────┬───────────────────────────────────┘
                            │
                            v
 ┌──────────────────────────────────────────────────────────────┐
 │               gpu_sequential_solver.py                        │
 │  Pass 1: Sequential per-bot DP (bot 0, then 1 w/ 0 locked…) │
 │  Pass 2+: Iterative refinement (re-plan each bot)            │
 └──────────────────────────┬───────────────────────────────────┘
                            │
                            v
 ┌──────────────────────────────────────────────────────────────┐
 │                  gpu_beam_search.py                            │
 │  GPUBeamSearcher — Exact DP via exhaustive BFS + dedup        │
 │  CUDA tensors, torch.compile(mode='default'), Triton 3.6.0   │
 │  Single-bot: explores ALL reachable states (provably optimal) │
 └──────────────────────────┬───────────────────────────────────┘
                            │
                            v
 ┌──────────────────────────────────────────────────────────────┐
 │                     precompute.py                              │
 │  GPU BFS all-pairs shortest paths, trip cost tables            │
 │  PrecomputedTables + TripTable, .npz disk cache               │
 └──────────────────────────────────────────────────────────────┘
```

---

## Token & Session Model

- **Token lifetime**: ~288 seconds (~5 minutes) from creation
- **Reusability**: Same token URL can start **multiple games** within the window
- **Determinism**: Same token = same seed = same map = same order sequence
- **Competition rules**: Only the **maximum score per map** counts on the leaderboard

This means within one token:
- Zig bot game: ~60s (120s game clock, but often finishes faster)
- Each replay: ~3s (5ms per round × 300 rounds)
- Each GPU optimize: ~20s (configurable)
- **Total cycles possible**: 6-8 optimize+replay cycles after the initial game

---

## Pipeline Modes

### Iterate Mode (Primary)

The fast iterate pipeline. Zig bot plays first to discover orders, then loops GPU optimize → replay → capture.

```
 0-60s:   Zig bot plays → ~80 score, discovers ~15 orders
 60-80s:  GPU optimize (20s, cold-start) → ~140 score
 80-83s:  Replay → ~140 live, discovers ~17 orders
 83-100s: GPU optimize (17s, warm-start) → ~160 score
100-103s: Replay → ~160 live, discovers ~19 orders
103-118s: GPU optimize (15s, warm-start) → ~175 score
... continue until ~280s
280-283s: Final replay → best score registered
```

### Single Mode

One live GPU stream game with extended post-game optimization. Used when you want the live GPU solver's real-time decisions rather than replay of precomputed plans.

```
 0-120s:  live_gpu_stream.py plays (GPU actions every round)
120-280s: Post-game GPU optimization (offline DP with captured orders)
```

---

## Iterate Pipeline — Full Walkthrough

### Step 1: User Starts Pipeline

User pastes a WebSocket token URL into the Pipeline page and clicks "Start Iterative". The UI sends:

```js
POST /api/pipeline/iterate
{
  url: "wss://game.ainm.no/ws?token=...",
  timeBudget: 280,         // seconds total
  gpuOptimizeTime: 20,     // seconds per GPU optimize pass
  postOptimizeTime: 30     // unused in iterate mode
}
```

### Step 2: Solution Cleanup

The server clears all existing solution files for all difficulties. New token = new game = old data is invalid.

```js
solutions/easy/{best.json, capture.json, meta.json}     → deleted
solutions/medium/{best.json, capture.json, meta.json}    → deleted
solutions/hard/{best.json, capture.json, meta.json}      → deleted
solutions/expert/{best.json, capture.json, meta.json}    → deleted
```

### Step 3: Iteration 0 — Zig Bot Play

The server spawns the Zig bot executable:

```
grocery-bot-zig/zig-out/bin/grocery-bot.exe wss://game.ainm.no/ws?token=...
```

The Zig bot:
- Connects to the WebSocket game server
- Auto-detects difficulty from grid dimensions and bot count
- Plays the full 300-round game using its built-in decision cascade
- Writes a `game_log_*.jsonl` file to `grocery-bot-zig/`
- Prints score updates and game over to stderr

The server:
- Parses stderr for round scores (`R150/300 Score:45`) and difficulty detection
- Copies the game log to `grocery-bot-gpu/` for the capture step
- Emits SSE events: `iter_start`, `round` (every 10 rounds), `log`

### Step 4: Order Capture from Zig Game

The server spawns the capture script:

```
python capture_from_game_log.py game_log_XXXX.jsonl hard
```

This script:
- Parses the JSONL game log (alternating game_state / action_response lines)
- Extracts grid, items, drop_off from round 0
- Accumulates all orders seen throughout the game
- Saves to `solutions/hard/capture.json` via `merge_capture()` (positional merge — keeps longer order list)
- Saves the Zig bot's action sequence as a warm-start solution via `save_solution()`

### Step 5: Iterations 1+ — GPU Optimize

The server spawns the GPU optimizer:

```
# Iteration 1: cold-start with fast settings
python -u optimize_and_save.py hard --max-time 20 --orderings 1 --refine-iters 2

# Iteration 2+: warm-only refine from previous solution
python -u optimize_and_save.py hard --max-time 20 --warm-only --refine-iters 2
```

#### Cold-Start (Iteration 1)

`optimize_and_save.py` calls `solve_sequential()` which:

1. **Precompute**: Builds `PrecomputedTables` (GPU BFS all-pairs shortest paths) + `TripTable`
2. **Pass 1 — Sequential DP**: For each bot in order:
   - Create `GPUBeamSearcher` instance
   - Upload locked trajectories of all previously-solved bots to GPU
   - Run `dp_search()`: exact DP via exhaustive BFS + state deduplication
   - Prune states to `max_states` if population exceeds budget
   - Extract best action sequence for this bot
3. **Pass 2 — Refinement**: For each bot (alternating order):
   - Lock ALL other bots to their current plans
   - Re-solve this bot with GPU DP
   - Keep result only if score improves
   - Repeat for `refine_iters` iterations
4. **CPU Verify**: Replay all bot actions through the Python game engine to validate

#### Warm-Only (Iteration 2+)

`optimize_and_save.py` with `--warm-only` calls `refine_from_solution()`:
- Loads the existing `best.json` solution
- Skips Pass 1 entirely
- Goes straight to refinement: re-plans each bot individually with others locked
- Uses the latest `capture.json` which now has more orders from the previous replay
- This is faster because it starts from a good solution and only needs to incorporate new orders

#### Solution Save

After solving, `save_solution()` checks if the new score exceeds the existing best. It **never overwrites a better score**.

### Step 6: Replay Optimized Solution

The server spawns the replay script:

```
python replay_solution.py wss://game.ainm.no/ws?token=... --difficulty hard
```

`replay_solution.py`:
1. Loads `best.json` (action sequences) and `capture.json` (map data)
2. Connects to the same WebSocket URL (same seed = same game!)
3. Runs the full 300-round game, executing precomputed DP actions
4. Handles desync:
   - **SYNCED**: Bot position matches expected → execute raw DP action
   - **SHORT DESYNC** (≤8 rounds): BFS navigate toward current goal
   - **PERSISTENT DESYNC** (>8 rounds): Switch to greedy mode
5. After DP plan is exhausted (all planned orders completed), switches to **greedy fallback** which reactively picks up items and delivers — this is where new orders are discovered
6. Writes `game_log_*.jsonl`
7. Merges newly discovered orders into `capture.json`
8. Imports game log to PostgreSQL

The server polls the game log every 200ms for round state events and streams them as SSE.

### Step 7: Capture New Orders

After replay completes, the server runs capture again:

```
python capture_from_game_log.py game_log_YYYY.jsonl hard
```

This merges any new orders discovered during the replay's greedy fallback phase into `capture.json`.

### Step 8: Loop or Final Replay

The loop continues while `remaining() > 25` seconds (minIterBudget). Each cycle:
- 20s GPU optimize (warm-only after iter 2)
- 3s replay
- 2s margin for capture + overhead

When the loop exits, if >5 seconds remain, one final replay is executed to register the best score on the leaderboard.

### Step 9: Pipeline Complete

The server emits:

```json
{
  "type": "pipeline_complete",
  "best_score": 185,
  "iterations": 7,
  "total_elapsed": 278.3,
  "difficulty": "hard"
}
```

---

## Single Pipeline — Full Walkthrough

### Step 1: User Starts Single Mode

```js
POST /api/pipeline/run
{
  url: "wss://game.ainm.no/ws?token=...",
  postOptimizeTime: 60
}
```

### Step 2: Live GPU Stream Game

The server spawns:

```
python -u live_gpu_stream.py wss://... --save --json-stream --post-optimize-time 60
```

The `AnytimeGPUStream` solver plays a live game with tiered action generation:

- **Tier 0 (immediate)**: Greedy BFS — always available, <1ms
- **Tier 1 (1-10s)**: MAPF planner in background thread
- **Tier 2+ (10-60s)**: GPU DP passes with increasing state budgets in background threads
- **Refinement**: Multi-bot sequential GPU DP with warm-start

Actions for each round come from the best available plan. Background workers continuously upgrade the plan. When a better plan is ready, it replaces the current one (thread-safe, generation-counted).

### Step 3: Post-Game Optimization

After the 300-round game ends, `live_gpu_stream.py` continues running the GPU solver for `postOptimizeTime` seconds on the captured data. This produces a solution that could be replayed on a subsequent token.

### Step 4: Database Import

On process exit, the server spawns `import_logs.py` to import the game log into PostgreSQL for the replay dashboard.

---

## SvelteKit Dashboard UI

**File**: `grocery-bot-zig/replay/app/src/routes/pipeline/+page.svelte`

### Layout

The page has three main sections:

1. **Input Panel** (collapsible): Token URL, mode selector, timing parameters, start/stop buttons
2. **Visualization Panel**: Grid view with bot positions, score chart, iteration info
3. **Log Panel**: Scrollable log console with filtered output

### State Management (Svelte 5 Runes)

```js
let mode        = $state('single');   // 'single' | 'iterate'
let wsUrl       = $state('');         // WebSocket token URL
let running     = $state(false);
let phase       = $state('idle');     // idle|playing|optimizing|post_optimizing|replaying|done
let timeBudget  = $state(280);        // iterate: total seconds
let gpuOptTime  = $state(20);         // iterate: seconds per GPU optimize
let postOptTime = $state(60);         // single: post-optimize seconds
let maxIters    = $state(99);         // iterate: safety cap
```

### Grid Visualization

Uses the `Grid.svelte` component with adaptive cell sizing:

```js
let adaptiveCell = $derived(
  gameInit ? Math.max(18, Math.min(32, Math.floor(560 / Math.max(gameInit.width, gameInit.height)))) : 24
);
```

Grid shows: walls, shelves, items (by type), drop-off, spawn, bot positions (colored), bot trails, order progress.

### Score Chart

SVG polyline chart tracking score progression across rounds. Points are added per `round` event.

### Iteration Tabs

In iterate mode, the UI shows tabs for each iteration with:
- Phase indicator (zig/optimize/replay)
- Score achieved
- Time elapsed/remaining
- Orders captured

### Bot Colors

```js
const BOT_COLORS = [
  '#f85149', '#58a6ff', '#39d353', '#d29922', '#bc8cff',
  '#3fb950', '#db6d28', '#8b949e', '#f778ba', '#79c0ff',
];
```

### Source Colors (plan origin)

```js
const SOURCE_COLORS = {
  greedy:      '#8b949e',
  mapf:        '#d29922',
  gpu_pass_0:  '#58a6ff',
  gpu_pass_1:  '#56d364',
  gpu_pass_2:  '#39d353',
  gpu_pass_3:  '#f85149',
  gpu_refine:  '#39d353',
  none:        '#484f58',
};
```

### Target Scores (leaderboard #1)

```js
const TARGETS = { easy: 150, medium: 225, hard: 260, expert: 310 };
```

---

## Server Endpoints

### `/api/pipeline/iterate` — Iterate Pipeline

**File**: `grocery-bot-zig/replay/app/src/routes/api/pipeline/iterate/+server.js`

**Method**: POST

**Input**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | required | WebSocket token URL |
| `timeBudget` | 280 | Total pipeline time (seconds) |
| `postOptimizeTime` | 30 | Unused in iterate mode |
| `gpuOptimizeTime` | 20 | GPU optimize time per iteration |

**Orchestration**:
- Manages child processes (`currentProcess`) with kill-on-cancel
- Safety timeout: `(timeBudget + 300) * 1000` ms
- Heartbeat: `: ping` every 10 seconds
- Time tracking: `elapsedSecs()`, `remaining()`
- `minIterBudget = 25` seconds (20s optimize + 3s replay + 2s margin)

**Internal functions**:
| Function | Spawns | Purpose |
|----------|--------|---------|
| `runZigBot(iter)` | `grocery-bot.exe` | Initial game play |
| `runLivePlay(iter, postOpt)` | `live_gpu_stream.py` | Live GPU game (unused in iterate) |
| `runGpuOptimize(diff, iter, time, opts)` | `optimize_and_save.py` | GPU DP optimization |
| `runReplay(diff, iter)` | `replay_solution.py` | Replay optimized solution |
| `captureOrders(diff, iter)` | `capture_from_game_log.py` | Extract orders from game log |

**Optimize options** (passed via `opts`):
- `warmOnly: true` → adds `--warm-only` flag (iter ≥ 2)
- `orderings: 1` → adds `--orderings 1`
- `refineIters: 2` → adds `--refine-iters 2`

### `/api/pipeline/run` — Single Pipeline

**File**: `grocery-bot-zig/replay/app/src/routes/api/pipeline/run/+server.js`

**Method**: POST

**Input**: `{ url, postOptimizeTime: 1800 }`

Spawns `live_gpu_stream.py` with `--save --json-stream --post-optimize-time N`. Forwards all JSON stdout events as SSE. On exit, spawns `import_logs.py` for PostgreSQL import.

### `/api/optimize/replay` — Standalone Replay

**File**: `grocery-bot-zig/replay/app/src/routes/api/optimize/replay/+server.js`

**Method**: POST

**Input**: `{ url, difficulty }`

Spawns `replay_solution.py`. Polls `game_log_*.jsonl` every 150ms for round-by-round state. Safety timeout: 3 minutes.

### `/api/optimize/play` — Zig Capture + GPU Solve

**File**: `grocery-bot-zig/replay/app/src/routes/api/optimize/play/+server.js`

**Method**: POST

**Input**: `{ url, difficulty }`

Spawns `capture_and_solve_stream.py` (Zig capture then GPU solve). Safety timeout: 15 minutes.

### `/api/optimize/learn` — CPU Optimizer

**File**: `grocery-bot-zig/replay/app/src/routes/api/optimize/learn/+server.js`

**Method**: POST

**Input**: `{ difficulty, time, workers }`

Spawns `learn_from_capture.py` (legacy CPU optimizer). Safety timeout: `(time + 30)s`.

### `/api/optimize/solutions` — Solution Management

**File**: `grocery-bot-zig/replay/app/src/routes/api/optimize/solutions/+server.js`

**Methods**:
- `GET`: Returns metadata for all 4 difficulties from `meta.json` files
- `DELETE`: Clears solution files for specific or all difficulties

### `/api/gpu/solve` — Offline GPU Solve

**File**: `grocery-bot-zig/replay/app/src/routes/api/gpu/solve/+server.js`

**Method**: POST

**Input**: `{ difficulty, seed }`

Spawns `gpu_multi_solve_stream.py` for pure offline GPU DP (no WebSocket). Safety timeout: 5 minutes.

### `/api/run-live` — General Live Game

**File**: `grocery-bot-zig/replay/app/src/routes/api/run-live/+server.js`

**Method**: POST

**Input**: `{ url, difficulty, solver }` where solver is `'zig'|'python'|'gpu'`

Dispatches to appropriate solver executable. GPU mode uses stdout streaming; Zig/Python use game log polling every 150ms.

---

## Python Components

### `optimize_and_save.py` — Offline GPU Optimizer

**CLI**:
```bash
python optimize_and_save.py <difficulty> [options]

Options:
  --max-time N        Time budget in seconds (default: 60)
  --max-states N      State budget per bot
  --refine-iters N    Refinement iterations (default: 20)
  --orderings N       Pass1 orderings (default: 3 for hard/expert, 1 for easy/medium)
  --warm-only         Skip cold-start, only refine existing solution
```

**Three operational modes**:

1. **Warm-only + existing solution**: Loads `best.json`, calls `refine_from_solution()` directly. Fastest mode — ideal for iterate cycles 2+.

2. **Warm-only fallback** (no existing solution): Falls back to `solve_sequential()`. Safety net for first iteration.

3. **Normal mode**: Runs `solve_sequential()` first, then if existing solution has higher score, tries `refine_from_solution()` with remaining time budget. Takes the better result.

**Critical flags always set**:
- `no_filler=True`: Only plan for real captured orders (not random filler)
- `no_compile=True`: Required for pipeline context (prevents torch.compile FX dynamo conflicts)

### `replay_solution.py` — Adaptive Replay

**CLI**:
```bash
python replay_solution.py <wss://...token> [--difficulty auto] [--log-dir DIR]
```

**Replay algorithm**:

For each round, for each bot:
1. Check if bot position matches expected position from DP plan
2. If **synced**: execute the raw DP action
3. If **desynced** (≤8 rounds): BFS navigate toward current goal (pickup or dropoff)
4. If **desynced** (>8 rounds): switch to greedy mode permanently for this bot

**Goal extraction**:
- Parses DP action sequence into high-level goals: `(pickup, item_idx)` or `(dropoff,)`
- Tracks goal progress via inventory changes
- Advances goal pointer when a pickup/dropoff is detected

**Greedy fallback**:
- BFS to nearest needed item → pick up → navigate to dropoff → deliver
- Discovers new orders that weren't in the original DP plan

**Post-game**:
- Merges newly seen orders into `capture.json`
- Imports game log to PostgreSQL via `import_logs.py`

### `capture_from_game_log.py` — Order Extraction

**CLI**:
```bash
python capture_from_game_log.py <game_log.jsonl> <difficulty>
python capture_from_game_log.py --latest                      # auto-find newest log
```

**Parsing**:
- JSONL format: alternating lines of game_state and action_response
- Round 0: extracts grid, items, drop_off, num_bots
- All rounds: accumulates orders seen (active + preview)
- Order merging is **positional**: order index determines identity

**Also extracts**:
- Game actions (moves/pickups/dropoffs) converted to internal `(act_type, item_idx)` format
- Saves as warm-start solution if score beats existing

### `live_gpu_stream.py` — Anytime Online Solver

**CLI**:
```bash
python live_gpu_stream.py <wss://...> [options]

Options:
  --save              Save game log
  --json-stream       Emit JSON events to stdout
  --record            Import to PostgreSQL
  --pipeline-mode     Skip heavy offline GPU (pipeline handles it)
  --preload-capture   Preload existing capture orders
  --post-optimize-time N  Seconds to continue GPU after game_over
  --max-states N      Override state budget
  --no-refine         Disable refinement
```

**Tiered architecture**:

| Tier | Source | Latency | Description |
|------|--------|---------|-------------|
| 0 | Greedy BFS | <1ms | Always available, reactive |
| 1 | MAPF planner | 1-10s | Multi-bot coordination |
| 2 | GPU DP pass 0 | 10-30s | Small state budget |
| 3 | GPU DP pass 1 | 30-60s | Medium state budget |
| 4 | GPU DP pass 2 | 60-120s | Large state budget |
| 5 | GPU refine | 60-180s | Multi-bot refinement |

**State budgets per difficulty**:
| Difficulty | Pass 0 | Pass 1 | Pass 2 |
|------------|--------|--------|--------|
| Easy | 50K | 500K | 2M |
| Medium | 20K | 200K | 1M |
| Hard | 10K | 100K | 500K |
| Expert | 5K | 50K | 200K |

**Thread safety**:
- `_update_plan()`: Atomic plan swap with generation counter
- `_solve_gen`: Bumped when new orders arrive, invalidates stale GPU results
- `no_compile=True` mandatory for all threaded GPU calls

---

## GPU DP Solver Deep Dive

### `gpu_beam_search.py` — Core Engine

**Class**: `GPUBeamSearcher`

The heart of the system. A fully vectorized CUDA dynamic programming engine that searches for optimal single-bot action sequences.

#### State Representation

All state is stored as GPU tensors with batch dimension B (number of active states):

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `bot_x` | [B] | int16 | Bot X position |
| `bot_y` | [B] | int16 | Bot Y position |
| `bot_inv` | [B, 3] | int8 | Inventory slots (-1=empty, 0-15=type_id) |
| `active_idx` | [B] | int32 | Current active order index |
| `active_del` | [B, 6] | int8 | Delivery status for active order items |
| `score` | [B] | int32 | Cumulative score |
| `orders_comp` | [B] | int32 | Orders completed count |

#### `dp_search()` — The Core Algorithm

For each round (0 to 299):
1. **Generate candidates**: For each existing state, generate all valid successor states (up to 7 actions: wait, 4 moves, pickup, dropoff)
2. **Apply transitions**: `_step_candidate_only()` processes moves, pickups, deliveries, order completion
3. **Deduplicate**: Hash states via `_hash()` (int64 packing), keep best score per unique state
4. **Prune**: If states exceed `max_states`, evaluate with `_eval()` heuristic and keep top-K

This is **exact DP** when the state space fits in memory — it explores ALL reachable states. For single-bot Easy, this is provably optimal.

#### State Hashing

```python
hash = bot_x | (bot_y << 8) | (inv[0] << 16) | (inv[1] << 24) |
       (inv[2] << 32) | (active_idx << 40) | (delivery_bits << 48)
```

Packed into a single int64 for O(1) dedup via `torch.unique()`.

#### Evaluation Heuristic (for pruning)

`_eval()` computes a heuristic score for beam pruning when states exceed budget:
- Base: current score
- Bonus: estimated value of inventory items (distance to delivery)
- Bonus: proximity to needed items
- Trip cost estimates from precomputed `TripTable`

#### Locked Trajectories

For multi-bot solving, previously-solved bots are "locked" — their positions per round are uploaded as tensors. The candidate bot:
- Cannot move to a cell occupied by a locked bot on that round
- Gets coordination bonuses/penalties for delivery timing

#### torch.compile Integration

```python
torch.compile(mode='default')  # Triton 3.6.0 kernel fusion
```

Hot paths (`_step_candidate_only`, `_eval`, `_hash`) are compiled with Triton, giving ~3.5x speedup. TF32 enabled for matmuls. Out-of-place ops (`torch.stack()` not slice assignment) for CUDA graph compatibility.

### `gpu_sequential_solver.py` — Multi-Bot Orchestration

**Two-phase approach**:

#### Pass 1 — Sequential DP

```
Bot 0: solve alone (full DP search)
Bot 1: solve with Bot 0 locked to its plan
Bot 2: solve with Bots 0+1 locked
...
Bot N: solve with Bots 0..N-1 locked
```

Multiple orderings can be tried (forward, reverse, random) with `num_pass1_orderings`.

#### Pass 2 — Iterative Refinement

```
For each refinement iteration:
  For each bot (alternating forward/reverse/random order):
    Lock ALL other bots to current plans
    Re-solve this bot with GPU DP
    If new score > old score: keep, else: revert
```

Features:
- **Perturbation escape**: If stuck for 3+ iterations, reset a random bot to idle
- **Async verify**: CPU verification overlapped with pre-simulation of next bot
- **Early stopping**: After 2+ consecutive no-improvement full passes
- **Time budget enforcement**: Checks `max_time_s` between bot solves

**Key functions**:

| Function | Purpose |
|----------|---------|
| `solve_sequential()` | Full pipeline: Pass 1 + Pass 2 + verify |
| `refine_from_solution()` | Warm-start: skip Pass 1, run refinement only |
| `pre_simulate_locked()` | CPU sim of locked bots (with optional Zig FFI) |
| `cpu_verify()` | Validate final score matches game engine |
| `solve_multi_restart()` | Try multiple random orderings, keep best |

---

## Game Engine

**File**: `grocery-bot-gpu/game_engine.py`

Pure Python deterministic game simulator. Replicates the server exactly.

### Action Constants

```python
ACT_WAIT      = 0
ACT_MOVE_UP   = 1
ACT_MOVE_DOWN = 2
ACT_MOVE_LEFT = 3
ACT_MOVE_RIGHT= 4
ACT_PICKUP    = 5
ACT_DROPOFF   = 6
```

### Cell Types

```python
CELL_FLOOR   = 0
CELL_WALL    = 1
CELL_SHELF   = 2
CELL_DROPOFF = 3
```

### Game Rules

- **Grid**: Rectangular with walls, shelves, floor, dropoff
- **Bots**: 1/3/5/10 (by difficulty), start at spawn `(w-2, h-2)`
- **Inventory**: 3 slots per bot, items are permanent (no discard)
- **Collision**: 1 bot per tile (except spawn tile)
- **Pickup**: Must be adjacent to shelf containing the item type
- **Dropoff**: Must be on the dropoff tile `(1, h-2)`
- **Scoring**: +1 per item delivered, +5 per completed order
- **Auto-delivery**: When an order completes, ALL bots on the dropoff tile get checked against the new active order
- **Orders**: Sequential — active + preview visible. New orders appear as active completes.
- **Duration**: 300 rounds, 120 seconds wall clock, 2 seconds per-round timeout

### Difficulty Configs

| Difficulty | Bots | Grid | Item Types | Order Size |
|------------|------|------|-----------|------------|
| Easy | 1 | 12×10 | 4 | 3-4 |
| Medium | 3 | 16×12 | 8 | 3-5 |
| Hard | 5 | 22×14 | 12 | 3-5 |
| Expert | 10 | 28×18 | 16 | 4-6 |

### Key Functions

| Function | Purpose |
|----------|---------|
| `build_map(difficulty)` | Build static map from config |
| `build_map_from_capture(capture)` | Build map from captured game data |
| `generate_all_orders(seed, map, diff, count)` | Pre-generate orders using same RNG as server |
| `init_game(seed, difficulty)` | Initialize game from seed |
| `init_game_from_capture(capture, n)` | Initialize from capture data |
| `step(state, actions, orders)` | Apply one round (mutates state) |

---

## Precomputation

**File**: `grocery-bot-gpu/precompute.py`

### `PrecomputedTables`

GPU-accelerated all-pairs shortest path computation using matrix-multiply BFS.

**Tables**:

| Table | Shape | Type | Description |
|-------|-------|------|-------------|
| `dist_matrix` | [N, N] | int16 | All-pairs shortest path distances |
| `next_step_matrix` | [N, N] | int8 | First action (1-4) from source to target |
| `dist_to_type` | [T, H, W] | int16 | Distance to nearest item of each type |
| `step_to_type` | [T, H, W] | int8 | First action toward nearest of each type |
| `dist_to_dropoff` | [H, W] | int16 | Distance to dropoff |
| `step_to_dropoff` | [H, W] | int8 | First action toward dropoff |

Where N = number of walkable cells, T = number of item types, H/W = grid dimensions.

**GPU BFS**: Uses `torch.mm()` with TF32-enabled matmuls. Computes all-pairs shortest paths for the entire map in <5ms by treating adjacency as a sparse matrix and iterating matrix multiplications.

### Caching

Three-level cache:
1. **Module-level memory cache**: `_tables_cache` dict keyed by grid hash
2. **Disk cache**: `.npz` files in `cache/` directory
3. **Recompute**: Full GPU BFS if neither cache hits

### `TripTable`

Precomputes exact multi-item trip costs for 1/2/3-item pickup combinations. Used by `GPUBeamSearcher._eval()` for accurate heuristic evaluation during beam pruning.

---

## Solution Storage

**File**: `grocery-bot-gpu/solution_store.py`

### Directory Structure

```
solutions/
  easy/
    best.json      — action sequences
    capture.json   — grid + items + orders
    meta.json      — score metadata
  medium/
    ...
  hard/
    ...
  expert/
    ...
```

### File Formats

**`best.json`**: Nested array of per-round, per-bot actions:
```json
[
  [[action_type, item_idx], [action_type, item_idx], ...],  // round 0, all bots
  [[action_type, item_idx], [action_type, item_idx], ...],  // round 1, all bots
  ...  // 300 rounds
]
```

**`capture.json`**: Full game capture data:
```json
{
  "grid": { "width": 22, "height": 14, "walls": [[x,y], ...] },
  "items": [{ "id": 0, "type": "apple", "position": [5, 3] }, ...],
  "orders": [{ "items": ["apple", "bread", "milk"] }, ...],
  "drop_off": [1, 12],
  "num_bots": 5
}
```

**`meta.json`**: Solution metadata:
```json
{
  "score": 185,
  "difficulty": "hard",
  "seed": 0,
  "num_bots": 5,
  "num_rounds": 300,
  "date": "2026-03-04T12:00:00",
  "capture_hash": "abc123",
  "optimizations": 5
}
```

### Key Invariant

**`save_solution()` NEVER overwrites a better score.** It checks `existing_meta.score >= score` and returns `False` without writing if the existing solution is better. This ensures the iterate pipeline only ratchets scores upward.

### Key Functions

| Function | Description |
|----------|-------------|
| `save_capture(diff, data)` | Save capture data |
| `merge_capture(diff, new)` | Merge new capture (positional order merge — keeps longer list) |
| `load_capture(diff)` | Load capture data |
| `save_solution(diff, score, actions)` | Save if better (or forced) |
| `load_solution(diff)` | Load best action sequence |
| `load_meta(diff)` | Load metadata |
| `increment_optimizations(diff)` | Bump optimization counter |
| `clear_solutions(diff)` | Delete solution files |
| `get_all_solutions()` | Get metadata for all difficulties |

---

## Replay System

**File**: `grocery-bot-gpu/replay_solution.py`

### Desync Handling

The replay system must handle timing differences between the DP plan (computed offline) and the live game. Causes of desync:
- Bot collision (another bot occupies expected cell)
- Round offset (server processes actions differently than expected)
- Order timing differences

**Strategy per bot per round**:

```
IF bot_position == expected_position:
    → SYNCED: execute raw DP action
ELIF desync_count ≤ 8:
    → SHORT DESYNC: BFS toward current goal
ELSE:
    → PERSISTENT: switch to greedy mode permanently
```

### Goal Extraction

The DP action sequence is parsed into high-level goals:

```python
goals = extract_goals(actions, capture)
# Result: [(pickup, item_3), (pickup, item_7), (dropoff,), (pickup, item_1), ...]
```

Goal advancement is tracked via inventory state changes rather than position matching, making it robust to path deviations.

### Greedy Fallback

When the DP plan is exhausted or desync is persistent:

```python
def greedy_action(bot, game_state, walkable):
    if bot has items matching active order and at dropoff:
        return DROPOFF
    if bot has items matching active order:
        return BFS toward dropoff
    if needed item is adjacent:
        return PICKUP
    return BFS toward nearest needed item
```

This phase is where **new orders are discovered** — the greedy play continues delivering items and progressing through orders beyond what the DP planned for.

---

## Order Capture & Discovery

**File**: `grocery-bot-gpu/capture_from_game_log.py`

### The Snowball Effect

Each game reveals orders up to about 2-3 beyond what was completed. The iterate pipeline exploits this:

```
Game 1 (Zig bot):     Score 80,  discovers orders 0-14 (15 orders)
GPU optimize:          Plans for orders 0-14, achieves score 140
Game 2 (replay):       Score 140, discovers orders 0-16 (17 orders)
GPU optimize:          Plans for orders 0-16, achieves score 160
Game 3 (replay):       Score 160, discovers orders 0-18 (19 orders)
...
```

### Order Merging

Orders are merged **positionally** — order index determines identity:

```python
def merge_capture(difficulty, new_capture):
    existing = load_capture(difficulty)
    if not existing:
        save_capture(difficulty, new_capture)
        return
    # Keep whichever has more orders
    if len(new_capture['orders']) > len(existing['orders']):
        existing['orders'] = new_capture['orders']
    save_capture(difficulty, existing)
```

This works because games are deterministic — order N is always the same order on the same seed.

---

## Live GPU Stream Solver

**File**: `grocery-bot-gpu/live_gpu_stream.py`

### Worker Threads

```
Main Thread (WebSocket game loop)
  │
  ├── _mapf_worker()         — MAPF multi-bot planner
  ├── _gpu_worker()          — Single-bot GPU DP (multiple passes)
  ├── _gpu_refine_worker()   — Multi-bot sequential refinement
  └── _per_round_gpu_worker()— Per-round immediate GPU action
```

### Plan Management

Plans are thread-safe with generation counting:

```python
def _update_plan(self, source, score, actions, gen):
    with self._plan_lock:
        if gen < self._solve_gen:
            return  # stale result, new orders arrived
        if score > self._plan_score:
            self._plan = actions
            self._plan_source = source
            self._plan_score = score
```

### Pipeline Mode

When `--pipeline-mode` is set, the live solver skips heavy offline GPU passes (the iterate pipeline handles optimization separately). It only runs greedy + MAPF + light GPU for real-time play.

---

## Local Iterate Script

**File**: `grocery-bot-gpu/iterate_local.py`

Offline testing tool that simulates the iterate pipeline without WebSocket connections.

**CLI**:
```bash
python iterate_local.py hard --seed 42 --max-time 600 --max-states 100000
```

### Two Modes

**Full foresight** (default):
1. Solve with N orders
2. Simulate → discover actual orders completed = M
3. Re-solve with M+2 orders (tighter focus)
4. Repeat until convergence

**Discovery mode** (`--discover`):
1. Solve with known orders
2. Simulate → discover new orders
3. Add new orders to capture
4. Re-solve with expanded order set
5. Repeat

### Key Insight

Fewer orders = less state space fragmentation = better beam search quality. Example: 25 orders + 50K states > 40 orders + 100K states.

---

## SSE Event Reference

All endpoints communicate via Server-Sent Events. Events are JSON objects with a `type` field.

### Pipeline Lifecycle Events

| Event | Fields | Source |
|-------|--------|--------|
| `pipeline_start` | time_budget, gpu_optimize_time | iterate endpoint |
| `iter_start` | iter, phase, elapsed, remaining | iterate endpoint |
| `iter_done` | iter, phase, score, game_score, captured_orders | iterate endpoint |
| `iter_summary` | iter, score, best_score, iterations_done | iterate endpoint |
| `iter_skip` | iter, reason, remaining | iterate endpoint |
| `pipeline_complete` | best_score, iterations, total_elapsed, difficulty | iterate endpoint |
| `pipeline_done` | final_score, difficulty | live_gpu_stream |

### Game Events

| Event | Fields | Source |
|-------|--------|--------|
| `init` | width, height, walls, shelves, items, drop_off, num_bots, difficulty | various |
| `round` | round, bots, orders, score | various |
| `game_over` | score | various |

### GPU Optimization Events

| Event | Fields | Source |
|-------|--------|--------|
| `optimize_start` | difficulty, max_time, prev_score, orders | optimize_and_save |
| `optimize_phase_start` | difficulty, max_time, warm_only | iterate endpoint |
| `gpu_bot_done` | bot, num_bots, score, elapsed | optimize_and_save |
| `gpu_phase` | phase, iteration, score, elapsed | optimize_and_save |
| `optimize_done` | score, prev_score, saved, elapsed, orders | optimize_and_save |
| `optimize_error` | message | optimize_and_save |

### Replay Events

| Event | Fields | Source |
|-------|--------|--------|
| `replay_phase_start` | difficulty | iterate endpoint |
| `replay_phase_done` | score, exit_code | iterate endpoint |

### Live Solver Events

| Event | Fields | Source |
|-------|--------|--------|
| `plan_upgrade` | from_source, to_source, score | live_gpu_stream |
| `gpu_pass_done` | pass_id, score, states, elapsed | live_gpu_stream |
| `post_optimize_start` | | live_gpu_stream |
| `post_optimize_progress` | score, elapsed | live_gpu_stream |
| `post_optimize_done` | score | live_gpu_stream |
| `seed_cracked` | seed | live_gpu_stream |

### General Events

| Event | Fields | Source |
|-------|--------|--------|
| `log` | text, _iter | all endpoints |
| `status` | message | various |
| `error` | message | various |

---

## Configuration & Tuning

### Time Budget Allocation (Iterate Mode)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeBudget` | 280s | Total pipeline time (token lasts ~288s) |
| `gpuOptimizeTime` | 20s | GPU optimize per iteration |
| `minIterBudget` | 25s | Minimum time for one optimize+replay cycle |
| Break condition | 6s | Remaining time to stop loop |
| Final replay | 5s | Minimum time to attempt final replay |

### GPU Solver Parameters

| Parameter | Easy | Medium | Hard | Expert |
|-----------|------|--------|------|--------|
| Default max_states | 500K | 500K | 100K | 100K |
| Default orderings | 1 | 1 | 3 | 3 |
| Default refine_iters | 0 | 3 | 10 | 10 |
| Pipeline orderings | 1 | 1 | 1 | 1 |
| Pipeline refine_iters | 2 | 2 | 2 | 2 |

### Critical Flags

| Flag | Value | Why |
|------|-------|-----|
| `no_filler` | True | DP wastes moves on fake filler orders without this |
| `no_compile` | True | Prevents torch.compile FX dynamo crash in multi-threaded contexts |
| `allow_tf32` | True | Enables TF32 for matmuls (precompute + BFS) |
| `mode='default'` | torch.compile | Triton kernel fusion (3.5x speedup over eager) |

### Warm-Only Mode Decision

```
Iteration 1: Cold-start (no existing solution to refine)
  → solve_sequential() with orderings=1, refine_iters=2

Iteration 2+: Warm-only (refine previous best with new orders)
  → refine_from_solution() with refine_iters=2
  → Falls back to cold-start if no existing solution
```

---

## File Reference

### SvelteKit Dashboard

| File | Purpose |
|------|---------|
| `grocery-bot-zig/replay/app/src/routes/pipeline/+page.svelte` | Pipeline UI page |
| `grocery-bot-zig/replay/app/src/routes/api/pipeline/iterate/+server.js` | Iterate pipeline endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/pipeline/run/+server.js` | Single run pipeline endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/optimize/play/+server.js` | Zig capture + GPU solve endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/optimize/replay/+server.js` | Standalone replay endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/optimize/learn/+server.js` | CPU optimizer endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/optimize/solutions/+server.js` | Solution management REST API |
| `grocery-bot-zig/replay/app/src/routes/api/gpu/solve/+server.js` | Offline GPU solve endpoint |
| `grocery-bot-zig/replay/app/src/routes/api/run-live/+server.js` | General live game endpoint |

### Python GPU Solver

| File | Purpose |
|------|---------|
| `grocery-bot-gpu/gpu_beam_search.py` | Core GPU DP engine (CUDA, Triton-compiled) |
| `grocery-bot-gpu/gpu_sequential_solver.py` | Multi-bot sequential DP with refinement |
| `grocery-bot-gpu/precompute.py` | GPU BFS all-pairs shortest paths + TripTable |
| `grocery-bot-gpu/game_engine.py` | Pure Python game simulator |
| `grocery-bot-gpu/optimize_and_save.py` | Offline GPU optimize (pipeline entry point) |
| `grocery-bot-gpu/replay_solution.py` | Adaptive replay with desync correction |
| `grocery-bot-gpu/capture_from_game_log.py` | Order extraction from game logs |
| `grocery-bot-gpu/solution_store.py` | Score-safe solution storage |
| `grocery-bot-gpu/live_gpu_stream.py` | Anytime online GPU solver |
| `grocery-bot-gpu/iterate_local.py` | Local iterate pipeline for offline testing |
| `grocery-bot-gpu/profile_gpu.py` | torch.profiler bottleneck analysis |

### Zig Bot

| File | Purpose |
|------|---------|
| `grocery-bot-zig/src/main.zig` | WebSocket client, game loop |
| `grocery-bot-zig/src/strategy.zig` | Core decision engine |
| `grocery-bot-zig/src/trip.zig` | Mini-TSP trip planner |
| `grocery-bot-zig/src/pathfinding.zig` | BFS distance maps |
| `grocery-bot-zig/zig-out/bin/grocery-bot.exe` | Compiled bot executable |
