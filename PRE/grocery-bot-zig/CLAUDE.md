# Grocery Bot (Zig) - AI Assistant Guide

## Project Overview
WebSocket-based grocery store bot that plays a 300-round item-picking game across 4 difficulty levels. Bots navigate a grid, pick items from shelves, and deliver to a dropoff point to score.

## Build & Run

```bash
# Build (ReleaseFast required for performance)
cd grocery-bot-zig
C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe build -Doptimize=ReleaseFast

# Single game
python run_easy.py          # or run_medium.py, run_hard.py, run_expert.py

# Statistical sweep (40 seeds, auto-records to PostgreSQL if DB is available)
python sweep_easy.py        # port 9850
python sweep_medium.py      # port 9860
python sweep_hard.py        # port 9870
python sweep_expert.py      # port 9880

# Custom sweep
python sweep.py expert --seeds 40 --port 9880
python sweep.py easy --seeds 40 --no-record    # skip DB recording (AVOID: always record to PostgreSQL)
```

## Replay Dashboard

Visual replay system with Docker + PostgreSQL + SvelteKit.

```bash
# Start PostgreSQL
cd replay && docker compose up -d db

# Record games (auto-inserts to DB)
python recorder.py easy --seeds 40
python recorder.py expert --seed 1001   # single seed

# Sweeps also auto-record when DB is available
python sweep.py hard --seeds 40

# Start SvelteKit dashboard
cd replay/app && npm run dev
# Open http://localhost:5173

# Full Docker deployment (DB + app)
cd replay && docker compose up -d
# Open http://localhost:5173
```

### Replay Files
| File | Purpose |
|------|---------|
| `replay/docker-compose.yml` | PostgreSQL + SvelteKit containers |
| `replay/init.sql` | Database schema (runs + rounds tables) |
| `replay/recorder.py` | Records games to PostgreSQL |
| `replay/app/` | SvelteKit dashboard with SVG grid visualization |
| `replay/SVG_REFERENCE.md` | All SVG elements with descriptions for reuse |

## Architecture

### Source Files (src/)
| File | Lines | Purpose |
|------|-------|---------|
| `main.zig` | 243 | WebSocket client, game loop, desync detection |
| `strategy.zig` | ~1360 | **Core decision engine** - orchestrator, trip execution, delivery |
| `trip.zig` | 359 | Mini-TSP trip planner (1/2/3-item route evaluation) |
| `pathfinding.zig` | 88 | BFS distance maps, collision-aware first-step avoidance |
| `parser.zig` | 120 | JSON game state parser |
| `ws.zig` | 376 | WebSocket client with TLS support |
| `types.zig` | 108 | Core types: Pos, Bot, GameState, NeedList, etc. |

### Decision Flow (strategy.zig)
Per-bot priority cascade each round:
1. **Drop off** - at dropoff with active items → deliver
2. **Evacuate** - at dropoff without active items → flee (multi-bot with inventory) or fall through
3. **Escape** - oscillation detected → escape movement
4. **Pick up** - adjacent needed item → pick (active pass first, then preview pass)
5. **Deliver** - has active items, should deliver → navigate to dropoff
6. **Follow trip** - existing trip plan → navigate to next pickup
7. **Plan trip** - no trip → evaluate candidates via trip.zig
8. **Deliver fallback** - has items → go to dropoff
9. **Pre-position** - idle → move near likely-needed items
10. **Dead inventory** - non-matching items → camp near dropoff or flee
11. **Wait** - nothing to do

### Key Systems
- **Orchestrator** (strategy.zig ~614-780): Assigns active items to bots greedily by distance. Phase 2 assigns preview to idle bots. Concentration bonus for 3-bot medium. max_pickers limits active bots (3 for 5-7, 4 for 8+).
- **Trip planner** (trip.zig): Evaluates 1/2/3-item trips, scores by `value*10000/cost`. Massive bonus for order-completing trips with preview items.
- **Anti-oscillation**: Position history (24 rounds), stall detection (threshold 6), escape mode (4 rounds), near-dropoff patience (dist<=2, max 3 waits).
- **Desync detection** (main.zig): Detects 1-round action offset via position mismatch. Shifts to effective positions when triggered.

## Game Mechanics (Official Rules 2026-03-10)

### Scoring
- **+1 per item** delivered to active order
- **+5 bonus** per completed order
- 4-item order completed = 4 + 5 = **9 points**
- 3-item order completed = 3 + 5 = **8 points**
- Leaderboard = **sum of best scores across all maps** (best score per map saved automatically)

### 5 Difficulty Levels
| Level | Grid | Bots | Aisles | Item Types | Drop Zones | Rounds | Wall-clock |
|-------|------|------|--------|------------|------------|--------|------------|
| Easy | 12x10 | 1 | 2 | 4 | 1 | 300 | 120s |
| Medium | 16x12 | 3 | 3 | 8 | 1 | 300 | 120s |
| Hard | 22x14 | 5 | 4 | 12 | 1 | 300 | 120s |
| Expert | 28x18 | 10 | 5 | 16 | 1 | 300 | 120s |
| Nightmare | 30x18 | 20 | 6 | 21 | 3 | 500 | 300s |

### Order Sizes
| Level | Items per Order |
|-------|----------------|
| Easy | 3-4 |
| Medium | 3-5 |
| Hard | 3-5 |
| Expert | 4-6 |
| Nightmare | 4-6 |

### Sequential Orders (Infinite)
- **Active order**: current order, full details visible, can deliver items
- **Preview order**: next order, full details visible, can pre-pick items but CANNOT deliver
- **Infinite**: when you complete an order, a new one appears. Orders never run out. Rounds are the only limit.
- Bad picks waste inventory slots — choose wisely

### Items Are Infinite
- Shelves never deplete. The items list stays constant throughout all rounds.
- Same `item_id` can be picked up repeatedly from the same shelf tile.
- No scarcity — optimize purely for shortest round-trip loops.

### Drop-Off Chain Reaction (DZ-ONLY — TRIPLE-CONFIRMED LIVE 2026-03-10)
- When the active order completes via drop_off, the preview order becomes active **immediately**.
- **ONLY bots physically AT a dropoff zone** get their inventories re-checked against the new active order. Bots elsewhere **keep their items**.
- Each DZ bot loses inventory items matching the current AND cascading orders in sequence.
- This can **cascade**: if auto-delivered items at DZ complete the new order, the next preview activates, and DZ bots are re-checked again.
- **Nightmare has 3 DZ zones** → place up to 3 bots (one per DZ cell) with known future order items before triggering.
- **MCP docs are WRONG** about "any items in bot inventories" — tested all-bots sim (1047pts) vs live (279pts, zero cascades). NEVER try all-bots again.
- **Throughput > cascade**: Leader at 1032 achieves ~100 orders in 500 rounds (5 rnd/order) via fast delivery, not deep cascades.

### Pickup Rules
- Bot must be **adjacent** (Manhattan distance 1) to the shelf containing the item
- Works from **all 4 cardinal directions**
- Bot inventory must not be full (max 3 items)
- Choose the pickup tile that minimizes the **TOTAL trip**, not just the approach distance.

### Dropoff Rules
- Bot must be standing **on** the drop-off cell
- Only items matching the **active order** are delivered — non-matching items **stay in inventory**
- Multiple drop-off zones on Nightmare (any zone works)

### Action Resolution
- Bot 0 moves first, then bot 1, etc. **Lower IDs have collision priority.**
- Invalid actions silently become wait — no penalty, but wastes a round.
- Bots block each other on all tiles **except the spawn tile**.

### Deterministic Per Day
- Same day = same map layout, item placement, **and full order sequence** (orders are fixed per day's seed).
- Orders are NOT random per game — they are deterministic for the day.
- Can pre-capture order sequences from prior runs and reuse.
- Item placement and orders change daily at midnight UTC. Grid structure stays the same.

### Limits
- 300 rounds max (500 Nightmare), 120s wall-clock (300s Nightmare), 2s response timeout per round
- Max 3 items per bot inventory (INV_CAP = 3)
- 60s cooldown between games, max 40/hour, 300/day per team
- Disconnect = game over (no reconnect)
- Drop-off at (1, h-2), spawn at (w-2, h-2)

## Token & Session Rules (CRITICAL for iterate pipeline)
- **Tokens last ~288 seconds** (~5 minutes) from creation
- **Same token URL can be reused for MULTIPLE games** within the 5-minute window
- **Same token = same seed = same map = same orders** — games are fully deterministic per token
- This enables the iterate pipeline: play → capture orders → GPU optimize → replay with same URL → capture more orders → repeat
- No need to fetch new tokens between iterations — reuse the same URL
- Competition scoring: only MAX score per map counts across all attempts

## Difficulty Configs
| | Bots | Grid | Types | Order Size | Rounds | Drop Zones |
|---|---|---|---|---|---|---|
| Easy | 1 | 12x10 | 4 | 3-4 | 300 | 1 |
| Medium | 3 | 16x12 | 8 | 3-5 | 300 | 1 |
| Hard | 5 | 22x14 | 12 | 3-5 | 300 | 1 |
| Expert | 10 | 28x18 | 16 | 4-6 | 300 | 1 |
| Nightmare | 20 | 30x18 | 21 | 4-6 | 500 | 3 |

## Critical Tuning Parameters
These parameters are tightly coupled. Changes often cause cascading regressions.

| Parameter | Value | Location | Notes |
|---|---|---|---|
| `max_pickers` | 3 (5-7 bots), 4 (8+) | strategy.zig ~653 | Concentrates items on fewer bots |
| `max_preview_carriers` | 1/2/3 (by bot count) | strategy.zig ~343 | KEY: prevents dead inventory |
| `max_orch_preview` | 2 (default), 4 (8+ bots) | strategy.zig ~724 | Preview assignments in orchestrator |
| `MAX_DROPOFF_ACTIVE` | bot_count/2 | strategy.zig ~556 | Limits dropoff congestion |
| `stall_count threshold` | 6 | strategy.zig ~483 | Triggers escape. Lower = too many false escapes |
| `escape_rounds` | 4 | strategy.zig ~490 | Too short = immediate re-stall |
| `anti-osc wait dist` | <=2 | strategy.zig ~1078,1201 | Near-dropoff patience |
| `anti-osc patience` | 3 | strategy.zig ~1078,1201 | Max wait rounds |
| `far_with_few` | dist>8, inv<2, 5+ bots | strategy.zig ~1002 | Prevents low-value deliveries |
| `concentration bonus` | 5, 3-bot only | strategy.zig ~675 | Batches items to fewer bots |
| `base_detour` | 5 (1-bot), 1 (multi) | strategy.zig ~1050 | Opportunistic pickup during delivery |
| `completing_detour` | 8 (1-bot), 4 (multi) | strategy.zig ~1053 | Extra detour to complete active order |
| `endgame` | dist + inv_len + 1 | strategy.zig ~1015 | Dynamic buffer scales with inventory |
| `type blocking` | bot_count < 5 | strategy.zig ~707 | Blocks excess type instances (Medium only) |
| `stuck-order escape` | 25 rounds, every 12 | strategy.zig ~539 | Forces physical escape on stuck orders |

## Optimization Workflow
1. Make a single targeted change
2. Build with ReleaseFast
3. Sweep 40 seeds on affected difficulties (**ALWAYS record to PostgreSQL** — never use --no-record)
4. Compare MAX score (peak matters most, variance is acceptable)
5. Keep improvements, revert regressions
6. Never combine multiple untested changes

## Known Failed Optimizations (DO NOT RETRY)
- `max_preview_carriers = bot_count/2` → massive dead inventory regression
- Concentration bonus for >=3 bots → -8.9% hard, -3.5% expert
- Round-trip sort for 5+ bots → -2.5% hard
- Round-trip orchestrator → -14.5% hard
- Stall threshold 5 → -5.3% hard
- Anti-osc wait dist<=1 → -7.8% hard
- Full-inv anti-osc bypass → -6% hard
- Rarity-sorted orchestrator → -4.5% medium
- Dead-inv bots flee at dist<=4 → too aggressive
- `far_with_few=false` → all bots rush delivery, congestion
- Quick deliver with 1 item → score 3
- Preview detour during single-bot delivery → catastrophic
- Pipeline preview (pick_remaining<=2) → dead inventory
- Completing detour range 4→6 → Hard regression
- Claim stealing for completing detours → chaos
- Assigned pre-positioning for 5-bot → Hard -12 max
- Bot processing order by distance → Hard -4 max, Expert -7 max
- Multi-step collision reservation (bfsReserved) → Hard -2, Expert -5 max
- Dropoff scheduling (round-robin) → Hard -8 max
- Scarcity-based orchestrator assignment → Hard -4 max
- Order completion rush (max_pickers bypass for last 1-2 items) → Expert -10 max (congestion), Hard -3 max
- Delivering bot detour for last item (dropoff_priority bypass) → Expert -13 max
- Completing detour 4→6 for multi-bot → no improvement, adds variance
- far_with_few disable when pick_remaining<=2 → logic bug (disables entirely), Expert regression
- Full active+preview blocking for all bot counts → Hard -7, Expert -15 (restricts backup pickers)
- Trip diversity +1 for all modes → single-bot picks duplicate types (2 yogurts when order needs 1)
- Trip diversity removed for all modes → Hard -3, Expert -6 (multi-bot needs alternatives)
- `far_with_few` expansion for non-priority bots (dist>5) → Hard -7, Expert -9 (restricts delivery)
- MAPF multi-step commitment for Expert (10 bots) → Expert -4% mean (over-constraining, too many reserved cells)

## Current Best Scores (40-seed sweep, seeds 7001-7040, production-style aisle walls)
| Difficulty | Mean | Max | Target |
|---|---|---|---|
| Easy | 115.1 | 131 | 175 |
| Medium | 108.4 | 131 | 175 |
| Hard | **94.7** | **125** | 175 |
| Expert | 65.8 | 96 | 175 |

Note: Hard improved via multi-step MAPF commitment (spacetime.zig). Expert unchanged (commitment disabled for 10 bots).

## Live Game Performance
- 15ms response delay in main.zig prevents 1-round action offset desync
- Aggressive desync detection: 1 mismatch threshold, 2 consecutive rounds
- Live Easy seed 7001: 116 (vs 121 sweep) — deterministic, no desync
- Previous live Medium seed 7002: 89 (vs 148 sweep) — massive desync before fix

## Production Pipeline: Iterative Live → GPU DP → Replay (within 5-min token)

The competition scoring uses **max score per map**. The optimal strategy within a single 5-minute token:
1. **Live GPU play** — `live_gpu_stream.py` plays with GPU actions at every round, discovers orders
2. **GPU DP optimizes offline** — `optimize_and_save.py` solves with all captured orders (sequential DP)
3. **Replay with SAME token** — `replay_solution.py` replays optimized solution → discovers MORE orders
4. **Repeat steps 2-3** until token expires (~288 seconds)

### Key Insight: Token Reuse
- **Same token URL = same seed = same map = same orders** — fully deterministic
- **Token can be reused multiple times** within its ~288s lifetime
- Each replay-capture cycle discovers ~2-3 new orders → higher GPU score → more orders → repeat
- No need to fetch new tokens between iterations

### Automated Pipeline (SvelteKit `/pipeline` page, iterate mode)
```
User pastes token URL → Start Iterative
  ├── Iter 0: live_gpu_stream.py (live play + post-optimize)
  │     └── capture_from_game_log.py (extract orders)
  ├── Iter 1: optimize_and_save.py (offline GPU DP)
  │     ├── replay_solution.py (replay with same URL)
  │     └── capture_from_game_log.py (more orders)
  ├── Iter 2: optimize → replay → capture (repeat)
  └── ... until time budget exhausted
```

### Manual Workflow
```bash
# 1. Live GPU play (captures orders into game log)
cd grocery-bot-gpu
python live_gpu_stream.py "wss://game.ainm.no/ws?token=..." --save --json-stream --post-optimize-time 30

# 2. Extract orders from game log
python capture_from_game_log.py game_log_XXXX.jsonl hard

# 3. GPU DP optimize offline
python optimize_and_save.py hard --max-time 45

# 4. Replay with SAME token URL
python replay_solution.py "wss://game.ainm.no/ws?token=..." --difficulty hard

# 5. Capture more orders from replay → go to step 3
python capture_from_game_log.py game_log_YYYY.jsonl hard
```

### Critical Notes
- `--no-filler` is MANDATORY for GPU DP: without it, DP wastes moves on fake orders
- `no_compile=True` required for all live/threaded GPU calls (prevents torch.compile FX dynamo crash in multi-threaded contexts)
- Offline single-threaded solves use `torch.compile(mode='default')` with Triton for 3.5x speedup
- Each capture-optimize-replay cycle discovers ~2-3 new orders
- `solution_store.py` NEVER overwrites better scores
- `replay_solution.py` has greedy fallback: after DP plan exhausts, switches to reactive item pickup

### GPU DP Solver (`grocery-bot-gpu/`)
| File | Purpose |
|------|---------|
| `gpu_beam_search.py` | Core GPU DP solver (CUDA, RTX 5090, Triton-compiled) |
| `gpu_sequential_solver.py` | Multi-bot sequential DP with refinement |
| `precompute.py` | All-pairs shortest paths + TripTable (GPU BFS, TF32) |
| `live_gpu_stream.py` | Live game: GPU actions at every round |
| `optimize_and_save.py` | Offline GPU optimize: load capture → solve → save (JSON events) |
| `capture_from_game_log.py` | Extract orders from game logs |
| `replay_solution.py` | Replay GPU solutions with greedy fallback |
| `solution_store.py` | Score-safe solution storage |
| `production_run.py` | CLI iterative pipeline (token fetch + optimize + replay) |
| `profile_gpu.py` | torch.profiler bottleneck analysis |

### GPU Compile Notes
- **Triton 3.6.0 works on Windows** with PyTorch 2.10+cu128
- `torch.compile(mode='default')`: Triton kernel fusion (3.5x faster than aot_eager)
- `reduce-overhead` mode (CUDA graphs) blocked by tensor lifetime issue — use `default`
- TF32 enabled: `allow_tf32=True` in both `precompute.py` and `gpu_beam_search.py`
- Out-of-place ops in `_vectorized_deliver`/pickup loop (`torch.stack()` not slice assign)

## File Dependencies
```
main.zig → ws.zig, types.zig, strategy.zig, parser.zig, pathfinding.zig, precomputed.zig
strategy.zig → types.zig, pathfinding.zig, trip.zig, precomputed.zig
trip.zig → types.zig, pathfinding.zig, precomputed.zig
parser.zig → types.zig
pathfinding.zig → types.zig
precomputed.zig → types.zig
```

## Precomputed Order Data (precomputed.zig)
Loads capture.json at startup via `--precomputed <path>`. Since orders are random per game,
this only helps when replaying known games (same token). For live play, it safely disables
(verification fails on round 0 when orders don't match).

**NOTE**: Precomputed features are kept for potential future use with deterministic replays
but do NOT improve live game performance.
