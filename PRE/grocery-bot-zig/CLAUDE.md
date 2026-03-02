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

## Game Rules
- 300 rounds, 120s wall clock
- INV_CAP = 3 items per bot
- Items stay on shelves permanently (never deplete)
- Sequential orders: active + preview visible
- Score: +1 per item delivered, +5 per order completed
- Auto-delivery: when order completes at dropoff, ALL bots on dropoff tile get inventory checked against NEW active order
- Bot collision: 1 bot per tile (except spawn)
- Drop-off at (1, h-2), spawn at (w-2, h-2)

## Difficulty Configs
| | Bots | Grid | Types | Order Size |
|---|---|---|---|---|
| Easy | 1 | 12x10 | 4 | 3-4 |
| Medium | 3 | 16x12 | 8 | 3-5 |
| Hard | 5 | 22x14 | 12 | 3-5 |
| Expert | 10 | 28x18 | 16 | 4-6 |

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

## Current Best Scores (40-seed sweep, seeds 7001-7040, production-style aisle walls)
| Difficulty | Mean | Max | Target |
|---|---|---|---|
| Easy | 115.1 | 131 | 175 |
| Medium | 108.4 | 131 | 175 |
| Hard | 86.5 | 119 | 175 |
| Expert | 65.8 | 96 | 175 |

Note: Sim server now generates production-style aisle walls. These scores are with accurate maps (tighter corridors than previous open-aisle layout).

## Live Game Performance
- 15ms response delay in main.zig prevents 1-round action offset desync
- Aggressive desync detection: 1 mismatch threshold, 2 consecutive rounds
- Live Easy seed 7001: 116 (vs 121 sweep) — deterministic, no desync
- Previous live Medium seed 7002: 89 (vs 148 sweep) — massive desync before fix

## File Dependencies
```
main.zig → ws.zig, types.zig, strategy.zig, parser.zig, pathfinding.zig
strategy.zig → types.zig, pathfinding.zig, trip.zig
trip.zig → types.zig, pathfinding.zig
parser.zig → types.zig
pathfinding.zig → types.zig
```
