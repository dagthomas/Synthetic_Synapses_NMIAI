# Grocery Bot Architecture & Strategy Guide

## Overview

This is a competitive AI bot built in **Zig** (0.15.2, compiled with `ReleaseFast`) that plays a WebSocket-based grocery store game. The bot controls 1-10 agents that navigate a grid map, pick items from shelves, and deliver them to a dropoff point to score. The game runs for 300 rounds with a 120-second wall clock limit.

The system uses a **centralized orchestrator** pattern — one brain makes decisions for all bots each round, with shared state tracking to prevent conflicts and optimize team coordination.

---

## Tech Stack

- **Language**: Zig 0.15.2 (systems language, zero-allocation game loop, compiled to native x86_64)
- **Build**: `zig build -Doptimize=ReleaseFast` for maximum runtime performance
- **Communication**: WebSocket client (custom Zig implementation with TLS support)
- **Simulation Server**: Python (sim_server.py) — hosts the game, processes actions, tracks state
- **Testing**: Python sweep scripts for statistical evaluation across 40 seed values per difficulty

---

## Game Rules

| Parameter | Value |
|---|---|
| Rounds | 300 per game |
| Inventory Cap | 3 items per bot |
| Scoring | +1 per item delivered, +5 per order completed |
| Orders | Sequential: 1 active + 1 preview visible at all times |
| Items | Stay on shelves permanently (never deplete, multiple bots can pick same shelf) |
| Movement | 4-directional (up/down/left/right), 1 tile per round |
| Collision | 1 bot per tile (except spawn point) |
| Actions | `move_*`, `pick_up`, `drop_off`, `wait` |

### Difficulty Configurations

| Difficulty | Bots | Grid Size | Item Types | Order Size |
|---|---|---|---|---|
| Easy | 1 | 12x10 | 4 | 3-4 |
| Medium | 3 | 16x12 | 8 | 3-5 |
| Hard | 5 | 22x14 | 12 | 3-5 |
| Expert | 10 | 28x18 | 16 | 4-6 |

### Key Mechanics
- **Auto-delivery**: When an order completes at the dropoff, ALL bots standing on the dropoff tile get their inventories checked against the NEW active order. Matching items are auto-delivered.
- **Dead inventory**: There is NO `drop` action. Once a bot picks an item that doesn't match any order, it's stuck forever. This is the #1 risk in multi-bot play.
- **Preview → Active**: The preview order becomes the next active order when the current one completes. Preview items are visible and can be pre-picked.

---

## Architecture

### Source Files

```
src/
  main.zig        (243 lines)  WebSocket client, game loop, desync detection
  strategy.zig    (~1360 lines) Core decision engine, orchestrator, trip execution
  trip.zig        (359 lines)  Mini-TSP trip planner (evaluates 1/2/3-item routes)
  pathfinding.zig (162 lines)  BFS distance maps, collision-aware navigation
  parser.zig      (120 lines)  JSON game state parser
  ws.zig          (376 lines)  WebSocket client with TLS support
  types.zig       (108 lines)  Core types: Pos, Bot, GameState, NeedList, etc.
```

### Dependency Graph
```
main.zig → ws.zig, types.zig, strategy.zig, parser.zig, pathfinding.zig
strategy.zig → types.zig, pathfinding.zig, trip.zig
trip.zig → types.zig, pathfinding.zig
parser.zig → types.zig
pathfinding.zig → types.zig
```

---

## How the Bot Thinks (Per Round)

Each round, the bot receives the full game state via WebSocket (JSON), parses it, and runs the decision engine to produce one action per bot. The entire decision process is **deterministic** — same state always produces same actions.

### Phase 1: Pre-Processing

1. **Parse game state**: Grid, bot positions, items on map, active order, preview order
2. **Desync detection**: Compare expected positions with actual — if mismatched, apply 1-round offset correction
3. **Build need lists**: What items the active order still needs (`pick_remaining`), what preview needs
4. **Count delivered items**: Track what each bot is carrying vs what's needed
5. **Persistent state update**: Each bot has persistent state (trip plan, delivering flag, oscillation counters)

### Phase 2: Orchestrator (Multi-Bot Only)

The orchestrator is the brain that decides **which bot picks which item**. It runs BEFORE individual bot decisions.

#### Active Item Assignment (Greedy by Distance)
```
For each item type still needed:
  While copies of this type still unassigned:
    Find the closest (bot, item-instance) pair where:
      - Bot isn't already delivering with priority
      - Bot has free inventory slots
      - Bot hasn't hit the max_pickers limit (NEW: concentrates items on fewer bots)
    Assign item to that bot (mark as "claimed")
```

**Max Pickers Limit**: For 5+ bots, only 3-4 bots get active item assignments. This concentrates items on fewer bots, creating multi-item trips instead of many single-item trips. Fewer delivery trips = less dropoff congestion.

#### Concentration Bonus (Medium Only)
For 3-bot games, bots that already have assignments get a distance bonus of 5, making them appear closer. This encourages the orchestrator to batch items onto the same bot.

#### Preview Assignment (Phase 2)
After active items are assigned, idle bots get preview item assignments:
- Gated by `max_preview_carriers` (prevents too many bots carrying potential dead inventory)
- Gated by `max_orch_preview` (limits total preview assignments per round)
- Only truly idle bots (no active items, not delivering) get preview

### Phase 3: Per-Bot Decision Cascade

Each bot runs through a priority cascade. The FIRST matching action is taken:

```
1. DROP OFF      → At dropoff with active items? → drop_off action
2. EVACUATE      → At dropoff without active items? → flee (multi-bot) or fall through (single)
3. ESCAPE        → Oscillation detected? → escape movement (pick items along the way)
4. PICK UP       → Adjacent to needed item? → pick_up (active pass first, then preview pass)
5. DELIVER       → Has active items + should deliver? → navigate to dropoff (with detour)
6. FOLLOW TRIP   → Has existing trip plan? → navigate to next pickup
7. PLAN TRIP     → No trip? → evaluate candidates via trip planner
8. DELIVER FALLBACK → Has any items? → go to dropoff
9. PRE-POSITION  → Idle? → move near likely-needed items
10. DEAD INVENTORY → Non-matching items? → camp near dropoff
11. WAIT         → Nothing to do
```

---

## Key Systems Deep Dive

### Trip Planner (trip.zig)

The trip planner is a **mini-TSP solver** that evaluates all possible 1, 2, and 3-item trips:

1. **Build candidates**: Find all reachable items matching needs (active + preview if allowed)
2. **Pre-compute BFS**: Distance maps from each candidate's position
3. **Evaluate all trips**:
   - Single items: `cost = dist_to_item + dist_to_dropoff`
   - Pairs: try both orderings (A→B→dropoff, B→A→dropoff)
   - Triples: try all 6 permutations
4. **Score each trip**: `value * 10000 / cost` (efficiency metric)
   - Active items worth 20 points each
   - Preview items worth 3 (or 18 if trip completes order)
   - Order completion bonus: +80 (triggers +5 in-game)
   - Completing trips with preview items: huge bonus (+150 per preview)

### Pathfinding (pathfinding.zig)

- **BFS distance maps**: Flood-fill from any position, respecting walls and shelves
- **Collision-aware navigation**: First step of BFS avoids tiles occupied by other bots
- **Adjacent-tile finding**: For items on shelves, find the best walkable tile adjacent to the shelf

### Anti-Oscillation System

Bots can get stuck oscillating between positions (moving back and forth). The system detects and breaks this:

1. **Position history**: Track last 24 positions per bot
2. **Stall detection**: If bot visits same position 6+ times in history → trigger escape
3. **Escape mode**: Bot moves in a random valid direction for 4 rounds
4. **During escape**: Bot still picks up adjacent items it encounters
5. **Near-dropoff patience**: Bots within 2 tiles of dropoff wait up to 3 rounds instead of escaping

### Delivery System

- **Opportunistic detour**: While heading to dropoff, pick up items that add minimal extra travel
  - Single bot: up to 5 extra rounds for detour
  - Multi-bot: 1 extra round (or 4 if it would complete the order)
- **Dropoff priority**: Only N closest delivering bots get priority (prevents pile-up)
- **Far-with-few filter**: Bots far from dropoff (>8) with <2 items don't deliver — they pick more items instead

### Desync Detection (main.zig)

The WebSocket protocol can cause a 1-round offset between actions and state. The bot detects this by comparing expected positions (based on last action) with actual positions. When a mismatch is detected, all positions are shifted to account for the offset.

---

## Critical Parameters

These parameters are tightly coupled. Changing one often causes cascading regressions across difficulties.

| Parameter | Value | Why It Matters |
|---|---|---|
| `max_pickers` | 3 (5-7 bots), 4 (8+) | Concentrates items on fewer bots, reducing deliveries |
| `max_preview_carriers` | 1/2/3 by bot count | Prevents dead inventory (THE most sensitive parameter) |
| `max_orch_preview` | 2-4 by bot count | Limits preview assignments in orchestrator |
| `MAX_DROPOFF_ACTIVE` | bot_count/2 | Limits dropoff congestion |
| `stall_count threshold` | 6 | Triggers escape (lower = too many false escapes) |
| `escape_rounds` | 4 | Duration of escape mode |
| `concentration_bonus` | 5 (3-bot only) | Batches items to fewer bots on small maps |
| `base_detour` | 5 (single), 1 (multi) | Opportunistic pickup during delivery |
| `far_with_few` | dist>8, inv<2, 5+ bots | Prevents low-value long-distance deliveries |

---

## What Makes This Hard

### The Dead Inventory Problem
Once a bot picks an item, it can NEVER drop it. If the item doesn't match any current or future order, it permanently occupies an inventory slot. With INV_CAP=3, losing even 1 slot to dead inventory reduces a bot's effectiveness by 33%.

Preview items become dead if:
- The bot picks a preview item
- The current order doesn't complete
- A new order replaces the preview before the bot can deliver
- The item is now permanently stuck

The entire `max_preview_carriers` / `max_orch_preview` system exists to carefully gate preview picking.

### The Multi-Bot Coordination Problem
With 10 bots but orders of only 4-6 items:
- Optimal: 2 bots handle everything (3+3 items for 6-item orders)
- Reality: orchestrator must carefully limit how many bots participate
- Idle bots (6-8 per round) waste compute and create pathfinding obstacles
- Per-bot efficiency: Expert gets 8.3 items/bot vs 137.3 items/bot (Easy) = 94% waste

### The Dropoff Bottleneck
Only ONE tile is the dropoff point. With collision (1 bot per tile), multiple bots trying to deliver simultaneously creates a traffic jam. Bots must queue, wait, and coordinate access.

---

## Performance Metrics (Current Best, 40-seed sweep)

| Difficulty | Max Score | Mean Score | Target |
|---|---|---|---|
| Easy (1 bot) | 152 | 136.7 | 175 |
| Medium (3 bots) | 167 | 139.1 | 175 |
| Hard (5 bots) | 150 | 117.9 | 175 |
| Expert (10 bots) | 122 | 79.8 | 175 |

---

## Optimization Methodology

1. Make a single targeted change
2. Build with ReleaseFast
3. Sweep 40 seeds on affected difficulties
4. Compare **MAX score** (peak matters most, variance is acceptable)
5. Keep improvements, revert regressions
6. Never combine multiple untested changes

### Known Failures (Do Not Retry)
- Increasing preview carriers broadly → massive dead inventory
- Concentration bonus for 5+ bots (only works for 3-bot)
- Round-trip metrics in orchestrator for multi-bot
- Reducing anti-oscillation parameters
- Dead-inventory bots navigating to dropoff
- Congestion penalties in orchestrator assignments
