# Grocery Bot - Findings History

## Version History & Scores (Easy map, 300 rounds)

| Version | Score | Key Changes | Issues |
|---------|-------|-------------|--------|
| v1 | 45 | Initial implementation | Baseline |
| v2 | 17 | Trip planning rewrite | WebSocket ping/pong not handled → early disconnect |
| v3 | 18 | Attempted fix | Same ping/pong issue |
| v4 | 71 | Fixed WS ping/pong in ws.zig | Oscillation in late game, left-side-only item usage |
| v5 | 37 | Major trip planner rewrite (tripScore, cached adjs, direction-aware adj) | Oscillation at dropoff (1,8) falsely triggered |
| v5b | 37 | Excluded dropoff from oscillation detection, added pickup blacklist | False blacklisting due to advanceTrip not shifting trip_adjs |
| v5c | 36 | Fixed advanceTrip to also shift trip_adjs | Still false blacklisting (10 entries), blacklist too aggressive |
| v5d | 37 | Disabled blacklist enforcement (logging only) | Deadlock: full inventory of preview items, can't pick active items |
| v5e | TBD | Fix order change detection, prevent pure-preview trips | Pending test |

**Target**: Easy 104 (current leader)

---

## Key Findings

### 1. WebSocket Ping/Pong (v4 fix)
- **Problem**: Server sends WebSocket ping frames (opcode 0x9). Bot didn't respond with pong → server disconnected bot mid-game.
- **Impact**: v2/v3 scored 17-18 instead of expected ~45+ because bot was disconnected early.
- **Fix**: In `ws.zig`, `recvMessage()` now handles ping (0x9 → sendPong), close (0x8 → error), and pong (0xA → ignore) frames.
- **Result**: v4 score jumped to 71.

### 2. Pickup Adjacency (v4 analysis)
- **Finding**: Pickups from position [3,7] targeting item [3,6] had 0% success rate (11 attempts). The correct adjacent position was [3,5] with 100% success rate.
- **Cause**: The adj finder picked a position that was adjacent but blocked by a wall/shelf on the path between the bot and the item.
- **Note**: This is a fundamental issue with `findBestAdj` - it finds floor cells adjacent to the item, but doesn't verify the pickup action actually works from that cell.

### 3. Left-Side Bias (v4 analysis)
- **Finding**: Bot only used left-side items (columns 1-5), leaving 8 right-side items (columns 7-11) completely unused.
- **Cause**: Trip planner picked closest items first, and left-side items were closer to the dropoff at (1,8).
- **Partial fix**: v5 rewrote trip planner to pick closest item per type-slot instead of first found.

### 4. Dropoff Oscillation Detection (v5 fix)
- **Problem**: Oscillation detection counted visits to the dropoff (1,8) as oscillation, because the bot naturally returns there after each delivery.
- **Impact**: Bot kept resetting trips at the dropoff, stuck in a loop.
- **Fix**: `if (isOscillating(pb, bot.pos) and !bot.pos.eql(state.dropoff))`

### 5. advanceTrip Adj Shift Bug (v5c fix)
- **Problem**: When `advanceTrip` removed a mid-trip item (e.g., item 0 of 3), it shifted `trip_ids` and `trip_id_lens` but NOT `trip_adjs`. This caused subsequent pickups to target the wrong adj position.
- **Example**: After removing item 0, `trip_adjs[0]` still pointed to the old item 0's adj instead of item 1's adj.
- **Fix**: Also shift `trip_adjs` in `advanceTrip`.

### 6. Pure-Preview Trip Deadlock (v5d analysis, v5e fix)
- **Problem**: Bot filled all 3 inventory slots with preview items (butter, butter, yogurt) while active order still needed milk. Full inventory = can't pick milk, can't deliver non-matching items = **permanent deadlock**.
- **Root cause**: Trip planner allowed pure-preview trips (ac=0, pc=3) even when active order items remained.
- **Fix**: `if (ac == 0 and active_remaining > 0) continue;` in all trip evaluations (1-item, 2-item, 3-item).

### 7. Order Change Detection (v5e fix)
- **Problem**: Order change detection compared `state.orders[0]` (always index 0) instead of using `active_order_index` from game state. When active order index changed (e.g., order 3 → order 4), the detection didn't notice because it always looked at array position 0.
- **Fix**: Parse `active_order_index` from game JSON into `state.active_order_idx`, use it for change detection.

---

## Game Mechanics Notes

- **Map**: 12x10 grid, deterministic per day (same code = same score within a day)
- **Easy**: 1 bot, 4 item types, 16 items on map, 3-4 items per order, inventory capacity 3
- **Scoring**: items_delivered × 1 + orders_completed × 5
- **Items persist**: items_on_map stays at 16 (items can be re-picked after delivery)
- **Orders are sequential**: Active + preview visible. When active completes, preview → active, new preview appears.
- **Drop-off**: Only matching items consumed. Non-matching items STAY in inventory (deadlock risk!)
- **Tokens**: Single-use JWTs that expire. One test per token.
- **Leaderboard**: Sum of best scores across all 4 maps (Easy/Medium/Hard/Expert)

---

## Architecture (main.zig)

- **decideActions()** (~line 650-850): Main decision engine, 6-step priority system
  - Step 1: Drop off items at dropoff
  - Step 2: Pick up item if adjacent to trip target
  - Step 3: Handle order change (invalidate trip)
  - Step 4: Handle oscillation (reset trip)
  - Step 5: Follow current trip (pathfind to next adj)
  - Step 6: Plan new trip (call planBestTrip)
- **planBestTrip()** (~line 350-610): Trip optimizer
  - Finds candidate items matching active/preview order needs
  - Evaluates all 1/2/3-item trip permutations with BFS distances
  - Uses `tripScore()` for value/cost scoring with order completion bonus
- **parseGameState()** (~line 880-1000): JSON parser for game state
- **BFS pathfinding**: Pre-computed distance maps, collision-aware navigation
