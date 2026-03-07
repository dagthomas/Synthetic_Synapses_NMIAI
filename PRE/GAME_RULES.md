# Grocery Bot — Complete Game Reference

## Overview

NM i AI 2026 pre-competition challenge. Build a bot that controls worker agents via WebSocket in a procedurally generated grocery store. Bots navigate the store, pick items from shelves, and deliver them to a drop-off zone to fulfill orders sequentially. The game runs for up to 300 rounds with a 120-second wall-clock limit.

Platform: app.ainm.no | Competition period: Feb 20 - Mar 16, 2026

## Difficulty Levels

| Level  | Grid  | Bots | Aisles | Item Types | Order Size |
|--------|-------|------|--------|------------|------------|
| Easy   | 12x10 | 1    | 2      | 4          | 3-4        |
| Medium | 16x12 | 3    | 3      | 8          | 3-5        |
| Hard   | 22x14 | 5    | 4      | 12         | 3-5        |
| Expert | 28x18 | 10   | 5      | 16         | 4-6        |

Item types (in order): milk, bread, eggs, butter, cheese, pasta, rice, juice, yogurt, cereal, flour, sugar, coffee, tea, oil, salt.

## Coordinate System

- Origin (0, 0) is the **top-left** corner
- X increases to the right
- Y increases downward

## Map Layout

- **Border walls** surround the entire grid
- **Aisles** are built starting at x=3 with 4-column spacing. Each aisle is: wall column, shelf column, walkway column, shelf column, wall column
- Shelves occupy rows from y=2 to mid_y-1 (top half) and mid_y+1 to h-3 (bottom half), where mid_y = h // 2
- **Drop-off** is at position (1, h-2) — bottom-left inside border
- **Spawn** is at position (w-2, h-2) — bottom-right inside border

## Grid Cell Types

| Cell    | Value | Walkable |
|---------|-------|----------|
| Floor   | 0     | Yes      |
| Wall    | 1     | No       |
| Shelf   | 2     | No       |
| Drop-off| 3     | Yes      |

Items sit ON shelf cells. Bots walk on floor and drop-off cells only.

## WebSocket Protocol

### Connection
```
wss://game.ainm.no/ws?token=<jwt_token>
```
Get a token by clicking "Play" on a map at app.ainm.no/challenge.

### Message Flow
```
Server -> Client: {"type": "game_state", ...}   (round 0)
Client -> Server: {"actions": [...]}
Server -> Client: {"type": "game_state", ...}   (round 1)
Client -> Server: {"actions": [...]}
...
Server -> Client: {"type": "game_over", ...}     (final)
```

### Game State Message
```json
{
  "type": "game_state",
  "round": 42,
  "max_rounds": 300,
  "grid": {
    "width": 14,
    "height": 10,
    "walls": [[1,1], [1,2], [3,1]]
  },
  "bots": [
    {"id": 0, "position": [3, 7], "inventory": ["milk"]},
    {"id": 1, "position": [5, 3], "inventory": []},
    {"id": 2, "position": [10, 7], "inventory": ["bread", "eggs"]}
  ],
  "items": [
    {"id": "item_0", "type": "milk", "position": [2, 1]},
    {"id": "item_1", "type": "bread", "position": [4, 1]}
  ],
  "orders": [
    {
      "id": "order_0",
      "items_required": ["milk", "bread", "eggs"],
      "items_delivered": ["milk"],
      "complete": false,
      "status": "active"
    },
    {
      "id": "order_1",
      "items_required": ["cheese", "butter", "pasta"],
      "items_delivered": [],
      "complete": false,
      "status": "preview"
    }
  ],
  "drop_off": [6, 9],
  "score": 12,
  "active_order_index": 0,
  "total_orders": 8
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Current round number (0-indexed) |
| `max_rounds` | int | Always 300 |
| `grid.width` | int | Grid width in cells |
| `grid.height` | int | Grid height in cells |
| `grid.walls` | int[][] | List of [x, y] wall positions (includes both walls AND shelves) |
| `bots` | object[] | All bots with id, position [x,y], and inventory (item type names) |
| `items` | object[] | All items on shelves with id, type (string), and position [x,y] |
| `orders` | object[] | Max 2 visible: one "active" + one "preview" |
| `drop_off` | int[] | [x, y] of the drop-off zone |
| `score` | int | Current score |
| `active_order_index` | int | Index of the current active order |
| `total_orders` | int | Total number of orders in the game |

### Bot Response
Send within **2 seconds** of receiving game state:
```json
{
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

### Game Over Message
```json
{
  "type": "game_over",
  "score": 47,
  "rounds_used": 200,
  "items_delivered": 22,
  "orders_completed": 5
}
```

## Actions

| Action | Extra Fields | Description |
|--------|-------------|-------------|
| `move_up` | — | Move one cell up (y-1) |
| `move_down` | — | Move one cell down (y+1) |
| `move_left` | — | Move one cell left (x-1) |
| `move_right` | — | Move one cell right (x+1) |
| `pick_up` | `item_id` | Pick up item from adjacent shelf |
| `drop_off` | — | Deliver matching items to active order |
| `wait` | — | Do nothing |

Invalid actions are silently treated as `wait`.

## Movement Rules

- Moves into walls, shelves, or out-of-bounds fail silently (treated as wait)
- Moves into a cell occupied by another bot fail silently (blocked_by_bot)
- Actions resolve in **bot ID order** — bot 0 moves first, then bot 1, etc.
- The spawn tile is **exempt from collision** — multiple bots can share it at game start

## Pickup Rules

- Bot must be **adjacent** (Manhattan distance 1) to the shelf containing the item
- Bot inventory must not be full (max **3 items**)
- `item_id` must match a valid item on the map
- Items are **permanent on shelves** — picking up gives you a copy, the shelf item remains
- Multiple bots can pick up the same item type from the same shelf

## Dropoff Rules

- Bot must be standing **on** the drop-off cell
- Bot must have items in inventory
- Only items matching the **active order** are delivered
- Non-matching items **stay in inventory**
- When the active order completes:
  1. The preview order becomes active
  2. A new preview order is generated
  3. **Auto-delivery**: all bots currently standing on the drop-off cell have their inventories re-checked against the new active order (items matching the new order are delivered immediately)

## Scoring

| Event | Points |
|-------|--------|
| Item delivered | +1 |
| Order completed | +5 bonus |

**Leaderboard score** = sum of best scores across all 4 maps.

## Orders

- **Active order**: the current order you must complete. Only this order accepts deliveries.
- **Preview order**: the next order. Visible but you cannot deliver to it yet. You CAN pre-pick items for it.
- **Infinite**: when you complete an order, a new one appears. Orders never run out. The 300-round cap is the only limit.
- Only 2 orders are visible at any time (active + preview).
- Order items are drawn from the available item types for that difficulty level.

## Constraints Summary

| Constraint | Value |
|-----------|-------|
| Max rounds | 300 |
| Wall-clock limit | 120 seconds |
| Response timeout | 2 seconds per round |
| Bot inventory capacity | 3 items |
| Bot collision | No two bots on same tile (except spawn) |
| Visibility | Full map visible every round |
| Game cooldown | 10 seconds between games per team |
| Disconnect | Game ends, score saved |
| Max visible orders | 2 (active + preview) |

## Daily Rotation

- Item placement on shelves and order contents change daily at midnight UTC
- The grid structure (walls, shelf positions) stays the same per difficulty
- Same day = same game = deterministic (same algorithm produces same score)
- This prevents hardcoding solutions

## Map Generation Details

Items are assigned to shelves in sorted order, cycling through item types. For a difficulty with N item types and S shelf positions, shelf i gets item type `i % N`. This means each item type appears on multiple shelves, spread across the map.

The RNG for order generation is seeded once per game. Orders are generated sequentially from this seed, meaning with the right seed you can predict ALL future orders (full foresight).

## Strategy Considerations

- Complete orders, don't just deliver items — the +5 order bonus is significant
- Focus on the active order first — you can only deliver to the current order
- Pre-pick items for the preview order — look ahead and pre-stage items
- Don't pick random items — non-matching items waste inventory slots
- Plan routes with BFS/A* — full map visible from round 1
- Coordinate bots — avoid duplicate pickups, assign bots to different tasks
- Consider the auto-delivery mechanic — having bots with useful items waiting at drop-off when an order completes gives free points
- Action resolution order matters — bot 0 moves first, plan accordingly to avoid collisions
