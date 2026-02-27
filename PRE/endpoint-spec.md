# WebSocket Protocol Specification

## Connection

Connect via WebSocket to the URL provided when you request a game token:

```
wss://game-dev.ainm.no/ws?token=<jwt_token>
```

Get a token by clicking "Play" on a map at [dev.ainm.no/challenge](https://dev.ainm.no/challenge), or by calling the `request_game(map_id)` MCP tool.

## Message Flow

```
Server → Client: {"type": "game_state", ...}     (round 0)
Client → Server: {"actions": [...]}
Server → Client: {"type": "game_state", ...}     (round 1)
Client → Server: {"actions": [...]}
...
Server → Client: {"type": "game_over", ...}       (final)
```

## Game State Message

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
| `max_rounds` | int | Maximum rounds (300) |
| `grid.width` | int | Grid width in cells |
| `grid.height` | int | Grid height in cells |
| `grid.walls` | int[][] | List of [x, y] wall positions |
| `bots` | object[] | All bots (1-10 depending on difficulty) with id, position [x,y], and inventory |
| `items` | object[] | All items on shelves with id, type, and position [x,y] |
| `orders` | object[] | Only active + preview orders (max 2). Each has `status`: `"active"` or `"preview"` |
| `drop_off` | int[] | [x, y] position of the drop-off zone |
| `score` | int | Current score |
| `active_order_index` | int | Index of the current active order |
| `total_orders` | int | Total number of orders in the game |

## Bot Response

Send within **2 seconds** of receiving the game state:

```json
{
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

### Actions

| Action | Extra Fields | Description |
|--------|-------------|-------------|
| `move_up` | — | Move one cell up (y-1) |
| `move_down` | — | Move one cell down (y+1) |
| `move_left` | — | Move one cell left (x-1) |
| `move_right` | — | Move one cell right (x+1) |
| `pick_up` | `item_id` | Pick up item from adjacent shelf |
| `drop_off` | — | Deliver matching items to active order at drop-off zone |
| `wait` | — | Do nothing |

### Move Rules

- Moves to walls, shelves, or out-of-bounds cells fail silently (treated as `wait`)
- Moves to a cell occupied by another bot fail silently (`blocked_by_bot`)
- Actions resolve in **bot ID order** — bot 0 moves first, then bot 1, etc.
- The spawn tile (bottom-right) is exempt from collision — bots can share it at game start

### Pickup Rules

- Bot must be **adjacent** (Manhattan distance 1) to the shelf containing the item
- Bot inventory must not be full (max 3 items)
- `item_id` must match an item on the map

### Dropoff Rules

- Bot must be standing **on** the drop-off cell
- Bot must have items in inventory
- Only items matching the **active order** are delivered — non-matching items **stay in inventory**
- Each delivered item = **+1 point**
- Completed order = **+5 bonus points**
- When the active order completes, the next order activates immediately and remaining items are re-checked

## Game Over Message

```json
{
  "type": "game_over",
  "score": 47,
  "rounds_used": 200,
  "items_delivered": 22,
  "orders_completed": 5
}
```

## Timeouts & Errors

- **2 second** timeout per round for your response
- Timeout → all bots wait (no action)
- Invalid actions → treated as `wait`
- Disconnect → game ends, score is saved
- **120 second** wall-clock limit per game

### Coordinate System

- Origin `(0, 0)` is the **top-left** corner
- X increases to the right
- Y increases downward
