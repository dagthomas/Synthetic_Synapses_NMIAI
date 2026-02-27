# Product Requirements

## Game Rules

- WebSocket-based multiplayer grocery delivery game
- **300 rounds** per game, **120 seconds** wall clock limit
- Grid-based store with walls, shelves, floor, and a single drop-off point

---

## Difficulty Tiers

| Difficulty | Bots | Grid Size |
|------------|------|-----------|
| Easy | 1 | 12x10 |
| Medium | 3 | 16x12 |
| Hard | 5 | 22x14 |
| Expert | 10 | 28x18 |

---

## Mechanics

### Inventory
- Each bot carries up to **3 items**
- Items are picked from adjacent shelf cells
- Items are dropped at the drop-off cell

### Orders
- Orders require **3-6 items** each
- Two orders visible: **active** (scoreable) and **preview** (next)
- When active order completes, preview becomes active, new preview appears
- Items on shelves **never deplete** (multiple bots can pick the same item type)

### Scoring
- **+1** per matching item delivered to drop-off
- **+5** bonus per completed order
- Non-matching items delivered are **not consumed** (dead inventory)

### Actions
Each bot submits one action per round:

| Action | Effect |
|--------|--------|
| `move up/down/left/right` | Move one cell in direction |
| `pickup <item_id>` | Pick item from adjacent shelf |
| `drop_off` | Deliver matching inventory items |

---

## Performance Targets

| Difficulty | Target Score |
|------------|-------------|
| Easy | 150 |
| Medium | 150 |
| Hard | 150 |
| Expert | 150 |

---

## Protocol

- WebSocket connection (ws:// or wss://)
- Server sends JSON game state each round
- Client responds with JSON action array
- Message types: `game_state`, `game_over`
