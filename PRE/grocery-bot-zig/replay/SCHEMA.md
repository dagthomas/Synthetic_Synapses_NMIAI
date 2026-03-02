# Grocery Bot Replay Database Schema

PostgreSQL database for recording and replaying grocery bot games.

## Connection

```
Host: localhost
Port: 5433
Database: grocery_bot
User: grocery
Password: grocery123
URL: postgres://grocery:grocery123@localhost:5433/grocery_bot
```

## Tables

### `runs` — One row per game

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing run ID |
| `seed` | `INTEGER NOT NULL` | Random seed used for map generation |
| `difficulty` | `TEXT NOT NULL` | `easy`, `medium`, `hard`, or `expert` |
| `grid_width` | `INTEGER NOT NULL` | Map width in tiles |
| `grid_height` | `INTEGER NOT NULL` | Map height in tiles |
| `bot_count` | `INTEGER NOT NULL` | Number of bots (1/3/5/10) |
| `item_types` | `INTEGER NOT NULL` | Number of distinct item types (4/8/12/16) |
| `order_size_min` | `INTEGER NOT NULL` | Minimum items per order |
| `order_size_max` | `INTEGER NOT NULL` | Maximum items per order |
| `walls` | `JSONB NOT NULL` | Array of `[x, y]` wall positions |
| `shelves` | `JSONB NOT NULL` | Array of `[x, y]` shelf positions |
| `items` | `JSONB NOT NULL` | Array of `{"id": str, "x": int, "y": int, "type": str}` |
| `drop_off` | `JSONB NOT NULL` | `[x, y]` dropoff position |
| `spawn` | `JSONB NOT NULL` | `[x, y]` spawn position |
| `final_score` | `INTEGER NOT NULL` | Final score at end of game |
| `items_delivered` | `INTEGER NOT NULL` | Total items delivered |
| `orders_completed` | `INTEGER NOT NULL` | Total orders completed (each gives +5 bonus) |
| `created_at` | `TIMESTAMPTZ` | Timestamp when recorded (default: NOW()) |

### `rounds` — One row per round per game

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing round ID |
| `run_id` | `INTEGER NOT NULL` | FK → `runs.id` (CASCADE delete) |
| `round_number` | `INTEGER NOT NULL` | Round number (0-299) |
| `bots` | `JSONB NOT NULL` | Array of bot states: `{"id": int, "x": int, "y": int, "inventory": [...]}` |
| `orders` | `JSONB NOT NULL` | Active + preview order state |
| `actions` | `JSONB DEFAULT '[]'` | Actions taken this round: `[{"bot": id, "action": "move"/"pick_up"/"drop_off"/"wait", ...}]` |
| `score` | `INTEGER NOT NULL` | Score at this round |
| `events` | `JSONB DEFAULT '[]'` | Game events (deliveries, order completions, etc.) |

**Unique constraint**: `(run_id, round_number)` — one entry per round per game.

## Indexes

| Index | Column(s) | Purpose |
|-------|-----------|---------|
| `idx_rounds_run_id` | `rounds.run_id` | Fast round lookup by game |
| `idx_runs_difficulty` | `runs.difficulty` | Filter games by difficulty |
| `idx_runs_seed` | `runs.seed` | Find games by seed |

## Scoring

- **+1** per item delivered
- **+5** bonus per order completed (all items in order delivered)
- `final_score = items_delivered + (orders_completed * 5)`

## Difficulty Configs

| Difficulty | Bots | Grid | Item Types | Order Size |
|------------|------|------|------------|------------|
| Easy | 1 | 12x10 | 4 | 3-4 |
| Medium | 3 | 16x12 | 8 | 3-5 |
| Hard | 5 | 22x14 | 12 | 3-5 |
| Expert | 10 | 28x18 | 16 | 4-6 |

## Docker Setup

```bash
cd replay
docker compose up -d db    # Start PostgreSQL only
docker compose up -d        # Start PostgreSQL + SvelteKit dashboard
docker compose down -v      # Stop and delete all data
```

## Example Queries

```sql
-- Best scores per difficulty
SELECT difficulty, MAX(final_score) as best, AVG(final_score)::int as avg,
       COUNT(*) as runs
FROM runs GROUP BY difficulty ORDER BY difficulty;

-- All runs for a specific seed
SELECT * FROM runs WHERE seed = 7001 ORDER BY final_score DESC;

-- Round-by-round replay for a game
SELECT round_number, score, bots, actions, events
FROM rounds WHERE run_id = 1 ORDER BY round_number;

-- Order completion rate
SELECT difficulty, AVG(orders_completed) as avg_orders,
       AVG(items_delivered) as avg_items
FROM runs GROUP BY difficulty;
```
