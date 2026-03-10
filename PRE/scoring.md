# Grocery Bot Scoring

## Score Formula

Per game:
```
score = items_delivered x 1 + orders_completed x 5
```

- **+1 point** for each item delivered to the drop-off
- **+5 bonus** for completing an entire order (all required items delivered)

## Leaderboard

Your **leaderboard score** is the **sum of your best scores across all 5 maps**.

- Play each map as many times as you want (60s cooldown, 40/hour, 300/day)
- Only your highest score per map is saved
- Deterministic within a day — same algorithm = same score
- To maximize your rank: get good scores on ALL maps

## Daily Rotation

Item placement on shelves and order contents change daily at midnight UTC. The grid structure (walls, shelf positions) stays the same. This prevents hardcoding solutions while keeping games deterministic within a single day.

## Infinite Orders

Orders never run out. When you complete the active order, the next one activates and a new preview appears. The only limit is the round cap. Score as much as you can before time runs out.

## Order Sizes

| Level | Items per Order |
|-------|----------------|
| Easy | 3-4 |
| Medium | 3-5 |
| Hard | 3-5 |
| Expert | 4-6 |
| Nightmare | 4-6 |

## Score Examples

| Scenario | Items | Orders | Score |
|----------|-------|--------|-------|
| Delivered 3 items, no complete orders | 3 | 0 | 3 |
| Delivered 4 items, completed 1 order | 4 | 1 | 9 |
| Delivered 15 items, completed 3 orders | 15 | 3 | 30 |
| Delivered 50 items, completed 10 orders | 50 | 10 | 100 |

## Game End Conditions

| Condition | Description |
|-----------|-------------|
| 300 rounds used (500 Nightmare) | Maximum rounds reached |
| Wall-clock timeout | 120 seconds elapsed (300s Nightmare) |
| Disconnect | Client disconnected |

## Strategy Tips

- **Complete orders, don't just deliver items** — the +5 order bonus is significant
- **Focus on the active order first** — you can only deliver to the current order
- **Pre-pick items for the preview order** — look ahead and pre-stage items in bot inventories
- **Exploit cascade chain reactions** — when active order completes, bots AT the dropoff zone get inventories auto-checked against the new active order (DZ-only, not all bots)
- **Don't pick random items** — non-matching items waste inventory slots permanently
- **Scale your strategy by bot count** — Easy (1 bot) is a different problem than Expert (10 bots)
- **Plan routes** — full map visible from round 1, use BFS/A* for pathfinding
- **Play all maps** — leaderboard is the sum of all maps
