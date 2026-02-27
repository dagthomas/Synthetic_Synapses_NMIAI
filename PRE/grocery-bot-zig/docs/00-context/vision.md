# Vision

Grocery-bot-zig is an autonomous agent that plays a competitive WebSocket-based grocery delivery game. The bot controls 1-10 robots navigating a grid-based store, picking items from shelves, and delivering them to a drop-off point to fulfill sequential orders.

---

## Goals

- Maximize score across all four difficulty tiers (Easy, Medium, Hard, Expert)
- Target: 150+ points per category
- Handle real-time WebSocket communication with sub-second decision latency
- Coordinate multiple bots without deadlock or congestion

---

## Constraints

- **300 rounds** per game, 120 seconds wall clock
- **3-item inventory** per bot
- Items persist on shelves (never deplete)
- Only the active order's items score on delivery; preview items become dead inventory if picked too early
- 1-round action offset from server (desync problem)

---

## Why Zig

- Zero-allocation game loop (fixed buffers, no GC pauses)
- Predictable latency for real-time WebSocket response
- Compiles to single binary with no runtime dependencies
