# System State

Current state of the grocery-bot-zig implementation as of 2026-02-27.

---

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.zig` | ~244 | Entry point, WebSocket client, game loop, desync detection |
| `src/types.zig` | ~109 | Type definitions and constants |
| `src/parser.zig` | ~121 | JSON game state parser |
| `src/ws.zig` | ~377 | WebSocket protocol (frame encode/decode, TLS) |
| `src/strategy.zig` | ~1213 | Decision engine, orchestrator, per-bot logic |
| `src/pathfinding.zig` | ~163 | BFS distance maps, collision-aware pathfinding |
| `src/trip.zig` | ~334 | Trip planning with mini-TSP optimization |

---

## Best Scores

| Difficulty | Bots | Score | Target |
|------------|------|-------|--------|
| Easy | 1 | 45 | 150 |
| Medium | 3 | -- | 150 |
| Hard | 5 | 90 | 150 |
| Expert | 10 | 77 | 150 |

---

## Known Issues

| Issue | Status | Description |
|-------|--------|-------------|
| WebSocket desync | Mitigated | 1-round action offset causes position mismatch; detected via 3-round threshold |
| Dead inventory | Partially fixed | Preview items stuck in inventory after order change; preview picking limited |
| Dropoff congestion | Partially fixed | Priority system limits to 2-3 active deliverers |
| Oscillation | Mitigated | Position history tracking with 4-round escape mechanism |
| Stale trips | Mitigated | Force reset after 30 rounds on same order |

---

## Build Toolchain

- Zig 0.15.2 (`C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe`)
- WebSocket dependency via `zig-pkg/`
- Build: `zig build -Doptimize=ReleaseFast`
