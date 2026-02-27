# Getting Started

## Prerequisites

- **Zig 0.15.2**: Install from [ziglang.org](https://ziglang.org/download/)
  - Current path: `C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe`
- **Python 3** (optional): For log analysis and local simulation

---

## Setup

```bash
cd grocery-bot-zig
zig build -Doptimize=ReleaseFast
```

This compiles the bot to `zig-out/bin/grocery-bot.exe`.

---

## Project Layout

```
grocery-bot-zig/
├── src/
│   ├── main.zig          # Entry point, game loop
│   ├── types.zig         # Shared types and constants
│   ├── parser.zig        # JSON game state parser
│   ├── ws.zig            # WebSocket client
│   ├── strategy.zig      # Decision engine (largest file)
│   ├── pathfinding.zig   # BFS pathfinding
│   └── trip.zig          # Trip optimization
├── build.zig             # Build configuration
├── build.zig.zon         # Package manifest
├── sim_server.py         # Local test server
├── analyze_log.py        # Log analysis tool
└── game_log.jsonl        # Game session logs
```

---

## Running a Game

```bash
./zig-out/bin/grocery-bot.exe wss://<server-url>/play
```

The bot will:
1. Connect via WebSocket
2. Receive game state each round
3. Compute and send actions for all bots
4. Log each round to `game_log.jsonl`
5. Print score every 50 rounds

---

## Key Concepts

- **Active vs Preview orders**: Only active order items score. Preview items become dead inventory if picked early.
- **Desync**: Server applies actions with 1-round delay. The bot detects and compensates for this.
- **Trip**: A planned sequence of 1-3 item pickups followed by delivery.
- **Orchestrator**: Centralized logic that assigns items to bots to prevent duplication.
