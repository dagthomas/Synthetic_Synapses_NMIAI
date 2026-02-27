# Development Workflow

## Build

```bash
cd grocery-bot-zig
C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe build -Doptimize=ReleaseFast
```

Output: `zig-out/bin/grocery-bot.exe`

---

## Run

### Live Game
```bash
./zig-out/bin/grocery-bot.exe wss://game-server-url/play
```

### Replay Analysis
```bash
./zig-out/bin/grocery-bot.exe --replay game_log.jsonl
```

Reports: total moves, pickups, dropoffs, useful action rate.

---

## Log Analysis

Game logs are saved as JSONL (`game_log.jsonl`). Analyze with:

```bash
python analyze_log.py game_log.jsonl
```

---

## Local Simulation

```bash
python sim_server.py
```

Runs a local WebSocket server for testing without the real game server.

---

## Debug Cycle

1. Run a live game or replay
2. Check score output (printed every 50 rounds)
3. Review `game_log.jsonl` for round-by-round state
4. Use `--replay` mode to test strategy changes against recorded games
5. Look for: stalls, oscillation, dead inventory, missed pickups
