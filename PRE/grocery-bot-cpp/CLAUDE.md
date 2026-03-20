# Grocery Bot C++ Solver

## Architecture

**C++ solver** (`solver.cpp`) for offline planning + **Python client** (`client.py`) for WebSocket + **Python orchestrator** (`run.py`) for pipeline.

## Build

```bash
# Compile (requires zig 0.15+)
C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe c++ -O2 -std=c++20 solver.cpp -o solver.exe

# Or via orchestrator
python run.py compile
```

## Usage

```bash
# Full pipeline: capture orders → solve → replay
python run.py <ws_url>

# Step by step:
python client.py capture <ws_url> --save capture.json    # Probe game, save map+orders
python run.py solve capture.json plan.txt                  # Offline solve
python client.py replay <ws_url> --plan plan.txt           # Replay solution

# Test with game_engine simulation
python test_solver.py easy medium hard expert

# Generate visualizer
python visualize.py easy 42
```

## Strategy (from user)

1. **Probe server** — Play greedy to discover orders (deterministic per day)
2. **Flatten items** — Map each required type to nearest shelf position
3. **Group by 3** — DP over bitmask groups items into trips of ≤3 (inventory capacity)
4. **Permutation search** — Generate all permutations of bots×groups, test top candidates
5. **A* with spacetime** — Plan each bot sequentially with W×H×T reservation table
6. **One-way traffic** — Lock horizontal corridor to left-only on nightmare
7. **Spawn parking** — Idle bots wait at spawn (allows overlap)

## Solver Algorithm

1. **BFS precompute** — All-pairs shortest paths on walkable grid (O(V²))
2. **Per order**: For each item type needed, find nearest shelf → DP group into trips → assign bots → A* paths with spacetime reservation → write action log
3. **Output**: Full action plan (per-round actions for all bots)

## Current Scores (seed 42 simulation)

| Diff | Score | Orders | Notes |
|---|---|---|---|
| Easy | 128 | 15 | Single bot, works perfectly |
| Medium | 19 | 2 | Spacetime divergence with 3 bots |
| Hard | 10 | 1 | Spacetime divergence with 5 bots |
| Expert | 62 | 6 | 2-bot trips work well |

## Known Issues

- **Multi-bot spacetime divergence**: Game engine resolves collisions in bot-ID order (sequential), but spacetime reservation model doesn't perfectly match. Bots at dropoff block others.
- **Move-away after dropoff** helps but doesn't fully solve congestion.
- **"Can't reach" warnings**: A* fails when corridors are over-reserved by earlier bots.

## Files

| File | Purpose |
|---|---|
| `solver.cpp` | C++ offline planner (BFS + DP grouping + A* spacetime) |
| `client.py` | Python WebSocket client (probe/capture/replay modes) |
| `run.py` | Orchestrator (compile + solve + replay pipeline) |
| `test_solver.py` | Test solver against game_engine simulation |
| `debug_plan.py` | Debug tool: trace plan execution round by round |
| `visualize.py` | Generate HTML/SVG visualizer |
