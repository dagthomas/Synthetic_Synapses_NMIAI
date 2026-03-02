# Next Steps — GPU Optimal Solver (2026-03-02)

## What Failed Tonight

### Problem: Live Score = 18 (Expected 130+)

Both `ReactiveSolver` and `LiveMultiSolver` fail catastrophically on live games:

1. **`LiveMultiSolver` precomputes all 300 rounds at round 0** using only 2 visible orders + 98 random filler orders. After the first 2 real orders complete, the precomputed actions target wrong items → **dead inventory** → bot stuck forever → score plateaus at 18.

2. **`ReactiveSolver`** is too basic — constant escape loops, only scored 26.

3. **Only 4-5 orders captured** because the bot barely completes any orders, so few new ones are revealed.

### Root Cause

The Python live solver (`LiveMultiSolver`) is NOT reactive. It precomputes a full 300-round action sequence at round 0 and replays it. This works on sim_server (where all orders are known) but fails on the live game server (where orders are revealed 2 at a time).

The Zig bot IS reactive (makes per-round decisions) and gets **115-131 on Easy live**. We should use the Zig bot for capture.

## Fix Plan (Priority Order)

### Fix 1: Use Zig Bot for Capture (Fastest Path)

The Zig bot already:
- Plays reactively (per-round decisions based on actual game state)
- Gets ~115-131 on Easy live
- Writes `game_log_*.jsonl` with full game state every round

**Desired workflow (single button, one or two tokens):**
1. Paste token → Run Zig bot → plays full 300-round game, writes game_log
2. Parse game_log → extract ALL orders seen during the game → save capture.json
3. GPU DP on capture → optimal solution → save best.json
4. Replay optimal with same token (if server allows reconnect) or new token

**New script needed**: `capture_from_game_log.py`
```python
# Parse a game_log_*.jsonl to extract capture data
# Accumulates all orders seen during the game
# Saves to solutions/<difficulty>/capture.json
```

**Update `/api/optimize/play/+server.js`**: Spawn Zig bot exe instead of Python live_solver. Parse its game_log for SSE streaming. After game ends, auto-extract capture and auto-run GPU DP.

**Ideal one-button flow on dashboard:**
1. User pastes token
2. Click "CAPTURE & SOLVE"
3. Backend: Zig bot plays game → parse game_log → capture.json → GPU DP → best.json
4. User pastes same/new token
5. Click "REPLAY" → replay_solution.py sends optimal actions

### Fix 2: Make LiveMultiSolver Reactive (Better Long-Term)

Make the Python solver re-plan when new orders appear:
- Track known orders
- When `data['orders']` contains a new order ID → re-run planner from current state
- Keep using MAPF planner but re-plan every time the order set changes

This is harder but would give the Python solver competitive live scores.

### Fix 3: GPU DP as Live Single-Connection Solver (Hardest)

The GPU DP takes ~4s for Easy. If we could:
1. Connect to game at round 0, read 2 orders
2. GPU DP those 2 orders → optimal first ~30 rounds
3. Play those rounds, get new orders as they appear
4. Re-run GPU DP with updated order set
5. Repeat

This requires incremental GPU DP (warm-start from current state). Complex but would be the ultimate solver.

## Dashboard Fixes Needed

### Terminal Log Reset
The system log in `/gpu` needs to reset when starting a new action. Currently old log lines accumulate.

**Fix**: In `startCapture()`, `startSolve()`, `startReplay()` — add `terminalLines = [];` at the start (already done for `startSolve()` but not for others consistently).

### Workflow
The 3-step workflow (CAPTURE → GPU SOLVE → REPLAY) is displayed but CAPTURE currently fails. After Fix 1, the CAPTURE button should spawn the Zig bot instead of the Python solver.

## GPU DP Performance (Working Correctly)

When given a GOOD capture (30+ orders from a full game), GPU DP works perfectly:
- Easy seed 7001 (yesterday): **score=137** (provably optimal, 0 pruning)
- 5-seed average: **mean=165.4, max=175**
- Time: ~4 seconds on RTX 5090
- Previous best CPU beam search: 139

The GPU DP is not the problem. The capture step is.

## Competition Context

- Competition: March 19, 2026
- Seeds fixed per day per difficulty
- Leaderboard = sum of best scores across all 4 maps
- Current leader: Easy=141, Medium=189, Hard=217, Expert=219
- Our previous best: Easy=148, Medium=139, Hard=160, Expert=125

## Files Reference

| File | Purpose |
|------|---------|
| `gpu_beam_search.py` | GPU DP solver (dp_search method) — WORKS |
| `gpu_solve_stream.py` | Streaming JSON wrapper for dashboard — WORKS |
| `live_solver.py` | Live game solver — BROKEN for capture (precomputes, not reactive) |
| `replay_solution.py` | Replay saved solution over WebSocket — WORKS |
| `solution_store.py` | Save/load solutions — WORKS |
| `replay/app/src/routes/gpu/+page.svelte` | Matrix dashboard — WORKS (needs terminal reset) |
| `replay/app/src/routes/api/gpu/solve/+server.js` | GPU solve SSE endpoint — WORKS |
| `replay/app/src/routes/api/optimize/play/+server.js` | Capture SSE endpoint — needs to use Zig bot |
| `replay/app/src/routes/api/optimize/solutions/+server.js` | Solutions CRUD — WORKS (GET + DELETE) |
