# Continue Prompt — Paste This to Resume

Read `grocery-bot-gpu/NEXT_STEPS.md` for full context. Here's the TL;DR:

## What's Working
- GPU DP solver gets **provably optimal** Easy scores (137-175) in ~4s
- Matrix-style SvelteKit dashboard at `/gpu` with capture/solve/replay buttons
- Solution storage (save/load/clear)
- Replay over WebSocket works

## What's Broken
- **Live capture scores 18 instead of 130+** — the Python `LiveMultiSolver` precomputes all 300 rounds at round 0 with only 2 known orders + fake fillers. After first 2 orders complete, bot picks wrong items → dead inventory → stuck.
- Only 4-5 orders get captured (bot barely completes anything)

## Priority Fix
**Use the Zig bot for capture** — it's reactive (per-round decisions) and scores 115-131 on Easy live. The Python solver precomputes all 300 rounds at round 0 with only 2 known orders, so it fails catastrophically (score 18).

The user wants a **single flow**: bot captures all orders by playing → GPU DP computes optimal → replay with same or new token.

1. Create `capture_from_game_log.py`: Parse Zig bot's `game_log_*.jsonl` to extract capture data (grid, items, ALL orders seen across 300 rounds). Save as `solutions/<difficulty>/capture.json`.

2. Update `/api/optimize/play/+server.js` to spawn the Zig bot (`grocery-bot-easy.exe`) instead of Python `live_solver.py`. After game ends, auto-run `capture_from_game_log.py` then auto-run `gpu_solve_stream.py`.

3. Flow: Paste token → Click "CAPTURE & SOLVE" → Zig bot plays (score ~120, captures all orders) → GPU DP (score ~160) → Paste new token → Click "REPLAY" → score ~160 hits leaderboard.

## Also Fix
- Terminal log in `/gpu` page should reset/clear when starting a new action (currently accumulates old lines)
- The `/gpu` page already has CLEAR ALL MAP DATA button and workflow guide (done last session)
- Consider making the Zig bot capture + GPU DP solve happen as one chained operation behind a single button

## Key Directories
- `grocery-bot-gpu/` — Python solver, GPU DP, solution store
- `grocery-bot-zig/` — Zig bot (reactive, scores well live)
- `grocery-bot-zig/replay/app/` — SvelteKit dashboard
- `grocery-bot-gpu/solutions/<difficulty>/` — Saved captures and solutions
