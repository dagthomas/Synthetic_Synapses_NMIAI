# Prompt for Next Session

Copy-paste this to continue where we left off:

---

Continue optimizing the grocery-bot-zig project. Read CLAUDE.md and NEXT_STEPS.md for full context.

## Current State
- **Per-difficulty build system ACTIVE**: `python build_all.py` builds 5 executables. Sweep scripts auto-select the right one.
- **Scores**: Easy 131 (mean 119.7), Medium 140 (mean 113.7), Hard 128 (mean 89.8), Expert 96 (mean 68.9). Total: **495/700**.
- Build: `python build_all.py` or `python build_all.py expert` for single
- Sweep: `python sweep_easy.py`, `sweep_medium.py`, `sweep_hard.py`, `sweep_expert.py`
- Competition in **18 days** (March 19).

## What Changed This Session (Session 3)
1. **Per-difficulty build system**: `-Ddifficulty=easy|medium|hard|expert` creates separate executables. `@import("config").difficulty` gives compile-time enum. Sweep scripts auto-select. SvelteKit runner updated.
2. **Preview delivery detour** (Easy/Medium only): Bots delivering to dropoff grab preview items on the way when `pick_remaining==0` (order guaranteed to complete). Easy mean +4.6, Medium mean +4.5. Expert EXCLUDED (caused -10 max regression).
3. **Endgame trip size reduction** (Expert/Hard): Last 40 rounds → max 2 items per trip, last 20 → max 1. Neutral (safe).
4. **Game log cleanup**: sweep.py now keeps only 15 latest game_log_*.jsonl files. Cleaned up 1847 old files.

## Key Architecture
- `src/strategy.zig` imports `@import("config")` for compile-time `DIFFICULTY` enum
- `src/trip.zig` also imports config for difficulty-aware candidate sorting
- `build.zig` accepts `-Ddifficulty` option, produces named executables
- `sweep.py` auto-selects `grocery-bot-{difficulty}.exe` if available
- `replay/app/.../+server.js` uses `getBotPath(difficulty)` for live runner

## Priority Tasks (in order)
1. **Expert-specific optimizations**: Expert has biggest gap (96 vs 175). Try PIBT-style collision resolution, better pre-positioning, reduced dead inventory.
2. **Easy-specific optimizations**: Easy has 131 max, needs single-bot route optimization. Try better trip scoring, pickup-while-delivering improvements.
3. **Medium-specific optimizations**: Round-trip cost + concentration bonus already help. Try higher concentration bonus values, different max_preview settings.
4. **Hard-specific optimizations**: Similar to Medium but with more congestion. Try graduated delivery improvements.
5. **PIBT-style conflict resolution**: Post-BFS collision detection across all bots. Estimated Expert +10-15 max.

## Rules
- Make ONE change at a time, sweep, compare MAX scores
- Use per-difficulty builds: `python build_all.py expert` to only rebuild expert
- Changes that RESTRICT movement always fail — only EXPAND options
- Dead inventory is PERMANENT (no drop action)
- Game logs auto-cleaned to 15 max

## Build & Test
```bash
cd grocery-bot-zig
python build_all.py          # Build all 5 executables
python build_all.py expert   # Build just expert
python sweep_expert.py       # Expert (port 9880)
python sweep_hard.py         # Hard (port 9870)
python sweep_easy.py         # Easy (port 9850)
python sweep_medium.py       # Medium (port 9860)
```

---
