# Next Steps - Grocery Bot Optimization (2026-03-01, Session 3)

## Current Scores (40-seed sweep, seeds 7001-7040)
| Difficulty | Mean | Max | Target | Gap |
|---|---|---|---|---|
| Easy | 119.7 | **131** | 175 | 44 |
| Medium | 113.7 | **140** | 175 | 35 |
| Hard | 89.8 | **128** | 175 | 47 |
| Expert | 68.9 | **96** | 175 | 79 |

**Total: 131 + 140 + 128 + 96 = 495** (target: 700, gap: 205)

## Per-Difficulty Build System (NEW)
- `python build_all.py` builds 5 separate executables
- Each uses compile-time `@import("config").difficulty` for per-difficulty parameters
- Sweep scripts auto-select the right executable
- Can now optimize each difficulty independently without cross-regression risk

## Changes Made This Session
1. **Preview delivery detour** (Easy/Medium): +4.6 mean Easy, +4.5 mean Medium
2. **Per-difficulty build system**: Separate executables with compile-time constants
3. **Endgame trip reduction** (Expert/Hard): Neutral safety margin
4. **Game log cleanup**: Auto-keeps only 15 latest files

## Changes From Previous Sessions (Still Active)
- Round-trip cost orchestrator (Medium/Hard): +9 max on both
- max_pickers=8 for Expert: +3.1 mean
- Concentration bonus=8 for Medium
- BFS pre-computation at round 0
- All other changes documented in CLAUDE.md

## Failed This Session
- Preview delivery detour for Expert: -10 max, -5 mean (restricted to bot_count < 5)
- Adjacent preview pickup removed from Expert: -4 max when accidentally removed
- BFS direction diversity (from last session): -20 on some seeds, reverted
- Global greedy orchestrator (from last session): worse than per-type assignment

## Priority Optimizations (Per-Difficulty)

### Expert (Gap: 79 points - BIGGEST PRIORITY)
1. **PIBT-style conflict resolution**: Post-BFS pass to detect and resolve head-on collisions in corridors. 10 bots in narrow corridors = massive time waste.
2. **Reduce dead inventory**: 45-65% of Expert inventory becomes dead. Need smarter preview limiting.
3. **Better pre-positioning**: When active order nearly done, move idle bots toward preview items.
4. **Corridor-aware routing**: Prefer wider paths to reduce blockages.

### Easy (Gap: 44 points)
1. **Better trip scoring**: Maximize items per trip (always do 3-item trips when possible).
2. **Pickup-while-delivering expansion**: The preview detour helped +4.6. Try expanding to pick_remaining <= 1 (nearly complete orders).
3. **Optimal route planning**: Single-bot TSP optimization — evaluate more permutations.

### Medium (Gap: 35 points)
1. **Higher concentration bonus**: Currently 8. Try 10, 12 for Medium-specific build.
2. **Preview pre-picking**: More aggressive preview pickup before order completes.
3. **Better max_preview_carriers**: Currently limited. Try expanding for Medium.

### Hard (Gap: 47 points)
1. **Round-trip cost tuning**: Already helped +9 max. Try different weights.
2. **max_pickers tuning**: Currently 3 for Hard. Try 4 (was tried globally, hurt Easy — but now per-difficulty!).
3. **Delivery batching**: Ensure bots don't rush with 1 item.

## Analysis Insights (from game log analysis)
- Expert first order takes ~100 rounds (1/3 of game!)
- Expert: 12.7% waits, 49 pickups, 20 drop_offs per game
- Order completion gaps: some orders take 58 rounds, others take 6 rounds
- Big variance in order difficulty — depends on item location distribution

## Competition Timeline
- Competition starts **March 19, 2026** (18 days away)
- Current total: **495** (target: 700, need +205)
- Roughly +51 per difficulty needed

## Build & Test
```bash
cd grocery-bot-zig
python build_all.py          # Build all 5 executables
python build_all.py expert   # Build just expert
python sweep_expert.py       # Expert (port 9880)
python sweep_hard.py         # Hard (port 9870)
```
