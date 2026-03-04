# Production Run Guide

Everything a developer needs to run each difficulty end-to-end, from initial capture to
optimized replay submission.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Game Rules & Difficulties](#game-rules--difficulties)
5. [The Production Pipeline (Capture → Optimize → Replay)](#the-production-pipeline)
6. [Running Each Difficulty](#running-each-difficulty)
   - [Easy (1 bot)](#easy-1-bot)
   - [Medium (3 bots)](#medium-3-bots)
   - [Hard (5 bots)](#hard-5-bots)
   - [Expert (10 bots)](#expert-10-bots)
7. [GPU DP Solver (Offline Optimization)](#gpu-dp-solver-offline-optimization)
8. [Zig Bot (Live Capture)](#zig-bot-live-capture)
9. [Replay](#replay)
10. [Solution Store](#solution-store)
11. [SvelteKit Dashboard](#sveltekit-dashboard)
12. [Sweeps & Statistical Testing](#sweeps--statistical-testing)
13. [Troubleshooting](#troubleshooting)

---

## Getting a Live Token

Tokens are obtained from https://app.ainm.no/challenge (requires login).
Claude Code can fetch tokens automatically using the Pinchtab browser skill:

1. Open https://app.ainm.no/challenge in Chrome
2. Click the difficulty button (e.g. "Hard")
3. Copy the `wss://game.ainm.no/ws?token=...` URL that appears

**Claude Code browser automation** (use `pinchtab` skill):
- Navigate to https://app.ainm.no/challenge
- Click the difficulty button
- Extract the token field value

Token format: `wss://game.ainm.no/ws?token=<JWT>`
JWT payload contains: `map_seed`, `difficulty`, `exp` (expiry ~10 min)

---

## Quick Reference

```bash
# 1. Build Zig bot (all difficulties)
cd grocery-bot-zig && python build_all.py

# 2. Capture a live game (Zig bot plays, logs all orders seen)
cd grocery-bot-gpu
python zig_capture.py "wss://game.ainm.no/ws?token=TOKEN" medium

# 3. Optimize offline with GPU DP
python gpu_sequential_solver.py medium --capture --no-filler --refine-iters 2

# 4. Replay the best saved solution
python replay_solution.py "wss://game.ainm.no/ws?token=TOKEN" --difficulty medium
```

---

## Architecture Overview

```
Live Server (WebSocket)
        │
        ▼
┌─────────────────┐      ┌──────────────────────────────────┐
│  Zig Bot        │──────│  game_log_<ts>.jsonl             │
│  (capture run)  │      │  (every round logged)            │
└─────────────────┘      └──────────────┬─────────────────┘
                                        │ zig_capture.py parses
                                        ▼
                         ┌──────────────────────────────────┐
                         │  solutions/<diff>/capture.json   │
                         │  (grid, items, drop_off, orders) │
                         └──────────────┬─────────────────┘
                                        │
                          gpu_sequential_solver.py
                                        │
                                        ▼
                         ┌──────────────────────────────────┐
                         │  solutions/<diff>/best.json      │
                         │  (300-round action sequence)     │
                         │  solutions/<diff>/meta.json      │
                         └──────────────┬─────────────────┘
                                        │
                          replay_solution.py
                                        │
                                        ▼
                         Live Server (replay with desync correction)
```

---

## Prerequisites

### Python (GPU solver side)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install websockets numpy
```

- Python 3.11+
- CUDA-capable GPU strongly recommended (RTX class). CPU fallback works but is much slower.
- All GPU scripts auto-detect `cuda` vs `cpu`.

### Zig (bot side)

- Zig 0.15.2 at `C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe`
- Build from `grocery-bot-zig/` directory.

### Build the Zig bot

```bash
cd grocery-bot-zig

# Build all 5 executables (auto, easy, medium, hard, expert)
python build_all.py

# Or build a single difficulty
C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe build -Doptimize=ReleaseFast -Ddifficulty=expert

# Output: zig-out/bin/grocery-bot-<difficulty>.exe
#         zig-out/bin/grocery-sim-<difficulty>.exe
```

---

## Game Rules & Difficulties

| Difficulty | Bots | Grid   | Item Types | Order Size | Port  |
|------------|------|--------|------------|------------|-------|
| Easy       | 1    | 12×10  | 4          | 3–4 items  | 9850  |
| Medium     | 3    | 16×12  | 8          | 3–5 items  | 9860  |
| Hard       | 5    | 22×14  | 12         | 3–5 items  | 9870  |
| Expert     | 10   | 28×18  | 16         | 4–6 items  | 9880  |

Key rules:
- **300 rounds**, 120s wall clock, 2s per-round timeout.
- **INV_CAP = 3** items per bot. Dead inventory is permanent (no discard).
- Items stay on shelves permanently — never deplete.
- Score: **+1 per item delivered**, **+5 per order completed**.
- Active + preview order visible at all times. Future orders revealed as previous complete.
- Auto-delivery: when an order completes, ALL bots on the dropoff tile auto-deliver
  matching inventory to the new active order.
- Competition seeds are **fixed per difficulty per day**. Seeds reset at midnight UTC.
- Leaderboard = sum of best scores across all 4 difficulties.

---

## The Production Pipeline

The full pipeline runs in three phases. Each phase builds on the last.

### Phase 1 — Live Capture

Run the Zig bot on the live server. It plays reactively (per-round decisions), scores
110–130, and — crucially — logs every order ID it sees across all 300 rounds.

```bash
cd grocery-bot-gpu
python zig_capture.py "wss://game.ainm.no/ws?token=TOKEN" medium
```

Output saved to `solutions/medium/capture.json`.

**Why Zig and not the Python planner?**
The Python planner is a precompute-all strategy — it plans 300 rounds at round 0 with
only 2 known orders. That gives it fake filler orders, so after the first 2 real orders
the plan is wrong → dead inventory → score ~18. The Zig bot is reactive, so it always
acts on real data and captures the full order stream.

**Iterative capture is valuable.** Each replay run also captures new orders not seen
before. Run: capture → optimize → replay → re-optimize to keep growing the order list.
More orders = better DP solutions.

### Phase 2 — GPU DP Optimization

With a full capture (30+ orders), run the GPU DP solver to find an (near-)optimal
sequence of 300 actions per bot.

```bash
cd grocery-bot-gpu
python gpu_sequential_solver.py medium --capture --no-filler --refine-iters 2
```

`--no-filler` is **critical**: without it the DP plans around fake filler orders and
wastes moves, causing sim scores to diverge from live (179 sim → 151 live vs
166 sim = 166 live with `--no-filler`).

If a solution already exists and you want to improve it:

```bash
python gpu_sequential_solver.py medium --capture --warm-start --no-filler --refine-iters 3
```

### Phase 3 — Replay

Send the saved optimal action sequence to the live server with adaptive desync
correction.

```bash
python replay_solution.py "wss://game.ainm.no/ws?token=TOKEN" --difficulty medium
```

The replayer also merges any new orders it sees back into `capture.json`, enabling
Phase 2 → 3 to be re-run with more data.

---

## Running Each Difficulty

### Easy (1 bot)

**Characteristics:** Single bot, small 12×10 grid, 4 item types. GPU DP is provably
optimal — it exhaustively explores all reachable states.

#### Capture

```bash
python zig_capture.py "wss://EASY_WS_URL" easy
```

#### Optimize

```bash
# Standard: exact DP, finds optimal
python gpu_sequential_solver.py easy --capture --no-filler

# Or with seed (no capture required):
python gpu_sequential_solver.py easy --seed 7001 --no-filler
```

No refinement iterations needed for single-bot (no inter-bot conflicts).

Typical solve time: 3–5 seconds on RTX 5090 with 500K max states.

**Expected scores:**
| Method | Score |
|--------|-------|
| Zig bot live | 116–131 |
| GPU DP (sim) | 163 |
| GPU DP (live) | 137 |

#### Replay

```bash
python replay_solution.py "wss://EASY_WS_URL" --difficulty easy
```

---

### Medium (3 bots)

**Characteristics:** 3 bots, 16×12 grid, 8 item types. Sequential DP: plan bot 0 solo,
bot 1 with bot 0 locked, bot 2 with both locked. Then iterative refinement.

#### Capture

```bash
python zig_capture.py "wss://MEDIUM_WS_URL" medium
```

Run 2–3 times to build up a larger order list (each run discovers 2–3 new orders).

#### Optimize

```bash
python gpu_sequential_solver.py medium --capture --no-filler --refine-iters 2
```

With warm start (improve existing solution):

```bash
python gpu_sequential_solver.py medium --capture --warm-start --no-filler --refine-iters 3
```

Multi-restart (try different bot planning orders):

```bash
python gpu_sequential_solver.py medium --capture --no-filler --restarts 3
```

Typical solve time: ~130s per bot pass on RTX 5090 with 1M max states.

**Expected scores:**
| Method | Score |
|--------|-------|
| Zig bot live | 110–122 |
| GPU DP (sim, 100K states) | 186 |
| GPU DP (live) | 166 |

#### Replay

```bash
python replay_solution.py "wss://MEDIUM_WS_URL" --difficulty medium
```

---

### Hard (5 bots)

**Characteristics:** 5 bots, 22×14 grid, 12 item types. More inter-bot conflicts make
sequential planning and refinement critical.

#### Capture

```bash
python zig_capture.py "wss://HARD_WS_URL" hard
```

#### Optimize

```bash
python gpu_sequential_solver.py hard --capture --no-filler --refine-iters 2
```

Default max_states for hard: 2,000,000. Reduce if OOM.

```bash
python gpu_sequential_solver.py hard --capture --no-filler --max-states 500000
```

**Expected scores:**
| Method | Score |
|--------|-------|
| Zig bot live | 86–119 |
| GPU DP (sim, 200K states) | 164 |
| GPU DP (live) | TBD |

#### Replay

```bash
python replay_solution.py "wss://HARD_WS_URL" --difficulty hard
```

---

### Expert (10 bots)

**Characteristics:** 10 bots, 28×18 grid, 16 item types. Most complex. Sequential DP
with restarts is the best approach. Each bot pass is fast because it plans one bot with
9 others locked; the challenge is collision displacement across refinement iterations.

#### Capture

```bash
python zig_capture.py "wss://EXPERT_WS_URL" expert
```

#### Optimize

```bash
# Standard
python gpu_sequential_solver.py expert --capture --no-filler --refine-iters 2

# With restarts (tries default, reversed, and random bot orderings)
python gpu_sequential_solver.py expert --capture --no-filler --restarts 3 --refine-iters 2

# Warm start from existing
python gpu_sequential_solver.py expert --capture --warm-start --no-filler --refine-iters 3
```

Default max_states for expert: 5,000,000.

**Expected scores:**
| Method | Score |
|--------|-------|
| Zig bot live | 65–96 |
| GPU DP (sim, 100K states) | 131 |
| GPU DP (live) | TBD |

#### Replay

```bash
python replay_solution.py "wss://EXPERT_WS_URL" --difficulty expert
```

---

## GPU DP Solver (Offline Optimization)

### Single-bot (Easy) — Exact DP

`GPUBeamSearcher.dp_search()` in `gpu_beam_search.py`.

For single-bot, this is a **provably optimal BFS/DP** on CUDA:
- Each round: expand all ~7 actions from all unique states.
- Dedup by int64 hash of `(x, y, inv[3], active_idx, active_del[6])`.
- State count peaks at ~200K. All states processed on GPU each round.
- Score is deterministic from state — no parallel tracking needed.
- ~3.4s for 300 rounds on RTX 5090.

### Multi-bot — Sequential DP + Iterative Refinement

`solve_sequential()` in `gpu_sequential_solver.py`.

**Pass 1 — Sequential planning:**
```
bot 0: solo DP → plan0
bot 1: DP with bot 0 locked at plan0 positions → plan1
bot 2: DP with bot 0,1 locked → plan2
...
```

**Pass 2+ — Per-bot refinement:**
```
For each bot i:
  Lock all other bots at their current best actions
  Re-run DP for bot i
  CPU verify: if new_score > best_score → keep; else revert
Repeat until no improvement or max_refine_iters
```

Locked positions are computed by CPU-simulating all locked bots, producing
`locked_pos_x[num_locked, 300]` and `locked_pos_y[num_locked, 300]` tensors
uploaded to GPU for collision exclusion.

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--capture` | off | Load from `solutions/<diff>/capture.json` |
| `--seed N` | — | Use synthetic game seed instead of capture |
| `--no-filler` | off | **CRITICAL**: use only captured orders, no random filler |
| `--refine-iters N` | 2 | Refinement iterations after Pass 1 |
| `--restarts N` | 1 | Try N bot orderings (default, reversed, random) |
| `--warm-start` | off | Load existing best.json and run refinement only |
| `--max-states N` | auto | Override per-bot state budget |

### State budget defaults

| Difficulty | Default max_states |
|------------|--------------------|
| Easy | 500,000 |
| Medium | 1,000,000 |
| Hard | 2,000,000 |
| Expert | 5,000,000 |

---

## Zig Bot (Live Capture)

The Zig bot is a reactive, per-round decision maker. It never precomputes anything —
each round it reads the current state, assigns items to bots, plans trips, and sends
actions.

### Running directly

```bash
# From grocery-bot-zig/
zig-out/bin/grocery-bot-expert.exe "wss://EXPERT_WS_URL"
```

### Running via Python capture wrapper

```bash
# From grocery-bot-gpu/
python zig_capture.py "wss://URL" expert
```

This wrapper:
1. Detects the right executable (`grocery-bot-expert.exe`).
2. Runs the bot and watches for new `game_log_*.jsonl` files.
3. Parses the log, extracting all unique order IDs seen.
4. **Merges** with `solutions/expert/capture.json` (never overwrites existing orders).

The game log is a JSONL file with one JSON object per line — alternating `game_state`
server messages and `{"actions": [...]}` client responses.

### Sweep (statistical testing)

```bash
cd grocery-bot-zig
python sweep_easy.py    # 40 seeds, port 9850
python sweep_medium.py  # 40 seeds, port 9860
python sweep_hard.py    # 40 seeds, port 9870
python sweep_expert.py  # 40 seeds, port 9880

# Custom
python sweep.py expert --seeds 40 --port 9880
```

Sweeps auto-record to PostgreSQL. Never use `--no-record`.

---

## Replay

`replay_solution.py` loads `solutions/<diff>/best.json` and replays it with adaptive
desync correction.

### Desync correction strategy

Before replay starts, the replayer pre-simulates all 300 rounds to compute
`expected_positions[round][bot_id]`.

Per round:
1. Check if all bot positions match expected (synced).
2. If not synced, check if positions match `round - 1` (missed round). If yes,
   increment `round_offset` and adjust the DP index.
3. If genuinely desynced (bot moved to wrong cell), still send the DP action — the
   goal-based BFS fallback handles navigation correction automatically on the next rounds.

```bash
python replay_solution.py "wss://URL" --difficulty easy
python replay_solution.py "wss://URL"                   # auto-detects from bot count
```

Replay also captures any new orders seen during the game and merges them into
`capture.json`.

---

## Solution Store

All per-difficulty solutions stored in `grocery-bot-gpu/solutions/<difficulty>/`:

| File | Content |
|------|---------|
| `capture.json` | Map grid, items, drop_off, all observed orders, bot count |
| `best.json` | 300-round action sequence `[[(action_int, item_idx), ...], ...]` |
| `meta.json` | Score, date, seed, capture hash, optimization count |

### Key behaviors

- `save_solution()` **never overwrites a better score** (even with `force=False`).
- Capture files are merged, never replaced — each run adds newly seen orders.
- `meta.json` tracks `optimizations_run` counter across refinement cycles.

### Inspecting current solutions

```python
from solution_store import get_all_solutions
for diff, meta in get_all_solutions().items():
    if meta:
        print(f"{diff}: score={meta['score']}, date={meta['date']}")
```

---

## SvelteKit Dashboard

Visual dashboard for live games, GPU solve progress, and replay.

```bash
# Start PostgreSQL (game history DB)
cd grocery-bot-zig/replay
docker compose up -d db

# Start dashboard
cd grocery-bot-zig/replay/app
npm run dev
# Open http://localhost:5173
```

Pages:
- `/` — replay viewer (grid visualization, action timeline)
- `/live` — live game via MAPF planner (auto-selects difficulty exe)
- `/gpu` — Matrix-style GPU solve dashboard (score graph, state exploration viz, terminal)

---

## Sweeps & Statistical Testing

Sweeps run the Zig bot across 40 seeds and record to PostgreSQL for statistical
comparison of parameter changes.

```bash
cd grocery-bot-zig
python sweep_easy.py     # seeds 7001–7040
python sweep_medium.py
python sweep_hard.py
python sweep_expert.py
```

Game log cleanup: sweeps keep only the 15 latest `game_log_*.jsonl` files per directory.

**Evaluation metric:** MAX score matters most (competition takes best per map). Mean
shows consistency. Use both when comparing changes.

---

## Troubleshooting

### `--no-filler` is essential

Without `--no-filler`, the DP solver pads the order list with 100 random fake orders.
The bot wastes moves on fake pickups. Sim score may be 179 but live is 151. With
`--no-filler`, sim = live (e.g., 166 = 166).

### Dead inventory (score ~18 on live)

Caused by precomputing 300 rounds at round 0 with only 2 known orders. The Python
precompute planner (`live_solver.py` without `--save-capture`) does this. Use:
- Zig bot for capture (reactive, never plans ahead).
- `live_solver.py --save-capture` for reactive Python capture.
- GPU DP for offline optimization.
- `replay_solution.py` for live replay.

### Score much lower live than sim

1. Check for desync: replay logs `SYNC` / `OFF<N>` per round. More than a few
   `OFF` rounds indicates the action sequence is misaligned.
2. Re-run capture: the capture may have too few orders. Run 3–5 Zig captures,
   each merges new orders.
3. Re-optimize with `--warm-start` after getting more orders.

### CUDA OOM

Reduce `--max-states`:

```bash
python gpu_sequential_solver.py expert --capture --no-filler --max-states 500000
```

### Zig build errors

- Use Zig 0.15.2 exactly. The API changed significantly in 0.14 → 0.15.
- `std.Thread.sleep()` not `std.time.sleep()`.
- `std.io.getStdOut()` does not exist — use `std.debug.print` (stderr).

### WebSocket connection issues

- The Zig bot adds a 15ms delay between receive and send. This prevents the 1-round
  action-offset desync that occurred on fast connections.
- If desyncs persist, check network latency. The replay script has zero send delay
  (`SEND_DELAY = 0.0`) — can be tuned if needed.
