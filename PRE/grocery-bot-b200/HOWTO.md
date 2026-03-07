# B200 Competition Pipeline

Automated pipeline for all 5 difficulties: Easy, Medium, Hard, Expert, Nightmare.
Handles token fetching, game capture, offline optimization, replay, and iteration.

## Prerequisites

1. **Google OAuth login** (one-time setup):
   ```bash
   cd grocery-bot-gpu
   python fetch_token.py hard --setup
   ```
   This opens a browser window. Log in with Google, then press Enter.

2. **PostgreSQL** running at `localhost:5433`:
   ```bash
   # Check connection
   psql postgres://grocery:grocery123@localhost:5433/grocery_bot -c "SELECT 1"
   ```

3. **Zig bot built** (for initial capture):
   ```bash
   cd grocery-bot-zig
   python build_all.py
   ```

4. **Python deps**: `torch`, `numpy`, `psycopg2`, `playwright`, `websockets`

## Quick Start: Full Competition Day

```bash
cd grocery-bot-b200
python competition_day.py --gpu 5090
```

This runs all difficulties sequentially with automatic time allocation:

| Difficulty | Time  | Method |
|-----------|-------|--------|
| Easy      | 6 min | Single-bot exact DP |
| Medium    | 1h24m | Stepladder: squad DP + joint DP + LNS |
| Hard      | 4h30m | Stepladder: 120-perm pass-1 + 3-bot joint + LNS |
| Expert    | 5h    | Stepladder: 200-perm pass-1 + 4-bot joint + LNS |
| Nightmare | 2h    | Stepladder: V3 + perturbation search (no GPU DP) |

Final replays run at the end for all difficulties.

## Per-Difficulty Commands

### Easy (trivially optimal)
```bash
python deep_optimize.py easy --budget 300
```

### Medium
```bash
python stepladder.py medium --hours 2
```

### Hard
```bash
python stepladder.py hard --hours 6 --max-states 500000
```

### Expert
```bash
python stepladder.py expert --hours 12 --max-states 200000
```

### Nightmare
```bash
# Option 1: Through b200 pipeline (uses stepladder)
python stepladder.py nightmare --hours 3

# Option 2: Direct sim training (known seed)
cd ../grocery-bot-gpu
python nightmare_offline.py --seed 7005 -v --train-time 300

# Option 3: Live iterate pipeline
cd ../grocery-bot-gpu
python nightmare_offline.py --ws-url "wss://game.ainm.no/ws?token=..." --time-budget 275

# Option 4: Multi-seed sweep
cd ../grocery-bot-gpu
python nightmare_offline.py --seeds 1000-1009 --train-time 120 -v
```

## How the Pipeline Works

### Stepladder Loop (Easy-Expert)

Each token window (~288s):

1. **Fetch token** via Playwright (auto-handles 60s cooldown)
2. **Capture/Replay**: First iteration runs Zig bot for initial capture.
   Subsequent iterations replay the best GPU solution to discover more orders.
3. **GPU optimize**: Sequential DP + squad joint DP within token window
4. **Deep train**: Between token windows, run hours-long offline optimization:
   - Phase 1 (30%): All-permutation pass-1 at explore_states
   - Phase 2 (50%): Deep refinement with squad joint DP
   - Phase 3 (20%): LNS destroy-repair to escape local optima

### Stepladder Loop (Nightmare)

Same token management, but optimization uses **NightmareTrainer**:

1. **Fetch token** + **V3 live game** (captures orders + map)
2. **Offline training**: V3 multi-restart with stochastic perturbations
   + checkpoint-based local search
3. **Replay** trained action sequence on live server
4. **Iterate** with expanded order set from replay discovery

### Why Nightmare is Different

GPU DP scores 1-3 on nightmare (20 bots cause catastrophic collision in
sequential DP). The NightmareTrainer uses:

- **V3 solver**: MRTA task allocation + PIBT pathfinding (baseline ~227 mean)
- **Stochastic restarts**: Randomized stall escapes create different congestion
  patterns, exploring diverse solutions
- **Checkpoint perturbation search**: At saved checkpoints, force one bot to
  take a different action, then re-run V3 to the end. Keeps improvements.
- **Result**: Mean 273 across 10 seeds (+20.6% over V3 baseline)

## Monitoring Progress

### Check Scores
```bash
cd grocery-bot-gpu
python solution_store.py
```

### View in Dashboard
```bash
cd replay/app
npm run dev
# Open http://localhost:5173
```

### Check Stepladder State
```bash
cat grocery-bot-b200/stepladder_state/nightmare.json
```

## Competition Day Schedule (Recommended)

Start at 07:00 UTC. The pipeline runs until ~20:30 UTC.

```bash
cd grocery-bot-b200
python competition_day.py --gpu 5090
```

Or run difficulties in parallel (if multiple GPUs):
```bash
# Terminal 1: GPU work
python stepladder.py hard --hours 6
# Terminal 2: CPU work (nightmare uses CPU only)
python stepladder.py nightmare --hours 6
```

**Note**: Only run ONE GPU task at a time. Nightmare is CPU-only so it can
run in parallel with GPU tasks.

## Manual Token + Replay

```bash
# 1. Get token
cd grocery-bot-gpu
python fetch_token.py nightmare

# 2. Run with token
python nightmare_offline.py --ws-url "wss://game.ainm.no/ws?token=..." --time-budget 275

# Or replay existing solution
python replay_solution.py "wss://game.ainm.no/ws?token=..." --difficulty nightmare
```

## Key Files

| File | Purpose |
|------|---------|
| `competition_day.py` | Full day automation (all difficulties) |
| `stepladder.py` | Token window management + iterate loop |
| `deep_optimize.py` | Hours-long offline optimization |
| `squad_solver.py` | Squad-based GPU DP solver (easy-expert) |
| `b200_config.py` | Per-GPU, per-difficulty parameters |
| `../grocery-bot-gpu/nightmare_offline.py` | Nightmare training pipeline |
| `../grocery-bot-gpu/nightmare_solver_v2.py` | V3 nightmare solver |
| `../grocery-bot-gpu/fetch_token.py` | Playwright token fetcher |
| `../grocery-bot-gpu/replay_solution.py` | WS solution replay |
| `../grocery-bot-gpu/solution_store.py` | PostgreSQL score-safe storage |
