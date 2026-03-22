# Running the Astar Island System

## Quick Start

```bash
cd astar-island-solution
cp .env.example .env   # Fill in ASTAR_TOKEN and GOOGLE_API_KEY

# Start everything (daemon manages autoloop automatically):
python -u daemon.py

# In separate terminals, start researchers:
python -u multi_researcher.py
python -u gemini_researcher.py
```

The `-u` flag is critical — it enables unbuffered output for real-time log monitoring.

## Architecture

```
daemon.py (orchestrator)
  ├── autoloop_fast.py (subprocess, auto-managed)
  │   └── Optimizes 44 prediction params → writes best_params.json
  └── Main loop (every 90s):
      ├── Detect active round → explore → submit
      ├── GPU sim re-submit every 10 min (iterative improvement)
      ├── Download calibration from completed rounds
      ├── Sync best params from autoloop → best_params.json
      └── Health-check autoloop (restart on crash)

multi_researcher.py (independent)
  └── Gemini Pro generates prediction code ideas, backtests them

gemini_researcher.py (independent)
  └── Gemini Flash proposes structural algorithm changes
```

## Per-Round Pipeline (fully automatic)

```
t=0:00  Round detected
t=0:02  Exploration complete (50 queries across 5 seeds)
t=0:02  Statistical prediction (predict_gemini.py) + GPU sim ensemble → submit
t=0:02  Autoloop resumes optimizing params
t=0:12  GPU re-submit iter 0 (2000 sims, different random seed)
t=0:22  GPU re-submit iter 1 (2500 sims, more CMA-ES evals)
  ...   (every 10 min, up to 10 iterations)
t=2:45  Round closes, daemon downloads GT for calibration
```

Key insight: the API allows **re-submission** while a round is active. We submit fast first, then iteratively improve with more GPU sim compute.

## Processes

### daemon.py — Main Orchestrator

```bash
python -u daemon.py                      # Full system (default)
python -u daemon.py --no-autoloop        # Monitor only, no optimization
python -u daemon.py --check-interval 60  # Faster polling (default: 90s)
```

**Manages:**
- Round detection and submission
- Autoloop subprocess (auto-restart on crash)
- GPU sim iterative re-submission
- Calibration data download
- Parameter sync (autoloop → best_params.json)

**Logs:** `data/daemon.log`

### autoloop_fast.py — Parameter Optimizer

```bash
python -u autoloop_fast.py              # Run indefinitely (managed by daemon)
python -u autoloop_fast.py --seeds 1    # Fast mode (1 seed/round)
python -u autoloop_fast.py --seeds 5    # Accurate mode (default)
python autoloop_fast.py --summary       # Print current best and exit
```

Continuously perturbs 44 prediction parameters, backtests against all calibration rounds, keeps the best. Writes results to `best_params.json` via daemon sync.

**Logs:** `data/autoloop_fast_output.log`, `data/autoloop_fast_log.jsonl`

### multi_researcher.py — AI Code Generator

```bash
python -u multi_researcher.py                # Run indefinitely
python -u multi_researcher.py --max-iters 50 # Limited iterations
python multi_researcher.py --summary         # Print results
```

Uses Gemini Pro to generate prediction function variants, backtests them, saves winning ideas to `data/multi_ideas/`.

**Requires:** `GOOGLE_API_KEY`
**Logs:** `data/multi_researcher_output.log`, `data/multi_research_log.jsonl`

### gemini_researcher.py — Structural Research

```bash
python -u gemini_researcher.py               # Run indefinitely
python -u gemini_researcher.py --max-iters 20
python gemini_researcher.py --summary
```

Proposes structural algorithm changes (not just parameter tweaks).

**Requires:** `GOOGLE_API_KEY` or `GEMINI_API_KEY`
**Logs:** `data/gemini_researcher_output.log`, `data/gemini_research_log.jsonl`

## Environment Variables

| Variable | Required By | Purpose |
|----------|------------|---------|
| `ASTAR_TOKEN` | daemon, submit, client | JWT token for Astar Island API |
| `GOOGLE_API_KEY` | researchers | Google Gemini API key |
| `GEMINI_API_KEY` | gemini_researcher | Fallback Gemini key |

All loaded from `.env` file in the solution directory.

## Key Files

| File | Purpose |
|------|---------|
| `best_params.json` | Current best prediction parameters (auto-updated) |
| `config.py` | Map dimensions (40x40), API base URL, terrain codes |
| `predict_gemini.py` | Production prediction function (reads best_params.json) |
| `sim_model_gpu.py` | GPU-accelerated Monte Carlo simulator (PyTorch CUDA) |
| `sim_inference.py` | CMA-ES parameter fitting for simulator |
| `calibration.py` | Hierarchical calibration model from historical GT |
| `explore.py` | Observation query strategies (adaptive/multi-sample) |
| `client.py` | API client (auth, query, submit) |
| `utils.py` | Feature extraction, observation processing |

## Data Directory Structure

```
data/
  ├── calibration/roundN/          # GT data per round (auto-downloaded)
  │   ├── round_detail.json        # Map, settlements, initial states
  │   └── analysis_seed_N.json     # Ground truth + submitted prediction + score
  ├── rounds/{uuid}/               # Observation data per round
  │   ├── obs_s{seed}_q{query}.json
  │   ├── estimates.json
  │   └── initial_states.json
  ├── sim_cache/                   # Autoloop simulation cache
  ├── sim_params/transfer_data.json # Fitted sim params for KNN warm-starts
  ├── multi_ideas/                 # Researcher winning ideas
  ├── daemon.log                   # Daemon activity log
  ├── autoloop_fast_output.log     # Autoloop stdout
  └── autoloop_fast_log.jsonl      # Autoloop experiment log (JSON lines)
```

## GPU Requirements

- **RTX 5090** (or any CUDA GPU with 8+ GB VRAM)
- PyTorch with CUDA support
- Used for: GPU simulator (sim_model_gpu.py), CMA-ES fitting
- Not required: autoloop and statistical model are pure CPU numpy

## Dependencies

```
numpy scipy cma torch
google-genai          # For researchers (Gemini API)
```

## Monitoring

```bash
# Watch daemon activity
tail -f data/daemon.log

# Watch autoloop progress
tail -f data/autoloop_fast_output.log

# Check current best params
python -c "import json; print(json.dumps(json.load(open('best_params.json')), indent=2))"

# Check autoloop best score
python autoloop_fast.py --summary

# Check researcher results
python multi_researcher.py --summary
```

## Troubleshooting

**Autoloop not starting:** Daemon auto-restarts it every 5 min. Check `data/daemon.log`.

**API rate limit (429):** Daemon handles this — waits and retries.

**GPU not detected:** Falls back to CPU simulator (slower but works). Check `torch.cuda.is_available()`.

**Stale autoloop:** If code changes aren't picked up, kill the autoloop PID — daemon will restart with fresh code.

**Token expired:** Update `ASTAR_TOKEN` in `.env`. The client auto-validates on startup.
