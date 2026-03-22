# Getting Started

Setup and operation guide for the Astar Island competition system.

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (RTX 5090 for full performance)
- PyTorch 2.10+ with CUDA support
- Google Cloud API key (for Gemini models in research agents)

---

## Setup

```bash
cd dagthomas/astar-island-solution
pip install -r requirements.txt
```

### Environment Variables

Create `.env` (gitignored):
```
GOOGLE_API_KEY=<your-key>
ASTAR_TOKEN=<competition-jwt>
GCP_PROJECT_ID=<project>
GCP_LOCATION=us-central1
```

---

## Running

### Full Autonomous Mode

```bash
python daemon.py
```

Runs 24/7: round detection, exploration, prediction, submission, re-submission, autoloop, and research agents.

### Individual Components

```bash
# Parameter optimization only
python autoloop_fast.py

# Multi-model research
python multi_researcher.py

# Gemini structural research
python gemini_researcher.py

# Manual submission for a specific round
python submit.py --round <round_id>

# Backtest current model
python eval_production.py
```

### Backtesting

```bash
# Evaluate against all historical rounds
python eval_production.py

# Test a specific experimental function
python test_ideas.py
```

---

## Key Files to Modify

| Goal | File |
|------|------|
| Change prediction logic | `predict_gemini.py` |
| Tune parameters | `best_params.json` (or let autoloop do it) |
| Change exploration strategy | `explore.py` |
| Modify GPU simulator | `sim_model_gpu.py` |
| Add calibration data | `calibration.py` |

---

## Monitoring

- Autoloop progress: `data/autoloop_fast_log.jsonl`
- Research ideas: `learnings/` directory
- Daemon health: `data/daemon.log`
- Round results: check competition dashboard at app.ainm.no
