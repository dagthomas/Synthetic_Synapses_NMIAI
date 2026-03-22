# Development Workflow

Daily development loop for each challenge.

---

## Astar Island

### Improving Predictions

1. **Check autoloop progress** -- review `best_params.json` score and experiment count
2. **Analyze weak rounds** -- identify boom vs non-boom score gaps
3. **Propose structural changes** -- modify `predict_gemini.py` or `calibration.py`
4. **Backtest locally** -- run against calibration data (17 rounds x 5 seeds)
5. **If improvement:** update autoloop parameter space if new knobs added
6. **Wait for next round** -- daemon auto-submits with latest params

### Adding New Features

1. Add parameter to `best_params.json` schema
2. Implement in `predict_gemini.py` (production) and `fast_predict.py` (vectorized)
3. Add to autoloop perturbation space in `autoloop_fast.py`
4. Backtest shows improvement before committing

### Monitoring Active Rounds

The daemon handles everything automatically. To intervene:
- Check `data/daemon.log` for submission status
- Use web dashboard for real-time scores
- Manual submission: `python submit.py` (for custom strategies)

---

## NorgesGruppen

### Improving Scores

1. **Evaluate current submission** -- `python evaluate.py` for train set metrics
2. **Identify weakness** -- detection mAP vs classification mAP
3. **Train improved model** -- one GPU job at a time
4. **Test locally** -- `python synth_test.py`
5. **Build submission** -- zip with required files
6. **Submit** -- upload at `app.ainm.no` (max 3/day)

### Pre-submission Checklist

- [ ] `run.py` starts with torch.load monkeypatch
- [ ] No blocked imports (`os`, `sys`, `subprocess`, etc.)
- [ ] Zip under 420 MB
- [ ] Max 3 weight files, max 10 Python files
- [ ] Weight files trained with `ultralytics==8.1.0` and `timm==0.9.12`
- [ ] Tested locally: completes in <285s

---

## General Rules

- **Never run concurrent GPU training** -- sequential only, CUDA OOM risk
- **Pin package versions** -- match sandbox exactly
- **Save ideas to IDEAS.md** -- document findings and hypotheses
- **Use `-u` flag** -- all Python daemon/autoloop processes need unbuffered output
