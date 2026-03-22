# Daemon System -- Technical Reference

Autonomous round detection, exploration, prediction, and iterative submission. Runs unattended 24/7.

---

## Architecture

```mermaid
stateDiagram-v2
    [*] --> Polling: daemon starts

    Polling --> RoundDetected: active round found
    Polling --> CalDownload: completed round found
    Polling --> Polling: no changes (60s sleep)

    RoundDetected --> Exploring: run_adaptive_exploration()
    Exploring --> SimFitting: fit_to_observations()
    SimFitting --> Predicting: gemini_predict() x 5 seeds
    Predicting --> Submitting: client.submit() x 5 seeds
    Submitting --> Resubmit: round still active?

    Resubmit --> SimFitting: yes (GPU resubmit loop)
    Resubmit --> Polling: no (round ended)

    CalDownload --> SaveGT: download analysis_seed_*.json
    SaveGT --> Polling: calibration data saved
```

```mermaid
flowchart LR
    subgraph Concurrent["Concurrent Processes"]
        D["daemon.py<br/>Round Monitor"]
        AL["autoloop_fast.py<br/>Parameter Optimizer"]
        RES["multi_researcher.py<br/>Algorithm Discovery"]
        WEB["Web Dashboard<br/>SvelteKit"]
    end

    D -->|"reads"| BP[best_params.json]
    AL -->|"writes"| BP
    D -->|"writes"| CAL[(calibration/)]
    AL -->|"reads"| CAL
    D -->|"logs"| DL[daemon.log]
    AL -->|"logs"| ALL[autoloop_fast_log.jsonl]
    RES -->|"writes"| IDEAS[multi_ideas/]
    WEB -->|"reads"| DL
    WEB -->|"reads"| ALL
```

---

## Round Monitor

```python
while True:
    rounds = client.get_rounds()
    for round in rounds:
        if round.status == "active" and round.id not in submitted:
            run_submission(client, round.id, detail)
            submitted.add(round.id)
        elif round.status == "completed" and round.number not in calibrated:
            download_round_analysis(client, round.id, round.number)
            calibrated.add(round.number)
    sleep(check_interval)  # default 60s
```

---

## Submission Pipeline (`run_submission()`)

```mermaid
sequenceDiagram
    participant D as Daemon
    participant CAL as CalibrationModel
    participant E as Explore
    participant API as Competition API
    participant SIM as GPU Simulator
    participant P as PredictGemini
    participant U as Utils

    D->>CAL: from_all_rounds() (17 rounds)
    D->>E: run_adaptive_exploration()
    E->>API: POST /simulate x 50 queries
    API-->>E: 15x15 viewport observations
    E->>E: Build FK Buckets + Multipliers
    E-->>D: observations, fk_buckets, global_mult

    D->>D: compute_expansion_radius()
    D->>D: estimate_vigor()

    D->>SIM: detect_regime_from_obs()
    SIM-->>D: regime (collapse/moderate/boom)
    D->>SIM: fit_to_observations(n_sims=5000)
    SIM-->>D: fitted params

    loop For each seed (0-4)
        D->>SIM: run(fitted_params, n_sims=10000)
        SIM-->>D: sim_prediction (40,40,6)
        D->>U: build_growth_front_map()
        D->>U: build_obs_overlay()
        D->>U: build_sett_survival()
        D->>P: gemini_predict(all evidence)
        P-->>D: prediction (40,40,6)
        D->>API: POST /submit(seed, prediction)
    end
```

### Step 1: Load Fresh Calibration
```python
cal = CalibrationModel.from_all_rounds()
predict._calibration = cal  # monkey-patch global singleton
```

### Step 2: Adaptive Exploration
```python
exploration = run_adaptive_exploration(client, round_id, detail)
# Returns: global_multipliers, fk_buckets, multi_store, variance_regime, observations
```

### Step 3: Compute Expansion Radius
```python
exp_radius = compute_expansion_radius(obs_list, detail)
# Returns: dict {distance: (sett_count, total_count)} per seed
```

Iterates through observations, computes Manhattan distance from each observed cell to nearest initial settlement, then accumulates settlement counts per distance bucket.

### Step 4: Estimate Vigor
```python
est_vigor = sett_observations / total_dynamic_observations
# Used for regime-conditional calibration
```

### Step 5: Simulator Inference
```python
# Detect regime from observations
regime = detect_regime_from_obs(obs_list, terrain)

# Choose GPU or CPU
use_gpu = torch.cuda.is_available()
fit_sims = 5000 if use_gpu else 500    # MC samples per CMA-ES eval
pred_sims = 10000 if use_gpu else 2000  # MC samples for final prediction

# Fit simulator parameters
sim_params, _ = fit_to_observations(rd, obs_list, n_sims=fit_sims,
                                     max_evals=200, use_gpu=use_gpu)

# Generate predictions per seed
for seed_idx in range(5):
    sim_predictions[seed_idx] = sim.run(sim_params, n_sims=pred_sims)
```

### Step 6: Build Per-Seed Evidence
```python
for seed_idx in range(5):
    growth_front_maps[seed_idx] = build_growth_front_map(seed_obs, terrain)
    obs_overlays[seed_idx] = build_obs_overlay(obs_list, terrain, seed_idx)
    sett_survivals[seed_idx] = build_sett_survival(obs_list, settlements, seed_idx)
```

### Step 7: Predict and Submit
```python
for seed_idx in range(5):
    prediction = gemini_predict(
        state, global_mult, fk_buckets,
        sim_pred=sim_predictions.get(seed_idx),
        sim_alpha=adaptive_alpha,
        growth_front_map=growth_front_maps.get(seed_idx),
        obs_overlay=obs_overlays.get(seed_idx),
        sett_survival=sett_survivals.get(seed_idx),
        est_vigor=est_vigor,
        obs_expansion_radius=exp_radius,
    )
    errors = validate_prediction(prediction)
    if errors:
        prediction = apply_floor(prediction)
    client.submit(round_id, seed_idx, prediction.tolist())
```

---

## GPU Resubmission Loop (`gpu_resubmit_round()`)

Called periodically while round is active. Each iteration uses a different strategy.

### Strategies by Iteration

| Iteration | Strategy |
|-----------|----------|
| 0 | Base: CMA-ES with default budget + alpha=0.25 |
| 1 | Tighter sigma (0.2), more evaluations |
| 2 | Different warm start (KNN neighbors only) |
| 3+ | Wider sigma (0.8), exploration mode |

### Iterative Improvement Process

```python
while round_is_active():
    # 1. Reload observations
    observations = load_all_obs(round_id)

    # 2. Rebuild multipliers and FK buckets (may have changed)
    gm = GlobalMultipliers()
    fk = FeatureKeyBuckets()
    for obs in observations:
        accumulate(gm, fk, obs)

    # 3. Re-fit simulator with different strategy per iteration
    sim_params = fit_varied_strategy(iteration)

    # 4. Predict and submit
    for seed in range(5):
        pred = gemini_predict(state, gm, fk,
                              sim_pred=sim.run(sim_params),
                              sim_alpha=adaptive_alpha)
        client.submit(round_id, seed, pred)

    iteration += 1
    wait(submission_cooldown)
```

---

## Calibration Download (`download_round_analysis()`)

```python
def download_round_analysis(client, round_id, round_number):
    cal_dir = data/calibration/round{N}/

    if analysis_seed_0.json exists:
        return  # Already downloaded

    detail = client.get_round_detail(round_id)
    save(cal_dir / "round_detail.json", detail)

    for seed in range(seeds_count):
        analysis = client.get_analysis(round_id, seed)
        save(cal_dir / f"analysis_seed_{seed}.json", analysis)
        # Contains: ground_truth (40,40,6), initial_grid, score
```

---

## Logging

All actions logged to `data/daemon.log` with timestamps:

```
[22:15:30] [INFO] Checking for active rounds...
[22:15:31] [INFO]   Round 18 detected (active)
[22:15:31] [INFO]   Calibration: 17 rounds, 136000 cells
[22:15:32] [INFO]   Starting adaptive exploration...
[22:15:45] [INFO]   Variance regime: MODERATE
[22:15:45] [INFO]   Multipliers: sett=0.823, port=0.156, forest=1.102
[22:15:45] [INFO]   Observed expansion radius: {0: (12, 15), 1: (8, 42), ...}
[22:15:46] [INFO]   Using params: prior_w=5.86, T_high=1.00, score_avg=89.389
[22:15:46] [INFO]   Estimated vigor: 0.0712
[22:15:47] [INFO]   Simulator: using GPU
[22:15:48] [INFO]   Simulator regime=moderate, alpha=0.35
[22:16:15] [INFO]   Simulator params fitted: base_surv=-0.47, exp_str=0.52
[22:16:35] [INFO]   Simulator predictions: 5 seeds, 10000 sims each
[22:16:36] [INFO]   Seed 0: ok (sett=0.0682)
[22:16:36] [INFO]   Seed 1: ok (sett=0.0715)
...
[22:16:38] [INFO]   All 5 seeds submitted for R18
```

---

## Error Handling

| Error | Handling |
|-------|----------|
| API 429 (rate limit) | Exponential backoff with retry |
| API timeout | Log warning, skip submission, retry next cycle |
| GPU OOM | Fall back to CPU simulator |
| Validation errors | Apply floor and re-normalize, then submit |
| Missing observations | Submit with calibration-only prediction (no obs enrichment) |
| Simulator fit failure | Log warning, submit without sim blend (sim_alpha=0) |
