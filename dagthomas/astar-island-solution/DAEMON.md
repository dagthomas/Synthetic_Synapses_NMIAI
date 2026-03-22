# The Daemon: Autonomous AI Competition System

## What It Does

The daemon is a fully autonomous system that competes in the Astar Island challenge — a real-time prediction competition where Norse civilization simulators run on hidden parameters and you have to predict the outcome from limited observations.

It runs 24/7 with zero human intervention. It detects new rounds, explores the simulation, fits a GPU-accelerated Monte Carlo simulator, generates predictions, submits them, and then **keeps improving and re-submitting** for the entire 165-minute round window.

While it does this, it simultaneously runs a continuous parameter optimization loop that has executed over **1 million experiments**, and two AI research agents that have generated **500+ algorithmic ideas**.

## System Architecture

```mermaid
graph TB
    subgraph Daemon["daemon.py — The Brain"]
        RM[Round Monitor<br/>Polls API every 90s]
        HM[Health Monitor<br/>Auto-restart crashed processes]
        PS[Param Sync<br/>Autoloop → production every 2 min]
    end

    subgraph LiveRound["Live Round Pipeline"]
        EX[Explore<br/>50 queries, 5 seeds<br/>Adaptive entropy-targeted]
        RD[Regime Detection<br/>Collapse / Moderate / Boom]
        SP[Statistical Prediction<br/>Calibration + FK + Multipliers<br/>+ Sett Survival]
        GP[GPU Simulator<br/>CMA-ES fitting, 10K Monte Carlo<br/>RTX 5090, 23x speedup]
        EN[Ensemble<br/>α adapts by regime<br/>0.15 collapse → 0.65 boom]
        SUB[Submit → API]
        RESUB[Iterative Re-submit<br/>Every 10 min, more compute<br/>Up to 10 iterations]
    end

    subgraph Background["Background Processes"]
        AL[Autoloop<br/>1M+ experiments<br/>44 params, 160K/hr]
        MR[Multi-Researcher<br/>Gemini Pro + Flash<br/>500+ ideas generated]
        GR[Gemini Researcher<br/>Structural algorithm changes<br/>1,247 iterations]
    end

    subgraph Data["Shared State"]
        BP[best_params.json<br/>44 tuned parameters]
        CAL[Calibration Data<br/>20 rounds, 160K cells]
        TD[Transfer Data<br/>GPU-fitted sim params<br/>KNN warm-starts]
    end

    RM -->|New round| EX
    EX --> RD
    RD --> SP
    RD --> GP
    SP --> EN
    GP --> EN
    EN --> SUB
    SUB --> RESUB
    RESUB -->|Every 10 min| GP

    AL -->|Writes| BP
    BP -->|Reads| SP
    CAL -->|Trains| SP
    TD -->|Warm-starts| GP
    HM -->|Restarts| AL

    RM -->|Completed round| CAL

    style Daemon fill:#1a1a2e,color:#e0e0e0
    style LiveRound fill:#16213e,color:#e0e0e0
    style Background fill:#0f3460,color:#e0e0e0
    style Data fill:#533483,color:#e0e0e0
```

## The Round Pipeline

```mermaid
sequenceDiagram
    participant API as Astar Island API
    participant D as Daemon
    participant E as Explorer
    participant S as Statistical Model
    participant G as GPU Simulator
    participant A as Autoloop

    D->>API: Poll for active round
    API-->>D: Round 21 detected!
    D->>A: Pause autoloop
    D->>E: Run adaptive exploration

    E->>API: 50 viewport queries
    API-->>E: Observed settlements, terrain
    E-->>D: Observations + regime (BOOM)

    par Statistical + GPU Sim
        D->>S: Predict (calibration + FK + multipliers)
        S-->>D: stat_pred (40x40x6)
        D->>G: CMA-ES fit from observations (8s)
        G-->>D: sim_pred (40x40x6)
    end

    D->>D: Ensemble (0.65*sim + 0.35*stat)
    D->>API: Submit prediction (t+2 min)
    D->>A: Resume autoloop

    loop Every 10 minutes
        D->>G: Re-fit with more compute
        G-->>D: Better sim_pred
        D->>D: New ensemble
        D->>API: Re-submit (overwrites previous)
    end

    API-->>D: Round closes (t+165 min)
    D->>API: Download ground truth
    D->>A: Restart with new calibration data
```

## The Iterative Re-submission Innovation

Most competition systems submit once and hope for the best. This system exploits the fact that predictions can be overwritten while a round is active:

```mermaid
flowchart TB
    OPEN[Round Opens] --> EXPLORE[Explore<br/>50 queries, ~2 min]
    EXPLORE --> SUBMIT1[Initial Submit<br/>Statistical + GPU sim<br/>t = 2 min]
    SUBMIT1 --> R0[GPU Resubmit iter 0<br/>2000 sims, 200 CMA evals<br/>t = 12 min]
    R0 --> R1[GPU Resubmit iter 1<br/>2500 sims, 300 CMA evals<br/>t = 22 min]
    R1 --> R2[GPU Resubmit iter 2<br/>3000 sims, 400 CMA evals<br/>t = 32 min]
    R2 --> RN[...<br/>Up to 10 iterations<br/>Each with more compute]
    RN --> FINAL[Final Resubmit<br/>6500 sims, 1100 CMA evals<br/>t = ~140 min]
    FINAL --> CLOSE[Round Closes<br/>t = 165 min<br/>Last submission wins]

    style SUBMIT1 fill:#e17055,color:#fff
    style FINAL fill:#00b894,color:#fff
    style CLOSE fill:#0984e3,color:#fff
```

Each iteration uses a different random seed and more compute, gradually converging on better simulator parameters. The system never regresses because each iteration also carries forward all the observation-corrected statistical model features.

## GPU Simulator

```mermaid
flowchart TB
    subgraph Input
        OBS[50 Observations<br/>Settlement positions<br/>Terrain snapshots]
        WS[KNN Warm Start<br/>3 nearest historical rounds<br/>Averaged params]
    end

    subgraph CMA["CMA-ES Optimizer (8s)"]
        EVAL["Evaluate params:<br/>Run 2000-6500 Monte Carlo sims<br/>Compare to observed cells<br/>Log-likelihood objective"]
        PERTURB["Perturb 16 parameters:<br/>base_survival, expansion_str,<br/>expansion_scale, decay_power,<br/>max_reach, forest_resist, ..."]
        EVAL --> PERTURB --> EVAL
    end

    subgraph GPU["RTX 5090 (124K sims/s)"]
        MC["Monte Carlo Simulation<br/>1. Sample settlement survival<br/>2. Compute nearest-alive distance<br/>3. Expansion with Gaussian decay<br/>4. Forest clearing/reclamation<br/>5. Accumulate cell counts"]
    end

    subgraph Output
        PRED["Prediction Tensor<br/>40x40x6 probabilities<br/>per settlement type"]
    end

    OBS --> CMA
    WS --> CMA
    CMA --> GPU
    GPU --> PRED

    style GPU fill:#76b900,color:#000
```

The heart of the system is a PyTorch CUDA Monte Carlo simulator that models the Norse civilization dynamics:

- **16 hidden parameters** per round: survival rates, expansion strength, decay power, max reach, coastal modifiers, forest resistance, ruin rates
- **Gaussian-power distance decay**: `P(expand|d) = str * exp(-(d/scale)^power)` with hard cutoff
- **5000-10000 Monte Carlo samples** per evaluation
- **23x speedup** over CPU (5000 sims in 40ms on RTX 5090)
- **CMA-ES fitting** from 50 viewport observations in ~8 seconds

The simulator captures spatial dynamics that the statistical model cannot — particularly the sharp settlement cutoff at distance boundaries that varies 14x between rounds.

## Regime Detection

```mermaid
flowchart TB
    OBS[25 Scout Queries] --> VIGOR{Observed<br/>Settlement Rate}
    VIGOR -->|"< 2%"| COLLAPSE[COLLAPSE<br/>α = 0.15<br/>Trust statistical model]
    VIGOR -->|"2-15%"| MODERATE[MODERATE<br/>α = 0.30<br/>Balanced ensemble]
    VIGOR -->|"> 15%"| BOOM[BOOM<br/>α = 0.65<br/>Trust GPU simulator]

    COLLAPSE --> PRED1[Low settlement prediction<br/>Conservative expansion]
    MODERATE --> PRED2[Mixed prediction<br/>Moderate expansion]
    BOOM --> PRED3[High settlement prediction<br/>Wide expansion with ports]

    style COLLAPSE fill:#2d3436,color:#dfe6e9
    style MODERATE fill:#636e72,color:#dfe6e9
    style BOOM fill:#d63031,color:#dfe6e9
```

This classification happens within the first 25 queries (~1 minute) and shapes the entire prediction strategy.

## Self-Improvement Loop

```mermaid
flowchart TB
    subgraph RN["Round N (Live)"]
        SUBMIT_N[Submit Prediction]
    end

    subgraph After["After Round Closes"]
        GT[Download Ground Truth]
        CAL_UPDATE[Update Calibration Model]
        SIM_FIT[GPU-fit Sim Params]
        AL_RESTART[Restart Autoloop<br/>with new data]
    end

    subgraph RN1["Round N+1 (Live)"]
        BETTER[Better Prediction<br/>More training data<br/>Better params<br/>Better KNN warm-starts]
    end

    SUBMIT_N --> GT
    GT --> CAL_UPDATE
    GT --> SIM_FIT
    CAL_UPDATE --> AL_RESTART
    SIM_FIT --> AL_RESTART
    AL_RESTART --> BETTER

    style RN fill:#00b894,color:#000
    style After fill:#fdcb6e,color:#000
    style RN1 fill:#00b894,color:#000
```

Every completed round makes the next round better. Ground truth is downloaded, calibration model is updated (now 20 rounds, 160K cells), simulator parameters are fitted for KNN warm-starts, and the autoloop restarts with the expanded dataset.

## Numbers

| Metric | Value |
|--------|-------|
| Autoloop experiments | 1,028,171 |
| Parameters optimized | 44 (continuous) |
| Calibration rounds | 20 |
| Ground truth cells | 160,000 |
| GPU sim speed | 124,000 sims/sec |
| CMA-ES fitting time | ~8 seconds |
| Round processing time | ~2 min (initial), then iterative |
| Re-submissions per round | Up to 10 |
| Uptime | 24/7 autonomous |

## What Makes This Special

1. **Fully autonomous** — no human touches the system during competition. It detects, explores, predicts, submits, and improves on its own.

2. **Self-improving** — the autoloop continuously finds better parameters. Each new round's ground truth is automatically downloaded and used to improve future predictions.

3. **GPU-accelerated** — the RTX 5090 runs a full Monte Carlo simulator 23x faster than CPU, enabling real-time parameter fitting during live rounds.

4. **Iterative** — instead of submit-and-pray, it keeps improving predictions for the entire round window. More compute = better predictions.

5. **Ensemble architecture** — combines a fast statistical model (milliseconds) with a physics-informed simulator (seconds), getting the best of both worlds.
