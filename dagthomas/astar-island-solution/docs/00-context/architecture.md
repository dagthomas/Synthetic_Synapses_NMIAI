# Architecture

Full system architecture for the Astar Island autonomous competition system.

---

## System Overview

```mermaid
flowchart TB
    subgraph Daemon["daemon.py"]
        RM[Round Monitor]
        HM[Health Monitor]
        PS[Param Sync]
    end

    subgraph Pipeline["Live Round Pipeline"]
        EX[explore.py]
        SP[predict_gemini.py]
        GP[sim_model_gpu.py]
        EN[Ensemble]
        SUB[submit.py]
    end

    subgraph Research["Background Research"]
        AL[autoloop_fast.py]
        MR[multi_researcher.py]
        GR[gemini_researcher.py]
    end

    subgraph Data["Shared State"]
        BP[best_params.json]
        CAL[Calibration Data]
    end

    RM -->|New round| EX
    EX --> SP
    EX --> GP
    SP --> EN
    GP --> EN
    EN --> SUB

    AL -->|Optimizes| BP
    BP -->|Reads| SP
    CAL -->|Trains| SP
    HM -->|Restarts| AL
    PS -->|Syncs| BP
```

---

## Data Flow Per Round

```mermaid
sequenceDiagram
    participant API as Astar Island API
    participant D as Daemon
    participant E as Explorer
    participant S as Statistical Model
    participant G as GPU Simulator

    D->>API: Poll for active round
    API-->>D: Round detected
    D->>E: Run 50 viewport queries

    par Statistical + GPU
        E->>S: Observations + multipliers
        S-->>D: stat_pred (40x40x6)
        E->>G: CMA-ES fit from observations
        G-->>D: sim_pred (40x40x6)
    end

    D->>D: Ensemble blend
    D->>API: Submit prediction

    loop Every 10 minutes
        D->>G: Re-fit with more compute
        D->>API: Re-submit (overwrites)
    end
```

---

## Key Files

| File | Purpose |
|------|---------|
| `daemon.py` | 24/7 orchestrator — round detection, health monitoring, param sync |
| `explore.py` | Adaptive viewport exploration (50 queries, entropy-targeted) |
| `predict_gemini.py` | Statistical prediction — calibration + FK buckets + multipliers |
| `sim_model_gpu.py` | PyTorch CUDA Monte Carlo simulator |
| `submit.py` | API submission with iterative re-submission |
| `autoloop_fast.py` | Continuous parameter optimization (160K experiments/hr) |
| `multi_researcher.py` | Gemini Flash + Pro research loop |
| `gemini_researcher.py` | Structural algorithm proposals |
| `calibration.py` | Hierarchical calibration from ground truth |
| `best_params.json` | 44 tuned continuous parameters |

---

## Memory Model

All shared state is file-based — no database, no message queue:

| File | Updated by | Read by |
|------|-----------|---------|
| `best_params.json` | Autoloop (every improvement) | Statistical model, researchers |
| `data/calibration_*.json` | Daemon (after round closes) | Statistical model |
| `data/sim_params_*.json` | GPU simulator | KNN warm-starts |
| `learnings/*.json` | Multi-researcher | Gemini researcher |

Processes coordinate through the filesystem. The daemon syncs autoloop parameters to production every 2 minutes.
