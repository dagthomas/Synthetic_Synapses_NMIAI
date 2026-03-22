# Autonomous AI Research: Machines That Do Science

> "The most interesting thing about AI right now is not that it can answer questions — it's that it can ask them."
> — Andrej Karpathy

## The Game: Astar Island

Astar Island is a beautifully designed AI challenge from the Norwegian AI Championship (NM i AI). It's a prediction game about Norse civilizations on a hidden simulator — and it's one of the best competitive AI problems we've ever seen.

### The Rules

You observe a 40x40 island through a 15x15 viewport. The island has terrain (land, forest, mountains, ocean) and Norse settlements. A hidden simulator runs the civilization forward — settlements survive or die, expand into new territory, clear forests, build ports on the coast, or collapse into ruins.

**You get 50 observation queries** across 5 random seeds of the same map. Each query shows you a snapshot of the simulation mid-run through your viewport. From these 50 glimpses, you must predict the **final probability distribution** of every cell on the 40x40 map — a 40x40x6 tensor where each cell has probabilities for: empty, settlement, port, ruin, forest, farmland.

**Scoring uses entropy-weighted KL divergence** — cells where the outcome is uncertain (high entropy) matter more than cells where the outcome is predictable. This rewards models that capture the uncertainty structure, not just the most likely outcome.

**5 seeds** share the same terrain but have different random outcomes. You submit a prediction for each seed. 50 queries total shared across all seeds — so you must balance coverage vs depth.

### Why It's Brilliant

```mermaid
mindmap
    root((Astar Island))
        Hidden Parameters
            16 simulator params vary per round
            Lambda varies 14x between rounds
            Can't be detected from initial state
            Must be estimated from observations
        Information Theory
            Entropy-weighted scoring
            High uncertainty cells matter most
            Rewards calibrated probabilities
            Punishes overconfidence
        Exploration vs Exploitation
            50 queries budget
            5 seeds to cover
            Viewport positioning matters
            Regime detection from early queries
        Stochastic Dynamics
            Settlement survival is probabilistic
            Expansion follows distance decay
            Forest clearing and reclamation
            Port formation on coasts
        Multi-Seed Reasoning
            Same terrain, different outcomes
            Cross-seed information sharing
            Variance estimation
            Regime classification
```

What makes Astar Island exceptional as a competitive AI problem:

1. **Hidden state estimation** — you can't see the simulator parameters. You must infer them from partial observations. This is the core challenge of science itself.

2. **Information-theoretic scoring** — KL divergence rewards well-calibrated probabilities, not just correct guesses. You must know what you don't know.

3. **Exploration under budget** — 50 queries is not enough to see everything. Where you look matters. This creates a natural explore-exploit tradeoff.

4. **Non-stationary dynamics** — each round has completely different hidden parameters. What worked last round might fail this round. The system must adapt in real-time.

5. **Multi-scale reasoning** — you need local features (what terrain surrounds each cell), global features (overall settlement vigor), and spatial dynamics (how far settlements expand).

It's the kind of problem where naive approaches score 20-40, good statistical models score 70-85, and the full autonomous research system we built scores 82-93. The gap between "good" and "great" requires understanding the hidden physics of the simulation.

## The Vision

Andrej Karpathy has been vocal about a paradigm shift: **AI systems that don't just execute tasks, but conduct research autonomously**. His vision of "auto-research" — systems that propose hypotheses, design experiments, run them, analyze results, and iterate — is exactly what we've built here.

This isn't prompt engineering. This isn't a chatbot. This is a system where AI models are the scientists, the code is the lab equipment, and the competition leaderboard is the peer review.

## What We Built

```mermaid
graph TB
    subgraph Agents["Three Autonomous Research Agents"]
        direction LR
        AL["Autoloop<br/>━━━━━━━━━━━━━━<br/>Brute Force Genius<br/>1,028,171 experiments<br/>44 parameters<br/>160K/hour"]
        MR["Multi-Researcher<br/>━━━━━━━━━━━━━━<br/>The Creative One<br/>617 iterations<br/>497 ideas generated<br/>32 breakthrough ideas"]
        GR["Gemini Researcher<br/>━━━━━━━━━━━━━━<br/>Structural Thinker<br/>1,247 iterations<br/>Algorithm redesigns<br/>Physics-informed"]
    end

    subgraph Shared["Shared Knowledge"]
        CODE[Prediction Codebase]
        PARAMS[best_params.json]
        IDEAS[Idea Archive<br/>500+ tested variants]
        CAL[Calibration Data<br/>20 rounds GT]
    end

    AL -->|Optimizes| PARAMS
    MR -->|Generates| IDEAS
    GR -->|Proposes changes to| CODE
    PARAMS -->|Baseline for| MR
    PARAMS -->|Baseline for| GR
    IDEAS -->|Inspires| GR
    CODE -->|Tests against| CAL

    style AL fill:#e17055,color:#fff
    style MR fill:#00b894,color:#fff
    style GR fill:#0984e3,color:#fff
```

### 1. The Autoloop — Brute Force Genius (1,028,171 experiments)

```mermaid
flowchart LR
    subgraph Loop["Infinite Loop (160K experiments/hour)"]
        direction TB
        BEST[Current Best Params<br/>44 continuous values]
        PERTURB[Perturb 1-3 params<br/>Gaussian noise]
        EVAL[Backtest against<br/>20 rounds × 5 seeds<br/>Vectorized numpy]
        COMPARE{Better?}
        ACCEPT[Accept + Sync<br/>to production]
        REJECT[Reject + Continue]
    end

    BEST --> PERTURB --> EVAL --> COMPARE
    COMPARE -->|Yes| ACCEPT --> BEST
    COMPARE -->|No| REJECT --> BEST

    style ACCEPT fill:#00b894,color:#fff
    style REJECT fill:#d63031,color:#fff
```

The autoloop takes the prediction function, parameterizes 44 continuous variables, and searches relentlessly for better configurations. It runs at **160,000 experiments per hour** using vectorized numpy operations.

What it found that humans wouldn't have:
- `mult_power_sett = 0.53` — settlement-specific multiplier power, different from general 0.19
- `cal_fine_divisor = 125` — calibration smoothing 25% higher than the default 100
- `growth_front_boost = 0.74` — young settlements signal 2.5x stronger than assumed
- `floor = 0.0034` — probability floor 2.4x lower than the safe default of 0.008

These aren't intuitive. No human would set `mult_power_sett` to exactly 0.53. But across a million experiments, this is what the data demands.

### 2. Multi-Model Researcher — The Creative One (617 iterations, 497 ideas)

```mermaid
sequenceDiagram
    participant F as Gemini Flash<br/>(Analyst)
    participant P as Gemini Pro<br/>(Coder)
    participant H as Backtest Harness<br/>(Judge)
    participant L as Idea Log<br/>(Memory)

    loop Every 25 seconds
        F->>L: Read experiment history
        F->>F: Identify error patterns
        F-->>P: "Settlement over-predicted at d=4+<br/>on collapse rounds. Try adding<br/>distance-dependent suppression."

        P->>P: Write complete prediction function
        P-->>H: experimental_pred_fn()

        H->>H: Backtest against 20 rounds
        H-->>L: Score: 86.7 ✓ GOOD

        L-->>F: Updated experiment history
    end
```

This is where it gets Karpathy-level interesting. Two Gemini models collaborate:

**Gemini Flash** (the analyst): Looks at the experiment log, identifies error patterns, and proposes a research direction in natural language:

> "The settlement class contributes 64% of KL error. Distance-ring analysis shows over-prediction at d=4+ on collapse rounds. Direction: add a distance-dependent settlement suppression factor that activates when observed vigor < 0.05."

**Gemini Pro** (the coder): Takes that direction and writes a complete prediction function — not a tweak, but a full reimplementation with the proposed change.

Out of 497 ideas generated:
- **32 scored "good"** (>86.6, beating the baseline)
- **166 scored "ok"** (>80.0, competitive)
- The best idea scored **87.0**

```mermaid
pie title 497 Generated Ideas by Outcome
    "Good (>86.6)" : 32
    "OK (>80.0)" : 166
    "Failed (error/crash)" : 149
    "Below baseline" : 150
```

### 3. Gemini Researcher — The Structural Thinker (1,247 iterations)

While the multi-researcher makes incremental improvements, the Gemini researcher proposes **structural algorithm changes**:

- "Replace distance-based multiplier with a diffusion field that accounts for terrain barriers"
- "Add a Dirichlet-Multinomial conjugate update for directly observed cells"
- "Implement cluster density as an inverted-U survival factor"

These are the kind of ideas a PhD student might have after weeks of thinking about the problem. The AI generates them in seconds, tests them in minutes, and moves on.

## The Research Speed Multiplier

```mermaid
graph LR
    subgraph Human["Human Researcher"]
        H1[Read data<br/>1 hour] --> H2[Form hypothesis<br/>30 min]
        H2 --> H3[Write code<br/>2 hours]
        H3 --> H4[Run experiment<br/>10 min]
        H4 --> H5[Analyze<br/>1 hour]
        H5 --> H1
    end

    subgraph AI["AI Researcher"]
        A1[Analyze errors<br/>2 sec] --> A2[Propose hypothesis<br/>5 sec]
        A2 --> A3[Generate code<br/>15 sec]
        A3 --> A4[Run backtest<br/>3 sec]
        A4 --> A5[Log + iterate<br/>instant]
        A5 --> A1
    end

    style Human fill:#d63031,color:#fff
    style AI fill:#00b894,color:#fff
```

| | Human | AI System | Speedup |
|---|---|---|---|
| Time per iteration | ~5 hours | ~25 seconds | **720x** |
| Ideas per day | 3-4 | 3,456 | **1,000x** |
| Ego / confirmation bias | Yes | None | - |
| Will test "stupid" ideas | Rarely | Always | - |
| Runs at 3 AM | No | Yes | - |

But it's not just speed. It's **fearlessness**. A human researcher has ego, intuition, and confirmation bias. They'll avoid testing "stupid" ideas. The AI has none of that. It will cheerfully test "increase port factor by 0.001" right after "completely replace the distance model with a gravity formulation." Some of those "stupid" ideas turn out to be the 87.0-scoring winners.

## The Compound Effect

```mermaid
flowchart TD
    AL1["Autoloop finds<br/>growth_front_boost = 0.74"]
    MR1["Multi-researcher discovers<br/>growth front + cluster density<br/>together score 86.7"]
    AL2["Autoloop optimizes<br/>cluster_optimal = 0.70<br/>cluster_quad_pen = -0.12"]
    GR1["Gemini researcher proposes<br/>terrain barriers for cluster model"]
    AL3["Autoloop optimizes<br/>barrier_strength = 0.13"]
    GPU["GPU simulator built from<br/>lambda discovery (varies 14x)"]
    ENSEMBLE["Ensemble: stat + sim<br/>+3 to +14 points"]

    AL1 --> MR1 --> AL2 --> GR1 --> AL3
    MR1 -.->|"Insight: hidden params<br/>vary 14x between rounds"| GPU
    GPU --> ENSEMBLE

    style AL1 fill:#e17055,color:#fff
    style AL2 fill:#e17055,color:#fff
    style AL3 fill:#e17055,color:#fff
    style MR1 fill:#00b894,color:#fff
    style GR1 fill:#0984e3,color:#fff
    style GPU fill:#76b900,color:#000
    style ENSEMBLE fill:#fdcb6e,color:#000
```

What makes this system more than the sum of its parts is how the three agents compound. Each works on a different timescale:

- **Autoloop**: milliseconds per experiment (brute force search)
- **Multi-researcher**: seconds per idea (creative code generation)
- **Gemini researcher**: minutes per structural proposal (deep algorithmic thinking)

They don't communicate directly — they share a codebase and a parameter file. The autoloop picks up structural changes from the researchers, and the researchers see autoloop-optimized baselines. **Emergent collaboration.**

## The GPU Simulator: When Research Becomes Engineering

The most impactful discovery from the research agents was identifying the **hidden parameter problem**: the simulation has 16 parameters that vary 14x between rounds, and they can't be predicted from the initial state.

```mermaid
graph LR
    subgraph Discovery["Research Discovery"]
        D1["Lambda varies 14x<br/>between rounds"]
        D2["Controls settlement<br/>expansion radius"]
        D3["Not detectable from<br/>initial state"]
        D4["Estimable from 50<br/>observations (~7% error)"]
    end

    subgraph Engineering["Engineering Response"]
        E1["PyTorch CUDA simulator<br/>16-param Monte Carlo"]
        E2["CMA-ES fitting<br/>8 seconds on RTX 5090"]
        E3["23x speedup<br/>124K sims/sec"]
        E4["Ensemble: +3 to +14 pts<br/>per round"]
    end

    D1 --> D2 --> D3 --> D4
    D4 --> E1 --> E2 --> E3 --> E4

    style Discovery fill:#0984e3,color:#fff
    style Engineering fill:#76b900,color:#000
```

This wasn't in any plan. It emerged from the research agents discovering that "lambda varies 14x and is the key differentiator between easy and hard rounds." The AI found the problem; the human-AI team built the solution.

## The Full System in Action

```mermaid
timeline
    title 48 Hours of Autonomous Competition
    section Hour 0-6 : System deployed : Autoloop starts optimizing : Researchers begin generating ideas
    section Hour 6-12 : 200K experiments completed : First breakthrough idea (86.7) : Calibration: 15 rounds
    section Hour 12-18 : Round 17 scores 93.0 (best ever) : Lambda discovery (varies 14x) : Settlement survival bias identified
    section Hour 18-24 : GPU simulator built (23x speedup) : Iterative re-submission added : Bug found and fixed (empty gm/fk)
    section Hour 24-36 : R19 submitted (collapse, 82.5) : R20 submitted (moderate, 89.4) : 900K+ experiments total
    section Hour 36-48 : R21 submitted automatically : 1M experiments milestone : 20 rounds calibration : System fully autonomous
```

## Results

| Round | Type | Score | What Happened |
|-------|------|-------|---------------|
| R17 | Boom | 93.0 | Best ever — well-calibrated boom prediction |
| R20 | Moderate | 89.4 | Solid — fixed obs correction bug |
| R19 | Collapse | 82.5 | GPU sim saved it from 54.6 stat-only |
| R18 | Boom | 70.3 | Hard round — unusual port expansion |

## The Numbers

```mermaid
graph LR
    subgraph Scale["Scale of Autonomous Research"]
        direction TB
        N1["1,028,171<br/>Autoloop experiments"]
        N2["1,864<br/>Research iterations"]
        N3["497<br/>Code variants generated"]
        N4["32<br/>Breakthrough ideas"]
        N5["20<br/>Rounds of calibration"]
        N6["160,000<br/>Ground truth cells"]
        N7["124,000<br/>GPU sims/second"]
        N8["24/7<br/>Autonomous operation"]
    end

    style Scale fill:#2d3436,color:#dfe6e9
```

## What This Proves (The Karpathy Thesis)

Karpathy is right. The future of AI isn't chatbots — it's autonomous research systems that:

```mermaid
mindmap
    root((Auto-Research))
        Identify Problems
            From data, not prompts
            Error pattern analysis
            Cross-round correlation
        Generate Hypotheses
            Ideas humans wouldn't think of
            No ego or confirmation bias
            Fearlessly tests "stupid" ideas
        Test Immediately
            Zero friction from idea to experiment
            25 seconds per full cycle
            3,456 ideas per day
        Iterate at Machine Speed
            1000x faster than human research
            Runs at 3 AM
            Never gets tired or discouraged
        Compound Discoveries
            Multiple agents, different timescales
            Emergent collaboration
            Each discovery enables the next
```

We built a system that competed in a real-time AI competition autonomously for 48+ hours, ran over a million experiments, generated 500 algorithmic ideas, and adapted its strategy based on what it learned from each round.

The game itself — Astar Island — is the perfect testbed for this. Its hidden parameters, information-theoretic scoring, and real-time adaptation requirements demand exactly the kind of autonomous, self-improving system that Karpathy envisions.

**This is auto-research. This is the future.**
