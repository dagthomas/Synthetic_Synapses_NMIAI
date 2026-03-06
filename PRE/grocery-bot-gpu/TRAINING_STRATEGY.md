# Grocery Bot GPU — Training, Replay & Strategy Guide

## Table of Contents

1. [Overview](#overview)
2. [The Core Loop](#the-core-loop)
3. [Order Discovery](#order-discovery)
4. [GPU DP Training (Offline)](#gpu-dp-training-offline)
5. [Sequential Per-Bot DP](#sequential-per-bot-dp)
6. [Multi-Bot Coordination](#multi-bot-coordination)
7. [Live Replay](#live-replay)
8. [Production Pipeline](#production-pipeline)
9. [Tuning Parameters](#tuning-parameters)
10. [Strategies & Findings](#strategies--findings)
11. [Current Scores & Targets](#current-scores--targets)
12. [Known Limitations](#known-limitations)
13. [Command Reference](#command-reference)

---

## Overview

The grocery bot competition (NM i AI, March 19, 2026) is a WebSocket-based game where bots navigate a grid, pick up items from shelves, and deliver them to complete orders. The score = items delivered + 5 per completed order.

**Key insight**: Games are **fully deterministic per day**. Same day + same difficulty = same map, same items, same order sequence. This means we can:
1. Play a game to discover orders
2. Train offline on those orders (unlimited time)
3. Replay the optimal solution on a new token for the same deterministic game

**Hardware**: RTX 5090 (32 GB VRAM), running CUDA via PyTorch 2.10+cu128 with torch.compile/Triton 3.6.0.

---

## The Core Loop

```mermaid
graph LR
    A[CAPTURE<br/>Zig bot plays live] --> B[TRAIN<br/>GPU DP offline]
    B --> C[REPLAY<br/>Send optimal actions]
    C --> D[DISCOVER<br/>2-5 new orders]
    D --> B

    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#ff6b6b,stroke:#333,color:#fff
    style C fill:#51cf66,stroke:#333,color:#fff
    style D fill:#ffd43b,stroke:#333,color:#000
```

> Each cycle discovers 2-5 new orders. More orders + better plans = higher scores.
> Repeat until score plateaus or token expires.

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Initial Capture (Zig Bot)"]
        P1A[Zig bot plays reactively] --> P1B[Score 50-130]
        P1B --> P1C["Discovers 8-20 orders"]
        P1C --> P1D["Output: game_log → capture.json"]
    end

    subgraph Phase2["Phase 2: GPU DP Training (Offline)"]
        P2A[Load capture data] --> P2B[Sequential per-bot DP on CUDA]
        P2B --> P2C[Optimal action sequence for 300 rounds]
    end

    subgraph Phase3["Phase 3: Live Replay"]
        P3A[Send pre-computed actions via WebSocket] --> P3B[Score higher → discover 2-5 more orders]
        P3B --> P3C[Merge new orders into capture data]
    end

    subgraph Phase4["Phase 4: Re-train + Re-replay"]
        P4A[More orders known] --> P4B[GPU DP finds better plans]
        P4B --> P4C[Each cycle builds on the last]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 -->|"Until token expires (~288s)"| Phase2

    style Phase1 fill:#e3f2fd,stroke:#1976d2
    style Phase2 fill:#fce4ec,stroke:#c62828
    style Phase3 fill:#e8f5e9,stroke:#2e7d32
    style Phase4 fill:#fff8e1,stroke:#f57f17
```

---

## Order Discovery

Orders are revealed 2 at a time: 1 active + 1 preview. New orders only appear when the active order is completed (all items delivered). You can only discover new orders by **completing more orders than your previous best run**.

### Order Accumulation

```mermaid
graph LR
    R1["Run 1 (Zig)<br/>Score: 57<br/>Orders 0-7"] -->|"+5 orders"| R2["Run 2 (GPU v1)<br/>Score: 120<br/>Orders 0-12"]
    R2 -->|"+6 orders"| R3["Run 3 (GPU v2)<br/>Score: 160<br/>Orders 0-18"]
    R3 -->|"+3 orders"| R4["Run 4 (GPU v3)<br/>Score: 180<br/>Orders 0-21"]

    style R1 fill:#ffcdd2,stroke:#c62828
    style R2 fill:#fff9c4,stroke:#f57f17
    style R3 fill:#c8e6c9,stroke:#2e7d32
    style R4 fill:#b3e5fc,stroke:#0277bd
```

### Storage
- `solutions/<diff>/capture.json` — capture data with all known orders
- `order_lists/<diff>_orders.json` — persistent order list (survives re-captures)
- Orders from different runs are MERGED — never lose previously discovered orders
- `extract_orders_from_logs.py` — scans all game_log files for missed orders

### Staleness
Orders change daily. Captures from March 5 do NOT work on March 6. Always re-capture on game day or same-day as competition.

---

## GPU DP Training (Offline)

### How It Works

The GPU DP solver (`gpu_beam_search.py`) does **exhaustive BFS with deduplication** on CUDA. For each bot:

```mermaid
flowchart TD
    Init["Round 0: Initial state<br/>(start pos, empty inventory)"] --> Expand

    subgraph BFS["BFS Loop (300 rounds)"]
        Expand["Expand all states<br/>× all valid actions<br/>(move/pickup/dropoff/wait)"] --> Dedup["Deduplicate by hash<br/>(x, y, inv[3], order_idx, delivery_mask)"]
        Dedup --> Prune["Prune to top max_states<br/>by eval score"]
        Prune --> NextRound{Round < 300?}
        NextRound -->|Yes| Expand
    end

    NextRound -->|No| Backtrack["Backtrack from best state<br/>→ optimal action sequence"]

    style Init fill:#e3f2fd,stroke:#1565c0
    style BFS fill:#fafafa,stroke:#9e9e9e
    style Backtrack fill:#c8e6c9,stroke:#2e7d32
```

### State Representation
- Position: (x, y) on the grid
- Inventory: 3 slots, each holding an item type or empty
- Order progress: which active order index, which items delivered
- Packed into int64 hash for GPU-efficient dedup

### Evaluation Heuristic (`_eval`)

```mermaid
flowchart TD
    State["State to Evaluate"] --> Base["Base Score<br/>items × 100K + orders × 500K"]
    State --> Progress["Active Order Progress<br/>partial delivery: 60-70K/item"]
    State --> Inventory["Inventory Value<br/>matching items: 50K each"]
    State --> Trip["Trip Planning<br/>distance gradient penalty"]
    State --> Preview["Preview Speculation<br/>preview items: 30K bonus"]
    State --> Chain["Chain Reaction Potential<br/>auto-delivery bonus"]
    State --> Coord["Coordination Signals"]

    Base --> Total((Total<br/>Eval<br/>Score))
    Progress --> Total
    Inventory --> Total
    Trip --> Total
    Preview --> Total
    Chain --> Total
    Coord --> Total

    Coord --> C1["+30K unique coverage"]
    Coord --> C2["-20K redundancy penalty"]
    Coord --> C3["-4K aisle congestion"]

    style Total fill:#ffd43b,stroke:#f57f17,color:#000
    style Coord fill:#e3f2fd,stroke:#1565c0
```

The heuristic scores states with 100K eval units per game point:
- **Base score**: items_delivered * 100K + orders_completed * 500K
- **Active order progress**: partial delivery value (60K-70K per item)
- **Inventory value**: items matching active order (50K each)
- **Trip planning**: distance to next pickup/dropoff (gradient penalty)
- **Preview speculation**: bonus for holding preview order items (30K)
- **Chain reaction potential**: bonus for items that auto-deliver on order completion
- **Coordination signals** (multi-bot):
  - Unique coverage bonus: +30K for types not held by any locked bot
  - Redundancy penalty: -20K for duplicating locked bot's active items
  - Aisle congestion penalty: -4K per locked bot in same narrow aisle

### torch.compile
- Enabled by default for offline training (3.5x speedup)
- Mode: `default` (not `reduce-overhead`, blocked by tensor lifetime bug)
- Must be DISABLED (`no_compile=True`) for live/threaded contexts

---

## Sequential Per-Bot DP

The multi-bot solver (`gpu_sequential_solver.py`) can't do joint state-space search (exponential in bot count). Instead:

### Pass 1: Sequential Planning

```mermaid
flowchart LR
    B0["Bot 0<br/>Plan alone"] --> B1["Bot 1<br/>Bot 0 locked"]
    B1 --> B2["Bot 2<br/>Bots 0,1 locked"]
    B2 --> B3["Bot 3<br/>Bots 0,1,2 locked"]
    B3 --> B4["Bot 4<br/>Bots 0-3 locked"]

    B0 -.->|"trajectory"| B1
    B1 -.->|"trajectories"| B2
    B2 -.->|"trajectories"| B3
    B3 -.->|"trajectories"| B4

    style B0 fill:#4a9eff,stroke:#333,color:#fff
    style B1 fill:#5cb3ff,stroke:#333,color:#fff
    style B2 fill:#6ec4ff,stroke:#333,color:#fff
    style B3 fill:#80d5ff,stroke:#333,color:#000
    style B4 fill:#92e6ff,stroke:#333,color:#000
```

Multiple orderings are tried (forward, reverse, random) to find the best initial plan. Each "pass1 ordering" produces a different solution, and the best is kept.

### Pass 2+: Iterative Refinement

```mermaid
flowchart TD
    Start([Start Iteration]) --> SelectBot[Select next bot in order]
    SelectBot --> Lock["Lock ALL other bots'<br/>trajectories"]
    Lock --> Replan["Re-plan this bot<br/>with GPU DP"]
    Replan --> Check{New plan<br/>improves total<br/>score?}
    Check -->|Yes| Keep[Keep new plan]
    Check -->|No| Revert[Revert to previous plan]
    Keep --> More{More bots<br/>to process?}
    Revert --> More
    More -->|Yes| SelectBot
    More -->|No| Done([Next Iteration])

    style Start fill:#e8f5e9,stroke:#2e7d32
    style Check fill:#fff3e0,stroke:#e65100
    style Keep fill:#c8e6c9,stroke:#2e7d32
    style Revert fill:#ffcdd2,stroke:#c62828
    style Done fill:#e3f2fd,stroke:#1565c0
```

### Refinement Bot Order
- Iteration 0: forward (bot 0, 1, 2, ...)
- Iteration 1: backward (bot N, N-1, ..., 0)
- Iteration 2+: **weakest-first** — compute marginal contribution of each bot, re-plan the weakest first

### Marginal Contribution
For each bot, simulate the game WITHOUT that bot (replace with wait actions). The contribution = total_score - score_without_bot. Bots with 0 or low contribution are re-planned first since they have the most room to improve.

### Perturbation Escape

```mermaid
flowchart TD
    Stall([Refinement Stalled<br/>No improvement for N iters]) --> Contrib[Compute marginal<br/>contributions of all bots]
    Contrib --> First{First stall?}
    First -->|Yes| Single["Reset 1 weakest bot<br/>to all-wait actions"]
    First -->|No| Pair["Reset 2 weakest bots<br/>to all-wait actions"]
    Single --> Reshuffle[Reshuffle type assignments]
    Pair --> Reshuffle
    Reshuffle --> Replan["Re-plan from<br/>perturbed state"]
    Replan --> Improved{Score<br/>improved?}
    Improved -->|Yes| Continue([Continue Refinement])
    Improved -->|No| Attempts{Escape attempts<br/>< limit?}
    Attempts -->|Yes| Contrib
    Attempts -->|"No (max 6)"| Stop([Accept best score])

    style Stall fill:#ffcdd2,stroke:#c62828
    style Continue fill:#c8e6c9,stroke:#2e7d32
    style Stop fill:#e0e0e0,stroke:#616161
```

### Locked Bot Simulation
When planning bot K, all other bots' trajectories are "locked":
- Pre-simulated to extract per-round positions
- Fed to GPU as collision obstacles
- Bot K's DP accounts for collisions with locked bots
- Zig FFI DLL does this 2.7x faster than Python

---

## Multi-Bot Coordination

### Current Approach (Sequential DP)

```mermaid
mindmap
  root((Multi-Bot<br/>Coordination))
    Eval Heuristic
      Unique coverage bonus
      Redundancy penalty
      Aisle congestion
    Type Specialization
      2-3 types per bot
      Minimize overlap
    Zone Assignments
      Preferred column ranges
      Reduce interference
    Iterative Refinement
      Re-plan with locked trajectories
      Weakest-first ordering
    Contribution Analysis
      Marginal value per bot
      Reset lowest contributors
```

Each bot is planned independently with others locked. Coordination happens through:
- **Eval heuristic**: unique coverage bonus, redundancy penalty, aisle congestion
- **Type specialization**: each bot is assigned 2-3 item types to focus on
- **Zone assignments**: bots given preferred column ranges to reduce physical interference
- **Iterative refinement**: re-planning each bot with updated locked trajectories
- **Contribution analysis**: weakest-first refinement ordering

### Why Sequential DP Has a Ceiling
- Bot 0 plans optimally for itself, but may block Bot 1's best path
- Refinement can fix some conflicts but can't find globally optimal multi-bot plans
- Joint 2-bot DP was attempted but state budget spreads too thin (49x action expansion)
- At 50K states, joint DP scored 72 vs 182 for sequential (on Hard)

### Coordination Signals in Eval
The heuristic tries to compensate for sequential planning:
- **Unique coverage bonus** (+30K + 3K * num_locked_bots per unique type):
  Reward picking up item types not held by any locked bot
- **Redundancy penalty** (-20K - 2K * num_locked_bots):
  Penalize duplicating active order items already in locked bots' inventory
- **Aisle congestion** (-4K per locked bot in same narrow aisle column):
  Avoid physical interference in tight spaces
- **Pipeline mode**: earlier bots get more state budget (depth-based)

### Type Specialization
Each bot is assigned 2-3 item types. The assignment is computed by analyzing order frequency and distributing types across bots to minimize overlap. During perturbation escape, assignments are reshuffled with a different random seed.

### LNS Order Assignment (EXPERIMENTAL — default OFF)
Round-robin order-to-bot assignment via `order_modulo`/`order_slot` on `GPUBeamSearcher`.
Dampens `active_inv_value` for non-assigned orders.

**Result: Hurts score.** Hard seed 42: 155 (LNS) vs 174 (baseline). The rigid modulo assignment
doesn't work with sequential DP — when bot 0 is planned first with no locked bots, it can't
defer delivery to not-yet-planned bots. Existing coordination signals (unique coverage, redundancy
penalty) handle work distribution better because they dynamically analyze locked bot state.

Enable with `use_order_assignment=True` in SolveConfig. May work better with joint optimization.

### Sparse 2-Bot Joint DP (NEW)

```mermaid
flowchart LR
    Dist{Bot Distance?} -->|"≤2 (close)"| Full["Full N×N<br/>action cross-product<br/>(49 combos)"]
    Dist -->|"3-5 (medium)"| Top3["Top-3 × Top-3<br/>proxy-scored<br/>(9 combos)"]
    Dist -->|">5 (far)"| Top1["Top-1 × Top-1<br/>best action each<br/>(1 combo)"]

    Full --> Sparse[".nonzero() sparse expansion<br/>10x fewer states"]
    Top3 --> Sparse
    Top1 --> Sparse

    style Full fill:#ffcdd2,stroke:#c62828
    style Top3 fill:#fff9c4,stroke:#f57f17
    style Top1 fill:#c8e6c9,stroke:#2e7d32
```

Distance-adaptive 2-bot DP for refinement (`GPUBeamSearcher2Bot`):
- Runs every 3 refinement iterations on the 2 weakest bots
- Uses `.nonzero()` sparse expansion → 10x fewer expanded states than dense grid

### What Would Help (Future Work)
- **Joint state-space search** with 200K+ states (needs more VRAM efficiency)
- **Communication between bot DPs** (share order completion timing)
- **Post-planning MAPF** (resolve collisions after DP instead of during)

---

## Live Replay

### `replay_solution.py` — Adaptive Replay

```mermaid
sequenceDiagram
    participant R as Replay Engine
    participant WS as WebSocket Server
    participant FB as Fallback Logic

    loop Each Round (1-300)
        WS->>R: Game state (positions, orders)
        R->>R: Compare actual vs expected state
        alt In sync
            R->>WS: Pre-computed action
        else Position desync
            R->>FB: Goal-based pathfinding
            FB->>WS: Corrective action
        else Order desync
            R->>FB: Adjust action for actual order state
            FB->>WS: Adapted action
        else Unplanned situation
            R->>FB: Greedy fallback
            FB->>WS: Best-effort action
        end
    end
```

### `live_gpu_stream.py` — Anytime Online Solver

```mermaid
flowchart TD
    subgraph Threads["Background Threads"]
        T1["MAPF Planning"]
        T2["GPU Refinement"]
        T3["Per-Round GPU"]
    end

    subgraph MultiBot["Multi-Bot Mode"]
        MB1["MAPF/gpu_refine plan"] --> MB2["Lenient 1/3 sync"]
    end

    subgraph SingleBot["Single-Bot Mode"]
        SB1["Strict sync"] --> SB2["Plan recovery"]
    end

    Threads --> MultiBot
    Threads --> SingleBot

    style Threads fill:#e3f2fd,stroke:#1565c0
    style MultiBot fill:#fff3e0,stroke:#e65100
    style SingleBot fill:#f3e5f5,stroke:#7b1fa2
```

GPU-powered per-round decisions (not pre-computed). Used for live play when no pre-computed solution exists.

### Desync Handling

```mermaid
flowchart LR
    D1["Bot collision drift"] --> Desync((DESYNC))
    D2["Simulator mismatch"] --> Desync
    D3["Network latency"] --> Desync
    Desync --> Recovery["Goal-based correction<br/>5-10 rounds to recover"]

    style Desync fill:#ff8a80,stroke:#c62828
    style Recovery fill:#c8e6c9,stroke:#2e7d32
```

---

## Production Pipeline

### `production_run.py` — Full Automated Pipeline

Within a single 288s token:

```mermaid
gantt
    title Production Pipeline (~288s token window)
    dateFormat s
    axisFormat %S s

    section Cycle 1
    Zig Capture (score ~50-130, 8-20 orders)    :c1, 0, 30s
    GPU Optimize (50K states, 3 orderings)       :g1, after c1, 25s
    Replay + Discover (2-5 new orders)           :r1, after g1, 15s

    section Cycle 2
    GPU Re-optimize (expanded order set)         :g2, after r1, 18s
    Replay + Discover (more orders)              :r2, after g2, 15s

    section Cycle 3+
    GPU Optimize (warm-start)                    :g3, after r2, 18s
    Replay + Discover                            :r3, after g3, 15s
    ... repeat until ~275s ...                   :done, after r3, 15s
```

### Key Flags
- `--no-filler`: CRITICAL — never add fake filler orders (wastes DP capacity)
- `--max-states 50000`: sweet spot for iteration speed vs quality
- `--speed-bonus 100`: reward finishing orders faster (enables more discovery)
- `--time-budget 275`: leave 13s margin before token expires

### `optimize_and_save.py` — Offline Deep Training

```mermaid
flowchart LR
    A["Load capture data<br/>(map, items, orders)"] --> B["GPU DP Solve<br/>(100K states, 3 orderings)"]
    B --> C["Iterative Refinement<br/>(20 iterations)"]
    C --> D{Score improved<br/>vs saved?}
    D -->|Yes| E["Save solution"]
    D -->|No| F["Discard"]

    style E fill:#c8e6c9,stroke:#2e7d32
    style F fill:#ffcdd2,stroke:#c62828
```

For unlimited-time offline optimization:
```bash
python optimize_and_save.py hard \
  --max-time 600 \
  --max-states 100000 \
  --speed-bonus 150 \
  --orderings 3 \
  --refine-iters 20
```

Loads capture data, runs GPU DP, saves only if score improves.

---

## Tuning Parameters

### State Budget (`max_states`)
| Value | Use Case | Notes |
|-------|----------|-------|
| 50K | Pipeline iterations | Best for 288s window (15-30s per bot) |
| 100K | Deep offline training | 7s per bot, diminishing returns above this |
| 200K | Single bot (Easy) | Provably optimal, too slow for multi-bot |

### Order Count
- **Fewer orders = less state fragmentation**
- 20 orders + 50K states > 40 orders + 50K states (proven on Hard)
- `order_cap` per bot limits visible orders (default: 8)

### Speed Bonus
- Rewards completing orders earlier in the game
- Higher = more aggressive early completion = more order discovery
- `speed_bonus=100, speed_decay=0.5` is default
- `speed_bonus=150-200` for offline training

### Refinement Iterations
- Default: easy=0, medium=3, hard=10, expert=10
- Deep training: 20-30 iterations
- Returns diminish sharply after 5-10 for most maps

### Escape Limit
- Pipeline (<60s): 2 escape attempts
- Normal (60-300s): 4 escapes
- Deep (>300s): 6 escapes

---

## Strategies & Findings

### What Works
1. **Iterative order discovery**: Each replay cycle discovers 2-5 new orders
2. **50K states, many iterations**: Better than 100K states, few iterations
3. **3 pass1 orderings**: Forward, reverse, random — essential for Hard/Expert
4. **Type specialization**: Assigning item types to bots reduces interference
5. **Weakest-first refinement**: Contribution analysis identifies which bots need replanning
6. **Chain reactions**: Pre-fetching preview items for 0-round order completions
7. **Aisle congestion penalty**: -4K per locked bot in same aisle column (+7% on Hard)
8. **Zig FFI**: 4.5x faster verification, 2.7x faster presim

### What Doesn't Work
1. **2-bot joint DP**: 49x action expansion starves state budget (72 vs 182)
2. **100K+ states**: Slower iterations, no score improvement over 50K
3. **More orders than needed**: 40 orders fragments state space (172 vs 182)
4. **1200s+ training budgets**: Sequential DP ceiling hit at ~180 regardless of time
5. **10 DP bots at 25K (Expert)**: Better to do 7 DP at 50K + 3 greedy
6. **Greedy bots touching active orders**: Catastrophic interference with DP plans
7. **Pair perturbation**: Resets 2 bots, but they converge to same local optimum
8. **LNS order assignment**: Rigid round-robin dampening (-12 to -19 points on Hard)
9. **Eval annealing**: Weakening coordination penalties in early iterations hurts scores

### Expert-Specific
- 7 DP bots + 3 greedy bots
- Greedy bots: ONLY fetch preview + high-frequency non-active types
- NEVER touch active order items (DP interference)
- Greedy ceiling: ~+18 points over DP-only

### Determinism Exploit ("Time Millionaire" Strategy)

```mermaid
flowchart TD
    subgraph Day["Competition Day Timeline"]
        direction LR
        Morning["Morning<br/>Initial Capture"] --> Train1["Deep Train<br/>(hours, 200K states)"]
        Train1 --> Replay1["Replay<br/>Discover orders"]
        Replay1 --> Train2["Deep Train<br/>(expanded orders)"]
        Train2 --> Replay2["Replay<br/>More orders"]
        Replay2 --> TrainN["... Deep Train ..."]
        TrainN --> Final["Final Replay<br/>Best Score"]
    end

    Key["Same day = same game = same orders<br/>Unlimited offline training between tokens"]

    style Morning fill:#fff9c4,stroke:#f57f17
    style Final fill:#c8e6c9,stroke:#2e7d32
    style Key fill:#e3f2fd,stroke:#1565c0
```

- Same day = same game = same orders
- Capture once, train for hours offline, replay next day (if same seed)
- Competition day: capture early, train all day, replay for final score
- **Stepladder discovery**: play → discover orders → deep train → replay → discover more → repeat
- Between tokens: unlimited offline training time (200K+ states, 50+ refine iters)
- Use `competition_day.py` for automated stepladder cycles
- 12 hours of stepladder cycles can discover 30-50+ orders per difficulty

---

## Current Scores & Targets

### As of 2026-03-06

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4a9eff'}}}%%
xychart-beta
    title "Our Score vs Leader (2026-03-06)"
    x-axis ["Easy", "Medium", "Hard", "Expert"]
    y-axis "Score" 0 --> 320
    bar [142, 188, 196, 139]
    bar [142, 214, 252, 303]
```

| Difficulty | Our Score | Source | Leader | Gap | Orders Known |
|-----------|-----------|--------|--------|-----|-------------|
| Easy | 142 | GPU DP | ~142 | 0 | 17 |
| Medium | 188 | GPU DP | 214 | -26 | 22 |
| Hard | **196** | **Live solver** | 252 | -56 | 21 |
| Expert | 139 | GPU DP | 303 | -164 | 15 |

Note: Hard 196 came from `live_gpu_stream.py` (reactive per-round), NOT offline GPU DP (max 180). The offline GPU DP solver has never beaten 180 on Hard.

### Bottlenecks
- **Hard**: Live solver produced 196 but offline DP caps at 180. Need to either improve live solver consistency or break the sequential DP ceiling.
- **Medium**: Sequential DP ceiling. Need better multi-bot coordination.
- **Expert**: Not enough orders + too many bots for current coordination quality.
- **All**: Order counts may be stale (orders change daily). Need fresh captures on competition day.

---

## Known Limitations

### The 196 Hard Deadlock — Live Solver vs Offline GPU DP

**Critical finding (2026-03-06):** Our best Hard score (196 points) was NOT produced by the offline GPU DP solver. It came from `live_gpu_stream.py` — the **anytime online solver** that makes per-round reactive GPU decisions. The offline sequential DP solver (`gpu_sequential_solver.py`) has never beaten 180 on Hard.

This is evident from `solutions/hard/meta.json` showing `optimizations_run: 0` — the solution was saved directly from a live game, never refined by offline DP.

**Why the live solver can beat offline DP**: The live solver makes per-round decisions with full knowledge of the current game state. It doesn't commit to a 300-round plan upfront, so it naturally adapts to congestion, order transitions, and bot interactions in real time. The offline DP, by contrast, locks trajectories sequentially — bot 0's plan is fixed before bot 1 is even considered.

**The idle bot problem**: In the 196 solution, bot 2 performs zero useful work for all 300 rounds. It executes 19 pickup actions, but ALL target items that are 4-20 tiles away (pickup requires Manhattan distance 1). Even in complete isolation, only 3 of those pickups would succeed, scoring 0 points. This happened because the live solver's per-round heuristic sent bot 2 chasing items it could never reach in time.

### Sequential DP Ceiling (the "Local Optimum Deadlock")

```mermaid
flowchart TD
    Problem["Sequential DP<br/>Local Optimum"] --> C1["Order Coupling<br/>Completing order N at round 100<br/>vs 105 cascades through game"]
    Problem --> C2["Single-Bot Scope<br/>Can only replan 1 bot<br/>Need 2-3 bot coordination"]
    Problem --> C3["Trajectory Locking<br/>Frozen bots can't cooperate<br/>No dynamic waiting"]
    Problem --> C4["Emergent Solutions<br/>Live solver found 196<br/>Sequential DP can't represent it"]

    C1 --> Stuck((Stuck at<br/>180 on Hard))
    C2 --> Stuck
    C3 --> Stuck
    C4 --> Stuck

    Stuck --> F1["Joint 2+ bot DP<br/>(needs 200K+ states)"]
    Stuck --> F2["Hybrid: live solver + DP refine"]
    Stuck --> F3["MAPF-style path planning"]
    Stuck --> F4["Population-based training"]

    style Problem fill:#ffcdd2,stroke:#c62828
    style Stuck fill:#ff8a80,stroke:#c62828,color:#fff
    style F1 fill:#c8e6c9,stroke:#2e7d32
    style F2 fill:#c8e6c9,stroke:#2e7d32
    style F3 fill:#c8e6c9,stroke:#2e7d32
    style F4 fill:#c8e6c9,stroke:#2e7d32
```

The fundamental limitation: each bot is planned independently with others locked. When the team is collectively near-optimal, **no single bot can improve** because:

1. The other 4 bots already complete all available orders (~21 orders = ~180-196 points)
2. Re-planning any single bot can only redistribute work, not create new capacity
3. Any change to one bot's timing shifts order completion boundaries, breaking the locked bots' plans

**Concrete evidence (Hard, 196 solution, refine attempt):**

| Bot Re-planned | DP Score (isolated) | Total Score (with others) | Delta |
|----------------|--------------------|-----------------------------|-------|
| Bot 0 | 196 | 196 | +0 |
| Bot 1 | 124 | 196 | +0 |
| Bot 2 | 94 | 196 | +0 |
| Bot 3 | 196 | 196 | +0 |
| Bot 4 | 179 | 196 | +0 |

Every bot's DP plan either matches the existing plan exactly (bots 0, 3) or produces a worse isolated score that can't improve the team total (bots 1, 2, 4).

**Bot 2's paradox**: Bot 2 can achieve DP score 94 in isolation (with others locked), but inserting that plan drops the team total from 196 to 88. Bot 2's pickups alter the order state progression — items it delivers shift when orders complete, breaking the timing assumptions embedded in the locked bots' pre-computed trajectories.

**Fresh solve also can't reach 196**: A clean `solve_sequential` run (no warm-start) scores only 162. The 196 solution occupies a region of the solution space that sequential DP cannot reach from scratch — it was found by the live solver's reactive exploration.

**Additional evidence of the ceiling:**
- 600s+ training with 100K states, 20 refine iters, pair perturbation, type reshuffling — all converge to ≤180
- Exhaustive orderings (43/120 tried on Hard) — best pass1 = 143
- 200K states per bot in refinement — reaches 178-179, never 180+
- 1200s budget with 22 orders, 5 orderings, 30 refine — scores 165

### Why Breaking Past the Deadlock Is Hard

The deadlock exists because sequential DP is trapped between two constraints:

1. **Order coupling**: Orders are sequential. Completing order N reveals order N+1. Bot A delivering item X at round 100 means order N completes at round 100, not round 105. Every other bot's plan depends on this exact timing. Moving one delivery by 1 round can cascade through the entire remaining game.

2. **Single-bot replanning scope**: Refinement can only change ONE bot at a time. To escape the local optimum, you'd need to simultaneously reassign work across 2-3 bots — "bot 0 stops picking up yogurt so bot 2 can pick it up faster from a closer shelf, while bot 3 shifts to pasta to compensate." Sequential DP cannot express this kind of coordinated reassignment.

3. **Trajectory locking is lossy**: When bot K is planned, other bots are frozen trajectories. But in reality, bots interact dynamically — bot 0 might wait 1 round for bot 1 to clear a doorway, then both benefit. The locked model can't represent these cooperative waits.

4. **The live solver found a solution that sequential DP can't represent**: The 196 solution emerged from 300 rounds of reactive per-round decisions where all 5 bots adapted to each other simultaneously. Translating this into "plan bot 0, then bot 1, then..." loses the emergent coordination.

**What might break through:**
- **True joint optimization**: Joint 2+ bot DP with enough state budget (needs 200K+ states per pair, currently impractical)
- **Hybrid approach**: Use live solver to generate candidate solutions, then refine with DP per-bot
- **MAPF-style coordination**: Plan collision-free paths first, then assign orders to paths
- **Population-based training**: Maintain multiple diverse solutions, crossover the best parts

### State Budget vs Bot Count
More bots = more sequential DP passes = less time per bot. With 10 Expert bots at 50K states, each bot gets ~6s. Joint optimization would need exponential state space.

### Order Staleness
Orders are deterministic PER DAY. Captures from March 5 are useless on March 6. Must re-capture each day.

### Token Expiry
JWT tokens expire after ~288s. All pipeline operations must complete within this window. Fetch new tokens for additional cycles.

### Known Bug: `verify_against_cpu` for Non-Zero Bot IDs
`gpu_beam_search.py` `verify_against_cpu()` always uses `cpu_state.bot_positions[0]` and puts the action at index 0, regardless of `candidate_bot_id`. This means verification is broken for bot IDs > 0. The GPU `_step()` function itself is correct (verified via round-by-round comparison) — only the verification helper is buggy.

---

## Command Reference

### Token Fetching
```bash
# First-time setup (Google OAuth login)
python fetch_token.py hard --setup

# Headless token fetch (after login)
python fetch_token.py hard

# With JSON output
python fetch_token.py hard --json
```

### Zig Capture
```bash
# Run Zig bot on live game to capture orders
python zig_capture.py "wss://game.ainm.no/ws?token=..." --difficulty hard
```

### GPU Training (Offline)
```bash
# Quick training (pipeline-speed)
python optimize_and_save.py hard --max-time 60 --max-states 50000

# Deep training
python optimize_and_save.py hard --max-time 600 --max-states 100000 \
  --speed-bonus 150 --orderings 3 --refine-iters 20

# Warm-start from existing solution
python optimize_and_save.py hard --max-time 300 --warm-only --refine-iters 10
```

### Live Replay
```bash
# Replay pre-computed solution
python replay_solution.py "wss://..." --difficulty hard

# Live GPU solver (anytime per-round)
python live_gpu_stream.py "wss://..." --save --max-states 50000
```

### Full Pipeline
```bash
# Automated: zig capture -> GPU optimize -> replay -> iterate (within 288s)
python production_run.py hard --ws-url "wss://..." \
  --time-budget 275 --max-states 50000
```

### Local Simulation
```bash
# Offline iterate loop (sim server, no token needed)
python iterate_local.py hard --seed 42 --max-states 50000 --time-budget 280
```

### Competition Day (Stepladder)
```bash
# Interactive: prompts for WS URLs between training cycles
python competition_day.py hard --deep-states 200000 --deep-refine 50

# Auto-token: fetches tokens automatically
python competition_day.py expert --auto-token --deep-states 100000

# With time limit per training pass
python competition_day.py hard --deep-time 600 --max-cycles 20
```

### Utilities
```bash
# Extract orders from game logs
python extract_orders_from_logs.py

# Check solution scores
python -c "import json; print(json.load(open('solutions/hard/meta.json')))"
```
