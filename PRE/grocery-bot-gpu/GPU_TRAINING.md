# GPU Training & Inference — Offline and Live

How the GPU DP solver works for both offline optimization and live game inference. Covers the CUDA pipeline from state representation through beam search to multi-bot coordination.

---

## Table of Contents

1. [Hardware & Software Stack](#hardware--software-stack)
2. [State Representation](#state-representation)
3. [GPU Beam Search DP](#gpu-beam-search-dp)
4. [Multi-Bot Sequential DP](#multi-bot-sequential-dp)
5. [Offline Training Pipeline](#offline-training-pipeline)
6. [Live Inference](#live-inference)
7. [Precomputation](#precomputation)
8. [Evaluation Heuristic](#evaluation-heuristic)
9. [Nightmare GPU Training](#nightmare-gpu-training)
10. [Performance Characteristics](#performance-characteristics)

---

## Hardware & Software Stack

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 5090, 32 GB VRAM |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.10+cu128 |
| Triton | 3.6.0 (kernel fusion via torch.compile) |
| TF32 | Enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) |
| torch.compile | Mode='default' (3.5x speedup; 'reduce-overhead' blocked by tensor lifetime issue) |
| Zig FFI | `grocery-sim.dll` for fast verification (4.5x) and pre-simulation (2.7x) |

**Critical Constraints**:
- Only one GPU task at a time — competing tasks degrade both
- `no_compile=True` required for all live/threaded GPU calls (prevents dynamo crash)
- `--no-filler` mandatory for DP (filler orders waste moves, 179 sim → 151 live without it)

---

## State Representation

Each DP state encodes the complete game situation for a single bot:

### Packed int64 Hash

```
Bits:  [bot_x:8][bot_y:8][inv0:8][inv1:8][inv2:8][active_idx:8][del_bitmask:16]
Total: 64 bits → single int64 for O(1) deduplication via torch.unique()
```

### Batch Tensor Layout (on CUDA)

```python
bot_x:       [B]       int16    # Bot x position
bot_y:       [B]       int16    # Bot y position
bot_inv:     [B, 3]    int8     # 3 inventory slots (item type IDs, 0=empty)
active_idx:  [B]       int32    # Current active order index
active_del:  [B, 128]  int8     # Delivery bitmask (7 items × 128 columns)
score:       [B]       int32    # Accumulated score
orders_comp: [B]       int32    # Orders completed count
```

Where `B` is the beam width (number of active states, typically 50K-500K).

### Delivery Bitmask

Each order can have up to MAX_ORDER_SIZE=7 items. The delivery bitmask uses 128 columns with weights `[1, 2, 4, 8, 16, 32, 64]` to track which items have been delivered.

---

## GPU Beam Search DP

### Algorithm Overview

Single-bot exhaustive BFS with pruning via beam width. Per round:

```
States[t] → Expand (all valid actions) → Deduplicate → Evaluate → Prune to top-K → States[t+1]
```

### Action Expansion

Three expansion strategies, selected per search phase:

| Strategy | Actions/State | Use Case |
|----------|--------------|----------|
| `_dp_expand()` | 6 + MAX_ADJ (≤10) | Position-aware: only valid moves + adjacent pickups |
| `_smart_expand()` | Variable | BFS-guided: move toward active/preview/preferred types |
| `_expand()` | 6 + num_items | Full: all moves + all item pickups (exhaustive) |

**`_dp_expand()` Details**:
- 5 movement actions (4 directions + wait) filtered by walkability
- Pickup actions only for items adjacent to current cell (precomputed `adj_items[cell]`)
- Dropoff action only at dropoff zones
- Output: B × (6+MAX_ADJ) candidate states

### Deduplication

After expansion, states are deduplicated by int64 hash:
```python
hashes = self._hash(expanded_states)
unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
# Keep highest-score duplicate
```

### Beam Pruning

Top-K selection by evaluation score:
```python
scores = self._eval(states)
_, top_indices = torch.topk(scores, k=min(max_states, len(scores)))
states = states[top_indices]
```

### Locked Bot Interleaving

In multi-bot sequential DP, locked bots' actions are interleaved at each round:

```python
# For each round t:
for locked_bot in locked_bots:
    apply locked_bot.action[t] to game state
apply current_bot candidates to game state
resolve collisions (lower ID wins)
```

This ensures the DP-optimized bot sees realistic game state including other bots' movements.

---

## Multi-Bot Sequential DP

### Algorithm (`gpu_sequential_solver.py`)

```
Pass 1 (Cold Start):
  Bot 0: GPU DP with no locked bots → solution_0
  Bot 1: GPU DP with bot 0 locked → solution_1
  Bot 2: GPU DP with bots 0,1 locked → solution_2
  ...
  Bot N: GPU DP with bots 0..N-1 locked → solution_N

Pass 2+ (Refinement):
  For each bot i in [0..N]:
    Re-plan bot i with ALL other bots locked
    If score improves: keep new plan
```

### Bot Ordering

Three orderings tried in Pass 1 (configurable via `num_pass1_orderings`):
1. **Forward**: Bot 0, 1, 2, ... (bot 0 gets most freedom)
2. **Reverse**: Bot N, N-1, ... (later bots get more freedom)
3. **Random**: Shuffled order (breaks ordering bias)

Best pass 1 result seeds the refinement phase.

### Locked Trajectory Management

```python
def pre_simulate_locked(actions_list, capture, locked_bot_ids):
    """Simulate full game with ALL bots' actions.
    Record positions of specified locked bots after each round.
    Returns: locked_positions[num_locked, num_rounds, 2] (x, y)
    """
```

- CPU simulation (Python game_engine or Zig FFI)
- Locked bots' positions fed into GPU `_step()` for collision resolution
- `_locked_remaining_planned[num_rounds, num_types]`: Suffix sum of locked bots' future pickups per round
- `_locked_all_planned_mask[num_types]`: Which types any locked bot ever picks

### Greedy Bots (Non-DP)

For Expert (10 bots), typically 7 DP + 3 greedy:
- **Never touch active order types** (catastrophic DP interference)
- Fetch preview + high-frequency non-active types only
- Co-simulated with DP bots for correct order progression
- Contribute ~+18 points over DP-only

### Verification

After solving, verify final score via:
1. **Python CPU**: `cpu_verify()` — replay all actions through game_engine
2. **Zig FFI**: `ffi_verify()` — 4.5x faster native replay
3. **Bot contributions**: `compute_bot_contributions()` — marginal value per bot

---

## Offline Training Pipeline

### Quick Pipeline (`production_run.py`)

Within 288s token window:

```
Phase 1: Zig Capture (30s)
  └─ Run Zig bot → discover initial orders (5-10)

Phase 2: Iterate Loop (5-8 iterations × 30-40s each)
  ├─ Iteration 0 (cold): solve_sequential(50K, 2 orderings, 2 refine)
  ├─ Iteration 1-2 (warm): refine_from_solution(50K, 1 ordering, 2 refine)
  ├─ Iteration 3 (deep): refine_from_solution(100K, 1 ordering, 3 refine)
  └─ Iteration 4+ (warm): refine_from_solution(50K, 1 ordering, 2 refine)

Per iteration:
  GPU Optimize (18-35s) → Replay via WS (30s) → Discover 2-5 new orders → Merge
```

**Speed Bonus Decay**: `speed_bonus × decay^iteration` (100 × 0.5^n by default). Earlier iterations prefer fast solutions; later iterations optimize score.

### Deep Training (`competition_day.py`)

Unlimited offline time between token windows:

```
Bootstrap (if needed):
  Zig bot + 3 quick GPU iterations (275s)

Discovery Cycles (unlimited):
  1. Deep Train: 100K states, 30 refine, 3 orderings (60% cold, 40% warm)
  2. Replay: New token → replay best solution → discover orders
  3. Retrain with expanded order set
  4. Repeat until stall (3 cycles with no new orders)
```

### Local Benchmarking (`iterate_local.py`)

Two modes for offline development:

**Full Foresight**: Start with 25 orders, solve → simulate → discover max_order_seen → adjust order count → re-solve. Key finding: 25 orders + 50K >> 40 orders + 100K.

**Iterative Discovery**: Start with 2 orders, solve → simulate → grow order set by 2-3. Simulates real pipeline behavior.

### Replay with Desync Correction (`replay_solution.py`)

```
Per round:
  if actual_position == expected_position:
    SYNCED → send cached DP action (zero computation)
  else:
    DESYNCED → BFS toward next DP goal (pickup/dropoff target)
  if DP plan exhausted:
    GREEDY → fallback strategy (deliver, pick nearest, pre-position)
```

Goal-following preserves multi-bot coordination even when individual bots drift.

---

## Live Inference

### Tiered Architecture (`live_gpu_stream.py`)

```
Round 0                                      Round 300/500
  |                                              |
  |-- Tier 0: Greedy BFS (<1ms) ----------------->|
  |                                              |
  |-- Tier 1: MAPF (1-10s) ------>|              |
  |                               |              |
  |-- Tier 2: GPU DP Pass 1 (20s) -->|           |
  |                                  |           |
  |-- Tier 3: GPU DP Pass 2 (40s) ------>|       |
  |                                      |       |
  |-- Per-Round GPU (continuous) ----------------->|
```

### Background Threads

| Thread | Purpose | Timing |
|--------|---------|--------|
| `_gpu_worker()` | Multi-pass GPU DP with growing budgets | Runs until game ends |
| `_gpu_refine_worker()` | Warm-start refinement from best solution | After initial pass |
| `_pr_worker()` | Per-round low-budget GPU DP | Every round |

### Per-Round GPU DP

Small-budget DP each round for reactive decisions:

| Difficulty | States | Horizon |
|-----------|--------|---------|
| Easy | 50K | 80 rounds |
| Medium | 30K | 60 rounds |
| Hard | 15K | 50 rounds |
| Expert | 10K | 40 rounds |
| Nightmare | 5K | 30 rounds |

### Plan Upgrade Logic

```python
def _update_plan(new_plan, new_score, generation):
    if new_score > current_score:
        current_plan = new_plan
        current_score = new_score
        round_offset = rounds_elapsed_since_computation_started
```

Only accepts score improvements. Stale plans from earlier generations are OK if they score higher.

### Nightmare Live Mode

Nightmare uses LMAPF solver instead of GPU DP for live games:
- `ws_action()` method in `NightmareSolverV3`
- Task allocation + PIBT pathfinding per round
- Opportunistic adjacent pickups for all bot types
- No GPU computation (too many bots for sequential DP)

---

## Precomputation

### All-Pairs BFS (`precompute.py`)

**GPU-accelerated** via matrix multiply:

```python
# Adjacency matrix A[N,N] (sparse)
# Distance matrix D initialized to infinity
# Frontier F[N,N] = identity (each cell reaches itself)
while frontier_has_new_cells:
    F_next = torch.mm(F, A)  # Expand frontier one step
    new_cells = (F_next > 0) & (D == INF)
    D[new_cells] = current_distance
    F = F_next
```

Output cached to `.npz` files keyed by grid MD5 hash.

### Trip Tables

Backward-chaining DP for O(1) multi-item trip cost estimation:

```
For combo (type_A, type_B, type_C):
  For each permutation of (A, B, C):
    cost = dist(last_pickup → dropoff) + 1
    cost += dist(second_pickup → last_pickup) + 1
    cost += dist(first_pickup → second_pickup) + 1
    cost += dist(start_cell → first_pickup) + 1
  trip_cost[cell][combo] = min over all permutations
```

Uploaded to GPU as `trip_cost_gpu[N_cells, N_combos]` int16 tensor.

### Item Adjacency Tables

```python
# For each walkable cell, which items can be picked up?
adj_items[cell_idx, 0..MAX_ADJ]  # item indices
adj_count[cell_idx]              # number of adjacent items (0-4)
```

Enables O(1) valid pickup enumeration during action expansion.

---

## Evaluation Heuristic

The `_eval()` function assigns a score to each DP state, guiding beam pruning:

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Accumulated score | 1.0 | Items delivered + orders completed |
| Active order value | 5 + remaining | Bonus for completing active order |
| Trip table cost | -1.0 | Estimated rounds to complete current pickup plan |
| Coordination penalty | -500 to -2000 | Per locked-bot type overlap (prevents redundant picks) |
| Aisle congestion | -4000 | Per locked bot in same narrow aisle column |
| Speed bonus | 100 × decay^iter | Prefer faster order completions (anytime) |
| Inventory value | Variable | Value of items matching active/preview orders |
| Preview targeting | +bonus | Items matching preview order get pickup priority |
| Zone preference | +bonus | Pickups in preferred geographic zone |

### Coordination Signals

When multiple bots plan simultaneously (sequential DP), coordination uses locked trajectory data:

```python
# Which types will locked bots pick in future rounds?
locked_remaining = _locked_remaining_planned[current_round:]  # [rounds_left, num_types]

# Penalty for picking a type that a locked bot will also pick
for type_id in current_bot_inventory:
    if locked_remaining[:, type_id].sum() > 0:
        penalty -= 500  # Reduces redundant pickups
```

---

## Nightmare GPU Training

### Why GPU DP Fails for Nightmare

Sequential DP scores 1-3 with 20 bots (vs 237 LMAPF):
- 20-bot collision is catastrophic for sequential planning
- Each locked trajectory constrains the next bot severely
- State space fragments across 20 sequential passes

### LMAPF Alternative (`nightmare_solver_v2.py`)

Reactive round-by-round planning instead of multi-round DP:

```
Per round:
  1. Identify active order shortfall
  2. Classify bots (active carrier / preview / dead / empty)
  3. Assign goals (deliver / pickup / stage / park / flee)
  4. PIBT pathfinding (recursive push chains, depth 4)
  5. Opportunistic adjacent pickups (all bot types)
  6. Return action per bot
```

### Offline Optimization (`nightmare_offline.py`)

Since GPU DP is ineffective, nightmare uses simulation-based optimization:

**Multi-Restart Search** (15% budget):
```
For each restart:
  Create PerturbedV3 (stochastic stall escape + jitter)
  Simulate 500 rounds
  If score > best: update best
```

**Checkpoint Local Search** (70% budget):
```
1. Run best solver, checkpoint state every 25 rounds
2. Pick random checkpoint (bias early rounds)
3. Mutate: force one random bot to take different action
4. Re-run solver from checkpoint to round 500
5. If improved: keep, rebuild checkpoints
6. Repeat until time budget exhausted
```

**Results**: Baseline 227 → Mean 273.7, Max 284 (+20.6% improvement).

---

## Performance Characteristics

### GPU Throughput

| Metric | Value |
|--------|-------|
| State evaluations | ~50M per 5s (RTX 5090) |
| Single-bot DP (50K states) | 6-10s |
| Single-bot DP (200K states) | 20-40s |
| Full 5-bot solve (50K) | 30-50s |
| Full 10-bot solve (50K) | 60-100s |
| torch.compile warmup | ~30s first call |

### Pipeline Throughput

| Phase | Time | Output |
|-------|------|--------|
| Zig capture | 30s | Initial orders (5-10) |
| GPU cold-start (50K) | 25s | First solution |
| GPU warm-refine (50K) | 18s | Improved solution |
| Replay + discover | 30s | +2-5 new orders |
| Full pipeline (275s) | 5-8 iterations | Converged solution |

### Memory Usage

| Component | VRAM |
|-----------|------|
| State tensors (50K beam) | ~200 MB |
| State tensors (500K beam) | ~2 GB |
| Precomputed tables | ~500 MB |
| Trip tables | ~100 MB |
| torch.compile cache | ~500 MB |
| **Total (50K typical)** | **~1.3 GB** |
| **Total (500K deep)** | **~3.1 GB** |

### Key Findings

| Finding | Impact |
|---------|--------|
| 50K states optimal for pipeline | More iterations > deeper search |
| `--no-filler` mandatory | +28 points on live (179→151 without it) |
| 3 pass1 orderings essential | Forward+reverse+random covers local optima |
| Speed bonus decay=0.5 | Early: fast solutions; late: optimal solutions |
| Sequential DP ceiling ~180 (Hard) | Cannot break regardless of time/states |
| 200K states confirms ceiling | 4x states has no effect at local optimum |
| Fewer orders = better | 25 orders + 50K >> 40 orders + 100K |
| Pair perturbation doesn't help | Converges to same local optimum |
| Joint 2-bot DP not viable | State budget spreads too thin (72 vs 182) |
| Joint 3-bot DP not viable | 512 combos × 50K = only 2 points |
