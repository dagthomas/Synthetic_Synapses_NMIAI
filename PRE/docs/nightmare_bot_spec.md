# Winning Grocery Bot: algorithms for 20-agent Nightmare mode

**PIBT (Priority Inheritance with Backtracking) combined with deterministic seed exploitation is the clearest path to a dominant Nightmare score.** The sub-millisecond MAPF algorithm leaves 99.9% of the 2-second budget free for task assignment optimization and anytime LNS refinement — a combination that should push well beyond the current Expert leaderboard ceiling of 303. The critical strategic insight is that daily determinism transforms this from a hard online planning problem into a tractable offline combinatorial optimization: discover the full order sequence on Run 1, pre-compute an optimal 500-round plan offline, then replay it.

The current sequential per-bot DP ceiling at ~180 points stems from a fundamental architectural flaw: planning each bot independently with others "locked" creates an ordering-dependent local optimum that no amount of beam-search budget can escape. The solution requires simultaneous multi-agent coordination (MAPF) combined with global task assignment optimization — both of which are now well-studied problems with practical, fast implementations.

---

## PIBT is the right MAPF algorithm — and it's not close

CBS (Conflict-Based Search), once the gold standard for optimal MAPF, is computationally infeasible for 20 agents on a 30×18 grid within a 2-second timeout. Empirical benchmarks show CBS timing out on 16×16 grids with just 20 agents. ECBS with bounded suboptimality (w=1.2) might barely fit, but replanning every round makes it impractical. The algorithm zoo collapses to a clear winner for this specific problem profile.

**PIBT runs in sub-millisecond time for 20 agents** — O(|A| × Δ(G)) per timestep where |A|=20 and Δ(G)=4 on a grid. It was designed explicitly for lifelong MAPF where goals change continuously, which exactly matches the Grocery Bot loop of pick-up → deliver → pick-up. PIBT won the 2023 League of Robot Runners competition handling 1,000+ agents per second. The algorithm works by processing agents in priority order each timestep: each agent selects its preferred adjacent cell (sorted by BFS distance to goal), and if that cell is occupied by a lower-priority agent, priority "inherits" upward, forcing the blocking agent to move. Backtracking resolves deadlocks when no valid move exists.

The key implementation detail is **priority rotation**: agents not at their goal get higher priority, and within that group, priorities cycle each timestep to ensure fairness and prevent starvation. On biconnected graphs (which a grocery-store grid with aisles typically satisfies), PIBT guarantees reachability.

For Python, two libraries stand out. **`w9_pathfinding`** (pip-installable, C++ backend via Cython) provides WHCA* and CBS with a clean grid API — WHCA* with a 10-step window solves 20 agents in under 100ms. **`pylacam`** (MIT license, by PIBT/LaCAM's original author Keisuke Okumura) provides a minimal Python LaCAM* that includes a PIBT branch. For maximum control, PIBT's core logic is roughly **100 lines of Python** and trivial to embed directly in the bot codebase.

The enhanced strategy is **PIBT + LNS refinement** (the WPPL pattern that won the 2023 competition): run PIBT in <1ms to get a feasible next move for all 20 bots, then spend the remaining ~1.9 seconds running Large Neighborhood Search — selecting subsets of 3-5 agents, replanning their windowed paths using A* while respecting others' plans. This consistently produces solutions **within 1.35% of optimal** on warehouse-style maps.

| Algorithm | Time per round (20 agents) | Solution quality | Lifelong MAPF | Recommendation |
|-----------|---------------------------|------------------|---------------|----------------|
| CBS | >>2s (timeout) | Optimal | No | ❌ Infeasible |
| ECBS (w=1.2) | 0.1–5s (borderline) | 1.2× optimal | Via RHCR | ⚠️ Marginal |
| **PIBT** | **<1ms** | Unbounded subopt. | **Native** | ✅ **Primary choice** |
| LaCAM* | 1–10ms | Eventually optimal | Via windowing | ✅ Excellent alt. |
| PIBT+LNS | <1ms + refinement | Near-optimal | Native | ✅ **Best quality** |

---

## Task assignment and order pipelining unlock the real score gains

The MAPF layer handles collision-free movement. The task assignment layer — which bot fetches which item, in what order, to which drop zone — is where the scoring ceiling lives. With 20 bots and orders requiring only 4-6 items, **14-16 bots are idle at any moment unless you pipeline aggressively**.

**Hungarian algorithm via `scipy.optimize.linear_sum_assignment`** solves the bot-to-item assignment optimally in under **0.01ms** for a 20×30 cost matrix. The cost matrix entries are BFS shortest-path distances from each bot to each item location (pre-computed once at game start since walls don't change). For handling the 3-item carry limit, use a two-phase approach: Phase 1 assigns individual items to bots via Hungarian, Phase 2 sequences each bot's assigned items via a mini-TSP. Alternatively, OR-Tools' CP-SAT solver handles the generalized assignment with capacity constraints directly, solving the 20-bot problem in under 1ms.

The **order pipelining strategy** is crucial. Split bots into four tiers operating simultaneously:

- **Tier 1 — Active pickers (4-6 bots)**: Assigned to current active order items via Hungarian
- **Tier 2 — Preview pre-pickers (4-6 bots)**: Already picking items for the preview order (allowed since you can pre-pick but not deliver)
- **Tier 3 — Delivery shuttles (3-4 bots)**: Positioned near drop zones, handling rapid drop-off
- **Tier 4 — Scouts (remaining bots)**: Pre-positioned across aisles for instant response to new orders

When the active order has ≤2 items remaining, transition preview pre-pickers to delivery mode. The moment an active order completes and the preview becomes active, pre-picked items are already in-hand — **reducing per-order cycle time by 30-40%**. This translates directly to more orders completed in 500 rounds and thus more points.

For **drop zone coordination** with 3 zones and 20 bots, use distance-plus-congestion scoring: `score(zone) = distance(bot, zone) + α × queue_length(zone)` where α ≈ 3-5. This naturally load-balances without complex centralized scheduling. Assign each drop zone to serve nearby aisles (e.g., Zone 1 → Aisles 1-2, Zone 2 → 3-4, Zone 3 → 5-6) to create spatial traffic separation. In PIBT, congestion near drop zones resolves naturally through priority inheritance — delivering bots get priority over waiting bots.

---

## Deterministic seed exploitation is the highest-leverage strategy

This is **the single most important insight** for competition performance. The game is deterministic per day: same seed produces the same map, same item placement, same order sequence. This means the entire game can be pre-computed.

**Run 1 (Discovery)**: Play with any fast greedy strategy. Complete orders as rapidly as possible — sacrificing per-item efficiency for speed — to discover the maximum number of orders in the sequence. Record the complete map layout, all item positions, and the full order sequence. Target: discover 30-50+ orders.

**Offline Phase (between runs)**: With full knowledge of all orders, item positions, and map layout, solve the entire 500-round game as a single offline optimization problem. This is now a **Multi-Vehicle Pickup-and-Delivery Problem with Time Windows (VRPTW)**, solvable with OR-Tools:

```python
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
manager = pywrapcp.RoutingIndexManager(num_locations, 20, starts, ends)
routing = pywrapcp.RoutingModel(manager)
# Add capacity constraints (3 items per bot)
# Add time dimension (500 rounds horizon)
# Solve with Guided Local Search metaheuristic
search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_params.time_limit.FromSeconds(60)
```

**Run 2 (Replay)**: Execute the pre-computed optimal plan. The plan is compact — just 10,000 action strings (500 rounds × 20 bots, <50KB). Load it, execute sequentially, collect the optimized score.

The key insight about order throughput: each order yields approximately **+10 points** (+5 completion bonus + ~5 item deliveries). Sacrificing 1 item delivery (-1 point) to complete an order 10 rounds faster means seeing 1 additional order over 500 rounds for a net gain of +9 points. **Always prioritize order completion speed over per-item efficiency.**

Conservative estimate with pipelining + Hungarian assignment: **250-300 points**. With deterministic replay + offline optimization: **350-500+ points**.

---

## LNS with simulated annealing breaks the planning ceiling

The current sequential DP approach gets stuck because each bot is planned independently — Bot 1's path constrains Bot 2, which constrains Bot 3, creating ordering-dependent local optima. **Adaptive Large Neighborhood Search (ALNS)** directly addresses this by iteratively destroying and repairing subsets of bot plans.

MAPF-LNS (Li et al., IJCAI 2021) provides the template. The algorithm starts with any feasible solution (from PIBT), then in each iteration: selects a subset of 3-5 agents using one of three destroy heuristics (random, agent-delay-based, or conflict-based), removes their paths, and replans using Prioritized Planning. **Adaptive heuristic selection via roulette wheel** consistently outperforms any single heuristic.

The critical addition for escaping local optima is **Simulated Annealing acceptance**: instead of only accepting improvements, accept worse solutions with probability `exp(-(new_cost - current_cost) / temperature)`. This prevents the search from getting trapped. Start temperature at ~10% of objective value, cool with rate **0.9995** per iteration. Combined with **Iterated Local Search** (strong perturbation every ~500 iterations, reassigning 30% of items randomly), this provides both intensification and diversification.

For the offline optimization pipeline after deterministic discovery:

1. **Build fast simulator** in Numba: ~10ms per full 500-round game evaluation, enabling ~12,000 LNS iterations per 120 seconds
2. **Two-level ALNS**: Outer loop destroys/repairs bot-to-item assignments, inner loop uses MAPF-LNS for collision-free paths
3. **Temporal neighborhoods**: Destroy and repair paths for windows of 50 rounds at a time
4. **Random restart ILS**: Every 500 iterations, apply strong perturbation and re-optimize

The **solution representation** is compact: store only bot-to-item assignments per order plus drop zone choices. Paths are deterministically reconstructable via simulation. A full game solution fits in a JSON file under 50KB.

---

## RL is the wrong primary approach — but hybrid methods add value

For a deterministic game where you can replay the exact scenario, **planning dominates pure RL**. QMIX scales poorly to 20 agents (documented 55% performance drop from 2→6 agents in warehouse tasks, with joint action space of 7^20 ≈ 8×10^16). MAPPO with parameter sharing is the best MARL option — all 20 bots share one policy network, making agent count irrelevant to model size — but even MAPPO typically requires 10M+ environment steps to converge on cooperative tasks, with no guarantee of outperforming a well-tuned planner.

Where learning adds value is as a **heuristic enhancement layer** for the planning stack:

- **CS-PIBT (Collision-Shield PIBT) with learned policies**: A small neural network predicts per-agent action probability distributions, and PIBT resolves 1-step collisions. Recent research (Veerapaneni et al., 2024) shows that even a model trained for 4 minutes on a single scene, combined with CS-PIBT, achieves state-of-the-art ML MAPF performance. The network learns tie-breaking preferences that PIBT's greedy heuristic misses.

- **GNN-based task assignment** (GRAND, 2024): A Graph Neural Network trained with RL predicts optimal bot-to-item assignments, converted to assignments via minimum-cost flow. This showed **10% throughput improvement** over the 2023 League of Robot Runners winning baseline. The GNN captures spatial patterns in item/bot distributions that the Hungarian algorithm's distance-only cost matrix misses.

- **Day-specific policy fine-tuning**: After discovering the day's scenario, train MAPPO for 1-2 hours on the exact known game state. Since the environment is deterministic, sample efficiency is extreme — no stochastic exploration needed. Use your reactive solver's trajectories as imitation learning warm-start, then fine-tune with RL.

The recommended hybrid architecture: **RL/learned heuristics for task assignment (Layer 1) → PIBT for collision-free pathfinding (Layer 2) → LNS for anytime refinement (Layer 3)**. Libraries: JaxMARL for fastest training (12,500× faster than CPU via GPU parallelization), or TorchRL/BenchMARL for integration with the existing PyTorch/CUDA codebase.

---

## The concrete implementation roadmap

The per-round decision pipeline should consume roughly **150ms of the 2-second budget**, leaving 1.85 seconds for anytime improvement:

| Phase | Time budget | Tool |
|-------|------------|------|
| Parse state + update distances | ~2ms | `orjson` + NumPy |
| Task assignment (Hungarian) | <0.01ms | `scipy.linear_sum_assignment` |
| MAPF path planning (PIBT or WHCA*) | ~100ms | `w9_pathfinding` (C++ backend) or custom PIBT |
| Conflict resolution | ~10ms | Priority rules |
| Anytime LNS improvement | ~1,850ms | Custom ALNS loop |
| Action formatting + send | ~1ms | `orjson` + `websockets` |

**Critical first-round setup**: Pre-compute all-pairs BFS distances on the 30×18 grid (~378 walkable cells). With Numba JIT: ~5-10ms. Store as a 378×378 NumPy array for O(1) distance lookups throughout the game.

The recommended tech stack: `scipy.optimize.linear_sum_assignment` for assignment, `w9_pathfinding` (pip-installable, C++ backend) for MAPF, Numba JIT for the offline simulator (~10ms per full game), `orjson` for fast JSON parsing, and `websockets` for the game loop with `time.monotonic()` deadline tracking and 200ms safety margin.

**Priority-ordered implementation plan**:

1. **P0 — Deterministic replay** (4-6 hours): Run once to discover orders, save state, replay with pre-computed plan. Expected gain: +100-200 points over baseline.
2. **P0 — Hungarian assignment + order pipelining** (4-8 hours): Replace greedy bot-item matching with optimal assignment. Split bots into active/preview/delivery/scout tiers. Expected gain: +30-50 points.
3. **P1 — PIBT integration** (4-8 hours): Replace sequential per-bot planning with proper multi-agent pathfinding. Expected gain: +15-25 points from eliminated congestion.
4. **P1 — ALNS offline optimization** (1-2 days): Build Numba simulator, implement destroy-repair loop with SA acceptance. Expected gain: +30-50 points.
5. **P2 — Learned heuristics** (2-3 days): Train GNN or small MLP for task assignment, integrate with CS-PIBT. Expected gain: +10-20 points.

## Conclusion

The path from 180 to 400+ points requires three architectural shifts, not incremental tuning. First, replace sequential per-bot DP with proper multi-agent pathfinding — PIBT solves 20-agent collision avoidance in microseconds, eliminating the congestion that wastes rounds. Second, exploit daily determinism ruthlessly: a discovery run followed by offline VRPTW optimization with ALNS can pre-compute a near-optimal 500-round plan. Third, pipeline order fulfillment so that 20 bots are never waiting — the preview order mechanism exists specifically to reward teams that keep all agents productive.

The deepest insight is that this is not primarily a pathfinding problem. On a 30×18 grid with 20 bots, paths are short (average ~10 steps). The scoring bottleneck is **throughput** — how many orders you complete in 500 rounds. Every round a bot spends idle, blocked, or walking to the wrong item is a round lost. The winning architecture is one that treats task assignment as the primary optimization target, pathfinding as a fast constraint-satisfaction subroutine, and the deterministic seed as an invitation to pre-compute everything.