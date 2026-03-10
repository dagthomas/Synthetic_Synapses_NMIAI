#!/usr/bin/env python3
"""Compare all nightmare solvers on a single seed."""
import sys, time, copy
from game_engine import init_game, step
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables

sys.stdout.reconfigure(encoding='utf-8')

SEED = 7005
NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def test_solver(name, solver_factory):
    """Run a solver and return score."""
    try:
        state, all_orders = init_game(SEED, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = solver_factory(ms, tables, all_orders)

        t0 = time.time()
        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            step(state, actions, all_orders)
        elapsed = time.time() - t0
        print(f"  {name:20s}: score={state.score:4d}  orders={state.orders_completed:3d}  "
              f"items={state.items_delivered:4d}  time={elapsed:.1f}s")
        return state.score
    except Exception as e:
        print(f"  {name:20s}: ERROR: {e}")
        return -1


results = []

# V2
try:
    from nightmare_solver_v2 import NightmareSolverV2
    score = test_solver("V2", lambda ms, t, o: NightmareSolverV2(ms, t))
    results.append(("V2", score))
except Exception as e:
    print(f"  V2 import error: {e}")

# V3
try:
    from nightmare_solver_v2 import NightmareSolverV3
    score = test_solver("V3", lambda ms, t, o: NightmareSolverV3(ms, t))
    results.append(("V3", score))
except Exception as e:
    print(f"  V3 import error: {e}")

# V4
try:
    from nightmare_solver_v4 import NightmareSolverV4
    score = test_solver("V4", lambda ms, t, o: NightmareSolverV4(ms, t, future_orders=o))
    results.append(("V4", score))
except Exception as e:
    print(f"  V4 import error: {e}")

# V5
try:
    from nightmare_solver_v5 import NightmareSolverV5
    score = test_solver("V5", lambda ms, t, o: NightmareSolverV5(ms, t, future_orders=o))
    results.append(("V5", score))
except Exception as e:
    print(f"  V5 import error: {e}")

# V6
try:
    from nightmare_solver_v6 import NightmareSolverV6
    score = test_solver("V6", lambda ms, t, o: NightmareSolverV6(ms, t, future_orders=o))
    results.append(("V6", score))
except Exception as e:
    print(f"  V6 import error: {e}")

# V7
try:
    from nightmare_solver_v7 import NightmareSolverV7
    score = test_solver("V7", lambda ms, t, o: NightmareSolverV7(ms, t, future_orders=o))
    results.append(("V7", score))
except Exception as e:
    print(f"  V7 import error: {e}")

# V7b (task queue)
try:
    from nightmare_v7 import TaskQueueSolver
    score = test_solver("V7-TaskQueue", lambda ms, t, o: TaskQueueSolver(ms, t, future_orders=o))
    results.append(("V7-TaskQueue", score))
except Exception as e:
    print(f"  V7-TaskQueue import error: {e}")

# V8
try:
    from nightmare_solver_v8 import NightmareSolverV8
    score = test_solver("V8", lambda ms, t, o: NightmareSolverV8(ms, t, future_orders=o))
    results.append(("V8", score))
except Exception as e:
    print(f"  V8 import error: {e}")

# V9
try:
    from nightmare_solver_v9 import NightmareSolverV9
    score = test_solver("V9", lambda ms, t, o: NightmareSolverV9(ms, t, future_orders=o))
    results.append(("V9", score))
except Exception as e:
    print(f"  V9 import error: {e}")

# V10
try:
    from nightmare_v10_solver import V10Solver
    score = test_solver("V10", lambda ms, t, o: V10Solver(ms, t, future_orders=o))
    results.append(("V10", score))
except Exception as e:
    print(f"  V10 import error: {e}")

# LMAPF (V4 live solver)
try:
    from nightmare_lmapf_solver import LMAPFSolver
    score = test_solver("LMAPF", lambda ms, t, o: LMAPFSolver(ms, t, future_orders=o))
    results.append(("LMAPF", score))
except Exception as e:
    print(f"  LMAPF import error: {e}")

# V6 with over-assign tweaks
try:
    from nightmare_v6_tweaks import TweakedAllocator
    from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
    def make_v6_oa(ms, t, o):
        solver = NightmareSolverV6(ms, t, future_orders=o)
        solver.allocator = TweakedAllocator(
            ms, t, solver.drop_zones,
            max_preview_pickers=99, drop_d_weight=0.4,
            over_assign_bonus=3, over_assign_threshold=2)
        return solver
    score = test_solver("V6+OverAssign", make_v6_oa)
    results.append(("V6+OverAssign", score))
except Exception as e:
    print(f"  V6+OverAssign import error: {e}")

# V5b pipeline
try:
    from nightmare_v5_solver import PipelineSolver
    score = test_solver("V5-Pipeline", lambda ms, t, o: PipelineSolver(ms, t, future_orders=o))
    results.append(("V5-Pipeline", score))
except Exception as e:
    print(f"  V5-Pipeline import error: {e}")

print(f"\n{'='*60}")
print(f"RESULTS (seed {SEED}):")
print(f"{'='*60}")
results.sort(key=lambda x: -x[1])
for name, score in results:
    bar = '#' * max(0, score // 5)
    print(f"  {name:20s}: {score:4d}  {bar}")
