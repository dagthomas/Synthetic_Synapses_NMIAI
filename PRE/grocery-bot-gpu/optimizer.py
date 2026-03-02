"""Iterative optimization solver for Grocery Bot.

Novel approach: instead of making decisions reactively (like Zig) or searching
forward (like beam search), we:
1. Generate an initial solution using any heuristic
2. Save checkpoints of game state every round
3. Systematically try different actions at random rounds
4. Re-simulate from changed point -> keep improvements
5. Repeat thousands of times

This leverages:
- Full order foresight (all orders pre-generated)
- Unlimited offline compute
- Deterministic simulation

The search is in the ACTION SEQUENCE space, not the state space.
"""
import time
import random as py_random
import numpy as np
from game_engine import (
    init_game, step, GameState, MapState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
)
from pathfinding import (
    precompute_all_distances, get_distance, get_first_step,
    get_nearest_item_cell,
)
from action_gen import find_items_of_type, get_active_needed_types


# ── Greedy Policy (fast, used for re-simulation) ─────────────────────

def greedy_action(state, bot_id, dist_maps, all_orders):
    """Fast greedy action for one bot. No persistent state needed."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    bot_pos = (bx, by)
    inv = state.bot_inv_list(bot_id)
    inv_count = len(inv)
    active = state.get_active_order()
    preview = state.get_preview_order()

    if not active:
        return (ACT_WAIT, -1)

    # 1. At dropoff with active items -> deliver
    if bx == ms.drop_off[0] and by == ms.drop_off[1] and inv_count > 0:
        if any(active.needs_type(t) for t in inv):
            return (ACT_DROPOFF, -1)

    # 2. Adjacent to needed item -> pickup
    if inv_count < INV_CAP:
        active_needs = get_active_needed_types(state)
        # Subtract other bots' inventories
        for bid2 in range(num_bots):
            if bid2 == bot_id:
                continue
            for t in state.bot_inv_list(bid2):
                if t in active_needs and active_needs[t] > 0:
                    active_needs[t] -= 1

        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                type_id = int(ms.item_types[item_idx])
                if active_needs.get(type_id, 0) > 0:
                    return (ACT_PICKUP, item_idx)

    # 3. Has active items -> deliver
    if inv_count > 0 and any(active.needs_type(t) for t in inv):
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
        if dist_to_drop > 0:
            act = get_first_step(dist_maps, bot_pos, ms.drop_off)
            if act > 0:
                return (act, -1)

    # 4. Has space -> move to nearest needed item
    if inv_count < INV_CAP:
        active_needs = get_active_needed_types(state)
        for bid2 in range(num_bots):
            if bid2 == bot_id:
                continue
            for t in state.bot_inv_list(bid2):
                if t in active_needs and active_needs[t] > 0:
                    active_needs[t] -= 1

        best_dist = 9999
        best_act = None
        for type_id, count in active_needs.items():
            if count <= 0:
                continue
            for item_idx in find_items_of_type(ms, type_id):
                result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                if result and result[2] < best_dist:
                    best_dist = result[2]
                    cell = (result[0], result[1])
                    act = get_first_step(dist_maps, bot_pos, cell)
                    if act and act > 0:
                        best_act = (act, -1)

        if best_act:
            return best_act

    # 5. Has items -> deliver (preview or cleanup)
    if inv_count > 0:
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
        if dist_to_drop > 0:
            act = get_first_step(dist_maps, bot_pos, ms.drop_off)
            if act > 0:
                return (act, -1)

    # 6. Preview pickup
    if inv_count < INV_CAP and preview:
        preview_needs = {}
        for tid in preview.needs():
            preview_needs[tid] = preview_needs.get(tid, 0) + 1
        for bid2 in range(num_bots):
            for t in state.bot_inv_list(bid2):
                if t in preview_needs and preview_needs[t] > 0:
                    preview_needs[t] -= 1

        best_dist = 9999
        best_act = None
        for type_id, count in preview_needs.items():
            if count <= 0:
                continue
            for item_idx in find_items_of_type(ms, type_id):
                result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                if result and result[2] < best_dist:
                    best_dist = result[2]
                    cell = (result[0], result[1])
                    act = get_first_step(dist_maps, bot_pos, cell)
                    if act and act > 0:
                        best_act = (act, -1)

        if best_act:
            return best_act

    return (ACT_WAIT, -1)


def greedy_all_bots(state, dist_maps, all_orders):
    """Greedy actions for all bots."""
    num_bots = len(state.bot_positions)
    return [greedy_action(state, bid, dist_maps, all_orders) for bid in range(num_bots)]


# ── Generate valid actions for a bot ──────────────────────────────────

def valid_actions(state, bot_id):
    """Return list of all valid (action_type, item_idx) for this bot."""
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    inv_count = state.bot_inv_count(bot_id)
    active = state.get_active_order()

    actions = [(ACT_WAIT, -1)]

    # Move actions
    for act_id, dx, dy in [(1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:
                actions.append((act_id, -1))

    # Pickup
    if inv_count < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                actions.append((ACT_PICKUP, item_idx))

    # Dropoff
    if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
            inv_count > 0 and active):
        actions.append((ACT_DROPOFF, -1))

    return actions


# ── Simulation from checkpoint ────────────────────────────────────────

def simulate_from(checkpoint, action_at_R, rest_policy, dist_maps, all_orders, R):
    """Simulate from checkpoint at round R with modified action, then greedy.

    Returns final score.
    """
    state = checkpoint.copy()
    state.round = R

    # Apply the modified action at round R
    step(state, action_at_R, all_orders)

    # Run greedy policy for remaining rounds
    for rnd in range(R + 1, MAX_ROUNDS):
        state.round = rnd
        actions = rest_policy(state, dist_maps, all_orders)
        step(state, actions, all_orders)

    return state.score


# ── Main Optimizer ────────────────────────────────────────────────────

def optimize(seed, difficulty, iterations=5000, time_limit=60.0, verbose=True,
             initial_actions=None):
    """Optimize action sequence using iterative local search.

    1. Generate initial solution (greedy or provided) with checkpoints
    2. For each iteration:
       a. Pick random round R, bot B
       b. Try different action for bot B at round R
       c. Re-simulate from round R with greedy policy
       d. If score improved, update solution
    3. Return best score and action log
    """
    t0 = time.time()
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Optimizer: {difficulty} seed={seed} bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  Distance maps: {len(dist_maps)} cells, {time.time()-t0:.1f}s")

    # Phase 1: Generate initial solution with checkpoints
    checkpoints = [None] * MAX_ROUNDS
    action_log = [None] * MAX_ROUNDS

    if initial_actions is not None:
        # Use provided initial actions
        current_state = state
        for rnd in range(MAX_ROUNDS):
            current_state.round = rnd
            checkpoints[rnd] = current_state.copy()
            actions = initial_actions[rnd]
            action_log[rnd] = list(actions)
            step(current_state, actions, all_orders)
    else:
        # Generate from beam search (greedy, beam_width=1)
        from beam_search import beam_search
        _, beam_actions, _ = beam_search(seed, difficulty, beam_width=1,
                                          max_per_bot=4, verbose=False)
        current_state = state
        for rnd in range(MAX_ROUNDS):
            current_state.round = rnd
            checkpoints[rnd] = current_state.copy()
            if rnd < len(beam_actions):
                actions = beam_actions[rnd]
            else:
                actions = greedy_all_bots(current_state, dist_maps, all_orders)
            action_log[rnd] = list(actions)
            step(current_state, actions, all_orders)

    best_score = current_state.score
    best_action_log = [list(a) for a in action_log]

    if verbose:
        print(f"  Initial score: {best_score} ({time.time()-t0:.1f}s)")

    # Phase 2: Iterative improvement
    improvements = 0
    rng = py_random.Random(42)

    for it in range(iterations):
        if time.time() - t0 > time_limit:
            break

        # Pick random round and bot
        R = rng.randint(0, MAX_ROUNDS - 10)  # avoid very last rounds
        B = rng.randint(0, num_bots - 1)

        # Get valid actions for this bot at this round
        checkpoint = checkpoints[R]
        va = valid_actions(checkpoint, B)
        if len(va) <= 1:
            continue

        # Current action for this bot
        current_act = best_action_log[R][B]

        # Pick a random different action
        alt_act = rng.choice(va)
        while alt_act == current_act and len(va) > 1:
            alt_act = rng.choice(va)

        if alt_act == current_act:
            continue

        # Build modified action set for round R
        modified_actions = list(best_action_log[R])
        modified_actions[B] = alt_act

        # Re-simulate from round R
        new_score = simulate_from(
            checkpoint, modified_actions, greedy_all_bots,
            dist_maps, all_orders, R,
        )

        if new_score > best_score:
            improvements += 1
            old_score = best_score
            best_score = new_score

            # Reconstruct the full action log from R onwards
            sim_state = checkpoint.copy()
            sim_state.round = R
            best_action_log[R] = modified_actions
            step(sim_state, modified_actions, all_orders)

            # Update checkpoints and action log from R+1 onwards
            for rnd in range(R + 1, MAX_ROUNDS):
                sim_state.round = rnd
                checkpoints[rnd] = sim_state.copy()
                actions = greedy_all_bots(sim_state, dist_maps, all_orders)
                best_action_log[rnd] = actions
                step(sim_state, actions, all_orders)

            if verbose:
                elapsed = time.time() - t0
                print(f"  it={it}: score {old_score} -> {best_score} "
                      f"(R={R}, B={B}, {alt_act}) [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nOptimization complete: score={best_score}, "
              f"improvements={improvements}/{iterations}, "
              f"time={elapsed:.1f}s")

    return best_score, best_action_log


def multi_pass_optimize(seed, difficulty, passes=3, iterations_per_pass=3000,
                        time_limit=120.0, verbose=True):
    """Run multiple optimization passes, each starting from the previous best.

    Between passes, we can also try different base policies.
    """
    t0 = time.time()

    # Pass 1: Optimize from beam search baseline
    best_score, best_actions = optimize(
        seed, difficulty,
        iterations=iterations_per_pass,
        time_limit=time_limit / passes,
        verbose=verbose,
        initial_actions=None,  # will use beam search
    )

    for p in range(1, passes):
        remaining_time = time_limit - (time.time() - t0)
        if remaining_time < 5:
            break

        if verbose:
            print(f"\n--- Pass {p+1} ---")

        # Re-optimize: start from current best, use different random seed
        state, all_orders = init_game(seed, difficulty)
        dist_maps = precompute_all_distances(state.map_state)

        # Replay best actions to get checkpoints
        checkpoints = [None] * MAX_ROUNDS
        current_state = state
        for rnd in range(MAX_ROUNDS):
            current_state.round = rnd
            checkpoints[rnd] = current_state.copy()
            step(current_state, best_actions[rnd], all_orders)

        # Now optimize from these checkpoints
        num_bots = len(state.bot_positions)
        rng = py_random.Random(42 + p * 1000)
        improvements = 0

        it_count = 0
        while it_count < iterations_per_pass:
            if time.time() - t0 > time_limit:
                break
            it_count += 1

            R = rng.randint(0, MAX_ROUNDS - 10)
            B = rng.randint(0, num_bots - 1)

            checkpoint = checkpoints[R]
            va = valid_actions(checkpoint, B)
            if len(va) <= 1:
                continue

            current_act = best_actions[R][B]
            alt_act = rng.choice(va)
            while alt_act == current_act and len(va) > 1:
                alt_act = rng.choice(va)
            if alt_act == current_act:
                continue

            modified_actions = list(best_actions[R])
            modified_actions[B] = alt_act

            new_score = simulate_from(
                checkpoint, modified_actions, greedy_all_bots,
                dist_maps, all_orders, R,
            )

            if new_score > best_score:
                improvements += 1
                old_score = best_score
                best_score = new_score

                sim_state = checkpoint.copy()
                sim_state.round = R
                best_actions[R] = modified_actions
                step(sim_state, modified_actions, all_orders)

                for rnd in range(R + 1, MAX_ROUNDS):
                    sim_state.round = rnd
                    checkpoints[rnd] = sim_state.copy()
                    actions = greedy_all_bots(sim_state, dist_maps, all_orders)
                    best_actions[rnd] = actions
                    step(sim_state, actions, all_orders)

                if verbose:
                    elapsed = time.time() - t0
                    print(f"  it={it_count}: score {old_score} -> {best_score} "
                          f"(R={R}, B={B}) [{elapsed:.1f}s]")

        if verbose:
            print(f"  Pass {p+1}: {improvements} improvements")

    if verbose:
        total_time = time.time() - t0
        print(f"\nFinal score: {best_score} (total time: {total_time:.1f}s)")

    return best_score, best_actions


if __name__ == '__main__':
    import sys

    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    time_lim = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

    score, actions = multi_pass_optimize(
        seed, difficulty,
        passes=3,
        iterations_per_pass=iters,
        time_limit=time_lim,
        verbose=True,
    )
