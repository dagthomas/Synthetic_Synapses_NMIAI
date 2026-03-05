"""DEPRECATED: Two-phase planner for expert: proven mab=3 base + additional bot actions.

This module is not part of the active production pipeline. Kept for reference.

Phase 1: Run v1 planner with mab=3 (reliable, no deadlock)
Phase 2: Simulate game, add actions for idle bots 3-9 by planning trips
         that avoid known positions of the phase 1 bots.

The result has 6-10 bots doing useful work, providing a much better
starting point for the optimizer.
"""
import time
import numpy as np
from collections import deque
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    DX, DY,
)
from pathfinding import precompute_all_distances, get_distance, get_first_step
from action_gen import find_items_of_type
from planner import (
    solve as v1_solve,
    BotController, ST_IDLE, ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF, ST_PARKED,
    find_best_adj_cell, optimize_trip_order, compute_remaining_needs,
    move_toward_avoiding, bfs_first_step_avoiding,
)


def simulate_and_record(game_factory, action_log):
    """Simulate game with given actions, record positions per round."""
    state, all_orders = game_factory()
    num_bots = len(state.bot_positions)
    positions = []  # positions[round] = [(x,y) for each bot]

    for rnd in range(min(len(action_log), MAX_ROUNDS)):
        state.round = rnd
        pos = [(int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
               for b in range(num_bots)]
        positions.append(pos)
        step(state, action_log[rnd], all_orders)

    # Fill remaining rounds
    while len(positions) < MAX_ROUNDS:
        pos = [(int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
               for b in range(num_bots)]
        positions.append(pos)

    return positions, state.score


def bfs_avoiding_bots(start, goal, ms, occupied_this_round):
    """BFS from start to goal, avoiding occupied cells this round.
    Returns first action or ACT_WAIT if no path."""
    if start == goal:
        return ACT_WAIT

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (x, y), first_act = queue.popleft()
        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            dx, dy = DX[act_id], DY[act_id]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < ms.width and 0 <= ny < ms.height):
                continue
            cell = ms.grid[ny, nx]
            if cell != 0 and cell != 3:
                continue
            npos = (nx, ny)
            if npos in visited:
                continue
            if npos in occupied_this_round and npos != ms.spawn:
                continue
            fa = first_act if first_act != 0 else act_id
            if npos == goal:
                return fa
            visited.add(npos)
            queue.append((npos, fa))

    return ACT_WAIT


def plan_additional_bots(game_factory, base_actions, phase1_bots, max_extra_deliverers=2, verbose=False):
    """Add actions for idle bots on top of phase 1 actions.

    Simulates the game round by round. For bots not in phase1_bots,
    plans trips (pick items → deliver) avoiding occupied positions.
    """
    state, all_orders = game_factory()
    ms = state.map_state
    num_bots = len(state.bot_positions)
    dist_maps = precompute_all_distances(ms)

    extra_bots = [b for b in range(num_bots) if b not in phase1_bots]
    if not extra_bots:
        return base_actions

    # Copy base actions
    new_actions = [list(a) for a in base_actions]
    while len(new_actions) < MAX_ROUNDS:
        new_actions.append([(ACT_WAIT, -1)] * num_bots)

    # Controllers for extra bots only
    controllers = {}
    for bid in extra_bots:
        controllers[bid] = BotController(bid)

    last_active_id = -1

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()
        active_id = active.id if active else -1

        # Order change: reset extra picking bots
        if active_id != last_active_id:
            for bid in extra_bots:
                bc = controllers[bid]
                if bc.state == ST_MOVING_TO_ITEM:
                    bc.set_idle()
            last_active_id = active_id

        # Stuck detection for extra bots
        for bid in extra_bots:
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)
            if bc.last_pos == bot_pos and bc.is_busy():
                bc.stuck_count += 1
            else:
                bc.stuck_count = 0
            bc.last_pos = bot_pos
            if bc.stuck_count > 6:
                bc.set_idle()

        # Build occupied set from ALL bots (phase1 committed + extra bots)
        occupied = set()
        for bid in range(num_bots):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            occupied.add((bx, by))

        # Also predict where phase1 bots will be after their action
        phase1_next = set()
        for bid in phase1_bots:
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            act = new_actions[rnd][bid][0] if rnd < len(new_actions) else ACT_WAIT
            if ACT_MOVE_UP <= act <= ACT_MOVE_RIGHT:
                dx, dy = DX[act], DY[act]
                nx, ny = bx + dx, by + dy
                phase1_next.add((nx, ny))
            else:
                phase1_next.add((bx, by))

        # Assign items to idle extra bots
        if active:
            # Count items already being picked by all bots (phase1 + extra)
            needs = compute_remaining_needs(state, list(controllers.values()), active, ms)
            remaining = []
            for tid, count in needs.items():
                for _ in range(count):
                    remaining.append(tid)

            if not remaining and preview:
                needs = compute_remaining_needs(state, list(controllers.values()), preview, ms)
                remaining = []
                for tid, count in needs.items():
                    for _ in range(count):
                        remaining.append(tid)

            # Assign to idle extra bots
            for bid in extra_bots:
                bc = controllers[bid]
                if bc.state != ST_IDLE or not remaining:
                    continue
                inv_count = state.bot_inv_count(bid)
                if inv_count >= INV_CAP:
                    continue

                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                bot_pos = (bx, by)

                # Find closest needed item
                best_cost = 9999
                best_item = None
                best_cell = None
                best_ni = -1

                for ni, type_id in enumerate(remaining):
                    for item_idx in find_items_of_type(ms, type_id):
                        adj_cell, d = find_best_adj_cell(dist_maps, bot_pos, item_idx, ms)
                        if adj_cell and d < best_cost:
                            best_cost = d
                            best_item = item_idx
                            best_cell = adj_cell
                            best_ni = ni

                if best_item is not None:
                    bc.assign_trip([(best_item, best_cell)])
                    remaining.pop(best_ni)

        # Generate actions for extra bots
        # Count current deliverers among extra bots
        extra_deliverers = sum(1 for bid in extra_bots
                               if controllers[bid].state == ST_MOVING_TO_DROPOFF)

        for bid in extra_bots:
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

            # Collision avoidance: don't move to where phase1 bots are going
            occ = occupied | phase1_next
            occ.discard(bot_pos)

            if bc.state == ST_MOVING_TO_DROPOFF:
                if bot_pos == ms.drop_off:
                    inv = state.bot_inv_list(bid)
                    if inv and active and any(active.needs_type(t) for t in inv):
                        new_actions[rnd][bid] = (ACT_DROPOFF, -1)
                        bc.set_idle()
                        continue
                    else:
                        bc.set_idle()

                if bc.state == ST_MOVING_TO_DROPOFF:
                    act = bfs_avoiding_bots(bot_pos, ms.drop_off, ms, occ)
                    if act != ACT_WAIT:
                        new_actions[rnd][bid] = (act, -1)
                    continue

            if bc.state == ST_MOVING_TO_ITEM:
                if bot_pos == bc.target and bc.item_to_pick >= 0:
                    new_actions[rnd][bid] = (ACT_PICKUP, bc.item_to_pick)
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        next_item, next_cell = bc.trip_items[bc.trip_idx]
                        bc.target = next_cell
                        bc.item_to_pick = next_item
                    else:
                        if extra_deliverers < max_extra_deliverers:
                            bc.assign_deliver(ms.drop_off)
                            extra_deliverers += 1
                        else:
                            bc.set_idle()
                    continue
                else:
                    act = bfs_avoiding_bots(bot_pos, bc.target, ms, occ)
                    if act != ACT_WAIT:
                        new_actions[rnd][bid] = (act, -1)
                    continue

            # Check if idle bot should deliver
            if bc.state == ST_IDLE:
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    if extra_deliverers < max_extra_deliverers:
                        bc.assign_deliver(ms.drop_off)
                        extra_deliverers += 1

        step(state, new_actions[rnd], all_orders)

    if verbose:
        print(f"  Phase 2 score: {state.score}")
    return new_actions, state.score


def solve_two_phase(seed=None, difficulty=None, verbose=True, max_active_bots=3,
                    max_extra_deliverers=2, game_factory=None):
    """Two-phase solve: v1 base + additional bot planning."""
    t0 = time.time()

    if game_factory is None:
        gf = lambda: init_game(seed, difficulty)
    else:
        gf = game_factory

    # Phase 1: v1 planner with mab=3
    if verbose:
        print(f"Phase 1: v1 planner mab={max_active_bots}")
    phase1_score, base_actions = v1_solve(
        game_factory=gf, verbose=False, max_active_bots=max_active_bots
    )
    if verbose:
        print(f"  Phase 1 score: {phase1_score}")

    # Determine which bots were active in phase 1
    # (bots that ever did non-WAIT actions)
    phase1_bots = set()
    for rnd_actions in base_actions:
        for bid, (act, _) in enumerate(rnd_actions):
            if act != ACT_WAIT:
                phase1_bots.add(bid)

    if verbose:
        print(f"  Phase 1 active bots: {sorted(phase1_bots)}")
        print(f"Phase 2: adding bots, max_extra_del={max_extra_deliverers}")

    # Phase 2: add extra bot actions
    new_actions, final_score = plan_additional_bots(
        gf, base_actions, phase1_bots,
        max_extra_deliverers=max_extra_deliverers,
        verbose=verbose,
    )

    if verbose:
        improvement = final_score - phase1_score
        print(f"\nFinal: {final_score} (phase1={phase1_score} +{improvement}) ({time.time()-t0:.1f}s)")

    return final_score, new_actions


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'expert'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    mab = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    med = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    score, actions = solve_two_phase(seed, difficulty, max_active_bots=mab,
                                      max_extra_deliverers=med)
