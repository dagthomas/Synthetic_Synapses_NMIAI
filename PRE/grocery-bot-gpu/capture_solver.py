"""Standalone expert-capable solver for capture games.

Fixes the planner deadlock issues for 10-bot expert games:
- Oscillation detection (not just same-position stuck)
- Strict delivery throttling (max 2 concurrent deliverers)
- Idle bots with non-useful items clear them at dropoff
- Jitter for stuck bots

Usage:
    # Test offline against game_engine:
    python capture_solver.py --test [--seed 7004] [--difficulty expert]

    # Use as capture bot (called from capture_game.py):
    from capture_solver import CaptureBot
"""
import random
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    DX, DY,
)
from collections import deque
from pathfinding import precompute_all_distances, get_distance, get_first_step
from planner import (
    BotController, ST_IDLE, ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF, ST_PARKED,
    assign_items_globally, optimize_trip_order, find_parking_spots,
    compute_remaining_needs, move_away_from,
)


def bfs_move(start, goal, ms, occupied):
    """BFS pathfind from start to goal, routing around occupied cells.

    Returns (action_id, -1) or (ACT_WAIT, -1) if no path.
    """
    if start == goal:
        return (ACT_WAIT, -1)

    queue = deque([(start, None)])
    visited = {start}

    while queue:
        (x, y), first_act = queue.popleft()
        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            nx, ny = x + DX[act_id], y + DY[act_id]
            if not (0 <= nx < ms.width and 0 <= ny < ms.height):
                continue
            cell = ms.grid[ny, nx]
            if cell != 0 and cell != 3:
                continue
            npos = (nx, ny)
            if npos in visited:
                continue
            if npos in occupied and npos != goal:
                continue
            fa = first_act if first_act is not None else act_id
            if npos == goal:
                return (fa, -1)
            visited.add(npos)
            queue.append((npos, fa))

    return (ACT_WAIT, -1)


def solve_for_capture(seed=None, difficulty='expert', game_factory=None,
                      max_active_bots=None, max_deliverers=2, verbose=True):
    """Solve game with focus on completing orders (for capture).

    Returns (score, action_log, orders_completed).
    """
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)

    ms = state.map_state
    num_bots = len(state.bot_positions)
    dist_maps = precompute_all_distances(ms)
    controllers = [BotController(bid) for bid in range(num_bots)]
    parking = find_parking_spots(ms, dist_maps, num_spots=num_bots * 2)
    rng = random.Random(seed if seed else 42)

    if max_active_bots is None:
        max_active_bots = min(num_bots, 2)

    if verbose:
        print(f"CaptureSolver: {num_bots} bots, map={ms.width}x{ms.height}, "
              f"pickers={max_active_bots}, deliverers={max_deliverers}")

    waiting_to_deliver = set()
    action_log = []

    # Oscillation detection: track last 4 positions per bot
    pos_history = [[] for _ in range(num_bots)]

    for rnd in range(MAX_ROUNDS):
        active = state.get_active_order()
        preview = state.get_preview_order()

        # --- Oscillation & stuck detection ---
        for bc in controllers:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            pos = (bx, by)

            hist = pos_history[bid]
            hist.append(pos)
            if len(hist) > 6:
                hist.pop(0)

            # Same-position stuck
            if bc.last_pos == pos and bc.is_busy():
                bc.stuck_count += 1
            else:
                bc.stuck_count = 0
            bc.last_pos = pos

            # Oscillation: A-B-A-B pattern
            is_oscillating = False
            if len(hist) >= 4 and bc.is_busy():
                if hist[-1] == hist[-3] and hist[-2] == hist[-4] and hist[-1] != hist[-2]:
                    is_oscillating = True

            if bc.stuck_count > 5 or is_oscillating:
                was_delivering = bc.state == ST_MOVING_TO_DROPOFF
                bc.set_idle()
                bc.stuck_count = 0
                if was_delivering and state.bot_inv_list(bid):
                    # Put stuck deliverer in waiting queue so a different bot
                    # gets promoted instead of this one re-assigned immediately
                    waiting_to_deliver.add(bid)
                else:
                    waiting_to_deliver.discard(bid)

        # --- Promote waiting-to-deliver bots ---
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        # Sort by distance to dropoff (closest first)
        def _wait_priority(bid):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            return int(get_distance(dist_maps, (bx, by), ms.drop_off))
        promote_list = sorted(waiting_to_deliver, key=_wait_priority)
        for bid in promote_list:
            if cur_del >= max_deliverers:
                break
            bc = controllers[bid]
            inv = state.bot_inv_list(bid)
            if inv and active and any(active.needs_type(t) for t in inv):
                bc.assign_deliver(ms.drop_off)
                waiting_to_deliver.discard(bid)
                cur_del += 1
            elif not inv:
                waiting_to_deliver.discard(bid)

        # --- Assign pickups ---
        assignments = assign_items_globally(
            state, dist_maps, all_orders, controllers, max_active_bots
        )
        for bid, items in assignments.items():
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            optimized = optimize_trip_order((bx, by), items, ms.drop_off, dist_maps)
            bc.assign_trip(optimized)
            waiting_to_deliver.discard(bid)

        # --- Idle bots with useful items -> deliver (throttled) ---
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        for bc in controllers:
            if bc.state in (ST_IDLE, ST_PARKED) and bc.bot_id not in waiting_to_deliver:
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    if cur_del < max_deliverers:
                        bc.assign_deliver(ms.drop_off)
                        cur_del += 1
                    else:
                        waiting_to_deliver.add(bid)

        # --- Idle bots with dead inventory -> deliver to clear ---
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        for bc in controllers:
            if bc.state == ST_IDLE and bc.bot_id not in waiting_to_deliver:
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                if inv and cur_del < max_deliverers:
                    # Has items but they're not useful for active order
                    # Deliver anyway to clear inventory
                    bc.assign_deliver(ms.drop_off)
                    cur_del += 1

        # --- Preview delivery ---
        if preview and active:
            active_needs = compute_remaining_needs(state, controllers, active, ms)
            uncovered = sum(v for v in active_needs.values() if v > 0)
            if uncovered == 0:
                cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
                for bc in controllers:
                    if bc.state in (ST_IDLE, ST_PARKED) and bc.bot_id not in waiting_to_deliver:
                        inv = state.bot_inv_list(bc.bot_id)
                        if inv and any(preview.needs_type(t) for t in inv):
                            if cur_del < max_deliverers:
                                bc.assign_deliver(ms.drop_off)
                                cur_del += 1
                            else:
                                waiting_to_deliver.add(bc.bot_id)

        # --- Park idle/waiting bots away from dropoff corridor ---
        parked = set(bc.park_target for bc in controllers
                     if bc.state == ST_PARKED and bc.park_target)
        any_deliverer = any(c.state == ST_MOVING_TO_DROPOFF for c in controllers)
        for bc in controllers:
            bid = bc.bot_id
            if bc.state not in (ST_IDLE, ST_PARKED):
                continue
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
            # Park bots near dropoff to clear corridor
            # Also park waiting-to-deliver bots if they're blocking the path
            should_park = False
            if bid not in waiting_to_deliver and d <= 6:
                should_park = True
            elif bid in waiting_to_deliver and d <= 3 and any_deliverer:
                # Waiting bot is too close to dropoff and blocking a deliverer
                # Keep in waiting_to_deliver so it can be promoted later
                should_park = True
            if should_park:
                for spot in parking:
                    if spot not in parked:
                        bc.assign_park(spot)
                        parked.add(spot)
                        break

        # === Generate actions ===
        actions = [(ACT_WAIT, -1)] * num_bots
        committed = {}

        def bot_priority(bc):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            if bc.state == ST_MOVING_TO_DROPOFF:
                if (bx, by) == ms.drop_off:
                    return (0, 0)
                return (1, int(get_distance(dist_maps, (bx, by), ms.drop_off)))
            if bc.state == ST_MOVING_TO_ITEM:
                return (2, 0)
            if bc.state == ST_PARKED:
                return (3, 0)
            return (4, 0)

        for bc in sorted(controllers, key=bot_priority):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            pos = (bx, by)

            # Use committed positions for bots already processed,
            # current positions for bots not yet processed
            occ = set()
            for b2 in range(num_bots):
                if b2 == bid:
                    continue
                if b2 in committed:
                    occ.add(committed[b2])
                else:
                    occ.add((int(state.bot_positions[b2, 0]),
                             int(state.bot_positions[b2, 1])))

            if bc.state == ST_MOVING_TO_DROPOFF:
                if pos == ms.drop_off:
                    inv = state.bot_inv_list(bid)
                    if inv:
                        actions[bid] = (ACT_DROPOFF, -1)
                        committed[bid] = pos
                        bc.set_idle()
                    else:
                        bc.set_idle()
                        act = move_away_from(pos, ms.drop_off, dist_maps, ms, occ)
                        actions[bid] = act
                        if act[0] != ACT_WAIT:
                            committed[bid] = (bx + DX[act[0]], by + DY[act[0]])
                        else:
                            committed[bid] = pos
                    continue

                act = bfs_move(pos, ms.drop_off, ms, occ)
                # If move_toward returns WAIT but we're not there, try jitter
                if act[0] == ACT_WAIT:
                    moves = _valid_moves(bx, by, ms, occ)
                    if moves:
                        act = (rng.choice(moves), -1)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    committed[bid] = (bx + DX[act[0]], by + DY[act[0]])
                else:
                    committed[bid] = pos
                continue

            if bc.state == ST_MOVING_TO_ITEM:
                if pos == bc.target and bc.item_to_pick >= 0:
                    actions[bid] = (ACT_PICKUP, bc.item_to_pick)
                    committed[bid] = pos
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        ni, nc = bc.trip_items[bc.trip_idx]
                        bc.target = nc
                        bc.item_to_pick = ni
                    else:
                        cur_del2 = sum(1 for c in controllers
                                       if c.state == ST_MOVING_TO_DROPOFF)
                        if cur_del2 < max_deliverers:
                            bc.assign_deliver(ms.drop_off)
                        else:
                            bc.set_idle()
                            waiting_to_deliver.add(bid)
                    continue

                # Move toward pickup target
                act = bfs_move(pos, bc.target, ms, occ)
                # Jitter if stuck or WAIT
                if act[0] == ACT_WAIT or bc.stuck_count >= 3:
                    moves = _valid_moves(bx, by, ms, occ)
                    if moves:
                        act = (rng.choice(moves), -1)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    committed[bid] = (bx + DX[act[0]], by + DY[act[0]])
                else:
                    committed[bid] = pos
                continue

            if bc.state == ST_PARKED:
                if bc.park_target and pos != bc.park_target:
                    act = bfs_move(pos, bc.park_target, ms, occ)
                    if act[0] == ACT_WAIT:
                        moves = _valid_moves(bx, by, ms, occ)
                        if moves:
                            act = (rng.choice(moves), -1)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        committed[bid] = (bx + DX[act[0]], by + DY[act[0]])
                    else:
                        committed[bid] = pos
                else:
                    committed[bid] = pos
                continue

            # IDLE
            committed[bid] = pos

        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 5 or rnd % 50 == 0 or rnd >= 295):
            n_pick = sum(1 for c in controllers if c.state == ST_MOVING_TO_ITEM)
            n_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
            n_wait = len(waiting_to_deliver)
            print(f'  R{rnd:3d}: score={state.score:3d} ord={state.orders_completed} '
                  f'pick={n_pick} del={n_del} wait={n_wait}')

    if verbose:
        print(f'\nFinal: score={state.score} orders_completed={state.orders_completed}')

    return state.score, action_log, state.orders_completed


def _valid_moves(bx, by, ms, occ):
    """Return list of valid move action IDs from (bx, by)."""
    moves = []
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = bx + DX[act], by + DY[act]
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell in (0, 3) and (nx, ny) not in occ:
                moves.append(act)
    return moves


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=7004)
    parser.add_argument('--difficulty', default='expert')
    parser.add_argument('--max-del', type=int, default=2)
    parser.add_argument('--seeds', type=int, default=1, help='Run multiple seeds')
    args = parser.parse_args()

    if args.test:
        if args.seeds > 1:
            scores = []
            orders = []
            for s in range(args.seed, args.seed + args.seeds):
                sc, _, oc = solve_for_capture(
                    seed=s, difficulty=args.difficulty,
                    max_deliverers=args.max_del, verbose=False)
                scores.append(sc)
                orders.append(oc)
                print(f'  seed={s}: score={sc} orders={oc}')
            print(f'\nMean score: {sum(scores)/len(scores):.1f}  '
                  f'Mean orders: {sum(orders)/len(orders):.1f}  '
                  f'Max score: {max(scores)}  Max orders: {max(orders)}')
        else:
            solve_for_capture(seed=args.seed, difficulty=args.difficulty,
                              max_deliverers=args.max_del)
