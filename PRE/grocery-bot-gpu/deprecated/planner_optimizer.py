"""DEPRECATED: Optimizer that uses the planner as both initial solution and rest policy.

This module is not part of the active production pipeline. Kept for reference.

Key difference from optimizer.py: when a perturbation is tried, the remaining
rounds use the PLANNER logic (with proper BotController state) instead of a
simple greedy policy. This means perturbations are evaluated in the context
of high-quality future play.
"""
import time
import math
import random as py_random
import copy
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS, DX, DY,
)
from pathfinding import precompute_all_distances, get_distance, get_first_step
from action_gen import find_items_of_type, get_active_needed_types, get_preview_needed_types
from planner import (
    BotController, ST_IDLE, ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF, ST_PARKED,
    find_best_adj_cell, optimize_trip_order, assign_items_globally,
    move_toward_avoiding, move_away_from, find_parking_spots,
    compute_remaining_needs,
)


def save_controllers(controllers):
    """Deep copy controller states."""
    saved = []
    for bc in controllers:
        s = BotController(bc.bot_id)
        s.state = bc.state
        s.trip_items = list(bc.trip_items)
        s.trip_idx = bc.trip_idx
        s.target = bc.target
        s.item_to_pick = bc.item_to_pick
        s.stuck_count = bc.stuck_count
        s.last_pos = bc.last_pos
        s.park_target = bc.park_target
        saved.append(s)
    return saved


def restore_controllers(saved):
    """Restore controllers from saved state."""
    controllers = []
    for s in saved:
        bc = BotController(s.bot_id)
        bc.state = s.state
        bc.trip_items = list(s.trip_items)
        bc.trip_idx = s.trip_idx
        bc.target = s.target
        bc.item_to_pick = s.item_to_pick
        bc.stuck_count = s.stuck_count
        bc.last_pos = s.last_pos
        bc.park_target = s.park_target
        controllers.append(bc)
    return controllers


def planner_round(state, controllers, dist_maps, all_orders, parking_spots, max_active_bots):
    """Execute one round of planner logic. Returns actions list.

    This is the planner's per-round logic extracted for reuse.
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    preview = state.get_preview_order()

    # Generate actions following planner logic
    # Order: stuck detection → assign → deliver → staging → park (matches planner.py)

    # Check for bots that completed trips or are stuck
    for bc in controllers:
        bid = bc.bot_id
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)

        if bc.last_pos == bot_pos and bc.is_busy():
            bc.stuck_count += 1
        else:
            bc.stuck_count = 0
        bc.last_pos = bot_pos

        if bc.stuck_count > 8:
            bc.set_idle()

    # Assign items to idle bots
    assignments = assign_items_globally(state, dist_maps, all_orders, controllers, max_active_bots)
    for bid, items in assignments.items():
        bc = controllers[bid]
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        optimized = optimize_trip_order((bx, by), items, ms.drop_off, dist_maps)
        bc.assign_trip(optimized)

    # Check deliveries
    for bc in controllers:
        if bc.state in (ST_IDLE, ST_PARKED):
            bid = bc.bot_id
            inv = state.bot_inv_list(bid)
            if inv and active and any(active.needs_type(t) for t in inv):
                bc.assign_deliver(ms.drop_off)

    # Auto-delivery staging: route bots with preview items to dropoff
    if preview:
        active_uncovered = 0
        if active:
            active_needs_left = {}
            for tid in active.needs():
                active_needs_left[tid] = active_needs_left.get(tid, 0) + 1
            for bid2 in range(num_bots):
                for t in state.bot_inv_list(bid2):
                    if t in active_needs_left and active_needs_left[t] > 0:
                        active_needs_left[t] -= 1
            for bc2 in controllers:
                if bc2.state == ST_MOVING_TO_ITEM:
                    for item_idx, _ in bc2.trip_items[bc2.trip_idx:]:
                        tid = int(ms.item_types[item_idx])
                        if tid in active_needs_left and active_needs_left[tid] > 0:
                            active_needs_left[tid] -= 1
            active_uncovered = sum(v for v in active_needs_left.values() if v > 0)

        if active_uncovered == 0:
            for bc in controllers:
                if bc.state in (ST_IDLE, ST_PARKED):
                    bid = bc.bot_id
                    inv = state.bot_inv_list(bid)
                    if inv and any(preview.needs_type(t) for t in inv):
                        bc.assign_deliver(ms.drop_off)

    # Park idle bots near dropoff or blocking delivery
    parked_spots = set()
    for bc in controllers:
        if bc.state == ST_PARKED and bc.park_target:
            parked_spots.add(bc.park_target)

    for bc in controllers:
        if bc.state == ST_IDLE:
            bid = bc.bot_id
            inv = state.bot_inv_list(bid)
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            dist_drop = int(get_distance(dist_maps, (bx, by), ms.drop_off))

            if inv and preview and any(preview.needs_type(t) for t in inv):
                continue

            blocking = False
            for bc2 in controllers:
                if bc2.state == ST_MOVING_TO_DROPOFF and bc2.bot_id != bid:
                    b2x = int(state.bot_positions[bc2.bot_id, 0])
                    b2y = int(state.bot_positions[bc2.bot_id, 1])
                    d_b2_drop = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                    d_b2_me = int(get_distance(dist_maps, (b2x, b2y), (bx, by)))
                    if d_b2_me + dist_drop <= d_b2_drop + 2:
                        blocking = True
                        break
            if blocking or dist_drop <= 3:
                for spot in parking_spots:
                    if spot not in parked_spots:
                        bc.assign_park(spot)
                        parked_spots.add(spot)
                        break

    # Priority ordering
    def bot_priority(bc):
        bid = bc.bot_id
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        if bc.state == ST_MOVING_TO_DROPOFF:
            if (bx, by) == ms.drop_off:
                return (0, 0)
            d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
            return (1, d)
        if bc.state == ST_MOVING_TO_ITEM:
            return (2, 0)
        if bc.state == ST_PARKED:
            return (3, 0)
        return (4, 0)

    sorted_bots = sorted(controllers, key=bot_priority)

    actions = [(ACT_WAIT, -1)] * num_bots
    committed_positions = {}

    for bc in sorted_bots:
        bid = bc.bot_id
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)

        occ_after = set()
        for bid2 in range(num_bots):
            if bid2 in committed_positions:
                occ_after.add(committed_positions[bid2])
            elif bid2 != bid:
                occ_after.add((int(state.bot_positions[bid2, 0]),
                               int(state.bot_positions[bid2, 1])))

        if bc.state == ST_MOVING_TO_DROPOFF:
            if bot_pos == ms.drop_off:
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                    committed_positions[bid] = bot_pos
                    bc.set_idle()
                    continue
                elif inv and preview and any(preview.needs_type(t) for t in inv):
                    # Preview items at dropoff. Yield if active delivery approaching.
                    must_yield = False
                    for bc2 in controllers:
                        if bc2.bot_id == bid:
                            continue
                        if bc2.state == ST_MOVING_TO_DROPOFF:
                            inv2 = state.bot_inv_list(bc2.bot_id)
                            if inv2 and active and any(active.needs_type(t) for t in inv2):
                                b2x = int(state.bot_positions[bc2.bot_id, 0])
                                b2y = int(state.bot_positions[bc2.bot_id, 1])
                                d = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                                if d <= 2:
                                    must_yield = True
                                    break
                    if must_yield:
                        act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                        actions[bid] = act
                        if act[0] != ACT_WAIT:
                            dx, dy = DX[act[0]], DY[act[0]]
                            committed_positions[bid] = (bx + dx, by + dy)
                        else:
                            committed_positions[bid] = bot_pos
                    else:
                        actions[bid] = (ACT_WAIT, -1)
                        committed_positions[bid] = bot_pos
                    continue
                else:
                    bc.set_idle()

            if bc.state == ST_MOVING_TO_DROPOFF:
                act = move_toward_avoiding(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    dx, dy = DX[act[0]], DY[act[0]]
                    committed_positions[bid] = (bx + dx, by + dy)
                else:
                    committed_positions[bid] = bot_pos
                continue

        if bc.state == ST_MOVING_TO_ITEM:
            if bot_pos == bc.target and bc.item_to_pick >= 0:
                actions[bid] = (ACT_PICKUP, bc.item_to_pick)
                committed_positions[bid] = bot_pos
                bc.trip_idx += 1
                if bc.trip_idx < len(bc.trip_items):
                    next_item, next_cell = bc.trip_items[bc.trip_idx]
                    bc.target = next_cell
                    bc.item_to_pick = next_item
                else:
                    bc.assign_deliver(ms.drop_off)
                continue
            else:
                act = move_toward_avoiding(bot_pos, bc.target, dist_maps, ms, occ_after)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    dx, dy = DX[act[0]], DY[act[0]]
                    committed_positions[bid] = (bx + dx, by + dy)
                else:
                    committed_positions[bid] = bot_pos
                continue

        if bc.state == ST_PARKED:
            if bc.park_target is None:
                act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    dx, dy = DX[act[0]], DY[act[0]]
                    new_pos = (bx + dx, by + dy)
                    committed_positions[bid] = new_pos
                    d_new = int(get_distance(dist_maps, new_pos, ms.drop_off))
                    if d_new > 4:
                        bc.set_idle()
                else:
                    committed_positions[bid] = bot_pos
            elif bot_pos != bc.park_target:
                act = move_toward_avoiding(bot_pos, bc.park_target, dist_maps, ms, occ_after)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    dx, dy = DX[act[0]], DY[act[0]]
                    committed_positions[bid] = (bx + dx, by + dy)
                else:
                    committed_positions[bid] = bot_pos
            else:
                any_heading_here = any(
                    bc2.bot_id != bid and bc2.target == bot_pos
                    for bc2 in controllers
                )
                if any_heading_here:
                    act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        committed_positions[bid] = bot_pos
                else:
                    committed_positions[bid] = bot_pos
            continue

        committed_positions[bid] = bot_pos

    return actions


def generate_trip_actions(state, bot_id, dist_maps, all_orders, rng):
    """Generate a complete trip plan for a bot: navigate to item, pickup, navigate to dropoff, dropoff.

    Returns list of (action_type, item_idx) for each round of the trip,
    or empty list if no valid trip found.
    """
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    inv_count = state.bot_inv_count(bot_id)
    active = state.get_active_order()

    if not active or inv_count >= INV_CAP:
        return []

    # Find a needed item type
    needed_types = list(active.needs())
    if not needed_types:
        preview = state.get_preview_order()
        if preview:
            needed_types = list(preview.needs())
    if not needed_types:
        return []

    target_type = rng.choice(needed_types)

    # Find all items of this type and pick a random one
    items_of_type = find_items_of_type(ms, target_type)
    if not items_of_type:
        return []

    item_idx = rng.choice(items_of_type)

    # Find best adjacent cell to the item
    adj_cell, d_to_item = find_best_adj_cell(dist_maps, (bx, by), item_idx, ms)
    if not adj_cell or d_to_item >= 9999:
        return []

    # Generate path: bot -> adj_cell -> pickup -> dropoff -> dropoff
    trip_actions = []

    # Phase 1: Navigate to adjacent cell
    pos = (bx, by)
    for _ in range(d_to_item + 5):  # allow some extra steps for detours
        if pos == adj_cell:
            break
        act = get_first_step(dist_maps, pos, adj_cell)
        if act <= 0:
            break
        trip_actions.append((act, -1))
        dx, dy = DX[act], DY[act]
        pos = (pos[0] + dx, pos[1] + dy)

    if pos != adj_cell:
        return []  # Couldn't reach item

    # Phase 2: Pickup
    trip_actions.append((ACT_PICKUP, item_idx))

    # Phase 3: Navigate to dropoff
    d_to_drop = int(get_distance(dist_maps, adj_cell, ms.drop_off))
    pos = adj_cell
    for _ in range(d_to_drop + 5):
        if pos == ms.drop_off:
            break
        act = get_first_step(dist_maps, pos, ms.drop_off)
        if act <= 0:
            break
        trip_actions.append((act, -1))
        dx, dy = DX[act], DY[act]
        pos = (pos[0] + dx, pos[1] + dy)

    if pos != ms.drop_off:
        return []  # Couldn't reach dropoff

    # Phase 4: Dropoff
    trip_actions.append((ACT_DROPOFF, -1))

    return trip_actions


def generate_valid_actions(state, bot_id):
    """All valid actions for one bot."""
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    inv_count = state.bot_inv_count(bot_id)
    active = state.get_active_order()

    actions = [(ACT_WAIT, -1)]
    for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        dx, dy = DX[act_id], DY[act_id]
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:
                actions.append((act_id, -1))
    if inv_count < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                actions.append((ACT_PICKUP, item_idx))
    if bx == ms.drop_off[0] and by == ms.drop_off[1] and inv_count > 0 and active:
        actions.append((ACT_DROPOFF, -1))

    return actions


def optimize_planner(seed=None, difficulty=None, iterations=10000, time_limit=240.0,
                     max_active_bots=None, verbose=True, game_factory=None,
                     random_seed=42):
    """Optimize using planner as both initial solution and rest policy."""
    t0 = time.time()
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Planner-Optimizer: bots={num_bots}")

    dist_maps = precompute_all_distances(ms)
    parking_spots = find_parking_spots(ms, dist_maps, num_spots=num_bots * 2)

    if max_active_bots is None:
        max_active_bots = min(num_bots, 2)

    # Phase 1: Generate initial solution with checkpoints
    controllers = [BotController(bid) for bid in range(num_bots)]
    checkpoints = [None] * MAX_ROUNDS  # (game_state, controller_state)
    action_log = [None] * MAX_ROUNDS

    current_state = state
    last_orders_completed = 0
    last_active_order_id = -1

    for rnd in range(MAX_ROUNDS):
        current_state.round = rnd
        active = current_state.get_active_order()
        active_id = active.id if active else -1

        # Order change detection
        order_changed = False
        if current_state.orders_completed > last_orders_completed:
            order_changed = True
            last_orders_completed = current_state.orders_completed
        if active_id != last_active_order_id:
            order_changed = True
            last_active_order_id = active_id

        if order_changed:
            for bc in controllers:
                if bc.state == ST_MOVING_TO_ITEM:
                    bc.set_idle()

        # Save checkpoint
        checkpoints[rnd] = (current_state.copy(), save_controllers(controllers))

        # Generate actions using planner
        actions = planner_round(current_state, controllers, dist_maps, all_orders,
                               parking_spots, max_active_bots)
        action_log[rnd] = list(actions)
        step(current_state, actions, all_orders)

    best_score = current_state.score
    best_action_log = [list(a) for a in action_log]

    # Best-ever tracking (SA may accept downgrades, but we always return the best)
    best_ever_score = best_score
    best_ever_action_log = [list(a) for a in best_action_log]

    if verbose:
        print(f"  Initial: {best_score} ({time.time()-t0:.1f}s)")

    # Phase 2: Iterative improvement with multi-round perturbation
    improvements = 0
    rng = py_random.Random(random_seed)
    stale_count = 0  # iterations since last improvement

    # Simulated annealing parameters
    sa_temperature = 5.0
    sa_alpha = 0.9995
    sa_floor = 0.1

    # Track the score at each round in the best solution for early termination
    best_round_scores = [0] * MAX_ROUNDS

    def rebuild_best_scores():
        """Rebuild score progression for the best solution."""
        game_cp, ctrl_cp = checkpoints[0]
        sim_state = game_cp.copy()
        sim_ctrls = restore_controllers(ctrl_cp)
        sim_oc = 0
        sim_aid = -1
        a = sim_state.get_active_order()
        if a:
            sim_aid = a.id
        for rnd in range(MAX_ROUNDS):
            sim_state.round = rnd
            a = sim_state.get_active_order()
            aid = a.id if a else -1
            if rnd > 0:
                oc_chg = False
                if sim_state.orders_completed > sim_oc:
                    oc_chg = True; sim_oc = sim_state.orders_completed
                if aid != sim_aid:
                    oc_chg = True; sim_aid = aid
                if oc_chg:
                    for bc in sim_ctrls:
                        if bc.state == ST_MOVING_TO_ITEM:
                            bc.set_idle()
            actions = planner_round(sim_state, sim_ctrls, dist_maps, all_orders,
                                   parking_spots, max_active_bots)
            step(sim_state, actions, all_orders)
            best_round_scores[rnd] = sim_state.score

    rebuild_best_scores()

    def simulate_from(start_round, modified_rounds, all_orders):
        """Simulate from start_round applying modified actions, planner for rest.

        modified_rounds: dict of {round: actions_list} for perturbed rounds.
        Returns final score.
        """
        game_cp, ctrl_cp = checkpoints[start_round]
        sim_state = game_cp.copy()
        sim_controllers = restore_controllers(ctrl_cp)

        sim_last_oc = sim_state.orders_completed
        sim_last_aid = -1
        a = sim_state.get_active_order()
        if a:
            sim_last_aid = a.id

        for rnd in range(start_round, MAX_ROUNDS):
            sim_state.round = rnd
            a = sim_state.get_active_order()
            aid = a.id if a else -1

            if rnd > start_round:
                order_changed = False
                if sim_state.orders_completed > sim_last_oc:
                    order_changed = True
                    sim_last_oc = sim_state.orders_completed
                if aid != sim_last_aid:
                    order_changed = True
                    sim_last_aid = aid
                if order_changed:
                    for bc in sim_controllers:
                        if bc.state == ST_MOVING_TO_ITEM:
                            bc.set_idle()

            if rnd in modified_rounds:
                actions = modified_rounds[rnd]
            else:
                actions = planner_round(sim_state, sim_controllers, dist_maps, all_orders,
                                       parking_spots, max_active_bots)

            step(sim_state, actions, all_orders)

            # Early termination: if falling behind best by too much, give up
            # Higher tolerance for SA (current solution may be below initial planner)
            max_deficit = 40 if num_bots >= 6 else 25
            if rnd > start_round + 30 and rnd < MAX_ROUNDS - 5:
                deficit = best_round_scores[rnd] - sim_state.score
                if deficit > max_deficit:
                    return sim_state.score  # Can't recover

        return sim_state.score

    def accept_improvement(start_round, modified_rounds):
        """Rebuild the full solution from start_round with accepted modifications."""
        game_cp, ctrl_cp = checkpoints[start_round]
        sim_state = game_cp.copy()
        sim_controllers = restore_controllers(ctrl_cp)

        sim_last_oc = sim_state.orders_completed
        sim_last_aid = -1
        a = sim_state.get_active_order()
        if a:
            sim_last_aid = a.id

        for rnd in range(start_round, MAX_ROUNDS):
            sim_state.round = rnd
            a = sim_state.get_active_order()
            aid = a.id if a else -1

            if rnd > start_round:
                order_changed = False
                if sim_state.orders_completed > sim_last_oc:
                    order_changed = True
                    sim_last_oc = sim_state.orders_completed
                if aid != sim_last_aid:
                    order_changed = True
                    sim_last_aid = aid
                if order_changed:
                    for bc in sim_controllers:
                        if bc.state == ST_MOVING_TO_ITEM:
                            bc.set_idle()

            checkpoints[rnd] = (sim_state.copy(), save_controllers(sim_controllers))

            if rnd in modified_rounds:
                actions = modified_rounds[rnd]
            else:
                actions = planner_round(sim_state, sim_controllers, dist_maps, all_orders,
                                       parking_spots, max_active_bots)

            best_action_log[rnd] = list(actions)
            step(sim_state, actions, all_orders)

    def pick_active_bot(R):
        """Pick a bot biased toward active ones (70% active, 30% random)."""
        game_cp, ctrl_cp = checkpoints[R]
        active_bots = [b for b in range(num_bots)
                       if best_action_log[R][b][0] != ACT_WAIT]
        if active_bots and rng.random() < 0.7:
            return rng.choice(active_bots)
        return rng.randint(0, num_bots - 1)

    def pick_stagnant_round():
        """Pick a round from a stagnant region (where score isn't increasing).

        For expert, score often plateaus for 200+ rounds. Targeting these
        regions finds deadlock-breaking perturbations faster.
        """
        # Build list of stagnant regions (consecutive rounds with same score)
        stagnant_starts = []
        for r in range(5, MAX_ROUNDS - 20):
            if best_round_scores[r] == best_round_scores[r - 5]:
                stagnant_starts.append(r)

        if stagnant_starts and rng.random() < 0.7:
            return rng.choice(stagnant_starts)
        return rng.randint(0, MAX_ROUNDS - 20)

    # Perturbation mix depends on number of bots
    # Expert (10 bots) needs more multi-bot and scramble perturbations
    is_expert = num_bots >= 6

    for it in range(iterations):
        if time.time() - t0 > time_limit:
            break

        # Choose perturbation type (weighted by difficulty)
        ptype = rng.random()

        if is_expert:
            # Expert: 20% single, 8% multi-round, 12% multi-bot, 8% scramble, 12% wide, 10% bot-swap, 12% delay, 18% trip
            thresholds = (0.20, 0.28, 0.40, 0.48, 0.60, 0.70, 0.82)
        else:
            # Normal: 40% single, 20% multi-round, 12% multi-bot, 0% scramble(guard), 5% wide, 10% bot-swap, 13% delay, 0% trip
            thresholds = (0.40, 0.60, 0.72, 1.01, 0.77, 0.87, 1.00)

        if ptype < thresholds[0]:
            # Single-round, single-bot perturbation
            R = pick_stagnant_round() if is_expert else rng.randint(0, MAX_ROUNDS - 20)
            B = pick_active_bot(R)

            game_cp, ctrl_cp = checkpoints[R]
            va = generate_valid_actions(game_cp, B)
            if len(va) <= 1:
                continue

            current_act = best_action_log[R][B]
            alt_act = rng.choice(va)
            while alt_act == current_act and len(va) > 1:
                alt_act = rng.choice(va)
            if alt_act == current_act:
                continue

            modified_actions = list(best_action_log[R])
            modified_actions[B] = alt_act
            modified_rounds = {R: modified_actions}

        elif ptype < thresholds[1]:
            # Multi-round perturbation: same bot, 2-5 consecutive rounds
            window = rng.randint(2, 5)
            R = rng.randint(0, MAX_ROUNDS - window - 10)
            B = pick_active_bot(R)

            modified_rounds = {}
            valid = True
            for r in range(R, R + window):
                if checkpoints[r] is None:
                    valid = False
                    break
                game_cp_r, _ = checkpoints[r]
                va = generate_valid_actions(game_cp_r, B)
                if len(va) <= 1:
                    valid = False
                    break
                alt_act = rng.choice(va)
                mod = list(best_action_log[r])
                mod[B] = alt_act
                modified_rounds[r] = mod

            if not valid or not modified_rounds:
                continue

        elif ptype < thresholds[2]:
            # Multi-bot perturbation: 2-4 bots in same round
            R = pick_stagnant_round() if is_expert else rng.randint(0, MAX_ROUNDS - 20)
            if num_bots < 2:
                continue

            n_pert_bots = min(rng.randint(2, 4), num_bots)
            pert_bots = rng.sample(range(num_bots), n_pert_bots)

            game_cp, ctrl_cp = checkpoints[R]
            modified_actions = list(best_action_log[R])
            any_valid = False
            for B in pert_bots:
                va = generate_valid_actions(game_cp, B)
                if len(va) > 1:
                    modified_actions[B] = rng.choice(va)
                    any_valid = True

            if not any_valid:
                continue
            modified_rounds = {R: modified_actions}

        elif ptype < thresholds[3] and is_expert:
            # SCRAMBLE: randomize ALL bots for 1-3 rounds (expert only)
            # Breaks total gridlock by giving all bots random actions
            window = rng.randint(1, 3)
            R = pick_stagnant_round()

            modified_rounds = {}
            for r in range(R, R + window):
                if checkpoints[r] is None:
                    break
                game_cp_r, _ = checkpoints[r]
                mod = list(best_action_log[r])
                for B in range(num_bots):
                    va = generate_valid_actions(game_cp_r, B)
                    if len(va) > 1:
                        mod[B] = rng.choice(va)
                modified_rounds[r] = mod

            if not modified_rounds:
                continue

        elif ptype < thresholds[4]:
            # Wider multi-round: single bot, longer window
            max_window = 30 if is_expert else 15
            window = rng.randint(5, max_window)
            R = rng.randint(0, MAX_ROUNDS - window - 10)
            B = rng.randint(0, num_bots - 1)  # Intentionally random for exploration

            modified_rounds = {}
            valid = True
            for r in range(R, R + window):
                if checkpoints[r] is None:
                    valid = False
                    break
                game_cp_r, _ = checkpoints[r]
                va = generate_valid_actions(game_cp_r, B)
                if len(va) <= 1:
                    valid = False
                    break
                alt_act = rng.choice(va)
                mod = list(best_action_log[r])
                mod[B] = alt_act
                modified_rounds[r] = mod

            if not valid or not modified_rounds:
                continue

        elif ptype < thresholds[5]:
            # BOT-SWAP: swap two bots' actions for a window of rounds
            if num_bots < 2:
                continue
            B1, B2 = rng.sample(range(num_bots), 2)
            window = rng.randint(3, 15)
            R = pick_stagnant_round() if is_expert else rng.randint(0, max(1, MAX_ROUNDS - window - 10))
            if R + window >= MAX_ROUNDS:
                R = max(0, MAX_ROUNDS - window - 10)

            modified_rounds = {}
            for r in range(R, min(R + window, MAX_ROUNDS)):
                if best_action_log[r] is None:
                    break
                mod = list(best_action_log[r])
                mod[B1], mod[B2] = mod[B2], mod[B1]
                modified_rounds[r] = mod

            if not modified_rounds:
                continue

        elif ptype < thresholds[6]:
            # DELAY: insert/remove wait actions to shift one bot's timing
            B = rng.randint(0, num_bots - 1)
            R = rng.randint(5, MAX_ROUNDS - 30)
            delay = rng.randint(1, 3)
            insert = rng.random() < 0.5

            modified_rounds = {}
            shift_window = delay + 15

            if insert:
                # Insert wait rounds, shift subsequent actions later
                for i in range(shift_window):
                    r = R + i
                    if r >= MAX_ROUNDS:
                        break
                    mod = list(best_action_log[r])
                    if i < delay:
                        mod[B] = (ACT_WAIT, -1)
                    else:
                        src_r = R + (i - delay)
                        if 0 <= src_r < MAX_ROUNDS and best_action_log[src_r] is not None:
                            mod[B] = best_action_log[src_r][B]
                    modified_rounds[r] = mod
            else:
                # Remove delay: skip ahead by delay rounds for this bot
                for i in range(shift_window):
                    r = R + i
                    src_r = R + i + delay
                    if r >= MAX_ROUNDS:
                        break
                    mod = list(best_action_log[r])
                    if src_r < MAX_ROUNDS and best_action_log[src_r] is not None:
                        mod[B] = best_action_log[src_r][B]
                    else:
                        mod[B] = (ACT_WAIT, -1)
                    modified_rounds[r] = mod

            if not modified_rounds:
                continue

        else:
            # TRIP INJECTION: generate complete trip plan for an idle bot (expert)
            # Picks an idle bot, plans a full pickup-deliver route, injects it
            R = pick_stagnant_round() if is_expert else rng.randint(0, MAX_ROUNDS - 60)
            if R > MAX_ROUNDS - 40:
                R = rng.randint(0, MAX_ROUNDS - 60)

            game_cp, ctrl_cp = checkpoints[R]

            # Find idle bots (not the ones the planner is using)
            idle_bots = [b for b in range(num_bots)
                         if best_action_log[R][b][0] == ACT_WAIT]
            if not idle_bots:
                # Try any bot
                idle_bots = list(range(num_bots))

            B = rng.choice(idle_bots)
            trip = generate_trip_actions(game_cp, B, dist_maps, all_orders, rng)
            if not trip or len(trip) < 3:
                continue

            # Cap trip length to remaining rounds
            max_trip = min(len(trip), MAX_ROUNDS - R - 5)
            trip = trip[:max_trip]

            modified_rounds = {}
            for i, trip_act in enumerate(trip):
                r = R + i
                if r >= MAX_ROUNDS:
                    break
                mod = list(best_action_log[r])
                mod[B] = trip_act
                modified_rounds[r] = mod

            if not modified_rounds:
                continue

        # Evaluate perturbation
        start_round = min(modified_rounds.keys())
        new_score = simulate_from(start_round, modified_rounds, all_orders)

        # Simulated annealing acceptance
        accepted = False
        if new_score > best_score:
            accepted = True
        elif sa_temperature > sa_floor and new_score >= best_score - sa_temperature * 10:
            delta = new_score - best_score
            if delta < 0 and math.exp(delta / sa_temperature) > rng.random():
                accepted = True

        if accepted:
            improvements += 1
            stale_count = 0
            old_score = best_score
            best_score = new_score
            accept_improvement(start_round, modified_rounds)

            # Track best-ever (SA may accept downgrades)
            if new_score > best_ever_score:
                best_ever_score = new_score
                best_ever_action_log = [list(a) for a in best_action_log]

            ptype_name = "single" if len(modified_rounds) == 1 else f"multi-{len(modified_rounds)}r"
            if verbose and new_score > old_score:
                print(f"  it={it}: {old_score} -> {best_score} ({ptype_name},R={start_round},T={sa_temperature:.2f}) [{time.time()-t0:.1f}s]")
        else:
            stale_count += 1

        sa_temperature = max(sa_floor, sa_temperature * sa_alpha)

    if verbose:
        print(f"  Final: {best_ever_score}, {improvements} improvements ({time.time()-t0:.1f}s)")

    return best_ever_score, best_ever_action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    time_lim = float(sys.argv[4]) if len(sys.argv) > 4 else 240.0

    score, actions = optimize_planner(seed, difficulty, iterations=iters,
                                       time_limit=time_lim, verbose=True)
