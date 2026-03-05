"""DEPRECATED: CPU beam search over game states.

This module is not part of the active production pipeline. Kept for reference.

Phase 1: Pure CPU implementation for correctness validation.
Phase 2: Replace step() with GPU kernel (gpu_sim.py).
"""
import time
import numpy as np
from game_engine import (
    init_game, step, GameState, MAX_ROUNDS, ACT_WAIT, INV_CAP,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_DROPOFF,
)
from pathfinding import precompute_all_distances, get_distance
from action_gen import (
    generate_joint_actions, get_active_needed_types, get_preview_needed_types,
    get_future_needed_types,
)


def generate_all_valid_actions_single_bot(state, dist_maps):
    """Generate ALL valid actions for a single bot (no heuristic pruning).

    Returns list of [(action_type, item_idx)] lists (one inner list per candidate).
    """
    ms = state.map_state
    bx = int(state.bot_positions[0, 0])
    by = int(state.bot_positions[0, 1])
    grid = ms.grid
    actions = []

    # Move actions (all 4 directions if walkable)
    for act, dx, dy in [(ACT_MOVE_UP, 0, -1), (ACT_MOVE_DOWN, 0, 1),
                         (ACT_MOVE_LEFT, -1, 0), (ACT_MOVE_RIGHT, 1, 0)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = grid[ny, nx]
            if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                actions.append([(act, -1)])

    # Pickup (if adjacent to item and have space)
    if state.bot_inv_count(0) < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                actions.append([(ACT_PICKUP, item_idx)])

    # Dropoff (if at dropoff with items)
    if bx == ms.drop_off[0] and by == ms.drop_off[1] and state.bot_inv_count(0) > 0:
        active = state.get_active_order()
        if active and any(active.needs_type(t) for t in state.bot_inv_list(0)):
            actions.append([(ACT_DROPOFF, -1)])

    # Wait always valid
    actions.append([(ACT_WAIT, -1)])

    return actions


def eval_state(state, all_orders, dist_maps):
    """Evaluate a game state for beam pruning.

    Higher is better. Considers:
    - Score (dominant factor)
    - Items in transit matching active/preview orders
    - Distance of useful bots to their targets
    - Dead inventory penalty
    """
    score = state.score * 100000

    active = state.get_active_order()
    preview = state.get_preview_order()
    active_needs = get_active_needed_types(state)
    preview_needs = get_preview_needed_types(state)

    # Find the current order index for future type check
    current_oi = 0
    for o in state.orders:
        if not o.complete and o.status == 'active':
            current_oi = o.id
            break
    future_types = get_future_needed_types(all_orders, current_oi)

    ms = state.map_state
    num_bots = len(state.bot_positions)

    # Count remaining active needs
    active_remaining = sum(1 for _ in active.needs()) if active else 0

    for bid in range(num_bots):
        inv = state.bot_inv_list(bid)
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))

        for type_id in inv:
            if active and active.needs_type(type_id):
                # Active item in transit: very good. Closer to dropoff = better.
                score += 2000 - dist_to_drop * 50
            elif preview and preview.needs_type(type_id):
                # Preview item: good if active is mostly covered
                if active_remaining <= 1:
                    score += 500 - dist_to_drop * 10
                else:
                    score -= 200
            elif type_id in future_types:
                score -= 300
            else:
                # Dead inventory
                score -= 2000

    # Bonus for completing active order items
    if active:
        needs = active.needs()
        total_req = len(active.required)
        delivered = total_req - len(needs)
        score += delivered * 1000
        if len(needs) == 0:
            score += 5000
        elif len(needs) == 1:
            score += 2000
        elif len(needs) == 2:
            score += 500

    # Tiebreaker: bots with active items should be close to dropoff
    for bid in range(num_bots):
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
        inv = state.bot_inv_list(bid)
        has_active = any(active.needs_type(t) for t in inv) if (active and inv) else False

        if has_active:
            score -= dist_to_drop * 5

    return score


def beam_search(seed=None, difficulty=None, beam_width=100, max_per_bot=3, max_joint=500, verbose=True,
                use_smart_policy=False, game_factory=None):
    """Run beam search to find optimal action sequence.

    Args:
        seed: Game seed
        difficulty: 'easy', 'medium', 'hard', 'expert'
        beam_width: Number of states to keep per round
        max_per_bot: Max candidate actions per bot
        max_joint: Max joint actions to evaluate per state
        verbose: Print progress
        game_factory: Optional callable returning (GameState, all_orders)

    Returns:
        (best_score, best_actions, stats_dict)
    """
    t0 = time.time()
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)
    ms = state.map_state

    if verbose:
        print(f"Building distance maps for {difficulty} ({ms.width}x{ms.height})...")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  {len(dist_maps)} walkable cells, distance maps built in {time.time()-t0:.1f}s")
        print(f"  Orders pre-generated: {len(all_orders)}")
        print(f"  First 5 orders: {[list(o.required) for o in all_orders[:5]]}")

    # Initialize beam with single start state
    beam = [(state, [])]  # list of (state, action_history)

    stats = {
        'total_candidates': 0,
        'peak_beam': 0,
        'round_times': [],
    }

    for rnd in range(MAX_ROUNDS):
        t_round = time.time()
        candidates = []

        for state_obj, action_history in beam:
            # Generate candidate joint actions
            if use_smart_policy:
                from smart_policy import smart_policy_with_variants
                joint_actions = smart_policy_with_variants(
                    state_obj, all_orders, dist_maps,
                    num_variants=max_per_bot,
                )
            else:
                joint_actions = generate_joint_actions(
                    state_obj, dist_maps, all_orders,
                    max_per_bot=max_per_bot,
                    max_joint=max_joint,
                )

            for actions in joint_actions:
                # Copy state and apply actions
                new_state = state_obj.copy()
                new_state.round = rnd
                step(new_state, actions, all_orders)

                ev = eval_state(new_state, all_orders, dist_maps)
                candidates.append((ev, new_state, action_history + [actions]))

        stats['total_candidates'] += len(candidates)

        # Sort by eval (descending) and prune to beam_width
        candidates.sort(key=lambda x: -x[0])
        beam = [(s, ah) for _, s, ah in candidates[:beam_width]]
        stats['peak_beam'] = max(stats['peak_beam'], len(beam))

        dt = time.time() - t_round
        stats['round_times'].append(dt)

        if verbose and (rnd < 10 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
            best_score = beam[0][0].score if beam else 0
            worst_score = beam[-1][0].score if beam else 0
            n_cand = len(candidates)
            print(f"  R{rnd:3d}: score={best_score:3d} ({worst_score}-{best_score}), "
                  f"cands={n_cand:5d}, beam={len(beam)}, dt={dt:.3f}s")

    # Find best final state
    best_idx = 0
    best_score = 0
    for i, (state_obj, _) in enumerate(beam):
        if state_obj.score > best_score:
            best_score = state_obj.score
            best_idx = i

    total_time = time.time() - t0
    best_actions = beam[best_idx][1]

    if verbose:
        print(f"\nBeam search complete: score={best_score}, "
              f"time={total_time:.1f}s, "
              f"avg_round={np.mean(stats['round_times']):.3f}s")

    stats['total_time'] = total_time
    return best_score, best_actions, stats


def greedy_search(seed, difficulty, verbose=True):
    """Simple greedy baseline: pick the best single action per bot each round.

    Equivalent to beam_width=1. Useful as a baseline.
    """
    return beam_search(seed, difficulty, beam_width=1, max_per_bot=3,
                       max_joint=500, verbose=verbose)


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    beam_width = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    use_smart = '--smart' in sys.argv

    mode = "smart+beam" if use_smart else "beam"
    print(f"{mode} search: {difficulty} seed={seed} beam_width={beam_width}")
    score, actions, stats = beam_search(seed, difficulty, beam_width=beam_width,
                                         use_smart_policy=use_smart)
    print(f"\nFinal score: {score}")
    print(f"Actions logged: {len(actions)} rounds")
