"""Sequential per-bot GPU DP solver for multi-bot games with iterative refinement.

Two-pass approach:
  Pass 1 (Sequential): Bot 0 solo, Bot 1 with 0 locked, Bot 2 with 0,1 locked, ...
  Pass 2+ (Refinement): Re-plan each bot with ALL other bots locked.
    This fixes collision displacement from Pass 1 (e.g., bot 0 blocked by bot 1).

Each bot's DP stays single-bot sized (~200K states), making it GPU-tractable.

Usage:
    from gpu_sequential_solver import solve_sequential
    score, actions = solve_sequential(capture, device='cuda')
"""
import sys
import time
import numpy as np
import torch

from game_engine import (
    init_game, init_game_from_capture, step as cpu_step,
    GameState, MapState, Order,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
    DX, DY,
)
from gpu_beam_search import GPUBeamSearcher
from configs import CONFIGS


def pre_simulate_locked(gs_template, all_orders, bot_actions, locked_bot_ids):
    """CPU simulation of locked bots to get their per-round positions.

    Args:
        gs_template: Initial GameState (all bots at spawn, round 0).
        all_orders: Full order list.
        bot_actions: Dict {bot_id: [(act, item)] * 300} for all planned bots.
        locked_bot_ids: Sorted list of bot IDs to lock.

    Returns:
        locked_trajectories dict ready for GPU upload:
            locked_actions: [num_locked, 300] int8
            locked_action_items: [num_locked, 300] int16
            locked_pos_x: [num_locked, 300] int16
            locked_pos_y: [num_locked, 300] int16
    """
    num_locked = len(locked_bot_ids)
    if num_locked == 0:
        return None

    gs = gs_template.copy()
    num_total_bots = len(gs.bot_positions)
    locked_id_set = set(locked_bot_ids)

    locked_actions = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int8)
    locked_action_items = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
    locked_pos_x = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
    locked_pos_y = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)

    # Record actions for locked bots
    for i, bid in enumerate(locked_bot_ids):
        for r in range(MAX_ROUNDS):
            a, item = bot_actions[bid][r]
            locked_actions[i, r] = a
            locked_action_items[i, r] = item

    # Simulate ALL bots with their current actions to get accurate positions
    # and order state. Bots without actions (including the candidate being
    # re-planned) wait. In refinement, the candidate bot's OLD actions are
    # included in bot_actions, giving correct order state for locked bots.
    for r in range(MAX_ROUNDS):
        round_actions = []
        for bid in range(num_total_bots):
            if bid in bot_actions:
                round_actions.append(bot_actions[bid][r])
            else:
                round_actions.append((ACT_WAIT, -1))

        gs.round = r
        cpu_step(gs, round_actions, all_orders)

        # Record positions AFTER this round
        for i, bid in enumerate(locked_bot_ids):
            locked_pos_x[i, r] = int(gs.bot_positions[bid, 0])
            locked_pos_y[i, r] = int(gs.bot_positions[bid, 1])

    return {
        'locked_actions': locked_actions,
        'locked_action_items': locked_action_items,
        'locked_pos_x': locked_pos_x,
        'locked_pos_y': locked_pos_y,
        'locked_bot_ids': locked_bot_ids,
    }


def cpu_verify(gs_template, all_orders, combined_actions, num_bots):
    """Verify final score by replaying all bots' combined actions on CPU.

    Args:
        gs_template: Initial GameState.
        all_orders: Full order list.
        combined_actions: List of 300 round_actions, each is [(act, item)] * num_bots.
        num_bots: Number of bots.

    Returns:
        Final score from CPU simulation.
    """
    gs = gs_template.copy()
    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)
    return gs.score


def _make_combined(bot_actions, num_bots):
    """Convert bot_actions dict to per-round combined format."""
    combined = []
    for r in range(MAX_ROUNDS):
        round_acts = []
        for bid in range(num_bots):
            round_acts.append(bot_actions[bid][r])
        combined.append(round_acts)
    return combined


def _fresh_gs(gs_orig, capture_data, no_filler=False):
    """Get a fresh copy of the game state."""
    if capture_data:
        num_orders = len(capture_data['orders']) if no_filler else 100
        return init_game_from_capture(capture_data, num_orders=num_orders)[0]
    return gs_orig.copy()


def solve_sequential(capture_data=None, seed=None, difficulty=None,
                     device='cuda', max_states=500000, verbose=True,
                     on_bot_progress=None, on_round=None, on_phase=None,
                     max_refine_iters=2, bot_order=None, no_filler=False):
    """Sequential per-bot GPU DP with iterative refinement.

    Pass 1: Sequential planning (bot 0 solo, bot 1 with 0 locked, ...).
    Pass 2+: Refinement — re-plan each bot with ALL other bots locked.
    Repeats up to max_refine_iters times or until no improvement.

    Args:
        capture_data: Captured game data (preferred).
        seed: Game seed (if no capture).
        difficulty: Game difficulty.
        device: 'cuda' or 'cpu'.
        max_states: Max states per bot DP search.
        verbose: Print progress.
        on_bot_progress: Optional callback(bot_id, num_bots, score, time).
        on_round: Optional callback(bot_id, rnd, score, unique, expanded, elapsed).
        on_phase: Optional callback(phase_name, iteration, cpu_score).
        max_refine_iters: Max refinement iterations (0 = no refinement).
        bot_order: Optional list of bot IDs for Pass 1 planning order.
                   Default: [0, 1, ..., num_bots-1].

    Returns:
        (final_score, combined_actions) where combined_actions is
        list of 300 round_actions, each is [(act, item)] * num_bots.
    """
    t0 = time.time()

    # Initialize game
    if capture_data:
        num_orders = len(capture_data['orders']) if no_filler else 100
        gs, all_orders = init_game_from_capture(capture_data, num_orders=num_orders)
        diff = capture_data.get('difficulty', difficulty or 'easy')
    elif seed and difficulty:
        gs, all_orders = init_game(seed, difficulty)
        diff = difficulty
    else:
        raise ValueError("Need capture_data or (seed + difficulty)")

    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    if verbose:
        filler_str = f", no_filler ({len(capture_data['orders'])} orders)" if (capture_data and no_filler) else ""
        print(f"Sequential GPU DP: {diff}, {num_bots} bots, "
              f"max_states={max_states}, refine_iters={max_refine_iters}{filler_str}",
              file=sys.stderr)

    # For single-bot, just run standard DP (no refinement needed)
    if num_bots == 1:
        searcher = GPUBeamSearcher(ms, all_orders, device=device, num_bots=num_bots)
        if verbose:
            gs_v = gs.copy()
            ok = searcher.verify_against_cpu(gs_v, all_orders, num_rounds=100)
            if not ok:
                print("VERIFICATION FAILED", file=sys.stderr)
                return 0, []

        def round_cb(rnd, score, unique, expanded, elapsed):
            if on_round:
                on_round(0, rnd, score, unique, expanded, elapsed)

        score, bot_acts = searcher.dp_search(
            gs.copy(), max_states=max_states, verbose=verbose, on_round=round_cb)

        wait_pad = [(ACT_WAIT, -1)] * (num_bots - 1)
        actions = [[(a, i)] + wait_pad for a, i in bot_acts]

        if on_bot_progress:
            on_bot_progress(0, num_bots, score, time.time() - t0)

        return score, actions

    # ===== Multi-bot: Sequential DP + Iterative Refinement =====
    bot_actions = {}  # bot_id -> [(act, item)] * 300

    # Planning order for Pass 1
    plan_order = bot_order if bot_order is not None else list(range(num_bots))

    # --- Pass 1: Sequential planning ---
    if on_phase:
        on_phase("pass1", 0, None)
    if verbose:
        order_str = ','.join(str(b) for b in plan_order)
        print(f"\n--- Pass 1: Sequential planning (order: {order_str}) ---",
              file=sys.stderr)

    for bot_id in plan_order:
        t_bot = time.time()
        if verbose:
            print(f"\n=== Pass 1, Bot {bot_id}/{num_bots} ===", file=sys.stderr)

        # Lock all previously-planned bots
        locked_ids = sorted(bot_actions.keys())
        locked = None
        if locked_ids:
            locked = pre_simulate_locked(
                _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions, locked_ids)

        searcher = GPUBeamSearcher(
            ms, all_orders, device=device, num_bots=num_bots,
            locked_trajectories=locked)

        # Verify on first bot planned only
        if bot_id == plan_order[0] and verbose:
            gs_v = _fresh_gs(gs, capture_data, no_filler)
            ok = searcher.verify_against_cpu(gs_v, all_orders, num_rounds=100)
            if not ok:
                print("VERIFICATION FAILED", file=sys.stderr)
                return 0, []

        def round_cb(rnd, score, unique, expanded, elapsed, _bid=bot_id):
            if on_round:
                on_round(_bid, rnd, score, unique, expanded, elapsed)

        gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
        dp_score, bot_acts = searcher.dp_search(
            gs_for_dp, max_states=max_states, verbose=verbose,
            on_round=round_cb, bot_id=bot_id)

        bot_actions[bot_id] = bot_acts

        bot_time = time.time() - t_bot
        if verbose:
            print(f"  Bot {bot_id}: DP score={dp_score}, time={bot_time:.1f}s",
                  file=sys.stderr)

        if on_bot_progress:
            on_bot_progress(bot_id, num_bots, dp_score, time.time() - t0)

        del searcher
        if device == 'cuda':
            torch.cuda.empty_cache()

    # CPU verify after Pass 1
    combined = _make_combined(bot_actions, num_bots)
    gs_v = _fresh_gs(gs, capture_data, no_filler)
    best_score = cpu_verify(gs_v, all_orders, combined, num_bots)
    best_actions = {k: list(v) for k, v in bot_actions.items()}

    if on_phase:
        on_phase("pass1_done", 0, best_score)
    if verbose:
        print(f"\nPass 1 CPU verify: score={best_score}", file=sys.stderr)

    # --- Pass 2+: Per-bot Refinement with Immediate CPU Verify ---
    # Re-plan one bot at a time, immediately verify, keep only improvements.
    for iteration in range(max_refine_iters):
        if on_phase:
            on_phase(f"refine", iteration + 1, best_score)
        if verbose:
            print(f"\n--- Refinement iteration {iteration+1}/{max_refine_iters} ---",
                  file=sys.stderr)

        iter_improved = False

        for bot_id in range(num_bots):
            t_bot = time.time()
            if verbose:
                print(f"\n=== Refine iter {iteration+1}, Bot {bot_id}/{num_bots} ===",
                      file=sys.stderr)

            # Lock ALL other bots (using current best actions)
            locked_ids = sorted(b for b in range(num_bots) if b != bot_id)
            locked = pre_simulate_locked(
                _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions, locked_ids)

            searcher = GPUBeamSearcher(
                ms, all_orders, device=device, num_bots=num_bots,
                locked_trajectories=locked)

            def round_cb(rnd, score, unique, expanded, elapsed, _bid=bot_id):
                if on_round:
                    on_round(_bid, rnd, score, unique, expanded, elapsed)

            gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
            dp_score, bot_acts = searcher.dp_search(
                gs_for_dp, max_states=max_states, verbose=verbose,
                on_round=round_cb, bot_id=bot_id)

            del searcher
            if device == 'cuda':
                torch.cuda.empty_cache()

            bot_time = time.time() - t_bot

            # Try replacing this bot's actions and immediately CPU verify
            old_acts = bot_actions[bot_id]
            bot_actions[bot_id] = bot_acts

            combined = _make_combined(bot_actions, num_bots)
            gs_v = _fresh_gs(gs, capture_data, no_filler)
            new_score = cpu_verify(gs_v, all_orders, combined, num_bots)

            if new_score > best_score:
                delta = new_score - best_score
                best_score = new_score
                best_actions = {k: list(v) for k, v in bot_actions.items()}
                iter_improved = True
                if verbose:
                    print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                          f"(+{delta}! best={best_score}), time={bot_time:.1f}s",
                          file=sys.stderr)
            else:
                # Revert this bot's actions
                bot_actions[bot_id] = old_acts
                if verbose:
                    print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                          f"(no improvement, reverted), time={bot_time:.1f}s",
                          file=sys.stderr)

            if on_bot_progress:
                on_bot_progress(bot_id, num_bots, best_score, time.time() - t0)

        if on_phase:
            on_phase(f"refine_done", iteration + 1, best_score)

        if verbose:
            print(f"\nRefine iter {iteration+1}: best_score={best_score}",
                  file=sys.stderr)

        if not iter_improved:
            if verbose:
                print(f"  Stopping refinement (no improvement)", file=sys.stderr)
            break

    # Final combined actions from best
    combined = _make_combined(best_actions, num_bots)

    total_time = time.time() - t0
    if verbose:
        print(f"\nSequential GPU DP: final_score={best_score}, "
              f"total_time={total_time:.1f}s", file=sys.stderr)

    return best_score, combined


def solve_multi_restart(capture_data=None, seed=None, difficulty=None,
                        device='cuda', max_states=500000, verbose=True,
                        max_refine_iters=2, num_restarts=3,
                        on_restart=None):
    """Run sequential solver with multiple random bot orderings, keep best.

    Args:
        num_restarts: Number of random bot orderings to try. First is always
                      the default [0,1,...,N-1] order.
        on_restart: Optional callback(restart_idx, bot_order, score, best_score).

    Returns:
        (best_score, best_actions).
    """
    import random

    # Determine num_bots
    if capture_data:
        gs_tmp, _ = init_game_from_capture(capture_data)
    elif seed and difficulty:
        gs_tmp, _ = init_game(seed, difficulty)
    else:
        raise ValueError("Need capture_data or (seed + difficulty)")
    num_bots = len(gs_tmp.bot_positions)
    del gs_tmp

    if num_bots == 1:
        return solve_sequential(
            capture_data=capture_data, seed=seed, difficulty=difficulty,
            device=device, max_states=max_states, verbose=verbose,
            max_refine_iters=0)

    # Generate bot orderings
    orderings = [list(range(num_bots))]  # default order first
    rng = random.Random(42)
    while len(orderings) < num_restarts:
        order = list(range(num_bots))
        rng.shuffle(order)
        if order not in orderings:
            orderings.append(order)

    best_score = -1
    best_actions = None

    for i, order in enumerate(orderings):
        if verbose:
            order_str = ','.join(str(b) for b in order)
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"RESTART {i+1}/{num_restarts}: order=[{order_str}]",
                  file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

        score, actions = solve_sequential(
            capture_data=capture_data, seed=seed, difficulty=difficulty,
            device=device, max_states=max_states, verbose=verbose,
            max_refine_iters=max_refine_iters, bot_order=order)

        if score > best_score:
            best_score = score
            best_actions = actions
            if verbose:
                print(f"\n*** New best: {score} (restart {i+1}, order {order}) ***",
                      file=sys.stderr)
        else:
            if verbose:
                print(f"\nRestart {i+1}: score={score} (best remains {best_score})",
                      file=sys.stderr)

        if on_restart:
            on_restart(i, order, score, best_score)

    return best_score, best_actions


def refine_from_solution(combined_actions, capture_data=None, seed=None,
                         difficulty=None, device='cuda', max_states=500000,
                         verbose=True, max_refine_iters=3,
                         on_bot_progress=None, on_round=None, on_phase=None):
    """Refine an existing multi-bot solution via GPU DP.

    Loads a pre-existing solution (e.g., from a previous GPU DP run or Python
    planner), then runs refinement iterations to improve individual bot paths.

    Args:
        combined_actions: List of 300 round_actions, each is [(act, item)] * num_bots.
        Other args same as solve_sequential.

    Returns:
        (best_score, best_combined_actions).
    """
    t0 = time.time()

    # Initialize game
    if capture_data:
        gs, all_orders = init_game_from_capture(capture_data)
        diff = capture_data.get('difficulty', difficulty or 'easy')
    elif seed and difficulty:
        gs, all_orders = init_game(seed, difficulty)
        diff = difficulty
    else:
        raise ValueError("Need capture_data or (seed + difficulty)")

    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    # Convert combined_actions to per-bot format
    bot_actions = {}
    for bid in range(num_bots):
        bot_actions[bid] = [(r_acts[bid][0], r_acts[bid][1])
                            for r_acts in combined_actions]

    # Verify starting score
    gs_v = gs.copy() if not capture_data else init_game_from_capture(capture_data)[0]
    best_score = cpu_verify(gs_v, all_orders, combined_actions, num_bots)
    best_actions = {k: list(v) for k, v in bot_actions.items()}

    if verbose:
        print(f"Warm-start refinement: {diff}, {num_bots} bots, "
              f"starting_score={best_score}, max_states={max_states}, "
              f"refine_iters={max_refine_iters}", file=sys.stderr)

    if on_phase:
        on_phase("warm_start", 0, best_score)

    # Run refinement iterations (same as in solve_sequential)
    for iteration in range(max_refine_iters):
        if on_phase:
            on_phase("refine", iteration + 1, best_score)
        if verbose:
            print(f"\n--- Refinement iteration {iteration+1}/{max_refine_iters} ---",
                  file=sys.stderr)

        iter_improved = False

        for bot_id in range(num_bots):
            t_bot = time.time()
            if verbose:
                print(f"\n=== Refine iter {iteration+1}, Bot {bot_id}/{num_bots} ===",
                      file=sys.stderr)

            locked_ids = sorted(b for b in range(num_bots) if b != bot_id)
            locked = pre_simulate_locked(
                _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions, locked_ids)

            searcher = GPUBeamSearcher(
                ms, all_orders, device=device, num_bots=num_bots,
                locked_trajectories=locked)

            def round_cb(rnd, score, unique, expanded, elapsed, _bid=bot_id):
                if on_round:
                    on_round(_bid, rnd, score, unique, expanded, elapsed)

            gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
            dp_score, bot_acts = searcher.dp_search(
                gs_for_dp, max_states=max_states, verbose=verbose,
                on_round=round_cb, bot_id=bot_id)

            del searcher
            if device == 'cuda':
                torch.cuda.empty_cache()

            bot_time = time.time() - t_bot

            # Try replacing and verify
            old_acts = bot_actions[bot_id]
            bot_actions[bot_id] = bot_acts

            combined = _make_combined(bot_actions, num_bots)
            gs_v = _fresh_gs(gs, capture_data, no_filler)
            new_score = cpu_verify(gs_v, all_orders, combined, num_bots)

            if new_score > best_score:
                delta = new_score - best_score
                best_score = new_score
                best_actions = {k: list(v) for k, v in bot_actions.items()}
                iter_improved = True
                if verbose:
                    print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                          f"(+{delta}! best={best_score}), time={bot_time:.1f}s",
                          file=sys.stderr)
            else:
                bot_actions[bot_id] = old_acts
                if verbose:
                    print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                          f"(no improvement, reverted), time={bot_time:.1f}s",
                          file=sys.stderr)

            if on_bot_progress:
                on_bot_progress(bot_id, num_bots, best_score, time.time() - t0)

        if on_phase:
            on_phase("refine_done", iteration + 1, best_score)

        if verbose:
            print(f"\nRefine iter {iteration+1}: best_score={best_score}",
                  file=sys.stderr)

        if not iter_improved:
            if verbose:
                print(f"  Stopping refinement (no improvement)", file=sys.stderr)
            break

    combined = _make_combined(best_actions, num_bots)
    total_time = time.time() - t0
    if verbose:
        print(f"\nWarm-start refinement: final_score={best_score}, "
              f"total_time={total_time:.1f}s", file=sys.stderr)

    return best_score, combined


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sequential per-bot GPU DP solver')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-states', type=int, default=500000)
    parser.add_argument('--refine-iters', type=int, default=2,
                        help='Max refinement iterations (default: 2)')
    parser.add_argument('--restarts', type=int, default=1,
                        help='Number of random bot orderings to try (default: 1)')
    parser.add_argument('--capture', action='store_true',
                        help='Use saved capture data')
    parser.add_argument('--warm-start', action='store_true',
                        help='Load existing best solution and refine it')
    parser.add_argument('--no-filler', action='store_true',
                        help='Use only captured orders (no random filler)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.warm_start:
        # Load saved solution and refine
        from solution_store import load_solution, load_capture, load_meta, save_solution
        solution = load_solution(args.difficulty)
        if not solution:
            print(f"No saved solution for {args.difficulty}", file=sys.stderr)
            sys.exit(1)
        meta = load_meta(args.difficulty)
        print(f"Loaded solution: score={meta.get('score', '?')}", file=sys.stderr)

        kwargs = dict(device=device, max_states=args.max_states,
                      max_refine_iters=args.refine_iters)
        if args.capture:
            capture = load_capture(args.difficulty)
            if not capture:
                print(f"No capture for {args.difficulty}", file=sys.stderr)
                sys.exit(1)
            kwargs['capture_data'] = capture
        elif args.seed:
            kwargs['seed'] = args.seed
            kwargs['difficulty'] = args.difficulty
        else:
            # Try capture first
            capture = load_capture(args.difficulty)
            if capture:
                kwargs['capture_data'] = capture
            else:
                print("Need --capture or --seed for warm-start", file=sys.stderr)
                sys.exit(1)

        score, actions = refine_from_solution(solution, **kwargs)

        # Save if improved
        old_score = meta.get('score', 0)
        if score > old_score:
            save_solution(args.difficulty, score, actions, force=True)
            print(f"\nFinal score: {score} (improved from {old_score}! saved)")
        else:
            print(f"\nFinal score: {score} (no improvement over {old_score})")
    else:
        solve_fn = solve_multi_restart if args.restarts > 1 else solve_sequential
        kwargs = dict(
            device=device, max_states=args.max_states,
            max_refine_iters=args.refine_iters,
            no_filler=args.no_filler)
        if args.restarts > 1:
            kwargs['num_restarts'] = args.restarts

        if args.capture:
            from solution_store import load_capture
            capture = load_capture(args.difficulty)
            if not capture:
                print(f"No capture for {args.difficulty}", file=sys.stderr)
                sys.exit(1)
            score, actions = solve_fn(capture_data=capture, **kwargs)
        elif args.seed:
            score, actions = solve_fn(
                seed=args.seed, difficulty=args.difficulty, **kwargs)
        else:
            print("Need --capture or --seed", file=sys.stderr)
            sys.exit(1)

        # Auto-save if using seed (sim games have stable solutions)
        from solution_store import save_solution, load_meta
        meta = load_meta(args.difficulty)
        old_score = meta.get('score', 0) if meta else 0
        if score > old_score:
            save_solution(args.difficulty, score, actions,
                          seed=args.seed or 0, force=True)
            print(f"\nFinal score: {score} (saved! was {old_score})")
        else:
            print(f"\nFinal score: {score} (best remains {old_score})")
