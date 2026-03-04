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
from concurrent.futures import ThreadPoolExecutor

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

_ZIG_AVAILABLE = False
try:
    import zig_ffi as _zig_ffi
    _ZIG_AVAILABLE = True
except (ImportError, OSError):
    pass

_DIFF_IDX = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}

# Difficulty-aware defaults: more refinement for harder difficulties with more bots
DEFAULT_REFINE_ITERS = {'easy': 0, 'medium': 3, 'hard': 10, 'expert': 10}
DEFAULT_PASS1_ORDERINGS = {'easy': 1, 'medium': 1, 'hard': 1, 'expert': 1}


def pre_simulate_locked(gs_template, all_orders, bot_actions, locked_bot_ids,
                        _zig_ctx=None):
    """CPU simulation of locked bots to get their per-round positions.

    Args:
        gs_template: Initial GameState (all bots at spawn, round 0).
        all_orders: Full order list.
        bot_actions: Dict {bot_id: [(act, item)] * 300} for all planned bots.
        locked_bot_ids: Sorted list of bot IDs to lock.
        _zig_ctx: Optional dict with 'diff_idx' and 'seed' for Zig FFI fast path.

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

    if _zig_ctx is not None and _ZIG_AVAILABLE:
        num_total_bots = len(gs_template.bot_positions)
        if _zig_ctx.get('mode') == 'live':
            return _zig_ffi.zig_presim_locked_live(
                _zig_ctx['capture_data'],
                all_orders, bot_actions, locked_bot_ids, num_total_bots)
        return _zig_ffi.zig_presim_locked(
            _zig_ctx['diff_idx'], _zig_ctx['seed'],
            all_orders, bot_actions, locked_bot_ids, num_total_bots)

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


def cpu_verify(gs_template, all_orders, combined_actions, num_bots,
               _zig_ctx=None):
    """Verify final score by replaying all bots' combined actions on CPU.

    Args:
        gs_template: Initial GameState.
        all_orders: Full order list.
        combined_actions: List of 300 round_actions, each is [(act, item)] * num_bots.
        num_bots: Number of bots.
        _zig_ctx: Optional dict with 'diff_idx' and 'seed' for Zig FFI fast path.

    Returns:
        Final score from CPU simulation.
    """
    if _zig_ctx is not None and _ZIG_AVAILABLE:
        if _zig_ctx.get('mode') == 'live':
            return _zig_ffi.zig_verify_live(
                _zig_ctx['capture_data'],
                all_orders, combined_actions, num_bots)
        return _zig_ffi.zig_verify(
            _zig_ctx['diff_idx'], _zig_ctx['seed'],
            all_orders, combined_actions, num_bots)

    gs = gs_template.copy()
    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)
    return gs.score


def cpu_verify_detailed(gs_template, all_orders, combined_actions, num_bots):
    """Like cpu_verify but returns (score, orders_completed, items_delivered)."""
    gs = gs_template.copy()
    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)
    return gs.score, gs.orders_completed, gs.items_delivered


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


DEFAULT_MAX_STATES = {
    'easy': 500_000,
    'medium': 500_000,
    'hard': 100_000,
    'expert': 100_000,
}


def compute_type_assignments(all_orders, num_bots, num_types, ms=None,
                             shuffle_seed=None):
    """Assign item types to bots for type specialization (round-robin by frequency).

    Round-robin by order frequency: bot 0 gets type 0 (most frequent), type N, type 2N...
    This gives early planners (no locked-bot coverage signal) a strong hint about
    which types to target, reducing inter-bot competition for high-value types.

    Args:
        shuffle_seed: If set, shuffle types_sorted with this seed before assigning.
                      Produces diverse assignments across multiple Pass 1 orderings.

    Returns:
        dict {bot_id: set[int]} — preferred item type IDs for each bot.
        Only populated for multi-bot (num_bots >= 3).
    """
    if num_bots < 3:
        return {}  # no specialization for 1-2 bots

    from collections import Counter
    import random as _r
    type_freq = Counter()
    for order in all_orders:
        for t in order.required:
            if t >= 0:
                type_freq[t] += 1

    if not type_freq:
        return {}

    # Sort types: most frequent first
    types_sorted = sorted(type_freq.keys(), key=lambda t: -type_freq[t])

    if shuffle_seed is not None:
        _rng = _r.Random(shuffle_seed)
        _rng.shuffle(types_sorted)

    # Assign types to bots round-robin
    assignments = {b: set() for b in range(num_bots)}
    for i, t in enumerate(types_sorted):
        assignments[i % num_bots].add(t)

    return assignments


def solve_sequential(capture_data=None, seed=None, difficulty=None,
                     device='cuda', max_states=None, verbose=True,
                     on_bot_progress=None, on_round=None, on_phase=None,
                     max_refine_iters=None, bot_order=None, no_filler=False,
                     pipeline_fraction=0.4, num_pass1_orderings=None,
                     pass1_states=None, max_pipeline_depth=3,
                     use_type_specialization=True,
                     all_orders_override=None, max_time_s=None,
                     no_compile=False):
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
        pipeline_fraction: Fraction of bots (last in plan_order) to plan with
                           pipeline_mode=True (pre-fetch preview order items).
                           0 = disabled; 0.4 = last 40% of bots are pipeline.
        num_pass1_orderings: Number of different Pass 1 orderings to try.
                             Best result is used as starting point for refinement.
                             Orderings: forward, reverse, random...
                             Default 1 (just forward). Use 3+ for better quality.
        pass1_states: Max states for Pass 1 (default: same as max_states).
                      Set lower to speed up multi-start pass 1.

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

    # Override all_orders when seed is cracked (full foresight, no filler)
    if all_orders_override is not None:
        all_orders = all_orders_override
        gs.orders = [all_orders[0].copy(), all_orders[1].copy()]
        gs.orders[0].status = 'active'
        gs.orders[1].status = 'preview'
        gs.next_order_idx = 2
        if verbose:
            print(f"  [all_orders_override] Using {len(all_orders)} pre-cracked orders",
                  file=sys.stderr)

    # Difficulty-aware default max_states
    if max_states is None:
        max_states = DEFAULT_MAX_STATES.get(diff, 500_000)

    # Difficulty-aware defaults for refine iters and pass1 orderings
    if max_refine_iters is None:
        max_refine_iters = DEFAULT_REFINE_ITERS.get(diff, 2)
    if num_pass1_orderings is None:
        num_pass1_orderings = DEFAULT_PASS1_ORDERINGS.get(diff, 1)

    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    # Type specialization: assign preferred item types to each bot
    # to reduce inter-bot competition for the same items.
    _type_assignments = {}
    if use_type_specialization and num_bots >= 3:
        _type_assignments = compute_type_assignments(
            all_orders, num_bots, ms.num_types, ms)
        if verbose and _type_assignments:
            assign_str = ', '.join(
                f"bot{b}:{sorted(ts)}" for b, ts in sorted(_type_assignments.items())
                if ts)
            print(f"  Type assignments: {assign_str}", file=sys.stderr)

    # Build Zig FFI context.
    # Priority: explicit seed > capture seed > live capture (no seed).
    _zig_seed = seed if seed else (capture_data.get('seed') if capture_data else None)
    if _ZIG_AVAILABLE and _zig_seed:
        _zig_ctx = {'diff_idx': _DIFF_IDX.get(diff, 0), 'seed': _zig_seed, 'mode': 'seed'}
    elif _ZIG_AVAILABLE and capture_data is not None:
        _zig_ctx = {'capture_data': capture_data, 'mode': 'live'}
    else:
        _zig_ctx = None

    if verbose:
        filler_str = f", no_filler ({len(capture_data['orders'])} orders)" if (capture_data and no_filler) else ""
        zig_str = " [ZIG FFI live]" if (_zig_ctx and _zig_ctx.get('mode') == 'live') else (" [ZIG FFI]" if _zig_ctx else "")
        print(f"Sequential GPU DP: {diff}, {num_bots} bots, "
              f"max_states={max_states}, refine_iters={max_refine_iters}{filler_str}{zig_str}",
              file=sys.stderr)

    # For single-bot, just run standard DP (no refinement needed)
    if num_bots == 1:
        searcher = GPUBeamSearcher(ms, all_orders, device=device, num_bots=num_bots,
                                    no_compile=no_compile)
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
    import math
    import random as _random_p1

    # Effective budget for Pass 1 (can be lower for multi-start).
    # When running multiple orderings, use 50% budget for faster screening.
    if pass1_states is not None:
        p1_states = pass1_states
    elif num_pass1_orderings > 1:
        p1_states = max(max_states // 2, 50_000)
    else:
        p1_states = max_states

    # Build list of Pass 1 orderings to try
    base_order = bot_order if bot_order is not None else list(range(num_bots))
    p1_orderings = [base_order]
    if num_pass1_orderings > 1:
        rev = list(reversed(base_order))
        if rev != base_order:
            p1_orderings.append(rev)
    if num_pass1_orderings > 2:
        _rng_p1 = _random_p1.Random(1337)
        while len(p1_orderings) < num_pass1_orderings:
            perm = list(base_order)
            _rng_p1.shuffle(perm)
            if perm not in p1_orderings:
                p1_orderings.append(perm)

    # Pipeline bots: last n_pipeline bots in plan_order get pipeline_mode=True.
    # n_pipeline is computed dynamically from active order item count.
    # Depths cycle through 1..max_pipeline_depth.
    # Example with 5 bots, active order needs 3 items → n_pipeline = max(0, 5-3) = 2:
    #   plan_order[-2]: pipeline_depth=1 (targets order+1)
    #   plan_order[-1]: pipeline_depth=2 (targets order+2)
    # If pipeline_fraction is set to 0 (disabled), n_pipeline = 0.
    _max_pd = max(1, max_pipeline_depth)
    if pipeline_fraction <= 0 or num_bots < 3:
        n_pipeline = 0
    else:
        # Dynamic: allocate primary bots to cover active order items, rest are pipeline.
        active_items_needed = len(all_orders[0].required) if all_orders else num_bots
        primary_bots_needed = min(num_bots, max(1, active_items_needed))
        n_pipeline_dynamic = max(0, num_bots - primary_bots_needed)
        # Also compute fixed fraction as a floor; use whichever is larger (more pipeline).
        n_pipeline_fixed = max(0, math.floor(num_bots * pipeline_fraction))
        n_pipeline = max(n_pipeline_dynamic, n_pipeline_fixed)

    # --- Order cap: limit how many orders any single bot handles in Pass 1 ---
    # Without this, bot 0 delivers 15+ orders, starving later bots of work.
    # Cap = ceil(visible_orders / num_primary_bots) + 1 headroom.
    # Pipeline bots get no cap (they pre-fetch, not deliver).
    # Only applies to Pass 1 (refinement is uncapped).
    _visible_orders = len(all_orders)
    _primary_bots = max(1, num_bots - n_pipeline)
    if num_bots >= 5 and _visible_orders > num_bots:
        _order_cap = max(3, (_visible_orders + _primary_bots - 1) // _primary_bots + 1)
    else:
        _order_cap = None  # small bot count or few orders: no cap
    if verbose and _order_cap is not None:
        print(f"  Order cap: {_order_cap} per bot (visible={_visible_orders}, "
              f"primary={_primary_bots})", file=sys.stderr)

    best_p1_score = -1
    best_p1_actions = None
    best_p1_pipeline_ids = {}

    # --- Pass 1: Sequential planning (possibly multiple orderings) ---
    _ta_for_ordering = _type_assignments if use_type_specialization and num_bots >= 3 else {}

    for p1_idx, plan_order in enumerate(p1_orderings):
        # Time check between pass1 orderings
        if max_time_s is not None and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"  [Pass 1] Time budget {max_time_s}s reached, skipping remaining orderings",
                      file=sys.stderr)
            break

        # Assign pipeline bots from the end of plan_order with cycling depths
        pipeline_bot_ids = {}  # bot_id -> pipeline_depth (0 = not pipeline)
        for i in range(1, n_pipeline + 1):
            if len(plan_order) >= i:
                bot_in_pipeline = plan_order[-i]
                depth = ((i - 1) % _max_pd) + 1  # cycles: 1,2,...,max_pd,1,2,...
                pipeline_bot_ids[bot_in_pipeline] = depth

        if on_phase:
            on_phase("pass1", p1_idx, None)
        if verbose:
            order_str = ','.join(str(b) for b in plan_order)
            print(f"\n--- Pass 1 ({p1_idx+1}/{len(p1_orderings)}): Sequential planning "
                  f"(order: {order_str}, budget: {p1_states:,}) ---",
                  file=sys.stderr)
        if verbose and pipeline_bot_ids:
            depths_str = ', '.join(f"bot{b}→depth{d}" for b, d in sorted(pipeline_bot_ids.items()))
            print(f"  Pipeline bots: {depths_str}", file=sys.stderr)

        bot_actions = {}  # reset for this ordering

        for bot_pos, bot_id in enumerate(plan_order):
            t_bot = time.time()
            if verbose:
                print(f"\n=== Pass 1 ({p1_idx+1}/{len(p1_orderings)}), "
                      f"Bot {bot_id}/{num_bots} ===", file=sys.stderr)

            # Lock all previously-planned bots
            locked_ids = sorted(bot_actions.keys())
            locked = None
            if locked_ids:
                locked = pre_simulate_locked(
                    _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions, locked_ids,
                    _zig_ctx=_zig_ctx)

            p_depth = pipeline_bot_ids.get(bot_id, 0)
            is_pipeline = p_depth > 0
            pref_types = _ta_for_ordering.get(bot_id) if _ta_for_ordering else None
            # Order cap: non-pipeline bots in Pass 1 only
            bot_order_cap = None if is_pipeline else _order_cap
            searcher = GPUBeamSearcher(
                ms, all_orders, device=device, num_bots=num_bots,
                locked_trajectories=locked, pipeline_mode=is_pipeline,
                pipeline_depth=p_depth, preferred_types=pref_types,
                no_compile=no_compile, order_cap=bot_order_cap)

            # Verify on first bot of first ordering only
            if p1_idx == 0 and bot_id == plan_order[0] and verbose:
                gs_v = _fresh_gs(gs, capture_data, no_filler)
                ok = searcher.verify_against_cpu(gs_v, all_orders, num_rounds=100)
                if not ok:
                    print("VERIFICATION FAILED", file=sys.stderr)
                    return 0, []

            def round_cb(rnd, score, unique, expanded, elapsed, _bid=bot_id):
                if on_round:
                    on_round(_bid, rnd, score, unique, expanded, elapsed)

            # Position-aware state budget: flatter scale so late bots
            # (delivery specialists) get fair budget.
            # pos 0 → 1.5x, pos N-1 → 0.8x
            if num_bots > 1:
                t = bot_pos / max(num_bots - 1, 1)  # 0.0 (first) to 1.0 (last)
                pos_scale = 2.0 - 1.5 * t  # 2.0 → 0.5
                effective_states = max(1000, int(p1_states * pos_scale))
            else:
                effective_states = p1_states

            gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
            dp_score, bot_acts = searcher.dp_search(
                gs_for_dp, max_states=effective_states, verbose=verbose,
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

            # Per-bot time check in pass1
            if max_time_s is not None and (time.time() - t0) > max_time_s:
                if verbose:
                    print(f"  [Pass 1] Time budget {max_time_s}s reached after bot {bot_id}",
                          file=sys.stderr)
                break

        # CPU verify after this Pass 1 ordering
        combined = _make_combined(bot_actions, num_bots)
        gs_v = _fresh_gs(gs, capture_data, no_filler)
        p1_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx=_zig_ctx)

        if on_phase:
            on_phase("pass1_done", p1_idx, p1_score)
        if verbose:
            print(f"\nPass 1 ordering {p1_idx+1}/{len(p1_orderings)} "
                  f"CPU verify: score={p1_score}", file=sys.stderr)

        if p1_score > best_p1_score:
            best_p1_score = p1_score
            best_p1_actions = {k: list(v) for k, v in bot_actions.items()}
            best_p1_pipeline_ids = dict(pipeline_bot_ids)  # save for refinement
            if verbose and len(p1_orderings) > 1:
                print(f"  New best Pass 1 score: {best_p1_score}", file=sys.stderr)

    # Use best Pass 1 result as starting point for refinement
    bot_actions = best_p1_actions
    best_score = best_p1_score
    best_actions = best_p1_actions
    pipeline_bot_ids = best_p1_pipeline_ids  # use best ordering's pipeline assignment

    if verbose and len(p1_orderings) > 1:
        print(f"\nBest Pass 1 score across {len(p1_orderings)} orderings: {best_score}",
              file=sys.stderr)

    # --- Pass 2+: Per-bot Refinement with Immediate CPU Verify ---
    # Re-plan one bot at a time, immediately verify, keep only improvements.
    import random as _random
    _rng = _random.Random(42)  # deterministic

    no_improve_iters = 0  # allow 1 escape attempt before stopping
    for iteration in range(max_refine_iters):
        # Time-budget check
        if max_time_s is not None and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"  [solve_sequential] Time budget {max_time_s}s reached at "
                      f"iter {iteration}, stopping", file=sys.stderr)
            break

        if on_phase:
            on_phase(f"refine", iteration + 1, best_score)

        # Alternate refinement order each iteration to escape local optima.
        # Iter 0: forward [0,1,...,N-1]
        # Iter 1: reverse [N-1,...,1,0]
        # Iter 2+: random shuffle
        if iteration == 0:
            refine_order = list(range(num_bots))
        elif iteration == 1:
            refine_order = list(range(num_bots - 1, -1, -1))
        else:
            refine_order = list(range(num_bots))
            _rng.shuffle(refine_order)

        if verbose:
            order_str = ','.join(str(b) for b in refine_order)
            print(f"\n--- Refinement iteration {iteration+1}/{max_refine_iters} "
                  f"(order: {order_str}) ---", file=sys.stderr)

        iter_improved = False

        # Two mini-passes per iteration: forward then backward.
        # Backward pass only runs if forward found improvements (propagates benefits
        # in the reverse direction immediately, rather than waiting for next iteration).
        for mini_idx, pass_order in enumerate([refine_order, list(reversed(refine_order))]):
            if mini_idx == 1 and not iter_improved:
                break  # Skip backward pass if forward made no progress

            # Async verify pool: overlaps cpu_verify(bot N) with pre_simulate_locked(bot N+1)
            _refine_pool = ThreadPoolExecutor(max_workers=1)
            # Pre-fetch locked trajectories for the first bot in pass_order
            # (runs in thread so GPU can immediately start after pool submits)
            _pending_locked = {}  # bot_id -> Future[locked_trajs]

            def _submit_presim(bid, actions_snapshot):
                """Submit pre_simulate_locked for bot bid to thread pool."""
                locked_ids = sorted(b for b in range(num_bots) if b != bid)
                return _refine_pool.submit(
                    pre_simulate_locked,
                    _fresh_gs(gs, capture_data, no_filler),
                    all_orders, actions_snapshot, locked_ids, _zig_ctx)

            # Kick off pre-sim for the first bot
            _first_bot = pass_order[0]
            _pending_locked[_first_bot] = _submit_presim(
                _first_bot, {k: list(v) for k, v in bot_actions.items()})

            consecutive_fails = 0
            for pass_i, bot_id in enumerate(pass_order):
                t_bot = time.time()
                if verbose:
                    print(f"\n=== Refine iter {iteration+1} "
                          f"({'fwd' if mini_idx == 0 else 'bwd'}), "
                          f"Bot {bot_id}/{num_bots} ===", file=sys.stderr)

                # Get (possibly pre-fetched) locked trajectories for this bot
                if bot_id in _pending_locked:
                    locked = _pending_locked.pop(bot_id).result()
                else:
                    locked_ids = sorted(b for b in range(num_bots) if b != bot_id)
                    locked = pre_simulate_locked(
                        _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions,
                        locked_ids, _zig_ctx=_zig_ctx)

                p_depth = pipeline_bot_ids.get(bot_id, 0)
                is_pipeline = p_depth > 0
                pref_types = _type_assignments.get(bot_id) if _type_assignments else None
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, pipeline_mode=is_pipeline,
                    pipeline_depth=p_depth, preferred_types=pref_types,
                    no_compile=no_compile)

                def round_cb(rnd, score, unique, expanded, elapsed, _bid=bot_id):
                    if on_round:
                        on_round(_bid, rnd, score, unique, expanded, elapsed)

                # Optimistically apply new plan (tentative) and submit cpu_verify.
                # Simultaneously pre-fetch locked trajectories for the next bot.
                old_acts = list(bot_actions[bot_id])

                gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
                dp_score, bot_acts = searcher.dp_search(
                    gs_for_dp, max_states=max_states, verbose=verbose,
                    on_round=round_cb, bot_id=bot_id)

                del searcher
                if device == 'cuda':
                    torch.cuda.empty_cache()

                # Apply tentative new plan and submit cpu_verify asynchronously
                bot_actions[bot_id] = bot_acts
                combined = _make_combined(bot_actions, num_bots)
                gs_v = _fresh_gs(gs, capture_data, no_filler)
                _verify_future = _refine_pool.submit(
                    cpu_verify, gs_v, all_orders, combined, num_bots, _zig_ctx)

                # Pre-fetch locked trajectories for next bot (CPU runs concurrently with verify)
                if pass_i + 1 < len(pass_order):
                    next_bot = pass_order[pass_i + 1]
                    # Snapshot current bot_actions (with optimistic update applied)
                    _snap = {k: list(v) for k, v in bot_actions.items()}
                    _pending_locked[next_bot] = _submit_presim(next_bot, _snap)

                # Wait for verify result
                new_score = _verify_future.result()
                bot_time = time.time() - t_bot

                if new_score > best_score:
                    delta = new_score - best_score
                    best_score = new_score
                    best_actions = {k: list(v) for k, v in bot_actions.items()}
                    iter_improved = True
                    consecutive_fails = 0
                    if verbose:
                        print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                              f"(+{delta}! best={best_score}), time={bot_time:.1f}s",
                              file=sys.stderr)
                else:
                    # Revert this bot's actions
                    bot_actions[bot_id] = old_acts
                    # Pre-fetched locked for next bot used the optimistic (reverted) actions.
                    # Cancel it — next bot will recompute with corrected bot_actions.
                    if pass_i + 1 < len(pass_order):
                        next_bot = pass_order[pass_i + 1]
                        if next_bot in _pending_locked:
                            # Wait for it to finish (can't interrupt), then discard
                            try:
                                _pending_locked[next_bot].result(timeout=5)
                            except Exception:
                                pass
                            del _pending_locked[next_bot]
                    if verbose:
                        print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                              f"(no improvement, reverted), time={bot_time:.1f}s",
                              file=sys.stderr)
                    consecutive_fails += 1
                    if consecutive_fails >= num_bots:
                        if verbose:
                            print(f"  Early stop: {num_bots} consecutive no-improvements",
                                  file=sys.stderr)
                        break

                if on_bot_progress:
                    on_bot_progress(bot_id, num_bots, best_score, time.time() - t0)

                # Per-bot time check in refinement
                if max_time_s is not None and (time.time() - t0) > max_time_s:
                    if verbose:
                        print(f"  [Refine] Time budget {max_time_s}s reached after bot {bot_id}",
                              file=sys.stderr)
                    break

            _refine_pool.shutdown(wait=True)

        if on_phase:
            on_phase(f"refine_done", iteration + 1, best_score)

        if verbose:
            print(f"\nRefine iter {iteration+1}: best_score={best_score}",
                  file=sys.stderr)

        if not iter_improved:
            no_improve_iters += 1
            _escape_limit = 4
            if no_improve_iters >= _escape_limit:
                if verbose:
                    print(f"  Stopping refinement ({no_improve_iters} consecutive "
                          f"no-improvement iters)", file=sys.stderr)
                break
            else:
                # Perturbation escape: fully reset a random bot to idle, then re-refine.
                # This forces other bots to plan without that bot's contribution,
                # creating a fundamentally different coordination pattern.
                perturb_bot = _rng.choice(list(range(num_bots)))
                bot_actions[perturb_bot] = [(ACT_WAIT, -1)] * MAX_ROUNDS
                if verbose:
                    print(f"  Perturbation escape: reset bot {perturb_bot} to idle "
                          f"({no_improve_iters}/{_escape_limit - 1})...", file=sys.stderr)
        else:
            no_improve_iters = 0

    # Final combined actions from best
    combined = _make_combined(best_actions, num_bots)

    total_time = time.time() - t0
    if verbose:
        print(f"\nSequential GPU DP: final_score={best_score}, "
              f"total_time={total_time:.1f}s", file=sys.stderr)

    return best_score, combined


def generate_orderings(num_bots, k=1000, seed=42):
    """Generate k random bot orderings via Fischer-Yates shuffle.

    First ordering is always [0, 1, ..., num_bots-1] (default).
    Second is reversed. Rest are random.

    Returns list of k lists, each of length num_bots.
    """
    import random
    rng = random.Random(seed)
    orderings = [list(range(num_bots))]
    if k >= 2:
        orderings.append(list(range(num_bots - 1, -1, -1)))
    seen = {tuple(orderings[0]), tuple(orderings[1])} if k >= 2 else {tuple(orderings[0])}
    attempts = 0
    while len(orderings) < k and attempts < k * 5:
        order = list(range(num_bots))
        rng.shuffle(order)
        t = tuple(order)
        if t not in seen:
            seen.add(t)
            orderings.append(order)
        attempts += 1
    return orderings[:k]


def batch_evaluate_orderings(capture_data, orderings_list, device='cuda', n_steps=25):
    """Screen K bot orderings via short greedy rollout on GPU.

    Runs n_steps of greedy simulation simultaneously for all K orderings.
    Higher-slot bots (lower priority in the ordering) yield in conflicts.
    This acts as a cheap proxy for which ordering gives first-mover advantage
    to the most useful bots (ones closest to needed items/dropoff).

    Returns list[float] of estimated scores for each ordering.
    """
    import random as _random
    K = len(orderings_list)
    if K == 0:
        return []

    gs, all_orders = init_game_from_capture(
        capture_data, num_orders=len(capture_data['orders']))
    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    if num_bots == 1:
        return [0.0] * K

    from precompute import PrecomputedTables
    tables = PrecomputedTables.get(ms)
    gpu_tables = tables.to_gpu_tensors(device)
    first_step_to_dropoff = gpu_tables['step_to_dropoff']   # [H, W] int8
    first_step_to_type = gpu_tables['step_to_type']         # [num_types, H, W] int8
    dist_to_type = gpu_tables['dist_to_type']               # [num_types, H, W] int16

    num_orders = len(all_orders)
    num_types = ms.num_types
    H, W = ms.height, ms.width
    drop_x, drop_y = int(ms.drop_off[0]), int(ms.drop_off[1])
    spawn_x, spawn_y = int(ms.spawn[0]), int(ms.spawn[1])

    order_req = torch.full((num_orders, MAX_ORDER_SIZE), -1, dtype=torch.int8, device=device)
    for i, o in enumerate(all_orders):
        for j, t in enumerate(o.required):
            order_req[i, j] = int(t)
    order_sizes = torch.tensor(
        [len(o.required) for o in all_orders], dtype=torch.int32, device=device)

    grid = torch.tensor(ms.grid, dtype=torch.int8, device=device)
    walkable_mask = ((grid == CELL_FLOOR) | (grid == CELL_DROPOFF))  # [H, W]

    # DX/DY indexed by action int (0=wait,1=up,2=down,3=left,4=right,5=dropoff,6=pickup)
    DX_T = torch.tensor([0, 0, 0, -1, 1, 0, 0], dtype=torch.int32, device=device)
    DY_T = torch.tensor([0, -1, 1, 0, 0, 0, 0], dtype=torch.int32, device=device)

    # Orderings tensor [K, num_bots]: orderings_list[k][s] = real bot id in slot s
    ord_t = torch.tensor(orderings_list, dtype=torch.int64, device=device)  # [K, N]

    # Initial state: all bots at spawn
    pos_x = torch.full((K, num_bots), spawn_x, dtype=torch.int16, device=device)
    pos_y = torch.full((K, num_bots), spawn_y, dtype=torch.int16, device=device)
    # Simplified inventory: track count of active items per bot
    carrying = torch.zeros((K, num_bots), dtype=torch.int32, device=device)
    active_idx = torch.zeros(K, dtype=torch.int32, device=device)
    active_del = torch.zeros((K, MAX_ORDER_SIZE), dtype=torch.int8, device=device)
    score = torch.zeros(K, dtype=torch.int32, device=device)

    item_px = torch.tensor(ms.item_positions[:, 0], dtype=torch.int32, device=device)
    item_py = torch.tensor(ms.item_positions[:, 1], dtype=torch.int32, device=device)
    item_ty = torch.tensor(ms.item_types, dtype=torch.int8, device=device)

    for rnd in range(n_steps):
        aidx = active_idx.long().clamp(0, num_orders - 1)   # [K]
        act_req = order_req[aidx]                            # [K, MAX_ORDER_SIZE]
        act_del = active_del                                 # [K, MAX_ORDER_SIZE]

        bx_l = pos_x.long()   # [K, N]
        by_l = pos_y.long()   # [K, N]
        has_active = carrying > 0               # [K, N]
        has_space = carrying < INV_CAP          # [K, N]
        at_drop = (pos_x == drop_x) & (pos_y == drop_y)  # [K, N]

        # --- Greedy action selection ---
        actions = torch.zeros((K, num_bots), dtype=torch.int8, device=device)  # wait

        # Dropoff if at dropoff with active items
        actions = torch.where(at_drop & has_active,
                              torch.tensor(ACT_DROPOFF, dtype=torch.int8, device=device),
                              actions)

        # Move toward dropoff if carrying active items
        move_to_drop = first_step_to_dropoff[by_l, bx_l]  # [K, N]
        actions = torch.where(has_active & ~at_drop & (move_to_drop > 0),
                              move_to_drop.to(torch.int8), actions)

        # Move toward nearest needed item type when not carrying
        best_move = torch.zeros((K, num_bots), dtype=torch.int8, device=device)
        best_dist = torch.full((K, num_bots), 9999, dtype=torch.int32, device=device)
        for os in range(MAX_ORDER_SIZE):
            needed_k = (act_req[:, os] >= 0) & (act_del[:, os] == 0)  # [K]
            if not needed_k.any():
                continue
            ntype = act_req[:, os].long().clamp(0, num_types - 1)   # [K]
            ntype_exp = ntype.unsqueeze(1).expand(K, num_bots)       # [K, N]
            d = dist_to_type[ntype_exp, by_l, bx_l]                  # [K, N]
            m = first_step_to_type[ntype_exp, by_l, bx_l]            # [K, N]
            needed_exp = needed_k.unsqueeze(1).expand(K, num_bots)   # [K, N]
            improve = needed_exp & has_space & ~has_active & (d.int() < best_dist) & (m > 0)
            best_dist = torch.where(improve, d.int(), best_dist)
            best_move = torch.where(improve, m.to(torch.int8), best_move)

        actions = torch.where(~has_active & (best_move > 0), best_move, actions)

        # --- Compute proposed positions ---
        act_l = actions.long()
        nx = pos_x.int() + DX_T[act_l]  # [K, N]
        ny = pos_y.int() + DY_T[act_l]
        in_bounds = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        nx_c = nx.clamp(0, W - 1).long()
        ny_c = ny.clamp(0, H - 1).long()
        walkable_here = in_bounds & walkable_mask[ny_c, nx_c]
        is_move = (actions >= ACT_MOVE_UP) & (actions <= ACT_MOVE_RIGHT)
        valid_move = is_move & walkable_here
        prop_x = torch.where(valid_move, nx.short(), pos_x)  # [K, N]
        prop_y = torch.where(valid_move, ny.short(), pos_y)

        # --- Collision resolution in slot-priority order ---
        # Reindex by slot order via gather
        slot_prop_x = torch.gather(prop_x, 1, ord_t)   # [K, N] slot-indexed
        slot_prop_y = torch.gather(prop_y, 1, ord_t)
        slot_cur_x = torch.gather(pos_x, 1, ord_t)
        slot_cur_y = torch.gather(pos_y, 1, ord_t)

        slot_final_x = slot_prop_x.clone()
        slot_final_y = slot_prop_y.clone()

        # Higher-slot (lower priority) bots yield to lower-slot bots
        for s in range(1, num_bots):
            my_x = slot_final_x[:, s]     # [K]
            my_y = slot_final_y[:, s]
            higher_x = slot_final_x[:, :s]   # [K, s]
            higher_y = slot_final_y[:, :s]
            conflict = ((my_x.unsqueeze(1) == higher_x) &
                        (my_y.unsqueeze(1) == higher_y)).any(dim=1)  # [K]
            slot_final_x[:, s] = torch.where(conflict, slot_cur_x[:, s], my_x)
            slot_final_y[:, s] = torch.where(conflict, slot_cur_y[:, s], my_y)

        # Scatter back to bot-indexed
        final_x = pos_x.clone()
        final_y = pos_y.clone()
        final_x.scatter_(1, ord_t, slot_final_x)
        final_y.scatter_(1, ord_t, slot_final_y)
        pos_x = final_x
        pos_y = final_y

        # --- Pickups: bot adjacent to item of needed type with space ---
        for item_idx in range(ms.num_items):
            ix = int(item_px[item_idx])
            iy = int(item_py[item_idx])
            itype = int(item_ty[item_idx])
            adj = ((pos_x.int() - ix).abs() + (pos_y.int() - iy).abs()) == 1  # [K, N]
            type_needed = torch.zeros(K, dtype=torch.bool, device=device)
            for os in range(MAX_ORDER_SIZE):
                type_needed = type_needed | (
                    (act_req[:, os] == itype) & (act_del[:, os] == 0))
            can_pick = adj & has_space & type_needed.unsqueeze(1)
            # Mark as picked (simplified: mark delivery slot as needed -> consumed)
            # We just increment carrying count
            carrying = carrying + can_pick.int()

        # --- Dropoffs: bot at dropoff with active items ---
        delivered = at_drop & has_active   # [K, N] — these bots deliver now
        delivery_count = delivered.int().sum(dim=1)  # [K] total items delivered
        score = score + delivery_count
        carrying = torch.where(delivered, torch.zeros_like(carrying), carrying)

        # Simplified order completion: if all items delivered (carrying all back to 0)
        # Check if current order is "done" — rough heuristic via delivery count
        # More accurately: track if total deliveries >= order size
        # For screening purposes, raw delivery count is sufficient signal

    return score.float().cpu().tolist()


def solve_multi_restart(capture_data=None, seed=None, difficulty=None,
                        device='cuda', max_states=500000, verbose=True,
                        max_refine_iters=2, num_restarts=3,
                        num_screen=1000, n_screen_steps=25,
                        on_restart=None, no_filler=False,
                        all_orders_override=None, no_compile=False):
    """Run sequential solver with multiple random bot orderings, keep best.

    Uses batch GPU greedy rollout to screen num_screen orderings cheaply,
    then runs full DP on the top num_restarts orderings only.

    Args:
        num_restarts: Number of orderings to run full DP on (after screening).
        num_screen: Number of orderings to generate and screen with greedy rollout.
        n_screen_steps: Rollout steps for screening (default 25).
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
            max_refine_iters=0, no_filler=no_filler,
            all_orders_override=all_orders_override,
            no_compile=no_compile)

    # Generate and screen orderings
    all_orderings = generate_orderings(num_bots, k=num_screen, seed=42)

    if num_bots > 1 and num_screen > num_restarts and capture_data is not None:
        if verbose:
            print(f"\nScreening {len(all_orderings)} orderings with "
                  f"{n_screen_steps}-step greedy rollout...", file=sys.stderr)
        t_screen = time.time()
        try:
            screen_scores = batch_evaluate_orderings(
                capture_data, all_orderings, device=device, n_steps=n_screen_steps)
            # Select top num_restarts orderings by greedy score
            ranked = sorted(range(len(all_orderings)),
                            key=lambda i: -screen_scores[i])
            orderings = [all_orderings[i] for i in ranked[:num_restarts]]
            if verbose:
                top_scores = [screen_scores[i] for i in ranked[:num_restarts]]
                print(f"  Screening done in {time.time()-t_screen:.1f}s. "
                      f"Top-{num_restarts} greedy scores: {top_scores}",
                      file=sys.stderr)
        except Exception as e:
            print(f"  Screening failed ({e}), falling back to random orderings",
                  file=sys.stderr)
            # Fallback: default orderings
            orderings = all_orderings[:num_restarts]
    else:
        orderings = all_orderings[:num_restarts]

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
            max_refine_iters=max_refine_iters, bot_order=order,
            no_filler=no_filler,
            all_orders_override=all_orders_override,
            no_compile=no_compile)

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
                         on_bot_progress=None, on_round=None, on_phase=None,
                         no_filler=False, pipeline_fraction=0.4, max_pipeline_depth=3,
                         all_orders_override=None, max_time_s=None,
                         no_compile=False):
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
        num_orders = len(capture_data['orders']) if no_filler else 100
        gs, all_orders = init_game_from_capture(capture_data, num_orders=num_orders)
        diff = capture_data.get('difficulty', difficulty or 'easy')
    elif seed and difficulty:
        gs, all_orders = init_game(seed, difficulty)
        diff = difficulty
    else:
        raise ValueError("Need capture_data or (seed + difficulty)")

    # Override all_orders when seed is cracked (full foresight, no filler)
    if all_orders_override is not None:
        all_orders = all_orders_override
        gs.orders = [all_orders[0].copy(), all_orders[1].copy()]
        gs.orders[0].status = 'active'
        gs.orders[1].status = 'preview'
        gs.next_order_idx = 2
        if verbose:
            print(f"  [all_orders_override] Using {len(all_orders)} pre-cracked orders",
                  file=sys.stderr)

    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    # Build Zig FFI context (seed or live capture)
    _zig_seed = seed if seed else (capture_data.get('seed') if capture_data else None)
    if _ZIG_AVAILABLE and _zig_seed:
        _zig_ctx = {'diff_idx': _DIFF_IDX.get(diff, 0), 'seed': _zig_seed, 'mode': 'seed'}
    elif _ZIG_AVAILABLE and capture_data is not None:
        _zig_ctx = {'capture_data': capture_data, 'mode': 'live'}
    else:
        _zig_ctx = None

    # Convert combined_actions to per-bot format
    bot_actions = {}
    for bid in range(num_bots):
        bot_actions[bid] = [(r_acts[bid][0], r_acts[bid][1])
                            for r_acts in combined_actions]

    # Pipeline bots: last n_pipeline bots get pipeline_mode=True (dynamic from active order).
    import math
    _max_pd_r = max(1, max_pipeline_depth)
    if pipeline_fraction <= 0 or num_bots < 3:
        n_pipeline = 0
    else:
        active_items_needed = len(all_orders[0].required) if all_orders else num_bots
        primary_bots_needed = min(num_bots, max(1, active_items_needed))
        n_pipeline_dynamic = max(0, num_bots - primary_bots_needed)
        n_pipeline_fixed = max(0, math.floor(num_bots * pipeline_fraction))
        n_pipeline = max(n_pipeline_dynamic, n_pipeline_fixed)
    pipeline_bot_ids = {}  # bot_id -> pipeline_depth
    for i in range(1, n_pipeline + 1):
        bid = num_bots - i
        if bid >= 0:
            depth = ((i - 1) % _max_pd_r) + 1  # cycles: 1,2,...,max_pd,1,2,...
            pipeline_bot_ids[bid] = depth

    # Type specialization for refinement
    _type_assignments_r = compute_type_assignments(
        all_orders, num_bots, ms.num_types) if num_bots >= 3 else {}

    # Verify starting score
    gs_v = _fresh_gs(gs, capture_data, no_filler)
    best_score = cpu_verify(gs_v, all_orders, combined_actions, num_bots, _zig_ctx=_zig_ctx)
    best_actions = {k: list(v) for k, v in bot_actions.items()}

    if verbose:
        print(f"Warm-start refinement: {diff}, {num_bots} bots, "
              f"starting_score={best_score}, max_states={max_states}, "
              f"refine_iters={max_refine_iters}", file=sys.stderr)

    if on_phase:
        on_phase("warm_start", 0, best_score)

    # Run refinement iterations (same as in solve_sequential)
    import random as _random_rfn
    _rng_rfn = _random_rfn.Random(42)

    no_improve_iters = 0  # allow 1 escape attempt before stopping
    for iteration in range(max_refine_iters):
        # Time-budget check
        if max_time_s is not None and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"  [refine_from_solution] Time budget {max_time_s}s reached at "
                      f"iter {iteration}, stopping", file=sys.stderr)
            break

        if on_phase:
            on_phase("refine", iteration + 1, best_score)

        if iteration == 0:
            refine_order = list(range(num_bots))
        elif iteration == 1:
            refine_order = list(range(num_bots - 1, -1, -1))
        else:
            refine_order = list(range(num_bots))
            _rng_rfn.shuffle(refine_order)

        if verbose:
            order_str = ','.join(str(b) for b in refine_order)
            print(f"\n--- Refinement iteration {iteration+1}/{max_refine_iters} "
                  f"(order: {order_str}) ---", file=sys.stderr)

        iter_improved = False

        # Two mini-passes: forward then backward (bidirectional per iteration)
        for mini_idx, pass_order in enumerate([refine_order, list(reversed(refine_order))]):
            if mini_idx == 1 and not iter_improved:
                break  # Skip backward pass if forward found no improvements

            consecutive_fails = 0
            for bot_id in pass_order:
                t_bot = time.time()
                if verbose:
                    print(f"\n=== Refine iter {iteration+1} "
                          f"({'fwd' if mini_idx == 0 else 'bwd'}), "
                          f"Bot {bot_id}/{num_bots} ===", file=sys.stderr)

                locked_ids = sorted(b for b in range(num_bots) if b != bot_id)
                locked = pre_simulate_locked(
                    _fresh_gs(gs, capture_data, no_filler), all_orders, bot_actions, locked_ids,
                    _zig_ctx=_zig_ctx)

                p_depth = pipeline_bot_ids.get(bot_id, 0)
                is_pipeline = p_depth > 0
                pref_types_r = _type_assignments_r.get(bot_id) if _type_assignments_r else None
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, pipeline_mode=is_pipeline,
                    pipeline_depth=p_depth, preferred_types=pref_types_r,
                    no_compile=no_compile)

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
                new_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx=_zig_ctx)

                if new_score > best_score:
                    delta = new_score - best_score
                    best_score = new_score
                    best_actions = {k: list(v) for k, v in bot_actions.items()}
                    iter_improved = True
                    consecutive_fails = 0
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
                    consecutive_fails += 1
                    if consecutive_fails >= num_bots:
                        if verbose:
                            print(f"  Early stop: {num_bots} consecutive no-improvements",
                                  file=sys.stderr)
                        break

                if on_bot_progress:
                    on_bot_progress(bot_id, num_bots, best_score, time.time() - t0)

        if on_phase:
            on_phase("refine_done", iteration + 1, best_score)

        if verbose:
            print(f"\nRefine iter {iteration+1}: best_score={best_score}",
                  file=sys.stderr)

        if not iter_improved:
            no_improve_iters += 1
            _escape_limit = 4
            if no_improve_iters >= _escape_limit:
                if verbose:
                    print(f"  Stopping refinement ({no_improve_iters} consecutive "
                          f"no-improvement iters)", file=sys.stderr)
                break
            elif verbose:
                print(f"  No improvement, trying escape attempt "
                      f"({no_improve_iters}/{_escape_limit - 1})...", file=sys.stderr)
        else:
            no_improve_iters = 0

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
    parser.add_argument('--max-states', type=int, default=None,
                        help='Max states per bot (default: difficulty-aware)')
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
                      max_refine_iters=args.refine_iters,
                      no_filler=args.no_filler)
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

        # Auto-save: save if improved, OR if capture hash changed (new map/seed)
        from solution_store import save_solution, load_meta, _capture_hash
        meta = load_meta(args.difficulty)
        old_score = meta.get('score', 0) if meta else 0
        cap_changed = not meta or meta.get('capture_hash') != _capture_hash(args.difficulty)
        if score > old_score or (cap_changed and score >= old_score):
            save_solution(args.difficulty, score, actions,
                          seed=args.seed or 0, force=True)
            reason = "saved! new best" if score > old_score else "saved (new capture, same score)"
            print(f"\nFinal score: {score} ({reason}, was {old_score})")
        else:
            print(f"\nFinal score: {score} (best remains {old_score})")

        # Record to PostgreSQL (with full round data for replay)
        if args.seed and score > 0:
            try:
                from synthetic_optimize import record_synthetic_score
                run_id = record_synthetic_score(
                    args.difficulty, args.seed, score,
                    max_states=args.max_states or 0,
                    refine_iters=args.refine_iters,
                    time_secs=0, actions=actions,
                )
                if run_id:
                    print(f"  -> db#{run_id}", file=sys.stderr)
            except Exception as e:
                print(f"  DB record skipped: {e}", file=sys.stderr)
