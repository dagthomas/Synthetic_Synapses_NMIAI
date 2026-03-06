"""Sequential per-bot GPU DP solver for multi-bot games with iterative refinement.

Two-pass approach:
  Pass 1 (Sequential): Bot 0 solo, Bot 1 with 0 locked, Bot 2 with 0,1 locked, ...
  Pass 2+ (Refinement): Re-plan each bot with ALL other bots locked.
    This fixes collision displacement from Pass 1 (e.g., bot 0 blocked by bot 1).

Each bot's DP stays single-bot sized (~200K states), making it GPU-tractable.

Error contract:
  - solve_sequential / refine_from_solution raise ValueError on invalid input
    (missing capture_data and seed+difficulty).
  - Both return (score, actions) tuple: score is int, actions is
    list of 300 round_actions where each round_actions is [(act, item)] * num_bots.
  - Returns (0, []) if GPU/CPU verification fails (solver internal check).
    Callers should check score > 0 before using the actions.
  - cpu_verify returns int score (0 on empty/failed simulation).

Usage:
    from gpu_sequential_solver import solve_sequential
    score, actions = solve_sequential(capture, device='cuda')
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from game_engine import (
    init_game, init_game_from_capture, step as cpu_step,
    GameState, MapState, Order, CaptureData,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
    DX, DY,
)
from gpu_beam_search import GPUBeamSearcher, GPUBeamSearcher2Bot
from configs import CONFIGS, DIFF_IDX as _DIFF_IDX, DIFF_ROUNDS as _DIFF_ROUNDS

_ZIG_AVAILABLE = False
try:
    import zig_ffi as _zig_ffi
    _ZIG_AVAILABLE = True
except (ImportError, OSError):
    pass  # zig_ffi DLL is optional; solver works without it (slower verification)

# Difficulty-aware defaults: more refinement for harder difficulties with more bots
DEFAULT_REFINE_ITERS = {'easy': 0, 'medium': 3, 'hard': 10, 'expert': 10, 'nightmare': 5}
DEFAULT_PASS1_ORDERINGS = {'easy': 1, 'medium': 1, 'hard': 3, 'expert': 3, 'nightmare': 3}
DEFAULT_MAX_DP_BOTS = {'easy': 1, 'medium': 3, 'hard': 5, 'expert': 7, 'nightmare': 5}


def pre_simulate_locked(gs_template: GameState, all_orders: list[Order],
                        bot_actions: dict[int, list[tuple[int, int]]],
                        locked_bot_ids: list[int],
                        _zig_ctx: dict[str, Any] | None = None) -> dict[str, np.ndarray] | None:
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
    _wait_act = (ACT_WAIT, -1)
    for i, bid in enumerate(locked_bot_ids):
        acts = bot_actions[bid]
        for r in range(MAX_ROUNDS):
            a, item = acts[r] if r < len(acts) else _wait_act
            locked_actions[i, r] = a
            locked_action_items[i, r] = item

    # Simulate ALL bots with their current actions to get accurate positions
    # and order state. Bots without actions (including the candidate being
    # re-planned) wait. In refinement, the candidate bot's OLD actions are
    # included in bot_actions, giving correct order state for locked bots.
    for r in range(MAX_ROUNDS):
        round_actions = []
        for bid in range(num_total_bots):
            if bid in bot_actions and r < len(bot_actions[bid]):
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


def cpu_verify(gs_template: GameState, all_orders: List[Order],
               combined_actions: List[List[Tuple[int, int]]], num_bots: int,
               _zig_ctx: Optional[Dict[str, Any]] = None) -> int:
    """Verify final score by replaying all bots' combined actions on CPU.

    Args:
        gs_template: Initial GameState.
        all_orders: Full order list.
        combined_actions: List of 300 round_actions, each is [(act, item)] * num_bots.
        num_bots: Number of bots.
        _zig_ctx: Optional dict with 'diff_idx' and 'seed' for Zig FFI fast path.

    Returns:
        Final score (int) from CPU simulation. Returns 0 on empty/trivial games.
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


def cpu_verify_detailed(gs_template: GameState, all_orders: List[Order],
                        combined_actions: List[List[Tuple[int, int]]],
                        num_bots: int) -> Tuple[int, int, int]:
    """Like cpu_verify but returns (score, orders_completed, items_delivered)."""
    gs = gs_template.copy()
    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)
    return gs.score, gs.orders_completed, gs.items_delivered


def _make_combined(bot_actions, num_bots):
    """Convert bot_actions dict to per-round combined format."""
    _wait = (ACT_WAIT, -1)
    combined = []
    for r in range(MAX_ROUNDS):
        round_acts = []
        for bid in range(num_bots):
            if bid in bot_actions and r < len(bot_actions[bid]):
                round_acts.append(bot_actions[bid][r])
            else:
                round_acts.append(_wait)
        combined.append(round_acts)
    return combined


def compute_bot_contributions(gs_template: GameState, all_orders: list[Order],
                              bot_actions: dict[int, list], num_bots: int,
                              dp_bot_ids: list[int],
                              _zig_ctx: dict | None = None,
                              capture_data: CaptureData | None = None,
                              no_filler: bool = False) -> dict[int, int]:
    """Compute marginal contribution of each DP bot.

    For each bot, simulate with that bot replaced by wait actions.
    Contribution = total_score - score_without_bot.

    Returns dict {bot_id: marginal_contribution}.
    """
    combined = _make_combined(bot_actions, num_bots)
    gs_v = gs_template.copy()
    total_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx)

    contributions = {}
    _wait_acts = [(ACT_WAIT, -1)] * MAX_ROUNDS
    for bid in dp_bot_ids:
        # Replace this bot with wait
        modified = dict(bot_actions)
        modified[bid] = _wait_acts
        mod_combined = _make_combined(modified, num_bots)
        if capture_data:
            num_orders = len(capture_data['orders']) if no_filler else 100
            gs_mod = init_game_from_capture(capture_data, num_orders=num_orders)[0]
        else:
            gs_mod = gs_template.copy()
        score_without = cpu_verify(gs_mod, all_orders, mod_combined, num_bots, _zig_ctx)
        contributions[bid] = total_score - score_without

    return contributions


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
    'nightmare': 50_000,
}


@dataclass
class SolveConfig:
    """Solver configuration parameters."""
    max_states: 'Optional[int]' = None
    max_time_s: 'Optional[float]' = None
    max_refine_iters: 'Optional[int]' = None
    num_pass1_orderings: 'Optional[int]' = None
    pass1_states: 'Optional[int]' = None
    pipeline_fraction: float = 0.4
    max_pipeline_depth: int = 3
    use_type_specialization: bool = True
    no_filler: bool = False
    no_compile: bool = False
    bot_order: 'Optional[list]' = None
    speed_bonus: float = 0.0
    max_dp_bots: 'Optional[int]' = None  # DP-plan only top N bots; rest get CPU greedy
    use_2bot_dp: bool = False  # Plan bot pairs jointly (2-bot DP)
    use_order_assignment: bool = False  # LNS order assignment (round-robin)


@dataclass
class SolveCallbacks:
    """Progress callbacks for streaming output."""
    on_bot_progress: 'Optional[Callable]' = None
    on_round: 'Optional[Callable]' = None
    on_phase: 'Optional[Callable]' = None




def compute_zone_assignments(ms: MapState, num_bots: int,
                             n_pipeline: int = 0) -> dict[int, tuple[int, ...]]:
    """Assign aisle zones to bots based on map geometry.

    Detects narrow aisle columns (1-tile wide between shelves) and distributes
    them across non-pipeline bots. Pipeline bots get no zone (they roam).

    Returns:
        dict {bot_id: tuple of aisle column indices} for each bot.
        Empty dict if < 3 bots or no aisles detected.
    """
    if num_bots < 3:
        return {}

    grid = ms.grid
    H, W = ms.height, ms.width

    # Detect corridor rows (wide horizontal passages)
    corridor_rows = set()
    for y in range(H):
        floor_count = sum(1 for x in range(W)
                          if grid[y, x] in (CELL_FLOOR, CELL_DROPOFF))
        if floor_count * 10 > W * 6:
            corridor_rows.add(y)

    # Detect aisle columns (narrow vertical passages between shelves)
    aisle_cols = []
    for x in range(1, W - 1):
        aisle_cells = 0
        total_floor = 0
        for y in range(H):
            if y in corridor_rows:
                continue
            if grid[y, x] not in (CELL_FLOOR, CELL_DROPOFF):
                continue
            total_floor += 1
            left = grid[y, x - 1]
            right = grid[y, x + 1]
            if left in (CELL_SHELF, CELL_WALL) and right in (CELL_SHELF, CELL_WALL):
                aisle_cells += 1
        if total_floor > 2 and aisle_cells * 4 > total_floor * 3:
            aisle_cols.append(x)

    if len(aisle_cols) < 2:
        return {}

    # Sort aisles by distance to dropoff (leftmost first on typical maps)
    drop_x = int(ms.drop_off[0])
    aisle_cols.sort(key=lambda c: abs(c - drop_x))

    # Number of bots that get zone assignments (exclude pipeline bots)
    n_primary = num_bots - n_pipeline
    if n_primary < 2:
        return {}

    # Distribute aisles across primary bots (round-robin, near-dropoff first)
    assignments: dict[int, list[int]] = {b: [] for b in range(n_primary)}
    for i, col in enumerate(aisle_cols):
        assignments[i % n_primary].append(col)

    return {b: tuple(cols) for b, cols in assignments.items() if cols}


def compute_type_assignments(all_orders: list[Order], num_bots: int,
                             num_types: int, ms: MapState | None = None,
                             shuffle_seed: int | None = None) -> dict[int, set[int]]:
    """Assign item types to bots for type specialization (round-robin by frequency).

    Round-robin by order frequency: bot 0 gets type 0 (most frequent), type N, type 2N...
    This gives early planners (no locked-bot coverage signal) a strong hint about
    which types to target, reducing inter-bot competition for high-value types.

    Args:
        shuffle_seed: If set, shuffle types_sorted with this seed before assigning.
                      Produces diverse assignments across multiple Pass 1 orderings.

    Returns:
        dict {bot_id: set[int]} -- preferred item type IDs for each bot.
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


def greedy_plan_bots(gs_template: GameState, all_orders: list[Order],
                     bot_ids: list[int], existing_bot_actions: dict[int, list],
                     ms: MapState, num_bots: int,
                     _zig_ctx: dict | None = None,
                     capture_data: CaptureData | None = None,
                     no_filler: bool = False) -> dict[int, list]:
    """Hybrid JIT + preview strategy for bots not covered by GPU DP.

    Strategy: fetch preview and high-frequency non-active types speculatively.
    NEVER pick active order types (interferes with DP bot plans).
    When items match the active order, deliver to dropoff.
    Stay away from dropoff when idle to avoid congesting DP bots.

    Co-simulates all bots together for correct order progression.
    """
    from precompute import PrecomputedTables

    tables = PrecomputedTables.get(ms)
    step_to_type = tables.step_to_type      # [num_types, H, W] int8
    step_to_dropoff = tables.step_to_dropoff  # [H, W] int8
    dist_to_type = tables.dist_to_type      # [num_types, H, W] int16
    dist_to_dropoff = tables.dist_to_dropoff  # [H, W] int16

    drop_x, drop_y = int(ms.drop_off[0]), int(ms.drop_off[1])
    greedy_set = set(bot_ids)

    # Build type frequency across ALL orders
    type_freq = {}
    for o in all_orders:
        for t in o.required:
            tid = int(t)
            if tid >= 0:
                type_freq[tid] = type_freq.get(tid, 0) + 1
    ranked_types = sorted(type_freq.keys(), key=lambda t: -type_freq[t])

    # Build item adjacency lookup: type_id -> [(item_idx, adj_x, adj_y)]
    type_adj_items = {}
    for item_idx in range(ms.num_items):
        type_id = int(ms.item_types[item_idx])
        ix = int(ms.item_positions[item_idx, 0])
        iy = int(ms.item_positions[item_idx, 1])
        if type_id not in type_adj_items:
            type_adj_items[type_id] = []
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            ax, ay = ix + ddx, iy + ddy
            if 0 <= ax < ms.width and 0 <= ay < ms.height:
                cell = ms.grid[ay, ax]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    type_adj_items[type_id].append((item_idx, ax, ay))

    def find_adjacent_item(bx, by, type_id):
        """Find item_idx of given type adjacent to (bx,by), or -1."""
        for item_idx, ax, ay in type_adj_items.get(type_id, []):
            if ax == bx and ay == by:
                return item_idx
        return -1

    # Co-simulate: DP bots replay, greedy bots decide greedily each round
    gs_sim = _fresh_gs(gs_template, capture_data, no_filler)
    greedy_actions = {bid: [] for bid in bot_ids}

    for r in range(MAX_ROUNDS):
        gs_sim.round = r

        # Get active order's needed types with multiplicity
        active_needed = {}
        for o in gs_sim.orders:
            if o.status == 'active':
                for i, t in enumerate(o.required):
                    if o.delivered[i] == 0:
                        tid = int(t)
                        active_needed[tid] = active_needed.get(tid, 0) + 1
                break

        # Get preview order types
        preview_needed = {}
        for o in gs_sim.orders:
            if o.status == 'preview':
                for i, t in enumerate(o.required):
                    tid = int(t)
                    preview_needed[tid] = preview_needed.get(tid, 0) + 1
                break

        # What DP bots are carrying
        dp_carrying = {}
        for bid in range(num_bots):
            if bid in greedy_set:
                continue
            for slot in range(INV_CAP):
                t = int(gs_sim.bot_inventories[bid][slot])
                if t >= 0:
                    dp_carrying[t] = dp_carrying.get(t, 0) + 1

        # Per-round claiming
        claimed = {}
        bot_decisions = {}

        for bid in bot_ids:
            bx = int(gs_sim.bot_positions[bid][0])
            by = int(gs_sim.bot_positions[bid][1])
            inv_count = int((gs_sim.bot_inventories[bid] >= 0).sum())
            has_items = inv_count > 0
            inv_full = inv_count >= INV_CAP

            my_types = []
            for slot in range(INV_CAP):
                t = int(gs_sim.bot_inventories[bid][slot])
                if t >= 0:
                    my_types.append(t)
            my_types_set = set(my_types)

            # Count active matches
            active_match_count = 0
            if active_needed:
                remaining_need = dict(active_needed)
                for t in my_types:
                    if remaining_need.get(t, 0) > 0:
                        active_match_count += 1
                        remaining_need[t] -= 1
            has_active_items = active_match_count > 0

            # ── 1. At dropoff → deliver
            if bx == drop_x and by == drop_y and has_items:
                bot_decisions[bid] = (ACT_DROPOFF, -1)
                continue

            # ── 2. Has active items → go deliver
            if has_active_items:
                step = int(step_to_dropoff[by, bx])
                bot_decisions[bid] = (step if step > 0 else ACT_WAIT, -1)
                continue

            # ── 3. Full inventory → wait away from dropoff
            if inv_full:
                dist_drop = int(dist_to_dropoff[by, bx])
                if dist_drop <= 2:
                    for tid in ranked_types:
                        if tid not in my_types_set:
                            d = int(dist_to_type[tid, by, bx])
                            if d > 3:
                                step = int(step_to_type[tid, by, bx])
                                if step > 0:
                                    bot_decisions[bid] = (step, -1)
                                    break
                    if bid not in bot_decisions:
                        bot_decisions[bid] = (ACT_WAIT, -1)
                else:
                    bot_decisions[bid] = (ACT_WAIT, -1)
                continue

            # ── 4. Fetch preview/high-frequency non-active types
            best_type = -1
            best_dist = 9999
            best_score = -1

            for tid in ranked_types:
                if tid in my_types_set:
                    continue
                if tid in active_needed:
                    continue  # never chase active types (DP handles)
                if claimed.get(tid, 0) >= 2:
                    continue
                d = int(dist_to_type[tid, by, bx])
                if d >= 9999:
                    continue
                if tid in preview_needed:
                    score = 20000 - d
                else:
                    score = 10000 - d
                if score > best_score:
                    best_score = score
                    best_dist = d
                    best_type = tid

            if best_type >= 0 and best_dist == 0:
                item_idx = find_adjacent_item(bx, by, best_type)
                if item_idx >= 0:
                    bot_decisions[bid] = (ACT_PICKUP, item_idx)
                    claimed[best_type] = claimed.get(best_type, 0) + 1
                else:
                    bot_decisions[bid] = (ACT_WAIT, -1)
            elif best_type >= 0:
                step = int(step_to_type[best_type, by, bx])
                bot_decisions[bid] = (step if step > 0 else ACT_WAIT, -1)
                claimed[best_type] = claimed.get(best_type, 0) + 1
            else:
                bot_decisions[bid] = (ACT_WAIT, -1)

        # Build round actions
        round_acts = []
        for bid in range(num_bots):
            if bid not in greedy_set:
                if bid in existing_bot_actions and r < len(existing_bot_actions[bid]):
                    round_acts.append(existing_bot_actions[bid][r])
                else:
                    round_acts.append((ACT_WAIT, -1))
            else:
                act_pair = bot_decisions.get(bid, (ACT_WAIT, -1))
                greedy_actions[bid].append(act_pair)
                round_acts.append(act_pair)

        cpu_step(gs_sim, round_acts, all_orders)

    return greedy_actions


def solve_sequential(capture_data: CaptureData | None = None,
                     seed: int | None = None, difficulty: str | None = None,
                     device: str = 'cuda', config: SolveConfig | None = None,
                     callbacks: SolveCallbacks | None = None,
                     verbose: bool = True,
                     all_orders_override: list[Order] | None = None,
                     # Legacy kwargs for backward compat - prefer config/callbacks
                     **kwargs) -> tuple[int, list[list[tuple[int, int]]]]:
    """Sequential per-bot GPU DP with iterative refinement.

    Args:
        capture_data: Captured game data (preferred).
        seed: Game seed (if no capture).
        difficulty: Game difficulty.
        device: 'cuda' or 'cpu'.
        config: SolveConfig with solver parameters (preferred over kwargs).
        callbacks: SolveCallbacks with progress callbacks (preferred over kwargs).
        verbose: Print progress.
        all_orders_override: Override all_orders with pre-cracked orders.
        **kwargs: Legacy keyword arguments for backward compatibility.

    Returns:
        (final_score, combined_actions) tuple.
    """
    # --- Merge config/callbacks with legacy kwargs ---
    if config is None:
        config = SolveConfig()
    if callbacks is None:
        callbacks = SolveCallbacks()
    # Allow legacy kwargs to override config fields
    for key in ('max_states', 'max_time_s', 'max_refine_iters', 'num_pass1_orderings',
                'pass1_states', 'pipeline_fraction', 'max_pipeline_depth',
                'use_type_specialization', 'no_filler', 'no_compile', 'bot_order',
                'speed_bonus', 'max_dp_bots', 'use_2bot_dp', 'use_order_assignment'):
        if key in kwargs:
            setattr(config, key, kwargs.pop(key))
    for key in ('on_bot_progress', 'on_round', 'on_phase'):
        if key in kwargs:
            setattr(callbacks, key, kwargs.pop(key))
    if kwargs:
        raise TypeError(f"solve_sequential() got unexpected keyword arguments: "
                        f"{list(kwargs.keys())}")

    # --- Unpack for local use ---
    max_states = config.max_states
    max_time_s = config.max_time_s if config.max_time_s else None
    max_refine_iters = config.max_refine_iters
    num_pass1_orderings = config.num_pass1_orderings
    pass1_states = config.pass1_states
    pipeline_fraction = config.pipeline_fraction
    max_pipeline_depth = config.max_pipeline_depth
    use_type_specialization = config.use_type_specialization
    no_filler = config.no_filler
    no_compile = config.no_compile
    bot_order = config.bot_order
    speed_bonus = config.speed_bonus
    max_dp_bots = config.max_dp_bots
    use_2bot_dp = config.use_2bot_dp
    use_order_assignment = config.use_order_assignment
    on_bot_progress = callbacks.on_bot_progress
    on_round = callbacks.on_round
    on_phase = callbacks.on_phase

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
    _max_rounds = _DIFF_ROUNDS.get(diff, 300)

    # max_dp_bots: only DP-plan this many bots; rest get CPU greedy plans.
    # For Expert (10 bots), planning all 10 takes ~130s cold + ~600s refine.
    # With max_dp_bots=5, cold ~65s + refine ~300s — fits in 288s token window.
    if max_dp_bots is None:
        max_dp_bots = DEFAULT_MAX_DP_BOTS.get(diff, num_bots)
    max_dp_bots = min(max_dp_bots, num_bots)
    _greedy_bot_ids = list(range(max_dp_bots, num_bots))  # bots planned by CPU greedy

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

    # Traffic flow: one-way lane system for medium (3-bot) maps.
    # Leftmost aisle col for going UP (after dropoff), second aisle col for going DOWN.
    _traffic_flow = None
    if diff == 'medium' and num_bots >= 2:
        _aisle_cols = sorted(x for x in range(ms.width)
                             if sum(1 for y in range(ms.height)
                                    if ms.grid[y, x] in (0, 3)) >= ms.height * 0.6)
        if len(_aisle_cols) >= 2:
            _traffic_flow = {'up_col': _aisle_cols[0], 'down_col': _aisle_cols[1]}
            if verbose:
                print(f"  Traffic flow: UP col={_aisle_cols[0]}, DOWN col={_aisle_cols[1]}",
                      file=sys.stderr)

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
                                    no_compile=no_compile, speed_bonus=speed_bonus,
                                    max_rounds=_max_rounds)
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

    # Build list of Pass 1 orderings to try (DP bots only)
    dp_bot_ids = list(range(max_dp_bots))  # bots 0..max_dp_bots-1 get GPU DP
    base_order = bot_order if bot_order is not None else dp_bot_ids
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

    # Zone assignments: assign bots to aisle zones for spatial separation
    _zone_assignments = compute_zone_assignments(ms, num_bots, n_pipeline=n_pipeline)
    if verbose and _zone_assignments:
        zone_str = ', '.join(
            f"bot{b}:cols{cols}" for b, cols in sorted(_zone_assignments.items()))
        print(f"  Zone assignments: {zone_str}", file=sys.stderr)

    # LNS Order Assignment: primary (non-pipeline) DP bots get round-robin order slots.
    # This distributes orders across bots so they don't all compete for the same active order.
    _n_primary_dp = max(1, max_dp_bots - n_pipeline)
    _use_order_assign = use_order_assignment and _n_primary_dp >= 3

    best_p1_score = -1
    best_p1_actions = None
    best_p1_pipeline_ids = {}

    # --- Pass 1: Sequential planning (possibly multiple orderings) ---
    _ta_for_ordering = _type_assignments if use_type_specialization and num_bots >= 3 else {}

    for p1_idx, plan_order in enumerate(p1_orderings):
        # Time-bound: stop if we've exceeded the budget (keep best so far)
        if max_time_s and (time.time() - t0) > max_time_s and best_p1_actions is not None:
            if verbose:
                print(f"\n  Pass 1 time limit ({max_time_s:.0f}s) exceeded after "
                      f"{p1_idx}/{len(p1_orderings)} orderings, keeping best",
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

        if use_2bot_dp and len(plan_order) >= 2:
            # === 2-Bot DP: plan pairs jointly ===
            # Build pairs: (0,1), (2,3), ... and optional solo last bot
            pairs = [(plan_order[i], plan_order[i + 1])
                     for i in range(0, len(plan_order) - 1, 2)]
            solo = [plan_order[-1]] if len(plan_order) % 2 == 1 else []

            if verbose:
                pair_str = ', '.join(f"({a},{b})" for a, b in pairs)
                solo_str = f", solo: {solo}" if solo else ""
                print(f"  2-Bot DP pairs: {pair_str}{solo_str}", file=sys.stderr)

            for pair_idx, (bot_a, bot_b) in enumerate(pairs):
                t_pair = time.time()
                if verbose:
                    print(f"\n=== Pass 1 ({p1_idx+1}/{len(p1_orderings)}), "
                          f"Pair ({bot_a},{bot_b}) ===", file=sys.stderr)

                locked_ids = sorted(bot_actions.keys())
                locked = None
                if locked_ids:
                    locked = pre_simulate_locked(
                        _fresh_gs(gs, capture_data, no_filler), all_orders,
                        bot_actions, locked_ids, _zig_ctx=_zig_ctx)

                searcher = GPUBeamSearcher2Bot(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked,
                    candidate_bot_ids=(bot_a, bot_b),
                    no_compile=no_compile, speed_bonus=speed_bonus)

                # Verify on first pair of first ordering only
                if p1_idx == 0 and pair_idx == 0 and verbose:
                    gs_v = _fresh_gs(gs, capture_data, no_filler)
                    ok = searcher.verify_2bot_against_cpu(
                        gs_v, all_orders, bot1_id=bot_a, bot2_id=bot_b,
                        num_rounds=50)
                    if not ok:
                        print("2-BOT VERIFICATION FAILED, falling back to 1-bot",
                              file=sys.stderr)
                        use_2bot_dp = False
                        bot_actions = {}
                        break

                # State budget for pair
                effective_states = p1_states

                gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
                dp_score, acts_a, acts_b = searcher.dp_search_2bot(
                    gs_for_dp, max_states=effective_states, verbose=verbose,
                    bot_ids=(bot_a, bot_b))

                bot_actions[bot_a] = acts_a
                bot_actions[bot_b] = acts_b

                pair_time = time.time() - t_pair
                if verbose:
                    print(f"  Pair ({bot_a},{bot_b}): DP score={dp_score}, "
                          f"time={pair_time:.1f}s", file=sys.stderr)

                if on_bot_progress:
                    on_bot_progress(bot_b, num_bots, dp_score, time.time() - t0)

                del searcher
                if device == 'cuda':
                    torch.cuda.empty_cache()

            # Plan solo bot (if odd number) using single-bot DP
            for bot_id in solo:
                t_bot = time.time()
                if verbose:
                    print(f"\n=== Pass 1 ({p1_idx+1}/{len(p1_orderings)}), "
                          f"Bot {bot_id} (solo) ===", file=sys.stderr)

                locked_ids = sorted(bot_actions.keys())
                locked = None
                if locked_ids:
                    locked = pre_simulate_locked(
                        _fresh_gs(gs, capture_data, no_filler), all_orders,
                        bot_actions, locked_ids, _zig_ctx=_zig_ctx)

                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, no_compile=no_compile,
                    speed_bonus=speed_bonus, traffic_flow=_traffic_flow,
                    max_rounds=_max_rounds)

                gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
                dp_score, bot_acts = searcher.dp_search(
                    gs_for_dp, max_states=p1_states, verbose=verbose,
                    bot_id=bot_id)

                bot_actions[bot_id] = bot_acts
                bot_time = time.time() - t_bot
                if verbose:
                    print(f"  Bot {bot_id}: DP score={dp_score}, "
                          f"time={bot_time:.1f}s", file=sys.stderr)

                del searcher
                if device == 'cuda':
                    torch.cuda.empty_cache()

            # If 2-bot verification failed earlier, restart with single-bot
            if not use_2bot_dp:
                continue

        else:
            # === Original single-bot sequential DP ===
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
                bot_zone = _zone_assignments.get(bot_id)
                # LNS: assign order slot for non-pipeline primary bots
                _om = _n_primary_dp if (_use_order_assign and not is_pipeline) else None
                _os = (bot_id % _n_primary_dp) if _om else None
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, pipeline_mode=is_pipeline,
                    pipeline_depth=p_depth, preferred_types=pref_types,
                    no_compile=no_compile, order_cap=bot_order_cap,
                    speed_bonus=speed_bonus, preferred_zone=bot_zone,
                    order_modulo=_om, order_slot=_os,
                    traffic_flow=_traffic_flow, max_rounds=_max_rounds)

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

                # Position-aware state budget: first bot needs more exploration
                # (no locked bots, must discover good paths from scratch).
                # Later bots are more constrained, need fewer states.
                # pos 0 → 2.0x, pos N-1 → 0.5x
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

        # Fill greedy bots (not DP-planned) with CPU greedy actions
        if _greedy_bot_ids:
            greedy_acts = greedy_plan_bots(
                gs, all_orders, _greedy_bot_ids, bot_actions, ms, num_bots,
                _zig_ctx=_zig_ctx, capture_data=capture_data, no_filler=no_filler)
            for bid, acts in greedy_acts.items():
                bot_actions[bid] = acts

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

    # Short time budgets (pipeline): stop quickly on plateau. Long budgets: allow more escapes.
    if max_time_s and max_time_s <= 60:
        _escape_limit = 2
    elif max_time_s and max_time_s >= 300:
        _escape_limit = 6  # deep training: more escape attempts
    else:
        _escape_limit = 4
    no_improve_iters = 0
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
        # Only refine DP bots; greedy bots are re-generated after each iteration.
        # Strategy: forward, backward, then WEAKEST-FIRST (contribution-based).
        if iteration == 0:
            refine_order = list(dp_bot_ids)
        elif iteration == 1:
            refine_order = list(reversed(dp_bot_ids))
        else:
            # Weakest-first: compute per-bot contributions and replan worst bots first
            try:
                contribs = compute_bot_contributions(
                    _fresh_gs(gs, capture_data, no_filler), all_orders,
                    bot_actions, num_bots, dp_bot_ids,
                    _zig_ctx=_zig_ctx, capture_data=capture_data,
                    no_filler=no_filler)
                refine_order = sorted(dp_bot_ids, key=lambda b: contribs.get(b, 0))
                if verbose:
                    contrib_str = ', '.join(f'b{b}:{contribs.get(b,0)}' for b in refine_order)
                    print(f"  Contributions (weakest-first): {contrib_str}", file=sys.stderr)
            except Exception:
                refine_order = list(dp_bot_ids)
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
                bot_zone = _zone_assignments.get(bot_id)
                # LNS: assign order slot for non-pipeline primary bots
                _om_r = _n_primary_dp if (_use_order_assign and not is_pipeline) else None
                _os_r = (bot_id % _n_primary_dp) if _om_r else None
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, pipeline_mode=is_pipeline,
                    pipeline_depth=p_depth, preferred_types=pref_types,
                    no_compile=no_compile, speed_bonus=speed_bonus,
                    preferred_zone=bot_zone,
                    order_modulo=_om_r, order_slot=_os_r,
                    traffic_flow=_traffic_flow, max_rounds=_max_rounds)

                # Eval annealing disabled — testing showed it hurts more than helps.
                # coord_temperature stays at 0.0 (full penalties).

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
                                pass  # Future may have already failed; we discard its result anyway
                            del _pending_locked[next_bot]
                    if verbose:
                        print(f"  Bot {bot_id}: DP={dp_score}, CPU={new_score} "
                              f"(no improvement, reverted), time={bot_time:.1f}s",
                              file=sys.stderr)
                    consecutive_fails += 1
                    if consecutive_fails >= max_dp_bots:
                        if verbose:
                            print(f"  Early stop: {max_dp_bots} consecutive no-improvements",
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

        # === Joint 2-bot refinement (periodic) ===
        # Every 3 iterations, try joint DP on the 2 weakest bots.
        # Uses distance-adaptive expansion for efficient state budget.
        if (iteration >= 2 and iteration % 3 == 2 and
                len(dp_bot_ids) >= 2 and
                (max_time_s is None or (time.time() - t0) < max_time_s * 0.85)):
            try:
                _j_contribs = compute_bot_contributions(
                    _fresh_gs(gs, capture_data, no_filler), all_orders,
                    bot_actions, num_bots, dp_bot_ids,
                    _zig_ctx=_zig_ctx, capture_data=capture_data,
                    no_filler=no_filler)
                _j_sorted = sorted(dp_bot_ids, key=lambda b: _j_contribs.get(b, 0))
                _j_pair = (_j_sorted[0], _j_sorted[1])
                if verbose:
                    print(f"\n=== Joint 2-bot refine: pair ({_j_pair[0]},{_j_pair[1]}) ===",
                          file=sys.stderr)

                _j_locked_ids = sorted(b for b in range(num_bots)
                                       if b != _j_pair[0] and b != _j_pair[1])
                _j_locked = None
                if _j_locked_ids:
                    _j_locked = pre_simulate_locked(
                        _fresh_gs(gs, capture_data, no_filler), all_orders,
                        bot_actions, _j_locked_ids, _zig_ctx=_zig_ctx)

                _j_searcher = GPUBeamSearcher2Bot(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=_j_locked,
                    candidate_bot_ids=_j_pair,
                    no_compile=no_compile, speed_bonus=speed_bonus)

                _j_states = max(max_states // 2, 25000)  # Half budget for pair
                gs_for_j = _fresh_gs(gs, capture_data, no_filler)
                _j_score, _j_acts_a, _j_acts_b = _j_searcher.dp_search_2bot(
                    gs_for_j, max_states=_j_states, verbose=verbose,
                    bot_ids=_j_pair)

                # Apply and verify
                _j_old_a = list(bot_actions[_j_pair[0]])
                _j_old_b = list(bot_actions[_j_pair[1]])
                bot_actions[_j_pair[0]] = _j_acts_a
                bot_actions[_j_pair[1]] = _j_acts_b
                _j_combined = _make_combined(bot_actions, num_bots)
                gs_jv = _fresh_gs(gs, capture_data, no_filler)
                _j_new_score = cpu_verify(gs_jv, all_orders, _j_combined, num_bots, _zig_ctx)
                if _j_new_score > best_score:
                    _j_delta = _j_new_score - best_score
                    best_score = _j_new_score
                    best_actions = {k: list(v) for k, v in bot_actions.items()}
                    iter_improved = True
                    if verbose:
                        print(f"  Joint 2-bot: CPU={_j_new_score} "
                              f"(+{_j_delta}! best={best_score})",
                              file=sys.stderr)
                else:
                    bot_actions[_j_pair[0]] = _j_old_a
                    bot_actions[_j_pair[1]] = _j_old_b
                    if verbose:
                        print(f"  Joint 2-bot: CPU={_j_new_score} "
                              f"(no improvement, reverted)", file=sys.stderr)

                del _j_searcher
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                if verbose:
                    print(f"  Joint 2-bot failed: {e}", file=sys.stderr)

        # Re-generate greedy bot plans after DP bots may have changed
        if _greedy_bot_ids and iter_improved:
            greedy_acts = greedy_plan_bots(
                gs, all_orders, _greedy_bot_ids, bot_actions, ms, num_bots,
                _zig_ctx=_zig_ctx, capture_data=capture_data, no_filler=no_filler)
            for bid, acts in greedy_acts.items():
                bot_actions[bid] = acts
            # Re-verify with updated greedy plans
            combined = _make_combined(bot_actions, num_bots)
            gs_v = _fresh_gs(gs, capture_data, no_filler)
            new_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx=_zig_ctx)
            if new_score > best_score:
                best_score = new_score
                best_actions = {k: list(v) for k, v in bot_actions.items()}
            elif new_score < best_score:
                # Greedy re-gen hurt score; restore best
                bot_actions = {k: list(v) for k, v in best_actions.items()}

        if on_phase:
            on_phase(f"refine_done", iteration + 1, best_score)

        if verbose:
            print(f"\nRefine iter {iteration+1}: best_score={best_score}",
                  file=sys.stderr)

        if not iter_improved:
            no_improve_iters += 1
            if no_improve_iters >= _escape_limit:
                if verbose:
                    print(f"  Stopping refinement ({no_improve_iters} consecutive "
                          f"no-improvement iters)", file=sys.stderr)
                break
            else:
                # Perturbation escape: reset weakest bot(s) + reshuffle type assignments.
                # After 2+ consecutive no-improvement, reset the 2 weakest bots
                # (pair perturbation) to allow re-coordination.
                try:
                    contribs = compute_bot_contributions(
                        _fresh_gs(gs, capture_data, no_filler), all_orders,
                        bot_actions, num_bots, dp_bot_ids,
                        _zig_ctx=_zig_ctx, capture_data=capture_data,
                        no_filler=no_filler)
                    sorted_bots = sorted(dp_bot_ids, key=lambda b: contribs.get(b, 0))
                    if no_improve_iters >= 2 and len(sorted_bots) >= 2:
                        # Pair perturbation: reset 2 weakest bots
                        perturb_bots = sorted_bots[:2]
                    else:
                        perturb_bots = [sorted_bots[0]]
                except Exception:
                    perturb_bots = [_rng.choice(dp_bot_ids)]
                for pb in perturb_bots:
                    bot_actions[pb] = [(ACT_WAIT, -1)] * MAX_ROUNDS
                # Reshuffle type assignments to explore different labor divisions
                if use_type_specialization and num_bots >= 3:
                    _type_assignments = compute_type_assignments(
                        all_orders, num_bots, ms.num_types, ms,
                        shuffle_seed=iteration * 1000 + no_improve_iters)
                if verbose:
                    pb_str = ','.join(str(b) for b in perturb_bots)
                    print(f"  Perturbation escape: reset bot(s) {pb_str} "
                          f"(weakest) + reshuffle types "
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


def generate_orderings(num_bots: int, k: int = 1000, seed: int = 42) -> list[list[int]]:
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


def batch_evaluate_orderings(capture_data: CaptureData, orderings_list: list[list[int]],
                             device: str = 'cuda', n_steps: int = 25) -> list[float]:
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


def solve_multi_restart(capture_data: CaptureData | None = None,
                        seed: int | None = None, difficulty: str | None = None,
                        device: str = 'cuda', max_states: int = 500000,
                        verbose: bool = True,
                        max_refine_iters: int = 2, num_restarts: int = 3,
                        num_screen: int = 1000, n_screen_steps: int = 25,
                        on_restart: Callable | None = None,
                        no_filler: bool = False,
                        all_orders_override: list[Order] | None = None,
                        no_compile: bool = False) -> tuple[int, list[list[tuple[int, int]]]]:
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


def refine_from_solution(combined_actions: list[list[tuple[int, int]]],
                         capture_data: CaptureData | None = None,
                         seed: int | None = None,
                         difficulty: str | None = None,
                         device: str = 'cuda',
                         config: SolveConfig | None = None,
                         callbacks: SolveCallbacks | None = None,
                         verbose: bool = True,
                         all_orders_override: list[Order] | None = None,
                         # Legacy kwargs for backward compat - prefer config/callbacks
                         **kwargs,
                         ) -> tuple[int, list[list[tuple[int, int]]]]:
    """Refine an existing multi-bot solution via GPU DP.

    Loads a pre-existing solution (e.g., from a previous GPU DP run or Python
    planner), then runs refinement iterations to improve individual bot paths.

    Args:
        combined_actions: List of 300 round_actions, each is [(act, item)] * num_bots.
        config: SolveConfig with solver parameters (or pass legacy kwargs).
        callbacks: SolveCallbacks with progress callbacks (or pass legacy kwargs).
        Other args same as solve_sequential.

    Returns:
        (best_score, best_combined_actions) tuple.
        Score is 0 if refinement produces no valid result (unlikely in practice).

    Raises:
        ValueError: If neither capture_data nor (seed + difficulty) is provided.
    """
    # Merge legacy kwargs into config/callbacks (same bridge as solve_sequential)
    if config is None:
        config = SolveConfig()
    if callbacks is None:
        callbacks = SolveCallbacks()

    _CONFIG_FIELDS = {f.name for f in config.__dataclass_fields__.values()}
    _CB_FIELDS = {f.name for f in callbacks.__dataclass_fields__.values()}
    for k, v in kwargs.items():
        if k in _CONFIG_FIELDS:
            setattr(config, k, v)
        elif k in _CB_FIELDS:
            setattr(callbacks, k, v)
        elif k.startswith('on_') and k.lstrip('on_') in ('bot_progress', 'round', 'phase'):
            setattr(callbacks, k, v)
        else:
            raise TypeError(f"refine_from_solution() got unexpected keyword argument '{k}'")

    max_states = config.max_states if config.max_states is not None else 500000
    max_refine_iters = config.max_refine_iters if config.max_refine_iters is not None else 3
    no_filler = config.no_filler
    no_compile = config.no_compile
    max_time_s = config.max_time_s
    speed_bonus = config.speed_bonus
    pipeline_fraction = config.pipeline_fraction
    max_pipeline_depth = config.max_pipeline_depth
    on_bot_progress = callbacks.on_bot_progress
    on_round = callbacks.on_round
    on_phase = callbacks.on_phase

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
    _max_rounds = _DIFF_ROUNDS.get(diff, 300)

    # max_dp_bots: only refine this many bots via GPU DP; rest get CPU greedy
    max_dp_bots_r = config.max_dp_bots
    if max_dp_bots_r is None:
        max_dp_bots_r = DEFAULT_MAX_DP_BOTS.get(diff, num_bots)
    max_dp_bots_r = min(max_dp_bots_r, num_bots)
    dp_bot_ids_r = list(range(max_dp_bots_r))
    _greedy_bot_ids_r = list(range(max_dp_bots_r, num_bots))

    # Build Zig FFI context (seed or live capture)
    _zig_seed = seed if seed else (capture_data.get('seed') if capture_data else None)
    if _ZIG_AVAILABLE and _zig_seed:
        _zig_ctx = {'diff_idx': _DIFF_IDX.get(diff, 0), 'seed': _zig_seed, 'mode': 'seed'}
    elif _ZIG_AVAILABLE and capture_data is not None:
        _zig_ctx = {'capture_data': capture_data, 'mode': 'live'}
    else:
        _zig_ctx = None

    # Convert combined_actions to per-bot format
    # Validate bot count matches — old solution may have wrong bot count
    solution_bots = len(combined_actions[0]) if combined_actions else 0
    if solution_bots != num_bots:
        raise ValueError(
            f"Solution has {solution_bots} bots but game needs {num_bots}. "
            f"Clear stale solution: rm solutions/{diff}/best.json")
    bot_actions = {}
    _wait_pad = (ACT_WAIT, -1)
    for bid in range(num_bots):
        acts = [(r_acts[bid][0], r_acts[bid][1]) for r_acts in combined_actions]
        while len(acts) < MAX_ROUNDS:
            acts.append(_wait_pad)
        bot_actions[bid] = acts

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
    _zone_assignments_r = compute_zone_assignments(ms, num_bots, n_pipeline=n_pipeline)

    # Traffic flow for medium maps
    _traffic_flow = None
    if diff == 'medium' and num_bots >= 2:
        _aisle_cols = sorted(x for x in range(ms.width)
                             if sum(1 for y in range(ms.height)
                                    if ms.grid[y, x] in (0, 3)) >= ms.height * 0.6)
        if len(_aisle_cols) >= 2:
            _traffic_flow = {'up_col': _aisle_cols[0], 'down_col': _aisle_cols[1]}

    # Generate greedy plans for non-DP bots
    if _greedy_bot_ids_r:
        greedy_acts = greedy_plan_bots(
            gs, all_orders, _greedy_bot_ids_r, bot_actions, ms, num_bots,
            _zig_ctx=_zig_ctx, capture_data=capture_data, no_filler=no_filler)
        for bid, acts in greedy_acts.items():
            bot_actions[bid] = acts

    # Verify starting score
    gs_v = _fresh_gs(gs, capture_data, no_filler)
    best_score = cpu_verify(gs_v, all_orders, _make_combined(bot_actions, num_bots),
                            num_bots, _zig_ctx=_zig_ctx)
    best_actions = {k: list(v) for k, v in bot_actions.items()}

    if verbose:
        dp_str = f" (DP bots: {dp_bot_ids_r}, greedy: {_greedy_bot_ids_r})" if _greedy_bot_ids_r else ""
        print(f"Warm-start refinement: {diff}, {num_bots} bots{dp_str}, "
              f"starting_score={best_score}, max_states={max_states}, "
              f"refine_iters={max_refine_iters}", file=sys.stderr)

    if on_phase:
        on_phase("warm_start", 0, best_score)

    # Run refinement iterations — only refine DP bots
    import random as _random_rfn
    _rng_rfn = _random_rfn.Random(42)

    no_improve_iters = 0
    _n_dp = len(dp_bot_ids_r)
    for iteration in range(max_refine_iters):
        # Time-budget check
        if max_time_s is not None and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"  [refine_from_solution] Time budget {max_time_s}s reached at "
                      f"iter {iteration}, stopping", file=sys.stderr)
            break

        if on_phase:
            on_phase("refine", iteration + 1, best_score)

        # Only refine DP bots; greedy bots re-generated after
        if iteration == 0:
            refine_order = list(dp_bot_ids_r)
        elif iteration == 1:
            refine_order = list(reversed(dp_bot_ids_r))
        else:
            refine_order = list(dp_bot_ids_r)
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
                bot_zone_r = _zone_assignments_r.get(bot_id)
                # LNS: assign order slot for non-pipeline primary bots
                _n_pri_r = max(1, max_dp_bots_r - len(pipeline_bot_ids))
                _use_oa_r = _n_pri_r >= 3
                _om_rfn = _n_pri_r if (_use_oa_r and not is_pipeline) else None
                _os_rfn = (bot_id % _n_pri_r) if _om_rfn else None
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=device, num_bots=num_bots,
                    locked_trajectories=locked, pipeline_mode=is_pipeline,
                    pipeline_depth=p_depth, preferred_types=pref_types_r,
                    no_compile=no_compile, speed_bonus=speed_bonus,
                    preferred_zone=bot_zone_r,
                    order_modulo=_om_rfn, order_slot=_os_rfn,
                    traffic_flow=_traffic_flow, max_rounds=_max_rounds)

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
                    if consecutive_fails >= _n_dp:
                        if verbose:
                            print(f"  Early stop: {_n_dp} consecutive no-improvements",
                                  file=sys.stderr)
                        break

                if on_bot_progress:
                    on_bot_progress(bot_id, num_bots, best_score, time.time() - t0)

        # Re-generate greedy bot plans after DP bots may have changed
        if _greedy_bot_ids_r and iter_improved:
            greedy_acts = greedy_plan_bots(
                gs, all_orders, _greedy_bot_ids_r, bot_actions, ms, num_bots,
                _zig_ctx=_zig_ctx, capture_data=capture_data, no_filler=no_filler)
            for bid, acts in greedy_acts.items():
                bot_actions[bid] = acts
            combined = _make_combined(bot_actions, num_bots)
            gs_v = _fresh_gs(gs, capture_data, no_filler)
            new_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx=_zig_ctx)
            if new_score > best_score:
                best_score = new_score
                best_actions = {k: list(v) for k, v in bot_actions.items()}
            elif new_score < best_score:
                bot_actions = {k: list(v) for k, v in best_actions.items()}

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


def duo_refine_from_solution(
    combined_actions: list[list[tuple[int, int]]],
    capture_data: CaptureData | None = None,
    seed: int | None = None,
    difficulty: str | None = None,
    device: str = 'cuda',
    max_states: int = 200_000,
    max_pairs: int = 5,
    max_time_s: float | None = None,
    speed_bonus: float = 100.0,
    no_filler: bool = True,
    no_compile: bool = False,
    verbose: bool = True,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Duo refinement: re-plan pairs of bots jointly via 2-bot DP.

    Pairs the weakest bot (by marginal contribution) with each stronger bot
    in turn, planning them jointly while all other bots are locked. Keeps
    the result only if it improves the total score.

    This breaks the sequential DP ceiling by allowing 2 bots to coordinate
    their paths simultaneously — one bot can yield a pickup to the other
    if it produces a better joint outcome.

    Args:
        combined_actions: Existing solution (300 rounds × num_bots).
        max_states: State budget for 2-bot joint DP (default 200K).
        max_pairs: Maximum number of pair attempts (default 5).
        max_time_s: Time budget in seconds.
        Other args: same as refine_from_solution.

    Returns:
        (best_score, best_combined_actions) tuple.
    """
    t0 = time.time()

    if capture_data:
        num_orders = len(capture_data['orders']) if no_filler else 100
        gs, all_orders = init_game_from_capture(capture_data, num_orders=num_orders)
        diff = capture_data.get('difficulty', difficulty or 'hard')
    elif seed and difficulty:
        gs, all_orders = init_game(seed, difficulty)
        diff = difficulty
    else:
        raise ValueError("Need capture_data or (seed + difficulty)")

    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    if num_bots < 2:
        if verbose:
            print("  Duo refine: need at least 2 bots, skipping", file=sys.stderr)
        return 0, combined_actions

    # Build Zig FFI context
    _zig_seed = seed if seed else (capture_data.get('seed') if capture_data else None)
    if _ZIG_AVAILABLE and _zig_seed:
        _zig_ctx = {'diff_idx': _DIFF_IDX.get(diff, 0), 'seed': _zig_seed, 'mode': 'seed'}
    elif _ZIG_AVAILABLE and capture_data is not None:
        _zig_ctx = {'capture_data': capture_data, 'mode': 'live'}
    else:
        _zig_ctx = None

    # Convert to per-bot format
    bot_actions = {}
    _wait_pad = (ACT_WAIT, -1)
    for bid in range(num_bots):
        acts = [(r_acts[bid][0], r_acts[bid][1]) for r_acts in combined_actions]
        while len(acts) < MAX_ROUNDS:
            acts.append(_wait_pad)
        bot_actions[bid] = acts

    # Verify starting score
    gs_v = _fresh_gs(gs, capture_data, no_filler)
    best_score = cpu_verify(gs_v, all_orders, _make_combined(bot_actions, num_bots),
                            num_bots, _zig_ctx=_zig_ctx)
    best_actions = {k: list(v) for k, v in bot_actions.items()}

    # Compute marginal contributions
    dp_bot_ids = list(range(num_bots))
    contribs = compute_bot_contributions(
        _fresh_gs(gs, capture_data, no_filler), all_orders,
        bot_actions, num_bots, dp_bot_ids,
        _zig_ctx=_zig_ctx, capture_data=capture_data, no_filler=no_filler)

    sorted_bots = sorted(dp_bot_ids, key=lambda b: contribs.get(b, 0))

    if verbose:
        contrib_str = ', '.join(f'b{b}:{contribs.get(b, 0)}' for b in sorted_bots)
        print(f"\nDuo refinement: {diff}, {num_bots} bots, "
              f"starting_score={best_score}, max_states={max_states}",
              file=sys.stderr)
        print(f"  Contributions (weak→strong): {contrib_str}", file=sys.stderr)

    # Build pair candidates: weakest bot paired with each stronger bot
    weakest = sorted_bots[0]
    partners = sorted_bots[1:]  # strongest last

    # Also try pairing 2nd-weakest with strongest
    pair_candidates = [(weakest, p) for p in reversed(partners)]
    if len(sorted_bots) >= 3:
        second_weakest = sorted_bots[1]
        strongest = sorted_bots[-1]
        if (second_weakest, strongest) not in pair_candidates:
            pair_candidates.append((second_weakest, strongest))

    pair_candidates = pair_candidates[:max_pairs]

    if verbose:
        pairs_str = ', '.join(f'({a},{b})' for a, b in pair_candidates)
        print(f"  Pair candidates: {pairs_str}", file=sys.stderr)

    for pair_idx, (bot_a, bot_b) in enumerate(pair_candidates):
        if max_time_s is not None and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"  Time budget reached after {pair_idx} pairs", file=sys.stderr)
            break

        t_pair = time.time()
        # Ensure bot_a < bot_b for consistent ID ordering
        if bot_a > bot_b:
            bot_a, bot_b = bot_b, bot_a

        if verbose:
            print(f"\n=== Duo pair {pair_idx+1}/{len(pair_candidates)}: "
                  f"bots ({bot_a},{bot_b}) "
                  f"[contrib: {contribs.get(bot_a, 0)}, {contribs.get(bot_b, 0)}] ===",
                  file=sys.stderr)

        # Lock all bots except this pair
        locked_ids = sorted(b for b in range(num_bots) if b != bot_a and b != bot_b)
        locked = pre_simulate_locked(
            _fresh_gs(gs, capture_data, no_filler), all_orders,
            bot_actions, locked_ids, _zig_ctx=_zig_ctx)

        searcher = GPUBeamSearcher2Bot(
            ms, all_orders, device=device, num_bots=num_bots,
            locked_trajectories=locked,
            candidate_bot_ids=(bot_a, bot_b),
            no_compile=no_compile, speed_bonus=speed_bonus)

        gs_for_dp = _fresh_gs(gs, capture_data, no_filler)
        dp_score, acts_a, acts_b = searcher.dp_search_2bot(
            gs_for_dp, max_states=max_states, verbose=verbose,
            bot_ids=(bot_a, bot_b))

        del searcher
        if device == 'cuda':
            torch.cuda.empty_cache()

        pair_time = time.time() - t_pair

        # Try replacing and verify
        old_a = bot_actions[bot_a]
        old_b = bot_actions[bot_b]
        bot_actions[bot_a] = acts_a
        bot_actions[bot_b] = acts_b

        combined = _make_combined(bot_actions, num_bots)
        gs_v = _fresh_gs(gs, capture_data, no_filler)
        new_score = cpu_verify(gs_v, all_orders, combined, num_bots, _zig_ctx=_zig_ctx)

        if new_score > best_score:
            delta = new_score - best_score
            best_score = new_score
            best_actions = {k: list(v) for k, v in bot_actions.items()}
            if verbose:
                print(f"  Pair ({bot_a},{bot_b}): DP={dp_score}, CPU={new_score} "
                      f"(+{delta}! best={best_score}), time={pair_time:.1f}s",
                      file=sys.stderr)
        else:
            # Revert
            bot_actions[bot_a] = old_a
            bot_actions[bot_b] = old_b
            if verbose:
                print(f"  Pair ({bot_a},{bot_b}): DP={dp_score}, CPU={new_score} "
                      f"(no improvement, reverted), time={pair_time:.1f}s",
                      file=sys.stderr)

    combined = _make_combined(best_actions, num_bots)
    total_time = time.time() - t0
    if verbose:
        print(f"\nDuo refinement: final_score={best_score}, "
              f"total_time={total_time:.1f}s", file=sys.stderr)

    return best_score, combined


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sequential per-bot GPU DP solver')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
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
