#!/usr/bin/env python3
"""Offline reactive solver: per-round GPU decisions against local simulator.

Replicates what live_gpu_stream.py does on the live server, but offline with
unlimited time per round and much higher GPU budgets. This is the approach
that produced our best Hard score (196) — reactive per-round decisions allow
all bots to adapt to each other simultaneously, avoiding the sequential DP
ceiling that caps offline planners at ~180.

Key advantages over offline DP (solve_sequential):
  - All bots see actual game state every round (no frozen trajectories)
  - Order transitions happen naturally (no timing assumptions)
  - Emergent coordination: bots adapt to each other's actions in real time

Key advantages over live solver (live_gpu_stream.py):
  - No 2-second per-round timeout (unlimited GPU budget per round)
  - Can run multiple attempts and keep the best
  - Uses pre-captured order data (full foresight)

Usage:
    # Standard run on Hard (capture data from DB)
    python offline_reactive.py hard --max-states 200000 --horizon 100

    # Multiple attempts to find best score
    python offline_reactive.py hard --attempts 5 --max-states 150000

    # Quick test
    python offline_reactive.py hard --max-states 50000 --horizon 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import numpy as np
import torch

from game_engine import (
    init_game_from_capture, step as cpu_step,
    GameState, MapState, Order,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_DROPOFF,
    DX, DY,
)
from gpu_beam_search import GPUBeamSearcher
from precompute import PrecomputedTables


def build_gpu_init_state(
    gs: GameState, bot_id: int, all_orders: list[Order],
    device: str = 'cuda',
    locked_positions: list[tuple[int, int]] | None = None,
    locked_inventories: list[list[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Build GPU initial state dict from a GameState for one candidate bot.

    This replaces live_gpu_stream's _build_bot_gpu_state, working directly
    from GameState instead of WebSocket JSON.
    """
    ms = gs.map_state
    bx = int(gs.bot_positions[bot_id, 0])
    by = int(gs.bot_positions[bot_id, 1])

    inv = [-1] * INV_CAP
    for s in range(INV_CAP):
        inv[s] = int(gs.bot_inventories[bot_id, s])

    # Find active order index and delivery mask
    active_idx = 0
    active_del = np.zeros(MAX_ORDER_SIZE, dtype=np.int8)
    for o in gs.orders:
        if o.status == 'active':
            # Find this order's index in all_orders by matching ID
            active_idx = o.id
            for j in range(len(o.delivered)):
                if j < MAX_ORDER_SIZE:
                    active_del[j] = int(o.delivered[j])
            break

    state = {
        'bot_x': torch.tensor([bx], dtype=torch.int16, device=device),
        'bot_y': torch.tensor([by], dtype=torch.int16, device=device),
        'bot_inv': torch.tensor([inv], dtype=torch.int8, device=device),
        'active_idx': torch.tensor([active_idx], dtype=torch.int32, device=device),
        'active_del': torch.tensor([active_del], dtype=torch.int8, device=device),
        'score': torch.tensor([0], dtype=torch.int32, device=device),
        'orders_comp': torch.tensor([0], dtype=torch.int32, device=device),
    }

    if locked_positions:
        num_locked = len(locked_positions)
        if locked_inventories is not None:
            linv_arr = np.array(locked_inventories, dtype=np.int8)
        else:
            linv_arr = np.full((num_locked, INV_CAP), -1, dtype=np.int8)
        state['locked_inv'] = torch.tensor(
            linv_arr[np.newaxis], dtype=torch.int8, device=device)
        state['locked_bx'] = torch.tensor(
            [[x for x, y in locked_positions]], dtype=torch.int16, device=device)
        state['locked_by'] = torch.tensor(
            [[y for x, y in locked_positions]], dtype=torch.int16, device=device)

    return state


def simulate_single_bot_trajectory(
    start_pos: tuple[int, int],
    planned_acts: list[tuple[int, int]],
    walkable: set[tuple[int, int]],
    horizon: int,
) -> list[tuple[int, int]]:
    """Simulate one bot's positions from planned actions (movement only)."""
    bx, by = start_pos
    positions = []
    for i in range(min(horizon, len(planned_acts))):
        act, _item = planned_acts[i]
        if ACT_MOVE_UP <= act <= ACT_MOVE_RIGHT:
            nx = bx + DX[act]
            ny = by + DY[act]
            if (nx, ny) in walkable:
                bx, by = nx, ny
        positions.append((bx, by))
    while len(positions) < horizon:
        positions.append((bx, by))
    return positions


def build_locked_trajs(
    planned_bots: list[tuple[int, list, list]],
    start_rnd: int,
) -> dict[str, np.ndarray]:
    """Build locked_trajectories dict from sequentially planned bots."""
    num_locked = len(planned_bots)
    locked_actions = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int8)
    locked_action_items = np.full((num_locked, MAX_ROUNDS), -1, dtype=np.int16)
    locked_pos_x = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
    locked_pos_y = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
    locked_bot_ids = []

    for li, (real_bid, planned_acts, positions) in enumerate(planned_bots):
        locked_bot_ids.append(real_bid)
        for i, ((act, item), (px, py)) in enumerate(zip(planned_acts, positions)):
            r = start_rnd + i
            if r >= MAX_ROUNDS:
                break
            locked_actions[li, r] = act
            locked_action_items[li, r] = item if item is not None and item >= 0 else -1
            locked_pos_x[li, r] = px
            locked_pos_y[li, r] = py
        # Fill before start_rnd
        if positions:
            fx, fy = positions[0]
        else:
            fx, fy = 0, 0
        for r in range(start_rnd):
            locked_pos_x[li, r] = fx
            locked_pos_y[li, r] = fy
        # Fill after planned horizon
        if positions:
            lx, ly = positions[-1]
            last_r = start_rnd + len(positions)
            for r in range(last_r, MAX_ROUNDS):
                locked_pos_x[li, r] = lx
                locked_pos_y[li, r] = ly

    return {
        'locked_actions': locked_actions,
        'locked_action_items': locked_action_items,
        'locked_pos_x': locked_pos_x,
        'locked_pos_y': locked_pos_y,
        'locked_bot_ids': locked_bot_ids,
    }


def bot_priority_key(gs: GameState, bid: int, tables: PrecomputedTables | None):
    """Sort key: bots carrying items first (by distance to dropoff), then others."""
    ms = gs.map_state
    bx = int(gs.bot_positions[bid, 0])
    by = int(gs.bot_positions[bid, 1])
    inv_count = gs.bot_inv_count(bid)
    drop_pos = tuple(ms.drop_off)

    if inv_count > 0:
        if tables is not None:
            try:
                dist = int(tables.get_distance((bx, by), drop_pos))
            except Exception:
                dist = abs(bx - drop_pos[0]) + abs(by - drop_pos[1])
        else:
            dist = abs(bx - drop_pos[0]) + abs(by - drop_pos[1])
        return (0, dist, bid)
    else:
        return (1, 0, bid)


def offline_reactive_solve(
    capture_data: dict,
    all_orders: list[Order],
    difficulty: str,
    device: str = 'cuda',
    max_states: int = 200_000,
    horizon: int = 100,
    speed_bonus: float = 100.0,
    replan_interval: int = 0,
    verbose: bool = True,
    attempt: int = 0,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Run one offline reactive solve with periodic full replanning.

    Strategy: Run full sequential DP from current game state, execute the
    plan for `replan_interval` rounds, then re-plan from actual state.
    This combines the long-horizon quality of offline DP with the adaptive
    advantage of reactive planning.

    If replan_interval=0, runs per-round reactive (original mode, slow).

    Returns (score, combined_actions) where combined_actions is
    list of 300 round_actions, each is [(act, item)] * num_bots.
    """
    from gpu_sequential_solver import (
        solve_sequential, pre_simulate_locked, cpu_verify,
        _make_combined, _fresh_gs, compute_bot_contributions,
        DEFAULT_MAX_DP_BOTS,
    )

    num_orders = len(capture_data.get('orders', []))
    gs, _ = init_game_from_capture(capture_data, num_orders=num_orders)
    ms = gs.map_state
    num_bots = len(gs.bot_positions)

    # Build walkable set for trajectory simulation
    walkable = set()
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                walkable.add((x, y))

    tables = PrecomputedTables.get(ms)

    t0 = time.time()

    if replan_interval > 0:
        # === PERIODIC REPLAN MODE ===
        # Full sequential DP from current state, execute N rounds, then replan
        all_round_actions = []
        current_plan = None  # per-bot actions for remaining rounds
        plan_rnd_offset = 0  # which round the plan starts from

        # Vary bot ordering across attempts
        import random
        rng = random.Random(42 + attempt * 1000)

        replan_count = 0

        for rnd in range(MAX_ROUNDS):
            remaining = MAX_ROUNDS - rnd

            # Do we need to (re)plan?
            need_replan = (current_plan is None or
                           (rnd - plan_rnd_offset) >= replan_interval)

            if need_replan and remaining > 5:
                replan_count += 1
                if verbose:
                    elapsed = time.time() - t0
                    print(f"\n  [Replan #{replan_count} at R{rnd}, "
                          f"score={gs.score}, remaining={remaining}, "
                          f"t={elapsed:.1f}s]", file=sys.stderr)

                # Build capture-like data from CURRENT game state
                # We need to create a modified capture with current positions/inventories
                plan_gs = gs.copy()

                # Plan all bots sequentially from current state
                bot_actions = {}
                max_dp_bots = DEFAULT_MAX_DP_BOTS.get(difficulty, num_bots)
                dp_bots = list(range(min(max_dp_bots, num_bots)))

                # Vary ordering
                if attempt == 0:
                    plan_order = list(dp_bots)
                elif attempt == 1:
                    plan_order = list(reversed(dp_bots))
                else:
                    plan_order = list(dp_bots)
                    rng.shuffle(plan_order)

                for bot_pos_idx, bot_id in enumerate(plan_order):
                    # Lock previously planned bots
                    locked_ids = sorted(bot_actions.keys())
                    locked = None
                    if locked_ids:
                        # Build per-bot actions for locked bots spanning remaining rounds
                        # For locked bots, extend their actions with wait from current plan
                        locked_bot_acts = {}
                        for lid in locked_ids:
                            # Build full 300-round action list (wait for past rounds,
                            # planned actions for future)
                            full_acts = [(ACT_WAIT, -1)] * MAX_ROUNDS
                            for r_off, act in enumerate(bot_actions[lid]):
                                r = rnd + r_off
                                if r < MAX_ROUNDS:
                                    full_acts[r] = act
                            locked_bot_acts[lid] = full_acts

                        locked = pre_simulate_locked(
                            plan_gs.copy(), all_orders, locked_bot_acts, locked_ids)

                    searcher = GPUBeamSearcher(
                        ms, all_orders, device=device, num_bots=num_bots,
                        locked_trajectories=locked,
                        no_compile=True, speed_bonus=speed_bonus)

                    init_state = build_gpu_init_state(
                        gs, bot_id, all_orders, device=device,
                        locked_positions=None, locked_inventories=None)

                    # If there are locked bots, set their current positions in init_state
                    if locked_ids:
                        lp = []
                        li = []
                        for lid in sorted(locked_ids):
                            lp.append((int(gs.bot_positions[lid, 0]),
                                       int(gs.bot_positions[lid, 1])))
                            inv_l = [int(gs.bot_inventories[lid, s]) for s in range(INV_CAP)]
                            li.append(inv_l)
                        init_state['locked_bx'] = torch.tensor(
                            [[x for x, y in lp]], dtype=torch.int16, device=device)
                        init_state['locked_by'] = torch.tensor(
                            [[y for x, y in lp]], dtype=torch.int16, device=device)
                        init_state['locked_inv'] = torch.tensor(
                            [li], dtype=torch.int8, device=device)

                    dp_score, acts = searcher.dp_search(
                        game_state=None,
                        max_states=max_states,
                        verbose=False,
                        start_rnd=rnd,
                        max_rounds=remaining,
                        init_state=init_state,
                        bot_id=bot_id)

                    bot_actions[bot_id] = acts if acts else [(ACT_WAIT, -1)] * remaining

                    del searcher
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                    if verbose:
                        print(f"    Bot {bot_id}: DP={dp_score}", file=sys.stderr)

                # Non-DP bots get wait
                for bid in range(num_bots):
                    if bid not in bot_actions:
                        bot_actions[bid] = [(ACT_WAIT, -1)] * remaining

                current_plan = bot_actions
                plan_rnd_offset = rnd

            # Execute one round from current plan
            plan_step = rnd - plan_rnd_offset
            round_actions = []
            for bid in range(num_bots):
                acts = current_plan.get(bid, [])
                if plan_step < len(acts):
                    round_actions.append(acts[plan_step])
                else:
                    round_actions.append((ACT_WAIT, -1))

            all_round_actions.append(round_actions)
            gs.round = rnd
            cpu_step(gs, round_actions, all_orders)

            if verbose and (rnd < 5 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
                elapsed = time.time() - t0
                print(f"  R{rnd:3d}: score={gs.score:3d}, orders={gs.orders_completed}, "
                      f"t={elapsed:.1f}s", file=sys.stderr)

    else:
        # === PER-ROUND REACTIVE MODE (original) ===
        all_round_actions = []

        base_searcher = GPUBeamSearcher(
            ms, all_orders, device=device, num_bots=num_bots,
            no_compile=True, speed_bonus=speed_bonus)

        for rnd in range(MAX_ROUNDS):
            remaining = MAX_ROUNDS - rnd
            rnd_horizon = min(horizon, remaining)

            if rnd_horizon <= 0:
                all_round_actions.append([(ACT_WAIT, -1)] * num_bots)
                gs.round = rnd
                cpu_step(gs, all_round_actions[-1], all_orders)
                continue

            priority_order = sorted(
                range(num_bots),
                key=lambda bid: bot_priority_key(gs, bid, tables))

            if attempt > 0 and rnd % (attempt + 2) == 0:
                if len(priority_order) >= 2:
                    priority_order[0], priority_order[1] = (
                        priority_order[1], priority_order[0])

            bot_acts_by_id = {}
            planned_bots = []
            locked_positions_cur = []
            locked_inventories_cur = []

            for bot_id in priority_order:
                bot_pos = (int(gs.bot_positions[bot_id, 0]),
                           int(gs.bot_positions[bot_id, 1]))

                locked_trajs = None
                if planned_bots:
                    locked_trajs = build_locked_trajs(planned_bots, rnd)

                init_state = build_gpu_init_state(
                    gs, bot_id, all_orders, device=device,
                    locked_positions=locked_positions_cur if locked_positions_cur else None,
                    locked_inventories=locked_inventories_cur if locked_inventories_cur else None)

                if locked_trajs:
                    searcher = GPUBeamSearcher(
                        ms, all_orders, device=device, num_bots=num_bots,
                        locked_trajectories=locked_trajs,
                        no_compile=True, speed_bonus=speed_bonus)
                else:
                    searcher = base_searcher

                score, acts = searcher.dp_search(
                    game_state=None,
                    max_states=max_states,
                    verbose=False,
                    start_rnd=rnd,
                    max_rounds=rnd_horizon,
                    init_state=init_state,
                    bot_id=bot_id)

                if locked_trajs:
                    del searcher

                full_acts = acts if acts else [(ACT_WAIT, -1)] * rnd_horizon
                positions = simulate_single_bot_trajectory(
                    bot_pos, full_acts, walkable, rnd_horizon)
                planned_bots.append((bot_id, full_acts, positions))
                locked_positions_cur.append(bot_pos)

                bot_inv = [int(gs.bot_inventories[bot_id, s]) for s in range(INV_CAP)]
                locked_inventories_cur.append(bot_inv)

                first_act = full_acts[0] if full_acts else (ACT_WAIT, -1)
                bot_acts_by_id[bot_id] = first_act

            round_actions = [bot_acts_by_id.get(bid, (ACT_WAIT, -1))
                             for bid in range(num_bots)]
            all_round_actions.append(round_actions)

            gs.round = rnd
            cpu_step(gs, round_actions, all_orders)

            if verbose and (rnd < 5 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
                elapsed = time.time() - t0
                print(f"  R{rnd:3d}: score={gs.score:3d}, orders={gs.orders_completed}, "
                      f"t={elapsed:.1f}s", file=sys.stderr)

            if rnd % 50 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

        del base_searcher

    torch.cuda.empty_cache()

    total_time = time.time() - t0
    if verbose:
        print(f"\n  Attempt {attempt}: score={gs.score}, orders={gs.orders_completed}, "
              f"time={total_time:.1f}s", file=sys.stderr)

    return gs.score, all_round_actions


def main():
    parser = argparse.ArgumentParser(
        description='Offline reactive solver: per-round GPU decisions against local sim')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--max-states', type=int, default=200_000,
                        help='GPU states per bot per round (default: 200K)')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Look-ahead rounds per decision (default: 100)')
    parser.add_argument('--speed-bonus', type=float, default=100.0,
                        help='Speed bonus for earlier completions (default: 100)')
    parser.add_argument('--replan-interval', type=int, default=30,
                        help='Re-plan every N rounds from actual state (default: 30, 0=per-round)')
    parser.add_argument('--attempts', type=int, default=1,
                        help='Number of attempts with varied priority order (default: 1)')
    parser.add_argument('--save', action='store_true',
                        help='Save best result to solution store if it improves')
    args = parser.parse_args()

    from solution_store import load_capture, load_meta, save_solution

    capture = load_capture(args.difficulty)
    if not capture:
        print(f"No capture data for {args.difficulty}. Run a capture first.",
              file=sys.stderr)
        sys.exit(1)

    n_orders = len(capture.get('orders', []))
    meta = load_meta(args.difficulty)
    prev_score = meta.get('score', 0) if meta else 0

    # Build order objects once
    _, all_orders = init_game_from_capture(capture, num_orders=n_orders)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  OFFLINE REACTIVE SOLVER: {args.difficulty.upper()}", file=sys.stderr)
    print(f"  Orders: {n_orders}, Current best: {prev_score}", file=sys.stderr)
    print(f"  States/bot/round: {args.max_states:,}", file=sys.stderr)
    print(f"  Horizon: {args.horizon} rounds", file=sys.stderr)
    print(f"  Replan interval: {args.replan_interval} rounds (0=per-round)", file=sys.stderr)
    print(f"  Attempts: {args.attempts}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    best_score = 0
    best_actions = None

    for attempt in range(args.attempts):
        print(f"\n--- Attempt {attempt + 1}/{args.attempts} ---", file=sys.stderr)

        score, actions = offline_reactive_solve(
            capture_data=capture,
            all_orders=all_orders,
            difficulty=args.difficulty,
            device='cuda',
            max_states=args.max_states,
            horizon=args.horizon,
            speed_bonus=args.speed_bonus,
            replan_interval=args.replan_interval,
            verbose=True,
            attempt=attempt)

        if score > best_score:
            best_score = score
            best_actions = actions
            print(f"  New best: {score}", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  RESULT: {args.difficulty} → {best_score} (prev: {prev_score})",
          file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if args.save and best_actions and best_score > 0:
        saved = save_solution(args.difficulty, best_score, best_actions)
        if saved:
            print(f"  Saved! (improved: {prev_score} → {best_score})", file=sys.stderr)
        else:
            print(f"  Not saved (existing {prev_score} >= {best_score})", file=sys.stderr)

    # Emit JSON for pipeline integration
    print(json.dumps({
        'type': 'offline_reactive_done',
        'difficulty': args.difficulty,
        'score': best_score,
        'prev_score': prev_score,
        'improved': best_score > prev_score,
    }))


if __name__ == '__main__':
    main()
