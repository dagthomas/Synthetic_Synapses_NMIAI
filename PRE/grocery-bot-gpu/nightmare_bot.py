"""PIBT-based multi-agent nightmare bot for Grocery Bot.

Centralized planner using:
- PIBT collision-free movement (<1ms for 20 agents)
- Hungarian assignment for optimal bot-to-item matching
- 4-tier role system with order pipelining
- Multi-drop-zone load balancing

Modes:
  Live discovery:  python nightmare_bot.py "wss://..." --save-capture cap.json
  Live replay:     python nightmare_bot.py "wss://..." --replay actions.json
  Local sim test:  python nightmare_bot.py --local 1772817802

Per-round pipeline (~2-3ms total):
  1. Parse WS JSON + update state
  2. Detect order transitions, reset stale assignments
  3. Task assignment via Hungarian + tier management
  4. Check immediate pickup/dropoff actions
  5. PIBT movement for remaining bots
  6. Format and send WS actions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import numpy as np

from game_engine import (
    MapState, Order, GameState,
    init_game_from_capture, step, build_map_from_capture, actions_to_ws_format,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import detect_difficulty, DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from pibt import pibt_step, compute_priorities, PRIO_DELIVERING, PRIO_CARRYING, PRIO_PICKING, PRIO_PREPICKING, PRIO_IDLE
from task_assigner import TaskAssigner, BotState, Goal
from live_solver import ws_to_capture

ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']


def _find_parking_spots(ms, tables, num_spots=40):
    """Find good parking spots: walkable cells away from drop zones and corridors.

    Prefers positions in aisles (vertical passages between shelves) at
    moderate distance from drop zones — not too close (blocking delivery)
    and not too far (slow to reach items).
    """
    drop_zones = getattr(ms, 'drop_off_zones', [ms.drop_off])
    mid_y = ms.height // 2
    spots = []

    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] != CELL_FLOOR:
                continue
            # Distance to nearest drop zone
            min_d = min(tables.get_distance((x, y), dz) for dz in drop_zones)
            if min_d < 4:
                continue
            # Penalize horizontal corridors (high traffic)
            corridor_penalty = 0
            if y == ms.height - 2 or y == 1:
                corridor_penalty = 10
            elif y == mid_y or y == mid_y - 1 or y == mid_y + 1:
                corridor_penalty = 5
            # Prefer moderate distance (not too far = slow response)
            dist_score = -abs(min_d - 15)  # peaks at dist=15
            score = dist_score - corridor_penalty
            spots.append((-score, x, y))

    spots.sort()
    return [(x, y) for _, x, y in spots[:num_spots]]


class NightmareBot:
    """Main nightmare bot with PIBT + Hungarian assignment."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ms: MapState | None = None
        self.tables: PrecomputedTables | None = None
        self.assigner: TaskAssigner | None = None
        self.bot_states: list[BotState] = []
        self.n_bots = 0
        self.round_num = 0
        self.capture_orders: list[dict] = []
        self._seen_order_ids: set[str] = set()
        self.parking_spots: list[tuple[int, int]] = []
        self._evac_until = -1  # spawn-evacuation active until this round
        self._carry_since: dict[int, int] = {}  # bot_id -> round first started carrying

    def init_from_ws(self, data: dict):
        """Initialize from round 0 WebSocket data."""
        capture = ws_to_capture(data)
        self.ms = build_map_from_capture(capture)
        self.tables = PrecomputedTables.get(self.ms)
        self.assigner = TaskAssigner(self.tables, self.ms)
        self.n_bots = len(data['bots'])
        self.bot_states = [BotState(bid) for bid in range(self.n_bots)]
        self.parking_spots = _find_parking_spots(self.ms, self.tables, self.n_bots * 2)

        # Record initial orders
        for order in data.get('orders', []):
            oid = order.get('id', '')
            if oid not in self._seen_order_ids:
                self._seen_order_ids.add(oid)
                self.capture_orders.append({
                    'items_required': order['items_required'],
                    'status': order.get('status', 'active'),
                })

        if self.verbose:
            diff = detect_difficulty(self.n_bots)
            print(f"NightmareBot init: {diff} {self.n_bots}bots "
                  f"{self.ms.width}x{self.ms.height} "
                  f"{self.ms.num_types}types {self.ms.num_items}items",
                  file=sys.stderr)

    def init_from_state(self, state: GameState, all_orders: list[Order]):
        """Initialize from local sim state."""
        self.ms = state.map_state
        self.tables = PrecomputedTables.get(self.ms)
        self.assigner = TaskAssigner(self.tables, self.ms)
        self.n_bots = len(state.bot_positions)
        self.bot_states = [BotState(bid) for bid in range(self.n_bots)]
        self.parking_spots = _find_parking_spots(self.ms, self.tables, self.n_bots * 2)

    def decide_ws(self, data: dict) -> list[dict]:
        """Decide actions from WebSocket game_state data. Returns WS-format actions."""
        rnd = data['round']
        self.round_num = rnd

        # Track new orders for capture
        for order in data.get('orders', []):
            oid = order.get('id', '')
            if oid not in self._seen_order_ids:
                self._seen_order_ids.add(oid)
                self.capture_orders.append({
                    'items_required': order['items_required'],
                    'status': order.get('status', 'active'),
                })

        # Build lightweight state from WS data
        state = self._ws_to_state(data)
        all_orders = self._state_orders(state)

        # Core decision
        actions = self._decide(state, all_orders)

        # Convert to WS format
        return actions_to_ws_format(actions, self.ms)

    def decide_local(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        """Decide actions for local sim. Returns internal action tuples."""
        self.round_num = rnd
        return self._decide(state, all_orders)

    def _decide(self, state: GameState, all_orders: list[Order]) -> list[tuple[int, int]]:
        """Core per-round decision pipeline."""
        ms = self.ms
        tables = self.tables
        n_bots = self.n_bots

        # Step 1: Task assignment
        goals, roles = self.assigner.assign(state, all_orders, self.bot_states)

        # Step 2: Check immediate pickup/dropoff actions
        actions = [(ACT_WAIT, -1)] * n_bots
        staying = set()  # bots that will stay in place this round (pickup/dropoff/wait)
        needs_movement = []  # bots that need movement resolution

        drop_zones = getattr(ms, 'drop_off_zones', [ms.drop_off])
        active = state.get_active_order()
        preview = state.get_preview_order()

        for bid in range(n_bots):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            pos = (bx, by)
            goal = goals[bid]
            inv = state.bot_inv_list(bid)

            # Opportunistic dropoff: any bot at a drop zone with active items
            at_drop = any(bx == dz[0] and by == dz[1] for dz in drop_zones)
            if at_drop and inv and active and any(active.needs_type(t) for t in inv):
                actions[bid] = (ACT_DROPOFF, -1)
                self.bot_states[bid].clear()
                staying.add(bid)
                continue

            if goal is not None and goal.type == Goal.PICK and goal.item_idx >= 0:
                # Check if adjacent to target item -> pickup
                ix = int(ms.item_positions[goal.item_idx, 0])
                iy = int(ms.item_positions[goal.item_idx, 1])
                if abs(bx - ix) + abs(by - iy) == 1:
                    actions[bid] = (ACT_PICKUP, goal.item_idx)
                    self.bot_states[bid].clear()
                    staying.add(bid)
                    continue
                # Also check opportunistic pickup of ANY needed adjacent item
                if state.bot_inv_count(bid) < INV_CAP and active:
                    pickup = self._check_adjacent_pickup(state, bid, active, preview)
                    if pickup is not None:
                        actions[bid] = pickup
                        self.bot_states[bid].clear()
                        staying.add(bid)
                        continue

            if goal is not None and goal.type == Goal.DELIVER:
                at_drop = any(bx == dz[0] and by == dz[1] for dz in drop_zones)
                if at_drop and inv:
                    if active and any(active.needs_type(t) for t in inv):
                        actions[bid] = (ACT_DROPOFF, -1)
                        self.bot_states[bid].clear()
                        staying.add(bid)
                        continue
                    # Non-active items at drop zone: clear goal, move away
                    self.bot_states[bid].clear()
                    goals[bid] = None
                    roles[bid] = PRIO_IDLE
                elif at_drop and not inv:
                    # Empty at drop zone - clear goal
                    self.bot_states[bid].clear()
                    goals[bid] = None
                    roles[bid] = PRIO_IDLE

            if goal is None:
                # Opportunistic pickup if adjacent to a needed item
                if state.bot_inv_count(bid) < INV_CAP and active:
                    pickup = self._check_adjacent_pickup(state, bid, active, preview)
                    if pickup is not None:
                        actions[bid] = pickup
                        staying.add(bid)
                        continue

            needs_movement.append(bid)

        # Step 3: Route idle bots — targeted evacuation near stuck deliverers
        # Track how long each bot has been carrying inventory
        for bid in range(n_bots):
            if state.bot_inv_count(bid) > 0:
                if bid not in self._carry_since:
                    self._carry_since[bid] = self.round_num
            else:
                self._carry_since.pop(bid, None)

        # Collect positions of truly stuck bots (carrying 60+ rounds or stuck>3)
        stuck_positions = []
        for bid, start in self._carry_since.items():
            if self.round_num - start > 60 and roles[bid] in (PRIO_DELIVERING, PRIO_CARRYING):
                stuck_positions.append(
                    (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])))
        for b in range(n_bots):
            if (self.bot_states[b].stuck_count > 3
                    and roles[b] in (PRIO_DELIVERING, PRIO_CARRYING, PRIO_PICKING)
                    and self.bot_states[b].last_pos):
                stuck_positions.append(self.bot_states[b].last_pos)

        idle_bids = [bid for bid in needs_movement if goals[bid] is None]
        spawn = ms.spawn
        parked_spots = set()
        for bid in idle_bids:
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            # Evacuate idle bots near stuck carriers to clear their path
            if stuck_positions and any(
                tables.get_distance(pos, sp) <= 5 for sp in stuck_positions
            ):
                if pos != spawn:
                    goals[bid] = Goal(Goal.MOVE, spawn)
                continue
            # Normal: park at distributed spots
            best_spot = None
            best_d = 9999
            for spot in self.parking_spots:
                if spot in parked_spots:
                    continue
                d = tables.get_distance(pos, spot)
                if d < best_d:
                    best_d = d
                    best_spot = spot
            if best_spot and pos != best_spot:
                goals[bid] = Goal(Goal.MOVE, best_spot)
                parked_spots.add(best_spot)
            elif best_spot:
                parked_spots.add(best_spot)

        # Step 4: Movement resolution for all bots that need to move
        positions = [(int(state.bot_positions[bid, 0]),
                      int(state.bot_positions[bid, 1]))
                     for bid in range(n_bots)]

        pibt_goals = [None] * n_bots
        for bid in needs_movement:
            goal = goals[bid]
            if goal and goal.target:
                pibt_goals[bid] = goal.target

        # All bots participate in priority resolution
        priorities = compute_priorities(
            [roles[bid] for bid in range(n_bots)],
            self.round_num,
            n_bots,
        )

        # Bots doing pickup/dropoff are "staying" - mark them at their
        # current position as highest priority so others route around
        for bid in staying:
            priorities[bid] = (-1, 0)  # highest possible priority
            pibt_goals[bid] = positions[bid]  # stay in place

        pibt_actions = pibt_step(positions, pibt_goals, priorities, tables, ms)

        # Only apply movement actions to bots that need movement
        for bid in needs_movement:
            actions[bid] = pibt_actions[bid]

        if self.verbose and (self.round_num < 20 or self.round_num % 25 == 0):
            score = state.score
            oc = state.orders_completed
            role_names = {PRIO_DELIVERING: 'D', PRIO_CARRYING: 'C',
                          PRIO_PICKING: 'P', PRIO_PREPICKING: 'V', PRIO_IDLE: 'I'}
            role_str = ''.join(role_names.get(roles[b], '?') for b in range(n_bots))
            act_names = ['W', 'U', 'D', 'L', 'R', 'PK', 'DO']
            # Show bots with non-idle roles first, then first few idle
            active_bots = [b for b in range(n_bots) if roles[b] != PRIO_IDLE]
            bot_details = []
            for b in active_bots[:12]:
                bx = int(state.bot_positions[b, 0])
                by = int(state.bot_positions[b, 1])
                a = act_names[actions[b][0]] if actions[b][0] < len(act_names) else '?'
                g = goals[b].target if goals[b] and goals[b].target else None
                inv = state.bot_inv_list(b)
                inv_str = f'[{",".join(str(t) for t in inv)}]' if inv else ''
                bot_details.append(f'B{b}@({bx},{by}){a}{inv_str}→{g}')
            # Show stuck count
            stuck = sum(1 for bs in self.bot_states if bs.stuck_count > 5)
            print(f"  R{self.round_num:3d} score={score:3d} orders={oc:2d} roles={role_str} stuck={stuck} "
                  f"| {' '.join(bot_details)}",
                  file=sys.stderr)

        return actions

    def _check_adjacent_pickup(
        self, state: GameState, bid: int, active: Order, preview: Order | None,
    ) -> tuple[int, int] | None:
        """Check if bot is adjacent to any active-order item. Returns (ACT_PICKUP, item_idx) or None.

        Only picks items matching the ACTIVE order to prevent inventory deadlocks
        (preview items can't be dropped off until the order transitions).
        """
        ms = self.ms
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                tid = int(ms.item_types[item_idx])
                if active.needs_type(tid):
                    return (ACT_PICKUP, item_idx)
        return None

    def _ws_to_state(self, data: dict) -> GameState:
        """Build GameState from WebSocket data for assignment decisions."""
        ms = self.ms
        state = GameState(ms)
        bots = data['bots']
        n_bots = len(bots)

        state.bot_positions = np.zeros((n_bots, 2), dtype=np.int16)
        state.bot_inventories = np.full((n_bots, INV_CAP), -1, dtype=np.int8)

        for bot in bots:
            bid = bot['id']
            state.bot_positions[bid] = bot['position']
            for j, item_name in enumerate(bot.get('inventory', [])):
                if j < INV_CAP:
                    name_lower = item_name.lower()
                    tid = ms.type_name_to_id.get(name_lower,
                           ms.type_name_to_id.get(item_name, -1))
                    if tid >= 0:
                        state.bot_inventories[bid, j] = tid

        state.score = data.get('score', 0)
        state.round = data.get('round', 0)
        state.orders_completed = data.get('orders_completed', 0)

        # Parse orders
        state.orders = []
        for i, order_data in enumerate(data.get('orders', [])):
            req_names = order_data['items_required']
            req_ids = []
            for name in req_names:
                name_lower = name.lower()
                tid = ms.type_name_to_id.get(name_lower,
                       ms.type_name_to_id.get(name, 0))
                req_ids.append(tid)

            status = order_data.get('status', 'active' if i == 0 else 'preview')
            # Use server-provided order ID so TaskAssigner detects transitions
            server_id = order_data.get('id', i)
            if isinstance(server_id, str):
                server_id = hash(server_id) & 0x7FFFFFFF
            order = Order(server_id, req_ids, status)
            order._required_names = req_names

            # Mark delivered items
            for del_name in order_data.get('items_delivered', []):
                del_lower = del_name.lower()
                del_tid = ms.type_name_to_id.get(del_lower,
                           ms.type_name_to_id.get(del_name, -1))
                if del_tid >= 0:
                    order.deliver_type(del_tid)

            state.orders.append(order)

        state.active_idx = 0
        return state

    def _state_orders(self, state: GameState) -> list[Order]:
        """Build all_orders list from state orders (for assignment compatibility)."""
        return list(state.orders)

    def get_capture_data(self, data: dict) -> dict:
        """Build capture_data dict from accumulated orders."""
        capture = ws_to_capture(data)
        capture['orders'] = self.capture_orders
        return capture


# ── Local sim mode ──────────────────────────────────────────────────────────

def run_local(seed: int, verbose: bool = True) -> int:
    """Test against local game simulator."""
    from game_engine import init_game, simulate_game

    difficulty = 'nightmare'
    num_rounds = DIFF_ROUNDS.get(difficulty, 500)

    print(f"NightmareBot local sim: seed={seed} difficulty={difficulty} "
          f"rounds={num_rounds}", file=sys.stderr)

    state, all_orders = init_game(seed, difficulty, num_orders=100)
    bot = NightmareBot(verbose=verbose)
    bot.init_from_state(state, all_orders)

    action_log = []
    last_order_count = 0
    order_start_round = 0
    order_times = []

    for rnd in range(num_rounds):
        state.round = rnd
        actions = bot.decide_local(state, all_orders, rnd)
        action_log.append(actions)
        step(state, actions, all_orders)

        # Track per-order completion timing
        if state.orders_completed > last_order_count:
            elapsed = rnd - order_start_round
            order_times.append(elapsed)
            if verbose:
                print(f"  ORDER {last_order_count} done at R{rnd} ({elapsed} rounds)",
                      file=sys.stderr)
            last_order_count = state.orders_completed
            order_start_round = rnd

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == num_rounds - 1):
            active = state.get_active_order()
            needs = active.needs() if active else []
            print(f"  R{rnd:3d}: score={state.score:3d} orders={state.orders_completed:2d} "
                  f"active_needs={len(needs)}", file=sys.stderr)

    print(f"\nFinal score: {state.score} (orders completed: {state.orders_completed})",
          file=sys.stderr)
    if order_times:
        avg = sum(order_times) / len(order_times)
        print(f"Order timing: {order_times}", file=sys.stderr)
        print(f"Avg rounds/order: {avg:.1f}, min={min(order_times)}, max={max(order_times)}",
              file=sys.stderr)
    return state.score


# ── WebSocket live mode ──────────────────────────────────────────────────────

async def run_live(ws_url: str, save_capture: str | None = None,
                   replay_file: str | None = None, verbose: bool = False) -> int:
    """Connect to game server and play live."""
    import websockets

    bot = NightmareBot(verbose=verbose)
    final_score = 0
    round0_data = None

    # Replay mode: load pre-computed actions
    replay_actions = None
    if replay_file and os.path.exists(replay_file):
        with open(replay_file) as f:
            replay_data = json.load(f)
        if isinstance(replay_data, list):
            replay_actions = replay_data
        elif isinstance(replay_data, dict) and 'actions' in replay_data:
            replay_actions = replay_data['actions']
        print(f"Loaded replay: {len(replay_actions)} rounds", file=sys.stderr)

    WS_CONNECT_TIMEOUT = 15
    WS_RECV_TIMEOUT = 10
    WS_SEND_TIMEOUT = 5

    # Game log for live page polling
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, f'game_log_{int(time.time())}.jsonl')
    log_file = open(log_path, 'w')

    print(f"Connecting to {ws_url}", file=sys.stderr)

    try:
        ws = await asyncio.wait_for(
            websockets.connect(ws_url),
            timeout=WS_CONNECT_TIMEOUT
        )
        print("WebSocket connected", file=sys.stderr)
    except (asyncio.TimeoutError, Exception) as e:
        print(f"ERROR: Connection failed: {e}", file=sys.stderr)
        return 0

    try:
        async with ws:
            game_over = False
            expected_rnd = 0

            while not game_over:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    print("WARNING: No message received", file=sys.stderr)
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"WebSocket closed: {e}", file=sys.stderr)
                    break

                data = json.loads(message)

                log_file.write(json.dumps(data) + '\n')
                log_file.flush()

                if data['type'] == 'game_over':
                    final_score = data.get('score', 0)
                    print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                    break

                if data['type'] != 'game_state':
                    continue

                # Drain stale messages
                while True:
                    try:
                        peek = await asyncio.wait_for(ws.recv(), timeout=0.002)
                        peek_data = json.loads(peek)
                        if peek_data.get('type') == 'game_over':
                            final_score = peek_data.get('score', 0)
                            print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                            game_over = True
                            break
                        if peek_data.get('type') == 'game_state':
                            data = peek_data
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        break
                if game_over:
                    break

                rnd = data['round']
                score = data.get('score', 0)
                max_rounds = data.get('max_rounds', 500)

                if rnd == 0:
                    bot.init_from_ws(data)
                    round0_data = data

                # Detect round gaps
                if rnd > expected_rnd and expected_rnd > 0:
                    gap = rnd - expected_rnd
                    print(f"R{rnd}: ROUND GAP {gap}", file=sys.stderr)
                expected_rnd = rnd + 1

                # Decide actions
                t0 = time.time()

                if replay_actions and rnd < len(replay_actions):
                    # Replay mode: use pre-computed actions with desync recovery
                    ws_actions = _replay_with_recovery(
                        data, replay_actions[rnd], bot, rnd)
                else:
                    # Live mode: PIBT + Hungarian
                    ws_actions = bot.decide_ws(data)

                dt = (time.time() - t0) * 1000

                if rnd < 5 or rnd % 25 == 0 or rnd >= max_rounds - 3:
                    print(f"R{rnd}/{max_rounds} Score:{score} dt={dt:.1f}ms",
                          file=sys.stderr)

                response = {'actions': ws_actions}
                log_file.write(json.dumps(response) + '\n')
                log_file.flush()
                try:
                    await asyncio.wait_for(
                        ws.send(json.dumps(response)),
                        timeout=WS_SEND_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    print(f"WARNING: Send timed out R{rnd}", file=sys.stderr)
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"WebSocket closed during send: {e}", file=sys.stderr)
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    log_file.close()
    print(f"Log saved: {log_path}", file=sys.stderr)

    # Save capture
    if save_capture and round0_data:
        capture = bot.get_capture_data(round0_data)
        capture['probe_score'] = final_score
        capture['captured_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(save_capture, 'w') as f:
            json.dump(capture, f, indent=2)
        print(f"Capture saved: {save_capture} ({len(bot.capture_orders)} orders)",
              file=sys.stderr)

    return final_score


def _replay_with_recovery(
    data: dict,
    planned_actions: list,
    bot: NightmareBot,
    rnd: int,
) -> list[dict]:
    """Replay pre-computed actions with PIBT-based desync recovery.

    If bot positions match expected, replay as-is.
    If desynced, fall back to live PIBT decisions.
    """
    bots = data['bots']
    ms = bot.ms

    # planned_actions can be either WS format or internal tuples
    if planned_actions and isinstance(planned_actions[0], dict):
        # Already WS format
        return planned_actions

    # Internal tuple format: [(act_type, item_idx), ...]
    # Check for desync: just use live decision for safety
    # (Full desync detection would need expected positions from simulation)
    return bot.decide_ws(data)


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PIBT-based nightmare bot for Grocery Bot')
    parser.add_argument('url', nargs='?', default=None,
                        help='WebSocket URL (wss://game.ainm.no/ws?token=...)')
    parser.add_argument('--local', type=int, default=None, metavar='SEED',
                        help='Test against local sim with given seed')
    parser.add_argument('--save-capture', type=str, default=None, metavar='FILE',
                        help='Save captured orders to file')
    parser.add_argument('--replay', type=str, default=None, metavar='FILE',
                        help='Replay pre-computed actions from file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Per-round status logging')
    args = parser.parse_args()

    if args.local is not None:
        score = run_local(args.local, verbose=True)
        print(f"Score: {score}")
        sys.exit(0 if score > 0 else 1)

    if args.url:
        score = asyncio.run(run_live(
            args.url,
            save_capture=args.save_capture,
            replay_file=args.replay,
            verbose=args.verbose,
        ))
        print(f"Score: {score}")
        sys.exit(0 if score > 0 else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
