"""Live game solver for SvelteKit dashboard integration.

Two modes:
1. Normal (default): Precompute all 300 rounds at round 0 using MAPF planner.
   Background optimizer improves actions during the game.
2. Reactive (--save-capture): Read actual orders each round, make greedy decisions.
   Captures all orders for the Learn & Replay workflow.

Usage:
    python live_solver.py <wss://game.ainm.no/ws?token=...>
    python live_solver.py <wss://...> --save-capture  # reactive + capture
"""
import asyncio
import json
import sys
import os
import time
import threading
from collections import deque

from game_engine import (
    MapState, Order, GameState, init_game_from_capture,
    step, build_map_from_capture, actions_to_ws_format,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
)
from planner import solve as planner_solve
from configs import detect_difficulty  # canonical source; re-exported for backward compat


def ws_to_capture(data):
    """Convert round 0 WebSocket data to capture_data format for init_game_from_capture."""
    capture = {
        'grid': data['grid'],
        'items': data['items'],
        'drop_off': data['drop_off'],
        'num_bots': len(data['bots']),
        'difficulty': detect_difficulty(len(data['bots'])),
        'orders': [],
    }

    for order in data.get('orders', []):
        capture['orders'].append({
            'items_required': order['items_required'],
            'items_delivered': order.get('items_delivered', []),
            'status': order['status'],
        })

    return capture


class ReactiveSolver:
    """Reactive per-round solver that reads actual orders from server.

    Uses BFS pathfinding + greedy assignment with:
    - Self-correcting navigation: detects failed moves, marks cells as walls
    - Anti-oscillation: detects position patterns, forces escape moves
    - Batched pickup: picks multiple items before delivering
    """

    def __init__(self, round0_data):
        self.capture = ws_to_capture(round0_data)
        self.difficulty = self.capture['difficulty']
        self.num_bots = self.capture['num_bots']
        self.ms = build_map_from_capture(self.capture)

        # Build walkable set from map_state
        self.walkable = set()
        ms = self.ms
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    self.walkable.add((x, y))

        # Build item lookup: type -> list of item info dicts
        # Use lowercase keys for case-insensitive matching
        self.items_by_type = {}
        for idx, item in enumerate(ms.items):
            t = item['type'].lower()
            if t not in self.items_by_type:
                self.items_by_type[t] = []
            self.items_by_type[t].append({
                'idx': idx,
                'id': item['id'],
                'pos': item['position'],
                'adj': list(ms.item_adjacencies.get(idx, [])),
            })

        self.drop_off = ms.drop_off

        # Item ID -> index mapping (for recording internal actions)
        self._id_to_idx = {item['id']: idx for idx, item in enumerate(ms.items)}

        # Self-correcting navigation state
        self.last_positions = {}    # bid -> (x, y)
        self.last_sent_actions = {} # bid -> action name string
        self.extra_walls = set()    # cells discovered to be walls at runtime
        self.wall_fail_count = {}   # (x,y) -> consecutive fail count
        self.visited_cells = set()  # cells the bot has actually visited (never mark as wall)
        self.visited_cells.add(ms.drop_off)  # dropoff is always safe

        # Anti-oscillation state
        self.bot_history = {}       # bid -> deque(maxlen=12) of positions

        # Persistent target tracking: prevents target-switching oscillation
        self.bot_goals = {}         # bid -> {'item_id': str, 'type': str, 'adj': (x,y)} or None

        print(f"  Reactive solver: {self.difficulty} {self.num_bots}bots "
              f"walkable={len(self.walkable)} items={len(ms.items)}", file=sys.stderr)

    def _detect_failed_moves(self, bots):
        """Detect moves that failed (bot didn't move) and mark target cells as walls.

        Conservative: requires 2+ consecutive fails, never marks visited cells or dropoff.
        """
        bot_pos_set = {(b['position'][0], b['position'][1]) for b in bots}
        move_deltas = {
            'move_right': (1, 0), 'move_left': (-1, 0),
            'move_up': (0, -1), 'move_down': (0, 1),
        }
        for bot in bots:
            bid = bot['id']
            bx, by = bot['position']
            # Track visited cells (these are definitely walkable)
            self.visited_cells.add((bx, by))

            if bid in self.last_positions and bid in self.last_sent_actions:
                prev = self.last_positions[bid]
                act_name = self.last_sent_actions[bid]
                if prev == (bx, by) and act_name in move_deltas:
                    dx, dy = move_deltas[act_name]
                    failed = (bx + dx, by + dy)
                    # Never mark visited cells, dropoff, or spawn as walls
                    if failed in self.visited_cells:
                        continue
                    if failed == self.drop_off:
                        continue
                    # Only mark as wall if no other bot was blocking
                    if failed not in bot_pos_set and failed in self.walkable:
                        # Require 2+ consecutive fails before marking
                        self.wall_fail_count[failed] = self.wall_fail_count.get(failed, 0) + 1
                        if self.wall_fail_count[failed] >= 2:
                            self.walkable.discard(failed)
                            self.extra_walls.add(failed)
                            print(f"  WALL CONFIRMED: {failed} ({act_name} from {prev} failed 2x)", file=sys.stderr)
                else:
                    # Move succeeded or no move — clear fail counts for the target
                    if act_name in move_deltas:
                        dx, dy = move_deltas[act_name]
                        target = (prev[0] + dx, prev[1] + dy)
                        self.wall_fail_count.pop(target, None)

    def _is_oscillating(self, bid):
        """Check if bot is visiting the same 2-3 positions repeatedly."""
        hist = self.bot_history.get(bid)
        if not hist or len(hist) < 6:
            return False
        last6 = list(hist)[-6:]
        return len(set(last6)) <= 3

    def _escape_move(self, bid, pos, occupied):
        """Find a move to break out of oscillation — prefer unvisited directions."""
        recent = set(list(self.bot_history.get(bid, []))[-4:])
        dirs = [(ACT_MOVE_RIGHT, (1, 0)), (ACT_MOVE_DOWN, (0, 1)),
                (ACT_MOVE_LEFT, (-1, 0)), (ACT_MOVE_UP, (0, -1))]
        # First pass: unvisited walkable cells
        for act, (dx, dy) in dirs:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in self.walkable and (nx, ny) not in occupied and (nx, ny) not in recent:
                return act
        # Second pass: any walkable cell
        for act, (dx, dy) in dirs:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in self.walkable and (nx, ny) not in occupied:
                return act
        return ACT_WAIT

    def _bfs_dist(self, start, goal):
        """BFS shortest path distance between two walkable cells."""
        if start == goal:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            (x, y), d = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == goal:
                    return d + 1
                if (nx, ny) in self.walkable and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), d + 1))
        return 9999

    def _bfs_next_step(self, start, goal, occupied):
        """BFS to find the first move toward goal, avoiding occupied cells."""
        if start == goal:
            return ACT_WAIT
        visited = {start}
        queue = deque([(start, None)])
        while queue:
            (x, y), first_act = queue.popleft()
            for act, (dx, dy) in [(ACT_MOVE_UP, (0, -1)), (ACT_MOVE_DOWN, (0, 1)),
                                   (ACT_MOVE_LEFT, (-1, 0)), (ACT_MOVE_RIGHT, (1, 0))]:
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if npos not in self.walkable or npos in visited:
                    continue
                if npos in occupied and npos != goal:
                    continue
                fa = first_act if first_act is not None else act
                if npos == goal:
                    return fa
                visited.add(npos)
                queue.append((npos, fa))
        return ACT_WAIT

    def _save_round_state(self, bots, actions):
        """Save positions and actions for next round's failed-move detection."""
        for bot in bots:
            self.last_positions[bot['id']] = (bot['position'][0], bot['position'][1])
        for act_data in actions:
            self.last_sent_actions[act_data['bot']] = act_data['action']

    def decide(self, data):
        """Decide actions for all bots based on current game state."""
        action_names = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']
        bots = data['bots']
        orders = data.get('orders', [])

        # Self-correcting: detect failed moves from last round
        self._detect_failed_moves(bots)

        # Update position history for oscillation detection
        for bot in bots:
            bid = bot['id']
            hist = self.bot_history.setdefault(bid, deque(maxlen=12))
            hist.append((bot['position'][0], bot['position'][1]))

        active_order = next((o for o in orders if o['status'] == 'active'), None)
        preview_order = next((o for o in orders if o['status'] == 'preview'), None)

        if not active_order:
            result = [{'bot': b['id'], 'action': 'wait'} for b in bots]
            self._save_round_state(bots, result)
            return result

        # What items are still needed for active order?
        needed = list(active_order['items_required'])
        for d in active_order.get('items_delivered', []):
            if d in needed:
                needed.remove(d)

        # Count what's already carried by all bots toward this order
        carried_toward_active = {}
        for bot in bots:
            for item_type in bot.get('inventory', []):
                t = item_type.lower()
                carried_toward_active[t] = carried_toward_active.get(t, 0) + 1

        # Items still needed that no bot is carrying yet
        still_need = list(needed)
        for item_type, count in carried_toward_active.items():
            for _ in range(count):
                if item_type in still_need:
                    still_need.remove(item_type)

        # Preview items: only pick if active order is fully covered by carried items
        preview_need = []
        if preview_order and not still_need:
            preview_need = [t.lower() for t in preview_order['items_required']]

        # Lowercase needed for matching
        needed_lower = [t.lower() for t in needed]
        still_need_lower = [t.lower() for t in still_need]

        bot_positions = {(b['position'][0], b['position'][1]) for b in bots}
        actions = []
        claimed_items = set()

        for bot in bots:
            bx, by = bot['position']
            inv = [t.lower() for t in bot.get('inventory', [])]
            bid = bot['id']

            # 1. At dropoff → drop off or evacuate (never idle-block the dropoff)
            if (bx, by) == self.drop_off:
                if inv:
                    has_match = any(t in needed_lower for t in inv)
                    if has_match:
                        actions.append({'bot': bid, 'action': 'drop_off'})
                        for t in inv:
                            if t in needed_lower:
                                needed_lower.remove(t)
                        continue
                    elif self.num_bots > 1:
                        # EVACUATE: dead inventory at dropoff blocks other bots
                        occupied = bot_positions - {(bx, by)}
                        act = self._escape_move(bid, (bx, by), occupied)
                        actions.append({'bot': bid, 'action': action_names[act]})
                        continue
                elif self.num_bots > 1:
                    # EVACUATE: empty bot idling at dropoff blocks deliveries
                    occupied = bot_positions - {(bx, by)}
                    act = self._escape_move(bid, (bx, by), occupied)
                    actions.append({'bot': bid, 'action': action_names[act]})
                    continue

            # 2. Adjacent to a NEEDED item → pick up
            picked = False
            if len(inv) < INV_CAP:
                # Multi-bot: only pick preview items if active is FULLY covered
                # and ALL items in bot's inventory are needed (no dead items) to prevent
                # picking 2 preview items + 1 active → deadlock after delivery
                use_preview = (not still_need_lower and preview_need and
                               (self.num_bots == 1 or
                                (any(t in needed_lower for t in inv) and
                                 all(t in needed_lower for t in inv))))
                pick_targets = still_need_lower if still_need_lower else (preview_need if use_preview else [])
                for item_type in pick_targets:
                    if item_type not in self.items_by_type:
                        continue
                    for item_info in self.items_by_type[item_type]:
                        iid = item_info['id']
                        if iid in claimed_items:
                            continue
                        ix, iy = item_info['pos']
                        if abs(ix - bx) + abs(iy - by) == 1:
                            actions.append({'bot': bid, 'action': 'pick_up', 'item_id': iid})
                            claimed_items.add(iid)
                            if item_type in still_need_lower:
                                still_need_lower.remove(item_type)
                            picked = True
                            break
                    if picked:
                        break

            if picked:
                self.bot_goals.pop(bid, None)  # Clear goal after pickup
                continue

            # 3. Deliver: go to dropoff when inventory full, or all needed items are covered
            has_active_items = any(t in needed_lower for t in inv)
            should_deliver = False
            if not inv:
                should_deliver = False
            elif has_active_items:
                should_deliver = True  # Has items to deliver
            elif len(inv) >= INV_CAP and self.num_bots == 1:
                should_deliver = True  # Single bot: go deliver even if not all match

            if should_deliver:
                occupied = bot_positions - {(bx, by)}
                act = self._bfs_next_step((bx, by), self.drop_off, occupied)
                actions.append({'bot': bid, 'action': action_names[act]})
                continue

            # 4. Navigate to closest needed item (batched: pick multiple before delivering)
            nav_targets = still_need_lower if still_need_lower else (preview_need if use_preview else [])
            if len(inv) < INV_CAP and nav_targets:
                # Check persistent goal first: keep pursuing same item if still valid
                goal = self.bot_goals.get(bid)
                use_existing_goal = False
                if goal and goal['type'] in nav_targets and goal['item_id'] not in claimed_items:
                    adj = goal['adj']
                    if adj in self.walkable:
                        use_existing_goal = True
                        best_item_id = goal['item_id']
                        best_target = adj
                        best_dist = self._bfs_dist((bx, by), adj)

                if not use_existing_goal:
                    best_dist = 9999
                    best_target = None
                    best_item = None
                    for item_type in nav_targets:
                        if item_type not in self.items_by_type:
                            continue
                        for item_info in self.items_by_type[item_type]:
                            if item_info['id'] in claimed_items:
                                continue
                            for adj in item_info['adj']:
                                if adj not in self.walkable:
                                    continue
                                d = self._bfs_dist((bx, by), adj)
                                if d < best_dist:
                                    best_dist = d
                                    best_item = item_info
                                    best_target = adj

                    if best_item:
                        self.bot_goals[bid] = {
                            'item_id': best_item['id'],
                            'type': next(t for t in nav_targets if t in self.items_by_type
                                         and any(ii['id'] == best_item['id'] for ii in self.items_by_type[t])),
                            'adj': best_target,
                        }
                        best_item_id = best_item['id']
                    else:
                        best_item_id = None

                if best_target and best_dist < 9999:
                    if best_item_id:
                        claimed_items.add(best_item_id)
                    occupied = bot_positions - {(bx, by)}
                    act = self._bfs_next_step((bx, by), best_target, occupied)
                    # Anti-oscillation: if we'd oscillate, escape instead
                    if self._is_oscillating(bid):
                        esc = self._escape_move(bid, (bx, by), occupied)
                        if esc != ACT_WAIT:
                            print(f"  Bot{bid} ESCAPE at ({bx},{by})", file=sys.stderr)
                            act = esc
                            self.bot_goals.pop(bid, None)  # Clear goal on escape
                    actions.append({'bot': bid, 'action': action_names[act]})
                    continue

            # 5. Have items → go to dropoff (for single bot or if items might match)
            if inv:
                if self.num_bots == 1 or has_active_items:
                    occupied = bot_positions - {(bx, by)}
                    act = self._bfs_next_step((bx, by), self.drop_off, occupied)
                    actions.append({'bot': bid, 'action': action_names[act]})
                    continue
                else:
                    # Multi-bot with dead inventory: wait near dropoff (don't camp ON it)
                    drop_dist = self._bfs_dist((bx, by), self.drop_off)
                    if drop_dist <= 2:
                        actions.append({'bot': bid, 'action': 'wait'})
                    else:
                        occupied = bot_positions - {(bx, by)}
                        act = self._bfs_next_step((bx, by), self.drop_off, occupied)
                        actions.append({'bot': bid, 'action': action_names[act]})
                    continue

            # 6. Wait (nothing to do — don't wander toward dropoff)
            actions.append({'bot': bid, 'action': 'wait'})

        self._save_round_state(bots, actions)
        return actions


class RouteSolver:
    """TSP-guided solver: route table distance matrix + optimal trip planning.

    Uses precomputed all-pairs distances for O(1) pathfinding. Plans optimal
    pickup sequences via TSP for each order. Packs preview items alongside
    active items for multi-trip orders to chain auto-deliveries.
    """

    def __init__(self, round0_data, route_table):
        self.capture = ws_to_capture(round0_data)
        self.difficulty = self.capture['difficulty']
        self.num_bots = self.capture['num_bots']
        self.rt = route_table

        self.ms = build_map_from_capture(self.capture)
        self.walkable = set()
        for y in range(self.ms.height):
            for x in range(self.ms.width):
                if self.ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    self.walkable.add((x, y))

        self.drop_off = tuple(self.ms.drop_off)
        self.drop_off_idx = self.rt.pos_to_idx.get(self.drop_off)

        # Build item lookup: type -> list of {id, pos, adj_cells}
        self.items_by_type = {}
        for idx, item in enumerate(self.ms.items):
            t = item['type'].lower()
            if t not in self.items_by_type:
                self.items_by_type[t] = []
            adj_list = list(self.ms.item_adjacencies.get(idx, []))
            self.items_by_type[t].append({
                'idx': idx,
                'id': item['id'],
                'pos': item['position'],
                'adj': adj_list,
                # Precompute best adj cell (closest to dropoff) for quick lookup
                'best_adj': min(adj_list, key=lambda a: self._dist_raw(a, self.drop_off))
                            if adj_list else None,
            })

        # Self-correcting navigation
        self.last_positions = {}
        self.last_sent_actions = {}
        self.wall_fail_count = {}
        self.visited_cells = set()
        self.visited_cells.add(self.drop_off)

        # Anti-oscillation
        self.bot_history = {}

        # Per-bot trip plan: persistent across rounds
        # plan = {'targets': [(item_id, adj_pos), ...], 'tidx': int, 'order_id': str}
        self.bot_plan = {}

        # Item ID → internal index for action recording
        self._id_to_idx = {item['id']: i for i, item in enumerate(self.ms.items)}

        print(f"  Route solver: {self.difficulty} {self.num_bots}bots "
              f"routes={len(route_table.routes)} cells={len(self.walkable)}", file=sys.stderr)

    def _dist_raw(self, a, b):
        """O(1) distance, handles missing positions."""
        ai = self.rt.pos_to_idx.get(a if isinstance(a, tuple) else tuple(a))
        bi = self.rt.pos_to_idx.get(b if isinstance(b, tuple) else tuple(b))
        if ai is None or bi is None:
            return 9999
        return int(self.rt.dist[ai, bi])

    def _dist(self, a, b):
        """O(1) distance between two positions."""
        return self._dist_raw(a, b)

    def _next_step(self, from_pos, to_pos, occupied):
        """Best move toward target using distance matrix, avoiding occupied."""
        if from_pos == to_pos:
            return 'wait'
        fi = self.rt.pos_to_idx.get(from_pos)
        ti = self.rt.pos_to_idx.get(to_pos)
        if fi is None or ti is None:
            return 'wait'
        best_act = 'wait'
        best_dist = int(self.rt.dist[fi, ti])
        for act, (dx, dy) in [('move_up', (0, -1)), ('move_down', (0, 1)),
                               ('move_left', (-1, 0)), ('move_right', (1, 0))]:
            nx, ny = from_pos[0] + dx, from_pos[1] + dy
            npos = (nx, ny)
            if npos not in self.walkable:
                continue
            if npos in occupied and npos != to_pos:
                continue
            ni = self.rt.pos_to_idx.get(npos)
            if ni is not None:
                d = int(self.rt.dist[ni, ti])
                if d < best_dist:
                    best_dist = d
                    best_act = act
        return best_act

    def _plan_trip(self, bot_pos, needed_types, claimed=None):
        """TSP-optimal pickup plan: which items to pick and in what order.

        Considers item assignment (which shelf item per type) and visit order
        (TSP) to minimize: bot→item1→item2→...→dropoff total distance.

        Returns [(item_id, adj_pos), ...] in optimal order, or None.
        """
        from itertools import product as iproduct, permutations
        if not needed_types:
            return None
        claimed = claimed or set()
        bot_idx = self.rt.pos_to_idx.get(bot_pos)
        if bot_idx is None or self.drop_off_idx is None:
            return None

        # Build candidate items per slot (each needed_type is one slot)
        K = 4  # top K candidates per type
        slots = []
        for t in needed_types:
            t_lower = t.lower()
            if t_lower not in self.items_by_type:
                return None
            cands = []
            for item_info in self.items_by_type[t_lower]:
                if item_info['id'] in claimed:
                    continue
                # Find best adj cell for this item from bot's perspective
                best_adj = None
                best_d = 9999
                for adj in item_info['adj']:
                    if adj not in self.walkable:
                        continue
                    adj_idx = self.rt.pos_to_idx.get(adj)
                    if adj_idx is None:
                        continue
                    # Score: distance from bot + distance to dropoff (total trip contribution)
                    d = int(self.rt.dist[bot_idx, adj_idx]) + int(self.rt.dist[adj_idx, self.drop_off_idx])
                    if d < best_d:
                        best_d = d
                        best_adj = adj
                if best_adj:
                    cands.append((item_info['id'], best_adj, best_d))
            cands.sort(key=lambda x: x[2])
            cands = cands[:K]
            if not cands:
                return None
            slots.append(cands)

        # Enumerate item selections × TSP orderings
        best_cost = 99999
        best_plan = None
        for selection in iproduct(*slots):
            adj_idxs = [self.rt.pos_to_idx[s[1]] for s in selection]
            n = len(selection)
            for perm in permutations(range(n)):
                cost = int(self.rt.dist[bot_idx, adj_idxs[perm[0]]])
                for i in range(n - 1):
                    cost += int(self.rt.dist[adj_idxs[perm[i]], adj_idxs[perm[i+1]]])
                cost += int(self.rt.dist[adj_idxs[perm[-1]], self.drop_off_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_plan = [(selection[p][0], selection[p][1]) for p in perm]

        return best_plan

    def _detect_failed_moves(self, bots):
        """Detect failed moves and mark cells as walls (conservative: 2+ fails, never dropoff)."""
        bot_pos_set = {(b['position'][0], b['position'][1]) for b in bots}
        move_deltas = {
            'move_right': (1, 0), 'move_left': (-1, 0),
            'move_up': (0, -1), 'move_down': (0, 1),
        }
        for bot in bots:
            bid = bot['id']
            bx, by = bot['position']
            self.visited_cells.add((bx, by))
            if bid in self.last_positions and bid in self.last_sent_actions:
                prev = self.last_positions[bid]
                act_name = self.last_sent_actions[bid]
                if prev == (bx, by) and act_name in move_deltas:
                    dx, dy = move_deltas[act_name]
                    failed = (bx + dx, by + dy)
                    # Never mark visited cells, dropoff, or spawn as walls
                    if failed in self.visited_cells or failed == self.drop_off:
                        continue
                    if failed not in bot_pos_set and failed in self.walkable:
                        self.wall_fail_count[failed] = self.wall_fail_count.get(failed, 0) + 1
                        if self.wall_fail_count[failed] >= 2:
                            self.walkable.discard(failed)
                            if failed in self.rt.pos_to_idx:
                                fidx = self.rt.pos_to_idx[failed]
                                self.rt.dist[fidx, :] = 9999
                                self.rt.dist[:, fidx] = 9999
                            print(f"  WALL CONFIRMED: {failed}", file=sys.stderr)
                else:
                    if act_name in move_deltas:
                        dx, dy = move_deltas[act_name]
                        target = (prev[0] + dx, prev[1] + dy)
                        self.wall_fail_count.pop(target, None)

    def _is_oscillating(self, bid):
        hist = self.bot_history.get(bid)
        if not hist or len(hist) < 6:
            return False
        return len(set(list(hist)[-6:])) <= 3

    def _escape_move(self, bid, pos, occupied):
        recent = set(list(self.bot_history.get(bid, []))[-4:])
        dirs = [('move_right', (1, 0)), ('move_down', (0, 1)),
                ('move_left', (-1, 0)), ('move_up', (0, -1))]
        for act, (dx, dy) in dirs:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in self.walkable and (nx, ny) not in occupied and (nx, ny) not in recent:
                return act
        for act, (dx, dy) in dirs:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in self.walkable and (nx, ny) not in occupied:
                return act
        return 'wait'

    def decide(self, data):
        """Greedy closest-item + preview pipelining during pickup phase."""
        bots = data['bots']
        orders = data.get('orders', [])

        self._detect_failed_moves(bots)

        for bot in bots:
            bid = bot['id']
            hist = self.bot_history.setdefault(bid, deque(maxlen=12))
            hist.append((bot['position'][0], bot['position'][1]))

        active_order = next((o for o in orders if o['status'] == 'active'), None)
        preview_order = next((o for o in orders if o['status'] == 'preview'), None)

        if not active_order:
            result = [{'bot': b['id'], 'action': 'wait'} for b in bots]
            self._save_round_state(bots, result)
            return result

        order_id = active_order.get('id', '')

        # What items are still needed for active order?
        needed = list(active_order['items_required'])
        for d in active_order.get('items_delivered', []):
            if d in needed:
                needed.remove(d)
        needed_lower = [t.lower() for t in needed]

        # Count what's carried by all bots
        carried = {}
        for bot in bots:
            for t in bot.get('inventory', []):
                carried[t.lower()] = carried.get(t.lower(), 0) + 1

        # Items still needed that no bot is carrying
        still_need = list(needed_lower)
        for t, count in carried.items():
            for _ in range(count):
                if t in still_need:
                    still_need.remove(t)

        # Preview types for pipelining (pack alongside active in multi-trip orders)
        preview_types = []
        if preview_order:
            preview_types = [t.lower() for t in preview_order['items_required']]

        # Pure preview targets: only when active is fully covered
        preview_nav = preview_types if not still_need else []

        bot_positions = {(b['position'][0], b['position'][1]) for b in bots}
        actions = []
        claimed_items = set()

        for bot in bots:
            bx, by = bot['position']
            inv = [t.lower() for t in bot.get('inventory', [])]
            bid = bot['id']

            # 1. At dropoff with matching items → drop off
            if (bx, by) == self.drop_off and inv:
                if any(t in needed_lower for t in inv):
                    actions.append({'bot': bid, 'action': 'drop_off'})
                    for t in inv:
                        if t in needed_lower:
                            needed_lower.remove(t)
                    self.bot_plan.pop(bid, None)
                    continue

            # 2. Adjacent to needed item → pick up
            picked = False
            if len(inv) < INV_CAP:
                # Active first; if active covered, preview; also preview if room + pipelining
                pick_targets = list(still_need) if still_need else preview_nav
                for item_type in pick_targets:
                    if item_type not in self.items_by_type:
                        continue
                    for item_info in self.items_by_type[item_type]:
                        iid = item_info['id']
                        if iid in claimed_items:
                            continue
                        ix, iy = item_info['pos']
                        if abs(ix - bx) + abs(iy - by) == 1:
                            actions.append({'bot': bid, 'action': 'pick_up', 'item_id': iid})
                            claimed_items.add(iid)
                            if item_type in still_need:
                                still_need.remove(item_type)
                            self.bot_plan.pop(bid, None)
                            picked = True
                            break
                    if picked:
                        break

            if picked:
                continue

            # 3. Deliver: full inventory, or all active covered (never delay)
            has_active = any(t in needed_lower for t in inv)
            should_deliver = False
            if not inv:
                should_deliver = False
            elif len(inv) >= INV_CAP:
                should_deliver = True
            elif has_active and not still_need:
                should_deliver = True

            if should_deliver:
                occupied = bot_positions - {(bx, by)}
                act = self._next_step((bx, by), self.drop_off, occupied)
                actions.append({'bot': bid, 'action': act})
                continue

            # 4. Navigate — greedy closest item (active priority, then preview)
            nav_targets = list(still_need) if still_need else preview_nav

            if nav_targets:
                # Persistent target — keep pursuing same item
                plan = self.bot_plan.get(bid)
                use_plan = False
                if plan and plan.get('order_id') == order_id:
                    tidx = plan.get('tidx', 0)
                    if tidx < len(plan.get('targets', [])):
                        target_id, target_adj = plan['targets'][tidx]
                        t_type = self._item_type(target_id)
                        if t_type and t_type in nav_targets and target_adj in self.walkable:
                            use_plan = True
                            best_item_id = target_id
                            best_target = target_adj

                if not use_plan:
                    best_dist = 9999
                    best_target = None
                    best_item_id = None
                    for item_type in nav_targets:
                        if item_type not in self.items_by_type:
                            continue
                        for item_info in self.items_by_type[item_type]:
                            if item_info['id'] in claimed_items:
                                continue
                            for adj in item_info['adj']:
                                if adj not in self.walkable:
                                    continue
                                d = self._dist((bx, by), adj)
                                if d < best_dist:
                                    best_dist = d
                                    best_item_id = item_info['id']
                                    best_target = adj

                    if best_item_id:
                        self.bot_plan[bid] = {
                            'targets': [(best_item_id, best_target)],
                            'tidx': 0,
                            'order_id': order_id,
                        }

                if best_target:
                    if best_item_id:
                        claimed_items.add(best_item_id)
                    occupied = bot_positions - {(bx, by)}
                    act = self._next_step((bx, by), best_target, occupied)
                    if self._is_oscillating(bid):
                        esc = self._escape_move(bid, (bx, by), occupied)
                        if esc != 'wait':
                            act = esc
                            self.bot_plan.pop(bid, None)
                    actions.append({'bot': bid, 'action': act})
                    continue

            # 5. Have items → deliver
            if inv:
                occupied = bot_positions - {(bx, by)}
                act = self._next_step((bx, by), self.drop_off, occupied)
                actions.append({'bot': bid, 'action': act})
                continue

            # 6. Wait
            actions.append({'bot': bid, 'action': 'wait'})

        self._save_round_state(bots, actions)
        return actions

    def _item_type(self, item_id):
        """Get lowercase type for an item_id."""
        for t, items in self.items_by_type.items():
            for info in items:
                if info['id'] == item_id:
                    return t
        return None

    def _save_round_state(self, bots, actions):
        for bot in bots:
            self.last_positions[bot['id']] = (bot['position'][0], bot['position'][1])
        for act_data in actions:
            self.last_sent_actions[act_data['bot']] = act_data['action']


def quick_multi_solve(game_factory, difficulty, time_budget=1.5):
    """Run planner with multiple mab values, return best (score, actions).

    Fast version for live play - no optimizer, just multi-strategy planner.
    """
    from configs import CONFIGS
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    results = []

    if num_bots == 1:
        mab_values = [1]
    elif num_bots <= 3:
        mab_values = [1, 2, 3]
    elif num_bots <= 5:
        mab_values = [1, 2, 3, 4]
    else:
        mab_values = [1, 2, 3]

    t0 = time.time()
    for mab in mab_values:
        if time.time() - t0 > time_budget:
            break
        try:
            score, actions = planner_solve(
                game_factory=game_factory, verbose=False, max_active_bots=mab
            )
            results.append((score, actions, mab))
        except Exception as e:
            print(f"  mab={mab}: FAILED ({e})", file=sys.stderr)

    if not results:
        score, actions = planner_solve(game_factory=game_factory, verbose=False)
        return score, actions

    results.sort(key=lambda x: -x[0])
    best_score, best_actions, best_mab = results[0]
    print(f"  Quick solve: best mab={best_mab} score={best_score} "
          f"({time.time()-t0:.2f}s, tried {len(results)} configs)",
          file=sys.stderr)
    return best_score, best_actions


class LiveMultiSolver:
    """Solve-then-replay live solver with parallel background optimization."""

    def __init__(self, round0_data, save_capture=False):
        t0 = time.time()
        self.capture = ws_to_capture(round0_data)
        self.difficulty = self.capture['difficulty']
        self.num_bots = self.capture['num_bots']
        self.save_capture = save_capture

        # Save capture data for later re-optimization
        if save_capture:
            from solution_store import save_capture as store_capture
            store_capture(self.difficulty, self.capture)
            print(f"  Capture saved to DB ({self.difficulty})", file=sys.stderr)

        # Build map_state for action format conversion
        self.ms = build_map_from_capture(self.capture)

        # Build game factory
        def game_factory():
            return init_game_from_capture(self.capture)

        self.game_factory = game_factory

        # Phase 1: Quick multi-strategy solve
        self.best_score, self.best_actions = quick_multi_solve(
            game_factory, self.difficulty, time_budget=1.5
        )

        self.lock = threading.Lock()
        self.current_round = 0
        self.optimizer_done = False

        elapsed = time.time() - t0
        print(f"  Init complete: {self.difficulty} {self.num_bots}bots "
              f"score={self.best_score} ({elapsed:.2f}s)", file=sys.stderr)

        # Phase 2: Start background parallel optimizer
        self.optimizer_thread = threading.Thread(
            target=self._run_optimizer, daemon=True
        )
        self.optimizer_thread.start()

    def _run_optimizer(self):
        """Run parallel optimizer in background, update best_actions when improved."""
        try:
            time.sleep(0.1)

            # Time budget: 100s total (leave 20s buffer for 120s wall clock)
            time_limit = 100.0
            num_workers = min(12, os.cpu_count() or 4)

            if self.num_bots > 1:
                # Use parallel optimizer for multi-bot games
                from parallel_optimizer import parallel_optimize
                opt_score, opt_actions = parallel_optimize(
                    capture_data=self.capture,
                    difficulty=self.difficulty,
                    time_limit=time_limit,
                    num_workers=num_workers,
                    verbose=False,
                )
            else:
                # Single bot: single optimizer is fine
                from planner_optimizer import optimize_planner
                opt_score, opt_actions = optimize_planner(
                    game_factory=self.game_factory,
                    iterations=100000,
                    time_limit=time_limit,
                    verbose=False,
                )

            with self.lock:
                if opt_score > self.best_score:
                    print(f"  Optimizer: {self.best_score} -> {opt_score} "
                          f"(+{opt_score - self.best_score})", file=sys.stderr)
                    self.best_score = opt_score
                    self.best_actions = opt_actions
                self.optimizer_done = True

        except Exception as e:
            print(f"  Optimizer FAILED: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.optimizer_done = True

    def get_actions(self, round_num):
        """Get pre-computed actions for a round. Returns WebSocket-format actions."""
        self.current_round = round_num

        with self.lock:
            if round_num < len(self.best_actions):
                internal_actions = self.best_actions[round_num]
            else:
                internal_actions = [(ACT_WAIT, -1)] * self.num_bots

        return actions_to_ws_format(internal_actions, self.ms)


async def play_live(ws_url, log_dir=None, save_capture=False, fast_mode=False):
    """Play a game using multi-strategy solver, write game_log.jsonl.

    Modes:
    - default: Precomputed planner + background optimizer
    - --save-capture: Reactive solver, captures orders for Learn workflow
    - --fast: Route-table solver, uses precomputed optimal routes for all orders
    """
    import websockets

    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(__file__))

    # Route table for fast mode (loaded at round 0 when difficulty is known)
    route_table = None
    if fast_mode:
        from route_table import RouteTable
        # Try to pre-load from JWT hint
        try:
            import base64
            token = ws_url.split('token=')[1] if 'token=' in ws_url else ''
            payload = token.split('.')[1]
            payload += '=' * (4 - len(payload) % 4)
            jwt_data = json.loads(base64.b64decode(payload))
            difficulty_hint = jwt_data.get('difficulty')
            if difficulty_hint:
                route_table = RouteTable.load(difficulty_hint)
                if route_table:
                    print(f"  Route table pre-loaded: {difficulty_hint} ({len(route_table.routes)} routes)",
                          file=sys.stderr)
        except Exception:
            pass  # Will load at round 0 from bot count

    timestamp = int(time.time())
    log_path = os.path.join(log_dir, f'game_log_{timestamp}.jsonl')
    log_file = open(log_path, 'w')

    print(f"Connecting to server...", file=sys.stderr)

    solver = None
    reactive = None
    route_solver = None
    seen_order_ids = set()
    all_orders_captured = []
    all_actions_recorded = []

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            log_file.write(json.dumps(data) + '\n')
            log_file.flush()

            if data["type"] == "game_over":
                score = data.get('score', 0)
                print(f"GAME_OVER Score:{score}", file=sys.stderr)
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]
            max_rounds = data.get("max_rounds", 300)
            score = data.get("score", 0)

            # Accumulate orders for capture
            if save_capture or fast_mode:
                for order in data.get('orders', []):
                    oid = order.get('id', f'order_{len(all_orders_captured)}')
                    if oid not in seen_order_ids:
                        seen_order_ids.add(oid)
                        all_orders_captured.append({
                            'items_required': order['items_required'],
                            'items_delivered': [],
                            'status': 'future',
                        })

            if rnd == 0:
                if fast_mode:
                    from route_table import RouteTable
                    # Always train fresh from current map (0.05-0.1s for Easy)
                    capture = ws_to_capture(data)
                    t_train = time.time()
                    route_table = RouteTable(capture)
                    route_table.train(verbose=True)
                    print(f"  Route table trained in {time.time()-t_train:.2f}s", file=sys.stderr)
                    route_solver = RouteSolver(data, route_table)
                elif save_capture:
                    reactive = ReactiveSolver(data)
                else:
                    solver = LiveMultiSolver(data, save_capture=False)

            print(f"R{rnd}/{max_rounds} Score:{score}", file=sys.stderr)

            if route_solver:
                ws_actions = route_solver.decide(data)

                # Record actions for save
                ws_to_internal = {
                    'wait': ACT_WAIT, 'move_up': ACT_MOVE_UP, 'move_down': ACT_MOVE_DOWN,
                    'move_left': ACT_MOVE_LEFT, 'move_right': ACT_MOVE_RIGHT,
                    'pick_up': ACT_PICKUP, 'drop_off': ACT_DROPOFF,
                }
                internal_actions = []
                for wa in ws_actions:
                    act = ws_to_internal.get(wa['action'], ACT_WAIT)
                    item_idx = -1
                    if act == ACT_PICKUP and 'item_id' in wa:
                        item_idx = route_solver._id_to_idx.get(wa['item_id'], -1)
                    internal_actions.append((act, item_idx))
                all_actions_recorded.append(internal_actions)
            elif reactive:
                ws_actions = reactive.decide(data)

                ws_to_internal = {
                    'wait': ACT_WAIT, 'move_up': ACT_MOVE_UP, 'move_down': ACT_MOVE_DOWN,
                    'move_left': ACT_MOVE_LEFT, 'move_right': ACT_MOVE_RIGHT,
                    'pick_up': ACT_PICKUP, 'drop_off': ACT_DROPOFF,
                }
                internal_actions = []
                for wa in ws_actions:
                    act = ws_to_internal.get(wa['action'], ACT_WAIT)
                    item_idx = -1
                    if act == ACT_PICKUP and 'item_id' in wa:
                        item_idx = reactive._id_to_idx.get(wa['item_id'], -1)
                    internal_actions.append((act, item_idx))
                all_actions_recorded.append(internal_actions)
            elif solver:
                ws_actions = solver.get_actions(rnd)
            else:
                ws_actions = [{'bot': b['id'], 'action': 'wait'}
                              for b in data['bots']]

            response = {"actions": ws_actions}

            log_file.write(json.dumps(response) + '\n')
            log_file.flush()

            await ws.send(json.dumps(response))

    log_file.close()
    print(f"Log saved: {log_path}", file=sys.stderr)

    # Save capture/solution
    active_solver = route_solver or reactive or solver
    if (save_capture or fast_mode) and active_solver:
        from solution_store import save_solution, merge_capture as do_merge, save_capture as store_capture

        capture = active_solver.capture if hasattr(active_solver, 'capture') else None
        difficulty = active_solver.difficulty if hasattr(active_solver, 'difficulty') else 'easy'

        if capture and all_orders_captured:
            new_capture = dict(capture)
            new_capture['orders'] = [{'items_required': o['items_required']} for o in all_orders_captured]
            merged, num_new, total = do_merge(difficulty, new_capture)
            if num_new > 0:
                print(f"  Capture merged: +{num_new} new orders ({total} total)", file=sys.stderr)
            else:
                print(f"  Capture unchanged ({total} orders)", file=sys.stderr)

        actions_to_save = all_actions_recorded if all_actions_recorded else \
                          (solver.best_actions if solver else [])
        if actions_to_save:
            saved = save_solution(difficulty, score, actions_to_save, force=False)
            if saved:
                print(f"  Solution saved: {difficulty} score={score}", file=sys.stderr)
            else:
                print(f"  Solution NOT saved (existing is better)", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Live game solver')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('log_dir', nargs='?', default=None, help='Log directory')
    parser.add_argument('--save-capture', action='store_true',
                        help='Save capture + solution for Learn & Replay workflow')
    parser.add_argument('--fast', action='store_true',
                        help='Use precomputed route table (train with route_table.py first)')
    args = parser.parse_args()

    asyncio.run(play_live(args.ws_url, args.log_dir,
                          save_capture=args.save_capture,
                          fast_mode=args.fast))
