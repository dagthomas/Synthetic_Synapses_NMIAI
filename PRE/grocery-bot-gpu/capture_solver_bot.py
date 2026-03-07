"""WebSocket-compatible wrapper around capture_solver for live games.

Maintains internal game engine state synced with server JSON,
runs capture_solver logic each round to generate actions.
"""
import numpy as np
import random
from collections import deque
from game_engine import (
    MapState, GameState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
    DX, DY,
)
from pathfinding import precompute_all_distances, get_distance, get_first_step
from planner import (
    BotController, ST_IDLE, ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF, ST_PARKED,
    assign_items_globally, optimize_trip_order, find_parking_spots,
    compute_remaining_needs, move_away_from,
)

ACT_TO_STR = {
    0: "wait", 1: "move_up", 2: "move_down",
    3: "move_left", 4: "move_right", 5: "pick_up", 6: "drop_off",
}


def bfs_move(start, goal, ms, occupied):
    """BFS pathfind routing around occupied cells."""
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
            if cell != CELL_FLOOR and cell != CELL_DROPOFF:
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


def _valid_moves(bx, by, ms, occ):
    moves = []
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = bx + DX[act], by + DY[act]
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell in (CELL_FLOOR, CELL_DROPOFF) and (nx, ny) not in occ:
                moves.append(act)
    return moves


class CaptureSolverBot:
    """WebSocket-compatible bot using BFS movement for expert games."""

    def __init__(self, max_active_bots=2, max_deliverers=1):
        self.max_active_bots = max_active_bots
        self.max_deliverers = max_deliverers
        self.initialized = False
        self.state = None
        self.ms = None
        self.dist_maps = None
        self.controllers = None
        self.all_orders = []
        self.num_bots = 0
        self.item_id_to_idx = {}
        self.type_name_to_id = {}
        self.item_type_names = []
        self.last_orders_completed = 0
        self.last_active_order_id = -1
        self.waiting_to_deliver = set()
        self.pos_history = []
        self.parking_spots = []
        self.rng = random.Random(42)

    def init_from_server(self, data):
        """Build internal state from server round 0 JSON."""
        grid_data = data["grid"]
        w, h = grid_data["width"], grid_data["height"]

        ms = MapState()
        ms.width = w
        ms.height = h
        ms.grid = np.zeros((h, w), dtype=np.int8)
        ms.drop_off = tuple(data["drop_off"])
        ms.spawn = tuple(data["bots"][0]["position"])

        for wx, wy in grid_data["walls"]:
            ms.grid[wy, wx] = CELL_WALL

        # Build item type mapping
        type_names = []
        type_map = {}
        for item in data["items"]:
            t = item["type"]
            if t not in type_map:
                type_map[t] = len(type_names)
                type_names.append(t)

        self.type_name_to_id = type_map
        self.item_type_names = type_names
        ms.item_type_names = type_names
        ms.type_name_to_id = type_map
        ms.num_types = len(type_names)

        num_items = len(data["items"])
        ms.num_items = num_items
        ms.item_positions = np.zeros((num_items, 2), dtype=np.int16)
        ms.item_types = np.zeros(num_items, dtype=np.int8)
        ms.items = []
        ms.item_adjacencies = {}

        for i, item in enumerate(data["items"]):
            ix, iy = item["position"]
            type_id = type_map[item["type"]]
            ms.item_positions[i] = [ix, iy]
            ms.item_types[i] = type_id
            ms.items.append({
                'id': item['id'], 'type': item['type'],
                'type_id': type_id, 'position': (ix, iy),
            })
            self.item_id_to_idx[item["id"]] = i
            ms.grid[iy, ix] = CELL_SHELF

            adj = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    cell = ms.grid[ny, nx]
                    if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                        adj.append((nx, ny))
            ms.item_adjacencies[i] = adj

        ms.grid[ms.drop_off[1], ms.drop_off[0]] = CELL_DROPOFF

        self.ms = ms
        self.num_bots = len(data["bots"])

        state = GameState(ms)
        state.bot_positions = np.zeros((self.num_bots, 2), dtype=np.int16)
        state.bot_inventories = np.full((self.num_bots, INV_CAP), -1, dtype=np.int8)
        for bot in data["bots"]:
            bid = bot["id"]
            state.bot_positions[bid] = bot["position"]
            for j, item_name in enumerate(bot["inventory"]):
                if j < INV_CAP:
                    state.bot_inventories[bid, j] = type_map[item_name]

        state.orders = []
        for order in data["orders"]:
            required_ids = [type_map[t] for t in order["items_required"]]
            o = Order(order["id"], required_ids, status=order.get("status", "active"))
            for t_name in order.get("items_delivered", []):
                o.deliver_type(type_map[t_name])
            if order.get("complete"):
                o.complete = True
            state.orders.append(o)
        state.active_idx = 0
        state.next_order_idx = len(state.orders)

        self.state = state
        self.all_orders = list(state.orders)
        self.dist_maps = precompute_all_distances(ms)
        self.parking_spots = find_parking_spots(ms, self.dist_maps, num_spots=self.num_bots * 2)
        self.controllers = [BotController(bid) for bid in range(self.num_bots)]
        self.pos_history = [[] for _ in range(self.num_bots)]
        self.initialized = True

        print(f"  CaptureSolverBot: {self.num_bots} bots, {num_items} items, "
              f"max_del={self.max_deliverers}")

    def sync_state(self, data):
        """Update internal state from server JSON."""
        state = self.state
        for bot in data["bots"]:
            bid = bot["id"]
            state.bot_positions[bid] = bot["position"]
            state.bot_inventories[bid] = -1
            for j, item_name in enumerate(bot["inventory"]):
                if j < INV_CAP:
                    state.bot_inventories[bid, j] = self.type_name_to_id[item_name]
        state.score = data.get("score", 0)
        state.orders_completed = data.get("orders_completed", 0)
        state.items_delivered = data.get("items_delivered", 0)
        state.round = data["round"]

        state.orders = []
        for order in data["orders"]:
            required_ids = [self.type_name_to_id[t] for t in order["items_required"]]
            o = Order(order["id"], required_ids, status=order.get("status", "active"))
            for t_name in order.get("items_delivered", []):
                o.deliver_type(self.type_name_to_id[t_name])
            if order.get("complete"):
                o.complete = True
            state.orders.append(o)
            if not any(ao.id == order["id"] for ao in self.all_orders):
                ao = Order(order["id"], required_ids, status=order.get("status", "active"))
                self.all_orders.append(ao)
        state.active_idx = 0
        state.next_order_idx = len(self.all_orders)

    def generate_actions(self, data):
        """Generate actions for one round. Returns server-format action list."""
        if not self.initialized:
            self.init_from_server(data)
        else:
            self.sync_state(data)

        state = self.state
        ms = self.ms
        controllers = self.controllers
        num_bots = self.num_bots
        dist_maps = self.dist_maps
        max_del = self.max_deliverers
        rng = self.rng

        active = state.get_active_order()
        preview = state.get_preview_order()
        active_id = active.id if active else -1

        # Order change detection
        if state.orders_completed > self.last_orders_completed or active_id != self.last_active_order_id:
            for bc in controllers:
                if bc.state == ST_MOVING_TO_ITEM:
                    bc.set_idle()
                    self.waiting_to_deliver.discard(bc.bot_id)
            self.last_orders_completed = state.orders_completed
            self.last_active_order_id = active_id

        # Oscillation & stuck detection
        for bc in controllers:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            pos = (bx, by)
            hist = self.pos_history[bid]
            hist.append(pos)
            if len(hist) > 6:
                hist.pop(0)
            if bc.last_pos == pos and bc.is_busy():
                bc.stuck_count += 1
            else:
                bc.stuck_count = 0
            bc.last_pos = pos
            is_osc = (len(hist) >= 4 and bc.is_busy() and
                      hist[-1] == hist[-3] and hist[-2] == hist[-4] and hist[-1] != hist[-2])
            if bc.stuck_count > 5 or is_osc:
                was_delivering = bc.state == ST_MOVING_TO_DROPOFF
                bc.set_idle()
                bc.stuck_count = 0
                if was_delivering and state.bot_inv_list(bid):
                    self.waiting_to_deliver.add(bid)
                else:
                    self.waiting_to_deliver.discard(bid)

        # Promote waiting bots (closest first)
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        def _wait_prio(bid):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            return int(get_distance(dist_maps, (bx, by), ms.drop_off))
        for bid in sorted(self.waiting_to_deliver, key=_wait_prio):
            if cur_del >= max_del:
                break
            inv = state.bot_inv_list(bid)
            if inv and active and any(active.needs_type(t) for t in inv):
                controllers[bid].assign_deliver(ms.drop_off)
                self.waiting_to_deliver.discard(bid)
                cur_del += 1
            elif not inv:
                self.waiting_to_deliver.discard(bid)

        # Assign pickups
        assignments = assign_items_globally(
            state, dist_maps, self.all_orders, controllers, self.max_active_bots)
        for bid, items in assignments.items():
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            opt = optimize_trip_order((bx, by), items, ms.drop_off, dist_maps)
            controllers[bid].assign_trip(opt)
            self.waiting_to_deliver.discard(bid)

        # Idle bots with useful items -> deliver (throttled)
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        for bc in controllers:
            if bc.state in (ST_IDLE, ST_PARKED) and bc.bot_id not in self.waiting_to_deliver:
                inv = state.bot_inv_list(bc.bot_id)
                if inv and active and any(active.needs_type(t) for t in inv):
                    if cur_del < max_del:
                        bc.assign_deliver(ms.drop_off)
                        cur_del += 1
                    else:
                        self.waiting_to_deliver.add(bc.bot_id)

        # Dead inventory: deliver to clear
        cur_del = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
        for bc in controllers:
            if bc.state == ST_IDLE and bc.bot_id not in self.waiting_to_deliver:
                inv = state.bot_inv_list(bc.bot_id)
                if inv and cur_del < max_del:
                    bc.assign_deliver(ms.drop_off)
                    cur_del += 1

        # Park idle/waiting bots away from dropoff corridor
        parked = set(bc.park_target for bc in controllers if bc.state == ST_PARKED and bc.park_target)
        any_deliverer = any(c.state == ST_MOVING_TO_DROPOFF for c in controllers)
        for bc in controllers:
            bid = bc.bot_id
            if bc.state not in (ST_IDLE, ST_PARKED):
                continue
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
            should_park = False
            if bid not in self.waiting_to_deliver and d <= 6:
                should_park = True
            elif bid in self.waiting_to_deliver and d <= 3 and any_deliverer:
                should_park = True
            if should_park:
                for spot in self.parking_spots:
                    if spot not in parked:
                        bc.assign_park(spot)
                        parked.add(spot)
                        break

        # === Generate actions ===
        actions_internal = [(ACT_WAIT, -1)] * num_bots
        committed = {}

        def prio(bc):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            if bc.state == ST_MOVING_TO_DROPOFF:
                if (bx, by) == ms.drop_off: return (0, 0)
                return (1, int(get_distance(dist_maps, (bx, by), ms.drop_off)))
            if bc.state == ST_MOVING_TO_ITEM: return (2, 0)
            if bc.state == ST_PARKED: return (3, 0)
            return (4, 0)

        for bc in sorted(controllers, key=prio):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            pos = (bx, by)
            occ = set()
            for b2 in range(num_bots):
                if b2 == bid: continue
                if b2 in committed: occ.add(committed[b2])
                else: occ.add((int(state.bot_positions[b2, 0]), int(state.bot_positions[b2, 1])))

            if bc.state == ST_MOVING_TO_DROPOFF:
                if pos == ms.drop_off:
                    inv = state.bot_inv_list(bid)
                    if inv:
                        actions_internal[bid] = (ACT_DROPOFF, -1)
                        committed[bid] = pos
                        bc.set_idle()
                    else:
                        bc.set_idle()
                        act = move_away_from(pos, ms.drop_off, dist_maps, ms, occ)
                        actions_internal[bid] = act
                        committed[bid] = (bx + DX[act[0]], by + DY[act[0]]) if act[0] != ACT_WAIT else pos
                    continue
                act = bfs_move(pos, ms.drop_off, ms, occ)
                if act[0] == ACT_WAIT:
                    moves = _valid_moves(bx, by, ms, occ)
                    if moves: act = (rng.choice(moves), -1)
                actions_internal[bid] = act
                committed[bid] = (bx + DX[act[0]], by + DY[act[0]]) if act[0] != ACT_WAIT else pos
                continue

            if bc.state == ST_MOVING_TO_ITEM:
                if pos == bc.target and bc.item_to_pick >= 0:
                    actions_internal[bid] = (ACT_PICKUP, bc.item_to_pick)
                    committed[bid] = pos
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        ni, nc = bc.trip_items[bc.trip_idx]
                        bc.target = nc
                        bc.item_to_pick = ni
                    else:
                        cd2 = sum(1 for c in controllers if c.state == ST_MOVING_TO_DROPOFF)
                        if cd2 < max_del:
                            bc.assign_deliver(ms.drop_off)
                        else:
                            bc.set_idle()
                            self.waiting_to_deliver.add(bid)
                    continue
                act = bfs_move(pos, bc.target, ms, occ)
                if act[0] == ACT_WAIT or bc.stuck_count >= 3:
                    moves = _valid_moves(bx, by, ms, occ)
                    if moves: act = (rng.choice(moves), -1)
                actions_internal[bid] = act
                committed[bid] = (bx + DX[act[0]], by + DY[act[0]]) if act[0] != ACT_WAIT else pos
                continue

            if bc.state == ST_PARKED:
                if bc.park_target and pos != bc.park_target:
                    act = bfs_move(pos, bc.park_target, ms, occ)
                    if act[0] == ACT_WAIT:
                        moves = _valid_moves(bx, by, ms, occ)
                        if moves: act = (rng.choice(moves), -1)
                    actions_internal[bid] = act
                    committed[bid] = (bx + DX[act[0]], by + DY[act[0]]) if act[0] != ACT_WAIT else pos
                else:
                    committed[bid] = pos
                continue

            committed[bid] = pos

        # Convert to server format
        server_actions = []
        for bid in range(num_bots):
            act_type, item_idx = actions_internal[bid]
            act_str = ACT_TO_STR.get(act_type, "wait")
            action = {"bot": bid, "action": act_str}
            if act_type == ACT_PICKUP and item_idx >= 0:
                action["item_id"] = ms.items[item_idx]["id"]
            server_actions.append(action)

        return server_actions
