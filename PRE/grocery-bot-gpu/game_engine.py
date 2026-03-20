"""Pure Python in-process game simulator. No WebSocket, no async.

Replicates sim_server.py logic exactly, with integer-based state for GPU readiness.
"""
from __future__ import annotations

import random
import copy
from typing import Any, Callable, List, TypedDict

import numpy as np
from configs import CONFIGS, ALL_TYPES, MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE


class CaptureOrder(TypedDict):
    """Schema for a single captured order."""
    items_required: List[str]


class CaptureData(TypedDict, total=False):
    """Schema for the capture_data interchange dict.

    Required keys are always present. Optional keys may be omitted
    depending on the capture source (live game, game log, or seed-based).
    """
    # Required
    difficulty: str
    grid: Any  # {"width": int, "height": int, "walls": list[list[int]]}
    items: List[Any]  # [{"type": str, "x": int, "y": int}, ...]
    drop_off: Any  # {"x": int, "y": int} or [x, y]
    num_bots: int
    orders: List[CaptureOrder]
    # Optional
    seed: int
    spawn: Any  # {"x": int, "y": int} or [x, y]
    captured_at: str
    probe_score: int


# Action encoding
ACT_WAIT = 0
ACT_MOVE_UP = 1
ACT_MOVE_DOWN = 2
ACT_MOVE_LEFT = 3
ACT_MOVE_RIGHT = 4
ACT_PICKUP = 5
ACT_DROPOFF = 6

# Grid cell types
CELL_FLOOR = 0
CELL_WALL = 1
CELL_SHELF = 2
CELL_DROPOFF = 3

# Direction deltas: indexed by action (1-4)
DX = [0, 0, 0, -1, 1, 0, 0]  # wait, up, down, left, right, pickup, dropoff
DY = [0, -1, 1, 0, 0, 0, 0]


class MapState:
    """Static map data, built once per difficulty."""
    __slots__ = [
        'width', 'height', 'grid', 'drop_off', 'drop_off_zones', 'spawn',
        'items', 'item_positions', 'item_types', 'item_type_names',
        'type_name_to_id', 'num_types', 'num_items',
        'item_adjacencies',  # walkable cells adjacent to each item
    ]

    def __init__(self):
        pass


class Order:
    __slots__ = ['id', 'required', 'delivered', 'complete', 'status', '_required_names']

    def __init__(self, order_id: int, required: list[int], status: str = 'active') -> None:
        self.id = order_id
        self.required = np.array(required, dtype=np.int8)  # item type IDs
        self.delivered = np.zeros(len(required), dtype=np.int8)  # 0=not delivered, 1=delivered
        self.complete = False
        self.status = status  # 'active' or 'preview'
        self._required_names: list[str] | None = None

    def needs(self) -> list[int]:
        """Return list of item type IDs still needed."""
        req = self.required
        delv = self.delivered
        return [int(req[i]) for i in range(len(req)) if delv[i] == 0]

    def needs_type(self, type_id: int) -> bool:
        """Check if this order still needs the given type."""
        req = self.required
        delv = self.delivered
        for i in range(len(req)):
            if req[i] == type_id and delv[i] == 0:
                return True
        return False

    def deliver_type(self, type_id: int) -> bool:
        """Deliver one item of given type. Returns True if accepted."""
        for i, r in enumerate(self.required):
            if r == type_id and self.delivered[i] == 0:
                self.delivered[i] = 1
                return True
        return False

    def is_complete(self) -> bool:
        return all(self.delivered)

    def copy(self) -> Order:
        o = Order.__new__(Order)
        o.id = self.id
        o.required = self.required.copy()
        o.delivered = self.delivered.copy()
        o.complete = self.complete
        o.status = self.status
        o._required_names = self._required_names
        return o


class GameState:
    """Mutable game state for one point in time."""
    __slots__ = [
        'round', 'score', 'items_delivered', 'orders_completed',
        'bot_positions', 'bot_inventories',
        'orders', 'active_idx', 'next_order_idx',
        'map_state',  # reference to static map (not copied)
    ]

    def __init__(self, map_state: MapState) -> None:
        self.map_state = map_state
        self.round = 0
        self.score = 0
        self.items_delivered = 0
        self.orders_completed = 0
        self.bot_positions: np.ndarray | None = None    # int16[N_bots, 2]
        self.bot_inventories: np.ndarray | None = None  # int8[N_bots, INV_CAP], -1 = empty
        self.orders: list[Order] = []             # list of Order objects
        self.active_idx = 0
        self.next_order_idx = 0

    def copy(self) -> GameState:
        s = GameState.__new__(GameState)
        s.map_state = self.map_state  # shared reference
        s.round = self.round
        s.score = self.score
        s.items_delivered = self.items_delivered
        s.orders_completed = self.orders_completed
        s.bot_positions = self.bot_positions.copy()
        s.bot_inventories = self.bot_inventories.copy()
        s.orders = [o.copy() for o in self.orders]
        s.active_idx = self.active_idx
        s.next_order_idx = self.next_order_idx
        return s

    def get_active_order(self) -> Order | None:
        for o in self.orders:
            if not o.complete and o.status == 'active':
                return o
        return None

    def get_preview_order(self) -> Order | None:
        for o in self.orders:
            if not o.complete and o.status == 'preview':
                return o
        return None

    def bot_inv_list(self, bot_id: int) -> list[int]:
        """Get list of item type IDs in bot inventory (excluding -1 empties)."""
        inv = self.bot_inventories[bot_id]
        # Unrolled for INV_CAP=3 — avoids list comprehension overhead
        result = []
        if inv[0] >= 0: result.append(int(inv[0]))
        if inv[1] >= 0: result.append(int(inv[1]))
        if inv[2] >= 0: result.append(int(inv[2]))
        return result

    def bot_inv_count(self, bot_id: int) -> int:
        """Count items in bot inventory."""
        inv = self.bot_inventories[bot_id]
        # Unrolled for INV_CAP=3 — avoids numpy.sum overhead
        return (1 if inv[0] >= 0 else 0) + (1 if inv[1] >= 0 else 0) + (1 if inv[2] >= 0 else 0)

    def bot_inv_add(self, bot_id: int, type_id: int) -> bool:
        """Add item to bot inventory. Returns True if successful."""
        for i in range(INV_CAP):
            if self.bot_inventories[bot_id, i] < 0:
                self.bot_inventories[bot_id, i] = type_id
                return True
        return False

    def bot_inv_remove_matching(self, bot_id: int, order: Order) -> int:
        """Remove items matching order needs, deliver them. Returns delivered count."""
        delivered = 0
        new_inv = []
        for i in range(INV_CAP):
            type_id = self.bot_inventories[bot_id, i]
            if type_id < 0:
                continue
            if order.deliver_type(int(type_id)):
                delivered += 1
            else:
                new_inv.append(type_id)
        # Rebuild inventory
        self.bot_inventories[bot_id] = -1
        for i, t in enumerate(new_inv):
            self.bot_inventories[bot_id, i] = t
        return delivered


def build_map(difficulty: str) -> MapState:
    """Build static map state from difficulty config. Returns MapState."""
    cfg = CONFIGS[difficulty]
    w, h = cfg['w'], cfg['h']

    grid = np.zeros((h, w), dtype=np.int8)

    walls = set()
    shelves = set()

    # Border walls
    for x in range(w):
        walls.add((x, 0))
        walls.add((x, h - 1))
    for y in range(h):
        walls.add((0, y))
        walls.add((w - 1, y))

    # Build aisles: shelf-walkway-shelf, 3 cols wide
    aisle_starts = []
    start_x = 3
    for _ in range(cfg['aisles']):
        aisle_starts.append(start_x)
        start_x += 4

    mid_y = h // 2
    shelf_rows_top = list(range(2, mid_y))
    shelf_rows_bot = list(range(mid_y + 1, h - 2))

    for ax in aisle_starts:
        for y in shelf_rows_top + shelf_rows_bot:
            walls.add((ax - 1, y))
            walls.add((ax + 3, y))
            shelves.add((ax, y))
            shelves.add((ax + 2, y))

    # Fill grid
    for (x, y) in walls:
        grid[y, x] = CELL_WALL
    for (x, y) in shelves:
        grid[y, x] = CELL_SHELF

    drop_off = (1, h - 2)
    spawn = (w - 2, h - 2)
    grid[drop_off[1], drop_off[0]] = CELL_DROPOFF

    # Items on shelves
    item_type_names = ALL_TYPES[:cfg['types']]
    type_name_to_id = {name: i for i, name in enumerate(item_type_names)}

    shelf_list = sorted(shelves)
    num_items = len(shelf_list)
    item_positions = np.zeros((num_items, 2), dtype=np.int16)
    item_types = np.zeros(num_items, dtype=np.int8)

    items = []
    for i, (sx, sy) in enumerate(shelf_list):
        type_name = item_type_names[i % len(item_type_names)]
        type_id = type_name_to_id[type_name]
        item_positions[i] = [sx, sy]
        item_types[i] = type_id
        items.append({
            'id': f'item_{i}',
            'type': type_name,
            'type_id': type_id,
            'position': (sx, sy),
        })

    # Precompute item adjacencies: for each item, which walkable cells are adjacent
    item_adjacencies = {}
    for i, (ix, iy) in enumerate(shelf_list):
        adj = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if grid[ny, nx] == CELL_FLOOR or grid[ny, nx] == CELL_DROPOFF:
                    adj.append((nx, ny))
        item_adjacencies[i] = adj

    ms = MapState()
    ms.width = w
    ms.height = h
    ms.grid = grid
    ms.drop_off = drop_off
    # Nightmare: 3 dropoff zones
    if cfg.get('dropoffs', 1) == 3:
        ms.drop_off_zones = [(1, h - 2), (w // 2, h - 2), (w - 3, h - 2)]
        for dz in ms.drop_off_zones:
            grid[dz[1], dz[0]] = CELL_DROPOFF
    else:
        ms.drop_off_zones = [drop_off]
    ms.spawn = spawn
    ms.items = items
    ms.item_positions = item_positions
    ms.item_types = item_types
    ms.item_type_names = item_type_names
    ms.type_name_to_id = type_name_to_id
    ms.num_types = cfg['types']
    ms.num_items = num_items
    ms.item_adjacencies = item_adjacencies
    return ms


def generate_order_from_rng(rng, order_id, item_type_names, order_size, status, available_counts=None):
    """Generate one order using the given RNG. Mirrors sim_server.py exactly."""
    n = rng.randint(order_size[0], order_size[1])

    if available_counts:
        avail_types = [t for t, c in available_counts.items() if c > 0]
        if not avail_types:
            avail_types = list(item_type_names)
        temp_counts = dict(available_counts)
        required = []
        for _ in range(n):
            usable = [t for t in avail_types if temp_counts.get(t, 0) > 0]
            if not usable:
                usable = avail_types
            t = rng.choice(usable)
            required.append(t)
            if t in temp_counts:
                temp_counts[t] -= 1
    else:
        required = [rng.choice(item_type_names) for _ in range(n)]
    return required


def generate_all_orders(seed: int, map_state: MapState, difficulty: str, count: int = 100) -> list[Order]:
    """Pre-generate ALL orders using the same RNG sequence as sim_server.py.

    The RNG is seeded with `seed` and then generates orders in the exact same
    sequence as the game server. This gives us full order foresight.

    Returns list of Order objects with integer type IDs.
    """
    cfg = CONFIGS[difficulty]
    order_size = cfg['order_size']

    # Count available items per type (same as sim_server get_available_counts)
    available_counts = {}
    for item in map_state.items:
        t = item['type']
        available_counts[t] = available_counts.get(t, 0) + 1

    rng = random.Random(seed)

    # sim_server.py seeds with random.seed(seed) at the start of run_game,
    # then calls build_map (no RNG), then generates orders.
    # But build_map uses sorted(shelves) which is deterministic.
    # So the first RNG call is for order generation.

    orders = []
    for i in range(count):
        status = 'active' if i == 0 else ('preview' if i == 1 else 'future')
        required_names = generate_order_from_rng(
            rng, i, map_state.item_type_names, order_size, status, available_counts
        )
        required_ids = [map_state.type_name_to_id[name] for name in required_names]
        order = Order(i, required_ids, status)
        order._required_names = required_names  # keep for debugging
        orders.append(order)

    return orders


def init_game(seed: int, difficulty: str, num_orders: int = 100) -> tuple[GameState, list[Order]]:
    """Initialize a game. Returns (GameState, all_orders)."""
    map_state = build_map(difficulty)
    all_orders = generate_all_orders(seed, map_state, difficulty, num_orders)
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    state = GameState(map_state)
    state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
    state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)

    # All bots start at spawn
    for i in range(num_bots):
        state.bot_positions[i] = [map_state.spawn[0], map_state.spawn[1]]

    # Initial orders: first is active, second is preview
    state.orders = [all_orders[0].copy(), all_orders[1].copy()]
    state.orders[0].status = 'active'
    state.orders[1].status = 'preview'
    state.next_order_idx = 2
    state.active_idx = 0

    return state, all_orders


def step(state: GameState, actions: list[tuple[int, int]], all_orders: list[Order]) -> int:
    """Apply one round of actions to state. Mutates state in-place.

    actions: list of (action_type, item_idx) tuples per bot.
        action_type: ACT_WAIT, ACT_MOVE_UP, etc.
        item_idx: index into map_state.items for ACT_PICKUP, -1 otherwise.

    Returns score delta for this round.
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    score_delta = 0

    # Build occupied map: (x,y) -> set of bot_ids
    occupied = {}
    for bid in range(num_bots):
        pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
        if pos not in occupied:
            occupied[pos] = set()
        occupied[pos].add(bid)

    # Process actions in bot ID order (matching sim_server.py)
    for bid in range(num_bots):
        act_type, item_idx = actions[bid]
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])

        if ACT_MOVE_UP <= act_type <= ACT_MOVE_RIGHT:
            dx = DX[act_type]
            dy = DY[act_type]
            nx, ny = bx + dx, by + dy

            # Check walkable
            if 0 <= nx < ms.width and 0 <= ny < ms.height:
                cell = ms.grid[ny, nx]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    # Check collision (spawn exempt)
                    target_occ = occupied.get((nx, ny), set())
                    if len(target_occ) == 0 or (nx, ny) == ms.spawn:
                        # Move
                        occupied[(bx, by)].discard(bid)
                        if not occupied[(bx, by)]:
                            del occupied[(bx, by)]
                        state.bot_positions[bid] = [nx, ny]
                        if (nx, ny) not in occupied:
                            occupied[(nx, ny)] = set()
                        occupied[(nx, ny)].add(bid)

        elif act_type == ACT_PICKUP:
            if item_idx >= 0 and state.bot_inv_count(bid) < INV_CAP:
                ix = int(ms.item_positions[item_idx, 0])
                iy = int(ms.item_positions[item_idx, 1])
                mdist = abs(bx - ix) + abs(by - iy)
                if mdist == 1:
                    type_id = int(ms.item_types[item_idx])
                    state.bot_inv_add(bid, type_id)

        elif act_type == ACT_DROPOFF:
            if any(bx == dz[0] and by == dz[1] for dz in ms.drop_off_zones):
                if state.bot_inv_count(bid) > 0:
                    active = state.get_active_order()
                    if active:
                        delivered = state.bot_inv_remove_matching(bid, active)
                        score_delta += delivered
                        state.score += delivered
                        state.items_delivered += delivered

                        # Chain reaction loop: complete order → auto-deliver → possibly complete again
                        while active and active.is_complete():
                            active.complete = True
                            score_delta += 5
                            state.score += 5
                            state.orders_completed += 1

                            # Activate preview
                            for o in state.orders:
                                if not o.complete and o.status == 'preview':
                                    o.status = 'active'
                                    break

                            # Add new preview from all_orders
                            if state.next_order_idx < len(all_orders):
                                new_order = all_orders[state.next_order_idx].copy()
                                new_order.status = 'preview'
                                state.orders.append(new_order)
                                state.next_order_idx += 1

                            # Auto-delivery: for each bot AT a dropoff zone,
                            # check ALL items in their inventory against the
                            # new active order and auto-deliver matching ones.
                            # (Live server confirmed DZ-only, 2026-03-11 + 2026-03-13)
                            active = state.get_active_order()
                            if active:
                                drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
                                for b2 in range(num_bots):
                                    b2_pos = (int(state.bot_positions[b2, 0]),
                                              int(state.bot_positions[b2, 1]))
                                    if b2_pos in drop_set:
                                        d = state.bot_inv_remove_matching(b2, active)
                                        score_delta += d
                                        state.score += d
                                        state.items_delivered += d

    state.round += 1
    return score_delta


def simulate_game(
    seed: int,
    difficulty: str,
    action_fn: Callable[[GameState, list[Order], int], list[tuple[int, int]]],
    verbose: bool = False,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Run a full game with the given action function.

    action_fn(state, all_orders, round_num) -> list of (action_type, item_idx) per bot

    Returns (final_score, action_log).
    """
    state, all_orders = init_game(seed, difficulty)
    action_log = []

    from configs import DIFF_ROUNDS
    _num_rounds = DIFF_ROUNDS.get(difficulty, 300)
    for rnd in range(_num_rounds):
        state.round = rnd
        actions = action_fn(state, all_orders, rnd)
        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 20 or rnd % 50 == 0):
            active = state.get_active_order()
            print(f"R{rnd} Score={state.score} Orders={state.orders_completed}")
            if active:
                print(f"  Active needs: {active.needs()}")

    return state.score, action_log


def state_to_ws_format(state: GameState, all_orders: list[Order]) -> dict[str, Any]:
    """Convert internal state to the WebSocket JSON format for replay client compatibility."""
    ms = state.map_state
    wall_list = []
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (CELL_WALL, CELL_SHELF):
                wall_list.append([x, y])

    item_list = []
    for i, item in enumerate(ms.items):
        item_list.append({
            'id': item['id'],
            'type': item['type'],
            'position': list(item['position']),
        })

    num_bots = len(state.bot_positions)
    bot_list = []
    for bid in range(num_bots):
        inv = []
        for j in range(INV_CAP):
            t = int(state.bot_inventories[bid, j])
            if t >= 0:
                inv.append(ms.item_type_names[t])
        bot_list.append({
            'id': bid,
            'position': [int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])],
            'inventory': inv,
        })

    # Visible orders (active + preview)
    order_list = []
    for o in state.orders:
        if o.complete:
            continue
        req_names = [ms.item_type_names[int(r)] for r in o.required]
        del_names = [ms.item_type_names[int(o.required[i])] for i in range(len(o.required)) if o.delivered[i]]
        order_list.append({
            'id': f'order_{o.id}',
            'items_required': req_names,
            'items_delivered': del_names,
            'complete': o.complete,
            'status': o.status,
        })
        if len(order_list) >= 2:
            break

    return {
        'type': 'game_state',
        'round': state.round,
        'max_rounds': 300,  # Default; callers should override with actual round count
        'grid': {'width': ms.width, 'height': ms.height, 'walls': wall_list},
        'bots': bot_list,
        'items': item_list,
        'orders': order_list,
        'drop_off': list(ms.drop_off),
        'drop_off_zones': [list(z) for z in ms.drop_off_zones] if len(ms.drop_off_zones) > 1 else None,
        'score': state.score,
    }


def build_map_from_capture(capture_data: dict[str, Any]) -> MapState:
    """Build MapState from captured server data (real item positions/types)."""
    width = capture_data['grid']['width']
    height = capture_data['grid']['height']

    grid = np.zeros((height, width), dtype=np.int8)

    # Walls from server
    for wx, wy in capture_data['grid']['walls']:
        grid[wy, wx] = CELL_WALL

    # Shelves from item positions
    for item in capture_data['items']:
        ix, iy = item['position']
        grid[iy, ix] = CELL_SHELF

    # Dropoff(s)
    drop_off = tuple(capture_data['drop_off'])
    drop_off_zones = [tuple(z) for z in capture_data.get('drop_off_zones', [drop_off])]
    for dz in drop_off_zones:
        grid[dz[1], dz[0]] = CELL_DROPOFF

    # Spawn = bottom-right inside border
    spawn = (width - 2, height - 2)

    # Item types from server
    type_names_set = set()
    for item in capture_data['items']:
        type_names_set.add(item['type'])
    item_type_names = sorted(type_names_set)
    type_name_to_id = {name: i for i, name in enumerate(item_type_names)}

    # Sort items by position for consistent internal indexing
    server_items = sorted(capture_data['items'],
                          key=lambda it: (it['position'][0], it['position'][1]))

    num_items = len(server_items)
    item_positions = np.zeros((num_items, 2), dtype=np.int16)
    item_types = np.zeros(num_items, dtype=np.int8)

    items = []
    for i, item in enumerate(server_items):
        ix, iy = item['position']
        type_name = item['type']
        type_id = type_name_to_id[type_name]
        item_positions[i] = [ix, iy]
        item_types[i] = type_id
        items.append({
            'id': item['id'],  # Preserve server item ID for replay
            'type': type_name,
            'type_id': type_id,
            'position': (ix, iy),
        })

    # Compute adjacencies
    item_adjacencies = {}
    for i, item in enumerate(server_items):
        ix, iy = item['position']
        adj = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny, nx] == CELL_FLOOR or grid[ny, nx] == CELL_DROPOFF:
                    adj.append((nx, ny))
        item_adjacencies[i] = adj

    ms = MapState()
    ms.width = width
    ms.height = height
    ms.grid = grid
    ms.drop_off = drop_off
    ms.drop_off_zones = drop_off_zones
    ms.spawn = spawn
    ms.items = items
    ms.item_positions = item_positions
    ms.item_types = item_types
    ms.item_type_names = item_type_names
    ms.type_name_to_id = type_name_to_id
    ms.num_types = len(item_type_names)
    ms.num_items = num_items
    ms.item_adjacencies = item_adjacencies
    return ms


def init_game_from_capture(capture_data: dict[str, Any], num_orders: int = 100) -> tuple[GameState, list[Order]]:
    """Initialize a game from captured server state.

    Uses captured orders first, then appends random filler orders
    for foresight beyond what the probe revealed.
    """
    map_state = build_map_from_capture(capture_data)

    # Orders from capture
    all_orders = []
    for i, order_data in enumerate(capture_data['orders']):
        required_names = order_data['items_required']
        required_ids = [map_state.type_name_to_id[name] for name in required_names]
        status = 'active' if i == 0 else ('preview' if i == 1 else 'future')
        order = Order(i, required_ids, status)
        order._required_names = required_names
        all_orders.append(order)

    # Filler orders beyond what we captured
    num_captured = len(all_orders)
    if num_captured < num_orders:
        filler_rng = random.Random(12345)
        cfg = CONFIGS[capture_data['difficulty']]
        order_size = cfg['order_size']
        available_counts = {}
        for item in map_state.items:
            t = item['type']
            available_counts[t] = available_counts.get(t, 0) + 1

        for i in range(num_captured, num_orders):
            required_names = generate_order_from_rng(
                filler_rng, i, map_state.item_type_names, order_size, 'future', available_counts
            )
            required_ids = [map_state.type_name_to_id[name] for name in required_names]
            order = Order(i, required_ids, 'future')
            order._required_names = required_names
            all_orders.append(order)

    # Game state
    num_bots = capture_data['num_bots']
    state = GameState(map_state)
    state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
    state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)

    for i in range(num_bots):
        state.bot_positions[i] = [map_state.spawn[0], map_state.spawn[1]]

    state.orders = [all_orders[0].copy(), all_orders[1].copy()]
    state.orders[0].status = 'active'
    state.orders[1].status = 'preview'
    state.next_order_idx = 2
    state.active_idx = 0

    return state, all_orders


def actions_to_ws_format(actions: list[tuple[int, int]], map_state: MapState) -> list[dict[str, Any]]:
    """Convert internal action tuples to WebSocket JSON format."""
    ws_actions = []
    action_names = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']
    for bid, (act_type, item_idx) in enumerate(actions):
        a = {'bot': bid, 'action': action_names[act_type]}
        if act_type == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
            a['item_id'] = map_state.items[item_idx]['id']
        elif act_type == ACT_PICKUP:
            a['action'] = 'wait'  # Invalid item index → wait instead of crash
        ws_actions.append(a)
    return ws_actions


def make_game_factory(capture_data: dict[str, Any], num_orders: int | None = None) -> Callable[[], tuple[GameState, list[Order]]]:
    """Create a game factory closure from capture data.

    Returns a callable that creates a fresh GameState each time.
    Use instead of defining ``def game_factory(): return init_game_from_capture(...)``
    in every orchestrator script.
    """
    def factory():
        kwargs = {}
        if num_orders is not None:
            kwargs['num_orders'] = num_orders
        return init_game_from_capture(capture_data, **kwargs)
    return factory
