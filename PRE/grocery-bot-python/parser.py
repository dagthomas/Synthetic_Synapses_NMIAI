"""Fast JSON parser - minimal object creation, flat arrays."""
from __future__ import annotations
import json
import sys

from types_ import FLOOR, WALL, SHELF, DROPOFF


class GameState:
    """Flat game state for maximum performance."""
    __slots__ = (
        "round", "max_rounds", "width", "height", "grid",
        "known_shelves", "bot_count", "bot_ids", "bot_positions",
        "bot_inventories", "item_count", "item_ids", "item_types",
        "item_positions", "orders", "dropoff", "score", "active_order_idx",
    )

    def __init__(self):
        self.round = 0
        self.max_rounds = 300
        self.width = 0
        self.height = 0
        self.grid: list[int] = []  # flat: grid[y * w + x]
        self.known_shelves: set[int] = set()  # flat indices
        self.bot_count = 0
        self.bot_ids: list[int] = []
        self.bot_positions: list[tuple[int, int]] = []
        self.bot_inventories: list[list[str]] = []
        self.item_count = 0
        self.item_ids: list[str] = []
        self.item_types: list[str] = []
        self.item_positions: list[tuple[int, int]] = []
        self.orders: list[tuple] = []  # (required, delivered, is_active, complete)
        self.dropoff = (0, 0)
        self.score = 0
        self.active_order_idx = 0


def parse_game_state(data: str, state: GameState) -> bool:
    msg = json.loads(data)
    msg_type = msg.get("type", "")

    if msg_type == "game_over":
        score = msg.get("score", 0)
        rounds_used = msg.get("rounds_used", 0)
        print(f"Game over! Score: {score}, Rounds: {rounds_used}", file=sys.stderr)
        return False

    if msg_type != "game_state":
        return False

    state.round = msg["round"]
    state.max_rounds = msg["max_rounds"]
    state.score = msg["score"]
    state.active_order_idx = msg["active_order_index"]

    grid_data = msg["grid"]
    w = grid_data["width"]
    h = grid_data["height"]
    state.width = w
    state.height = h

    # Flat grid
    size = w * h
    grid = [FLOOR] * size
    state.grid = grid

    if state.round == 0:
        state.known_shelves = set()

    for wall in grid_data.get("walls", []):
        grid[wall[1] * w + wall[0]] = WALL

    drop = msg["drop_off"]
    dx, dy = drop[0], drop[1]
    state.dropoff = (dx, dy)
    grid[dy * w + dx] = DROPOFF

    # Items - parallel arrays
    items_data = msg.get("items", [])
    ic = len(items_data)
    state.item_count = ic
    ids = [None] * ic
    types = [None] * ic
    positions = [None] * ic
    ks = state.known_shelves
    for i in range(ic):
        item = items_data[i]
        pos = item["position"]
        px, py = pos[0], pos[1]
        ids[i] = item["id"]
        types[i] = item["type"]
        positions[i] = (px, py)
        fi = py * w + px
        ks.add(fi)
    state.item_ids = ids
    state.item_types = types
    state.item_positions = positions

    # Mark known shelves
    for fi in ks:
        if grid[fi] == FLOOR:
            grid[fi] = SHELF

    # Bots - parallel arrays
    bots_data = msg.get("bots", [])
    bc = len(bots_data)
    state.bot_count = bc
    bids = [0] * bc
    bpos = [None] * bc
    binv = [None] * bc
    for i in range(bc):
        b = bots_data[i]
        pos = b["position"]
        bids[i] = b["id"]
        bpos[i] = (pos[0], pos[1])
        binv[i] = list(b.get("inventory", []))
    state.bot_ids = bids
    state.bot_positions = bpos
    state.bot_inventories = binv

    # Orders
    orders_data = msg.get("orders", [])
    state.orders = []
    for o in orders_data:
        state.orders.append((
            list(o.get("items_required", [])),
            list(o.get("items_delivered", [])),
            o.get("status", "") == "active",
            o.get("complete", False),
        ))

    if state.round == 0:
        print(f"R0: {w}x{h} grid, {bc} bots, {ic} items, dropoff ({dx},{dy})", file=sys.stderr)

    return True
