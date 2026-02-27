"""Core types - optimized for speed with minimal object creation."""
from __future__ import annotations

# Constants
UNREACHABLE = 9999
INV_CAP = 3
MAX_BOTS = 10
HIST_LEN = 24

# Cell types (plain ints)
FLOOR = 0
WALL = 1
SHELF = 2
DROPOFF = 3

# Directions (plain ints)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIR_NAMES = ("move_up", "move_down", "move_left", "move_right")
DIR_DX = (0, 0, -1, 1)
DIR_DY = (-1, 1, 0, 0)
DIR_REVERSE = (1, 0, 3, 2)  # UP->DOWN, DOWN->UP, LEFT->RIGHT, RIGHT->LEFT


class NeedList:
    """Counter-based need tracker. O(1) contains/remove/add."""
    __slots__ = ("_counts", "_total")

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._total = 0

    @property
    def count(self) -> int:
        return self._total

    def contains(self, t: str) -> bool:
        return self._counts.get(t, 0) > 0

    def remove(self, t: str) -> None:
        c = self._counts.get(t, 0)
        if c > 0:
            if c == 1:
                del self._counts[t]
            else:
                self._counts[t] = c - 1
            self._total -= 1

    def add(self, t: str) -> None:
        self._counts[t] = self._counts.get(t, 0) + 1
        self._total += 1

    def copy(self) -> NeedList:
        nl = NeedList()
        nl._counts = self._counts.copy()
        nl._total = self._total
        return nl

    def unique_with_counts(self) -> list[tuple[str, int]]:
        return [(t, c) for t, c in self._counts.items() if c > 0]


class PersistentBot:
    __slots__ = (
        "trip_ids", "trip_adjs", "trip_pos", "has_trip", "delivering",
        "stall_count", "last_pos", "pos_hist", "pos_hist_idx",
        "osc_count", "last_active_order_idx", "last_tried_pickup",
        "last_pickup_pos", "last_pickup_ipos", "last_inv_len",
        "rounds_on_order", "last_dir", "escape_rounds",
    )

    def __init__(self):
        self.trip_ids: list[str] = []
        self.trip_adjs: list[tuple[int, int]] = []
        self.trip_pos = 0
        self.has_trip = False
        self.delivering = False
        self.stall_count = 0
        self.last_pos = (-1, -1)
        self.pos_hist: list[tuple[int, int]] = []
        self.pos_hist_idx = 0
        self.osc_count = 0
        self.last_active_order_idx = -1
        self.last_tried_pickup = False
        self.last_pickup_pos = (-1, -1)
        self.last_pickup_ipos = (-1, -1)
        self.last_inv_len = 0
        self.rounds_on_order = 0
        self.last_dir = -1
        self.escape_rounds = 0

    @property
    def trip_count(self) -> int:
        return len(self.trip_ids)
