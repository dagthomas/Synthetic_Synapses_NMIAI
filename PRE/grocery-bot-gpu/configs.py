"""Difficulty configurations matching sim_server.py exactly."""
from __future__ import annotations

CONFIGS = {
    "easy":   {"w": 12, "h": 10, "bots": 1, "aisles": 2, "types": 4, "order_size": (3, 4)},
    "medium": {"w": 16, "h": 12, "bots": 3, "aisles": 3, "types": 8, "order_size": (3, 5)},
    "hard":   {"w": 22, "h": 14, "bots": 5, "aisles": 4, "types": 12, "order_size": (3, 5)},
    "expert": {"w": 28, "h": 18, "bots": 10, "aisles": 5, "types": 16, "order_size": (4, 6)},
    "nightmare": {"w": 30, "h": 18, "bots": 20, "aisles": 6, "types": 21, "order_size": (4, 7), "dropoffs": 3},
}

ALL_TYPES = [
    "milk", "bread", "eggs", "butter", "cheese", "pasta", "rice", "juice",
    "yogurt", "cereal", "flour", "sugar", "coffee", "tea", "oil", "salt",
    "honey", "beans", "corn", "soup", "cream", "oats", "apples", "lettuce",
]

MAX_ROUNDS = 500  # Global max (nightmare uses 500, others use 300)
DIFF_ROUNDS = {'easy': 300, 'medium': 300, 'hard': 300, 'expert': 300, 'nightmare': 500}
INV_CAP = 3
MAX_BOTS = 24
MAX_ITEMS = 200
MAX_ORDERS = 100
MAX_ORDER_SIZE = 7

DIFF_IDX = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3, 'nightmare': 4}

# Reverse lookup: (width, height, num_bots) -> difficulty name
_DIFFICULTY_DIMS = {
    (cfg["w"], cfg["h"], cfg["bots"]): name
    for name, cfg in CONFIGS.items()
}

# Reverse lookup: num_bots -> difficulty name
_BOTS_TO_DIFF = {cfg["bots"]: name for name, cfg in CONFIGS.items()}


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed specification: '7001-7003', '42,7001', or '3' (count from 7001)."""
    if '-' in seeds_str and ',' not in seeds_str:
        parts = seeds_str.split('-')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if end < 100:
                end = start + end - 1
            return list(range(start, end + 1))
    if ',' in seeds_str:
        return [int(s.strip()) for s in seeds_str.split(',')]
    n = int(seeds_str)
    if n < 100:
        return list(range(7001, 7001 + n))
    return [n]


def detect_difficulty(num_bots: int, width: int | None = None, height: int | None = None) -> str | None:
    """Detect difficulty from bot count, optionally cross-checked with map dimensions.

    Args:
        num_bots: Number of bots in the game.
        width: Optional grid width for exact (w, h, bots) lookup.
        height: Optional grid height for exact (w, h, bots) lookup.

    Returns:
        Difficulty string ('easy', 'medium', 'hard', 'expert') or None.
    """
    if width is not None and height is not None:
        return _DIFFICULTY_DIMS.get((width, height, num_bots))
    return _BOTS_TO_DIFF.get(num_bots)
