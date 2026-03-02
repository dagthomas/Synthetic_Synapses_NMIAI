"""Difficulty configurations matching sim_server.py exactly."""

CONFIGS = {
    "easy":   {"w": 12, "h": 10, "bots": 1, "aisles": 2, "types": 4, "order_size": (3, 4)},
    "medium": {"w": 16, "h": 12, "bots": 3, "aisles": 3, "types": 8, "order_size": (3, 5)},
    "hard":   {"w": 22, "h": 14, "bots": 5, "aisles": 4, "types": 12, "order_size": (3, 5)},
    "expert": {"w": 28, "h": 18, "bots": 10, "aisles": 5, "types": 16, "order_size": (4, 6)},
}

ALL_TYPES = [
    "milk", "bread", "eggs", "butter", "cheese", "pasta", "rice", "juice",
    "yogurt", "cereal", "flour", "sugar", "coffee", "tea", "oil", "salt",
]

MAX_ROUNDS = 300
INV_CAP = 3
MAX_BOTS = 10
MAX_ITEMS = 200
MAX_ORDERS = 100
MAX_ORDER_SIZE = 6
