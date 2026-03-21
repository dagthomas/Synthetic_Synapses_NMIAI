import os
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

ASTAR_TOKEN = os.environ.get("ASTAR_TOKEN", "")
BASE_URL = "https://api.ainm.no"

MAP_W = 40
MAP_H = 40
NUM_CLASSES = 6
PROB_FLOOR = 0.01  # Hard minimum per scoring rules (0.015 wastes too much mass)

# Internal terrain code → prediction class index
TERRAIN_TO_CLASS = {
    10: 0,  # Ocean → Empty
    11: 0,  # Plains → Empty
    0: 0,   # Empty → Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
