"""Data infrastructure for the parametric simulator.

Loads round data (initial states, ground truth, observations) and extracts
per-settlement spatial features used by the transition model.
"""
import json
from pathlib import Path

import numpy as np

from config import MAP_H, MAP_W, NUM_CLASSES, TERRAIN_TO_CLASS

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

ROUND_IDS = {
    "round1": None,  # No observations for round1
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
    "round8": None,
    "round9": "2a341ace-0f57-4309-9b89-e59fe0f09179",
    "round10": "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    "round11": "324fde07-1670-4202-b199-7aa92ecb40ee",
    "round12": "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    "round13": "7b4bda99-6165-4221-97cc-27880f5e6d95",
    "round14": "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    "round15": "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    "round16": "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
    "round17": "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
}

ALL_ROUNDS = [f"round{i}" for i in range(1, 18)]


def terrain_to_class(code: int) -> int:
    return TERRAIN_TO_CLASS.get(code, 0)


def compute_settlement_features(terrain: np.ndarray, settlements: list[dict]) -> dict:
    """Compute per-settlement spatial features.

    Returns dict with arrays indexed by settlement index:
      - positions: (n_sett, 2) array of (x, y)
      - is_coastal: (n_sett,) bool
      - food_adj: (n_sett,) int - food-producing neighbors (forest + plains)
      - cluster_r3: (n_sett,) int - settlements within Manhattan distance 3
      - cluster_r5: (n_sett,) int - settlements within Manhattan distance 5
      - has_port: (n_sett,) bool
      - dist_maps: (n_sett, H, W) - Manhattan distance from each settlement to every cell
    """
    H, W = terrain.shape
    n = len(settlements)
    positions = np.array([(s["x"], s["y"]) for s in settlements], dtype=int)
    has_port = np.array([s.get("has_port", False) for s in settlements], dtype=bool)

    is_coastal = np.zeros(n, dtype=bool)
    food_adj = np.zeros(n, dtype=int)

    for i, s in enumerate(settlements):
        x, y = s["x"], s["y"]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if terrain[ny, nx] == 10:
                    is_coastal[i] = True
                if terrain[ny, nx] in (4, 11):  # forest or plains
                    food_adj[i] += 1

    cluster_r3 = np.zeros(n, dtype=int)
    cluster_r5 = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = abs(positions[i, 0] - positions[j, 0]) + abs(positions[i, 1] - positions[j, 1])
            if d <= 3:
                cluster_r3[i] += 1
            if d <= 5:
                cluster_r5[i] += 1

    # Per-settlement distance maps
    dist_maps = np.zeros((n, H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        sx, sy = positions[i]
        dist_maps[i] = np.abs(xx - sx) + np.abs(yy - sy)

    return {
        "positions": positions,
        "is_coastal": is_coastal,
        "food_adj": food_adj,
        "cluster_r3": cluster_r3,
        "cluster_r5": cluster_r5,
        "has_port": has_port,
        "dist_maps": dist_maps,
    }


def compute_cell_features(terrain: np.ndarray, settlements: list[dict]) -> dict:
    """Compute per-cell features for the entire grid.

    Returns dict with (H, W) arrays:
      - is_coastal: bool
      - forest_adj: int (0-4)
      - min_sett_dist: float - distance to nearest initial settlement
      - is_ocean: bool
      - is_mountain: bool
      - is_forest: bool
      - is_settlement: bool
      - is_buildable: bool - can become settlement (not ocean/mountain)
    """
    H, W = terrain.shape
    sett_points = [(s["x"], s["y"]) for s in settlements]

    is_coastal = np.zeros((H, W), dtype=bool)
    forest_adj = np.zeros((H, W), dtype=int)

    for y in range(H):
        for x in range(W):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if terrain[ny, nx] == 10:
                        is_coastal[y, x] = True
                    if terrain[ny, nx] == 4:
                        forest_adj[y, x] += 1

    # Min distance to nearest settlement
    if sett_points:
        yy, xx = np.mgrid[0:H, 0:W]
        min_dist = np.full((H, W), 999.0, dtype=np.float32)
        for sx, sy in sett_points:
            d = np.abs(xx - sx) + np.abs(yy - sy)
            min_dist = np.minimum(min_dist, d)
    else:
        min_dist = np.full((H, W), 999.0, dtype=np.float32)

    return {
        "is_coastal": is_coastal,
        "forest_adj": forest_adj,
        "min_sett_dist": min_dist,
        "is_ocean": terrain == 10,
        "is_mountain": terrain == 5,
        "is_forest": terrain == 4,
        "is_settlement": (terrain == 1) | (terrain == 2),
        "is_buildable": ~((terrain == 10) | (terrain == 5)),
    }


class RoundData:
    """All data for one round, one seed."""

    def __init__(self, round_name: str, seed_idx: int,
                 terrain: np.ndarray, settlements: list[dict],
                 ground_truth: np.ndarray | None = None):
        self.round_name = round_name
        self.seed_idx = seed_idx
        self.terrain = terrain
        self.settlements = settlements
        self.ground_truth = ground_truth
        self.sett_features = compute_settlement_features(terrain, settlements)
        self.cell_features = compute_cell_features(terrain, settlements)


def load_round(round_name: str, seed_idx: int = 0) -> RoundData | None:
    """Load a single round+seed from calibration data."""
    rd = DATA_DIR / round_name
    if not (rd / "round_detail.json").exists():
        return None

    detail = json.loads((rd / "round_detail.json").read_text())
    if seed_idx >= len(detail["initial_states"]):
        return None

    state = detail["initial_states"][seed_idx]
    terrain = np.array(state["grid"], dtype=int)
    settlements = state["settlements"]

    gt = None
    ap = rd / f"analysis_seed_{seed_idx}.json"
    if ap.exists():
        analysis = json.loads(ap.read_text())
        gt = np.array(analysis["ground_truth"], dtype=np.float64)

    return RoundData(round_name, seed_idx, terrain, settlements, gt)


def load_all_rounds(seeds_per_round: int = 5) -> list[RoundData]:
    """Load all rounds with ground truth."""
    rounds = []
    for rname in ALL_ROUNDS:
        for si in range(seeds_per_round):
            rd = load_round(rname, si)
            if rd is not None and rd.ground_truth is not None:
                rounds.append(rd)
    return rounds


def load_observations(round_name: str) -> list[dict]:
    """Load API observations for a round (from the rounds data dir)."""
    rid = ROUND_IDS.get(round_name)
    if rid is None:
        return []
    obs_dir = OBS_DIR / rid
    if not obs_dir.exists():
        return []
    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    observations = []
    for f in obs_files:
        observations.append(json.loads(f.read_text()))
    return observations


if __name__ == "__main__":
    # Quick validation
    rounds = load_all_rounds(seeds_per_round=1)
    print(f"Loaded {len(rounds)} rounds")
    for rd in rounds:
        n_sett = len(rd.settlements)
        n_dynamic = rd.cell_features["is_buildable"].sum() - rd.cell_features["is_settlement"].sum()
        gt_sett = rd.ground_truth[:, :, 1].mean() * 100 if rd.ground_truth is not None else 0
        print(f"  {rd.round_name} seed{rd.seed_idx}: "
              f"{n_sett} settlements, {n_dynamic} expansion cells, "
              f"GT sett avg={gt_sett:.2f}%")
