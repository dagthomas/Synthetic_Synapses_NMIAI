"""Smart viewport selection — focus queries on dynamic cells.

Strategy:
  Phase 1 (35 queries): 7 viewports/seed covering the most settlement-dense areas
  Phase 2 (15 queries): Re-query the 3 most dynamic viewports per seed for 2nd observations

vs Current (explore.py):
  Phase 1 (45 queries): 9 viewports/seed in fixed 3x3 grid (100% coverage)
  Phase 2 (5 queries): 1 adaptive per seed

The tradeoff: ~85% coverage vs 100%, but 3x more 2nd observations on dynamic cells.
"""
import numpy as np

from config import MAP_H, MAP_W


def score_viewport(grid: np.ndarray, settlements: list[dict],
                   x: int, y: int, w: int = 15, h: int = 15) -> float:
    """Score a viewport position by how many dynamic cells it covers.

    Dynamic cells = non-ocean, non-mountain, within distance 8 of a settlement.
    Higher score = more valuable to observe.
    """
    sett_pos = [(s["y"], s["x"]) for s in settlements if s.get("alive", True)]
    score = 0.0

    for vy in range(y, min(y + h, MAP_H)):
        for vx in range(x, min(x + w, MAP_W)):
            code = int(grid[vy, vx])
            if code == 10 or code == 5:  # ocean/mountain = 0 value
                continue

            # Distance to nearest settlement
            if sett_pos:
                min_dist = min(abs(vy - sy) + abs(vx - sx) for sy, sx in sett_pos)
            else:
                min_dist = 99

            # Score: settlement cells most valuable, nearby cells next
            if code in (1, 2):  # settlement/port
                score += 5.0
            elif min_dist <= 2:
                score += 3.0
            elif min_dist <= 5:
                score += 2.0
            elif min_dist <= 8:
                score += 1.0
            else:
                score += 0.1  # far from settlements, low value

    return score


def select_viewports(grid: np.ndarray, settlements: list[dict],
                     n_viewports: int = 7) -> list[dict]:
    """Select the n best non-overlapping viewport positions.

    Returns list of {x, y, w, h, score} dicts.
    """
    # Score all possible positions
    candidates = []
    for y in range(0, MAP_H - 14):
        for x in range(0, MAP_W - 14):
            score = score_viewport(grid, settlements, x, y)
            candidates.append({"x": x, "y": y, "w": 15, "h": 15, "score": score})

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Greedy selection with overlap constraint (<50%)
    selected = []
    for c in candidates:
        if len(selected) >= n_viewports:
            break
        # Check overlap with already selected
        overlaps = False
        for s in selected:
            # Compute intersection
            ix1 = max(c["x"], s["x"])
            iy1 = max(c["y"], s["y"])
            ix2 = min(c["x"] + c["w"], s["x"] + s["w"])
            iy2 = min(c["y"] + c["h"], s["y"] + s["h"])
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                min_area = min(c["w"] * c["h"], s["w"] * s["h"])
                if intersection / min_area > 0.50:
                    overlaps = True
                    break
        if not overlaps:
            selected.append(c)

    return selected


def plan_smart_queries(detail: dict, budget: int = 50) -> list[dict]:
    """Plan all queries for a round using smart viewport selection.

    Returns list of {seed_index, x, y, w, h, phase} dicts.
    """
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Phase 1: Coverage — 7 viewports per seed
    coverage_per_seed = max(5, (budget - seeds_count * 3) // seeds_count)
    adaptive_budget = budget - coverage_per_seed * seeds_count

    queries = []
    seed_viewports = {}  # seed -> list of viewports with scores

    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        grid = np.array(state["grid"], dtype=int)
        settlements = state["settlements"]

        viewports = select_viewports(grid, settlements, n_viewports=coverage_per_seed)
        seed_viewports[seed_idx] = viewports

        for vp in viewports:
            queries.append({
                "seed_index": seed_idx,
                "x": vp["x"], "y": vp["y"],
                "w": vp["w"], "h": vp["h"],
                "phase": "coverage",
                "score": vp["score"],
            })

    # Phase 2: Double-sample — re-query the most dynamic viewports
    # Collect all viewports with scores, pick top N
    all_viewports = []
    for seed_idx, vps in seed_viewports.items():
        for vp in vps:
            all_viewports.append({
                "seed_index": seed_idx,
                "x": vp["x"], "y": vp["y"],
                "w": vp["w"], "h": vp["h"],
                "phase": "double_sample",
                "score": vp["score"],
            })

    all_viewports.sort(key=lambda v: v["score"], reverse=True)

    for vp in all_viewports[:adaptive_budget]:
        queries.append(vp)

    return queries


def get_coverage_stats(queries: list[dict], detail: dict) -> dict:
    """Compute coverage statistics for a query plan."""
    seeds_count = detail["seeds_count"]
    coverage = {}
    obs_count = {}

    for seed_idx in range(seeds_count):
        covered = np.zeros((MAP_H, MAP_W), dtype=int)
        for q in queries:
            if q["seed_index"] == seed_idx:
                for y in range(q["y"], min(q["y"] + q["h"], MAP_H)):
                    for x in range(q["x"], min(q["x"] + q["w"], MAP_W)):
                        covered[y, x] += 1
        coverage[seed_idx] = (covered > 0).sum() / (MAP_H * MAP_W)
        obs_count[seed_idx] = covered.sum() / max((covered > 0).sum(), 1)

    return {
        "avg_coverage": np.mean(list(coverage.values())),
        "min_coverage": min(coverage.values()),
        "avg_obs_per_cell": np.mean(list(obs_count.values())),
        "per_seed_coverage": coverage,
    }
