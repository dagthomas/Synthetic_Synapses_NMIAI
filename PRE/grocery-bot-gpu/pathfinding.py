"""BFS distance maps and pathfinding utilities.

Precomputed once per map, shared across all beam search states.
"""
import numpy as np
from collections import deque
from game_engine import CELL_FLOOR, CELL_DROPOFF, MapState


def bfs_distance(grid, source):
    """BFS from source cell, returns 2D distance array. -1 = unreachable."""
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=np.int16)
    sx, sy = source
    if grid[sy, sx] != CELL_FLOOR and grid[sy, sx] != CELL_DROPOFF:
        return dist
    dist[sy, sx] = 0
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        d = dist[y, x]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and dist[ny, nx] < 0:
                cell = grid[ny, nx]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    dist[ny, nx] = d + 1
                    queue.append((nx, ny))
    return dist


def bfs_first_step(grid, source, target):
    """BFS from source to target, return the first step direction (action ID 1-4) or 0 for wait."""
    if source == target:
        return 0
    h, w = grid.shape
    sx, sy = source
    tx, ty = target
    # BFS from target to source (reverse) so we can read first step
    dist = np.full((h, w), -1, dtype=np.int16)
    dist[ty, tx] = 0
    queue = deque([(tx, ty)])
    while queue:
        x, y = queue.popleft()
        d = dist[y, x]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and dist[ny, nx] < 0:
                cell = grid[ny, nx]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    dist[ny, nx] = d + 1
                    queue.append((nx, ny))
    if dist[sy, sx] < 0:
        return 0  # unreachable
    # Find the neighbor of source with minimum distance
    best_d = dist[sy, sx]
    best_act = 0
    # Map: (dx,dy) -> action ID
    dir_to_act = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}  # up, down, left, right
    for (dx, dy), act in dir_to_act.items():
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < w and 0 <= ny < h:
            if dist[ny, nx] >= 0 and dist[ny, nx] < best_d:
                best_d = dist[ny, nx]
                best_act = act
    return best_act


def precompute_all_distances(map_state):
    """Precompute BFS distances from every walkable cell.

    Returns dict: (x,y) -> dist_map[h,w] (numpy int16 array).

    Uses GPU-accelerated PrecomputedTables with disk caching when available.
    Falls back to per-cell BFS if precompute module is not available.
    """
    try:
        from precompute import PrecomputedTables
        tables = PrecomputedTables.get(map_state)
        return tables.as_dist_maps()
    except ImportError:
        pass

    # Fallback: original per-cell BFS
    grid = map_state.grid
    h, w = grid.shape
    dist_maps = {}
    for y in range(h):
        for x in range(w):
            if grid[y, x] == CELL_FLOOR or grid[y, x] == CELL_DROPOFF:
                dist_maps[(x, y)] = bfs_distance(grid, (x, y))
    return dist_maps


def precompute_item_pickup_cells(map_state):
    """For each item index, list walkable cells from which pickup is possible (manhattan dist 1).

    Returns dict: item_idx -> list of (x, y, first_step_from_dropoff_dist).
    """
    return map_state.item_adjacencies


def get_distance(dist_maps, source, target):
    """Get BFS distance from source to target using precomputed maps."""
    dm = dist_maps.get(source)
    if dm is None:
        return 9999
    d = dm[target[1], target[0]]
    return d if d >= 0 else 9999


def get_first_step(dist_maps, source, target):
    """Get first step action ID (1-4) from source toward target, 0 if at target or unreachable."""
    if source == target:
        return 0
    # Use target's distance map to find neighbor of source with minimum distance
    dm = dist_maps.get(target)
    if dm is None:
        return 0
    sx, sy = source
    if dm[sy, sx] < 0:
        return 0

    best_d = dm[sy, sx]
    best_act = 0
    for dx, dy, act in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < dm.shape[1] and 0 <= ny < dm.shape[0]:
            if dm[ny, nx] >= 0 and dm[ny, nx] < best_d:
                best_d = dm[ny, nx]
                best_act = act
    return best_act


def get_nearest_item_cell(dist_maps, source, item_idx, map_state):
    """Find the nearest walkable cell adjacent to item_idx from source.

    Returns (cell_x, cell_y, distance) or None if unreachable.
    """
    adj = map_state.item_adjacencies.get(item_idx, [])
    best = None
    best_d = 9999
    for (cx, cy) in adj:
        d = get_distance(dist_maps, source, (cx, cy))
        if d < best_d:
            best_d = d
            best = (cx, cy, d)
    return best
