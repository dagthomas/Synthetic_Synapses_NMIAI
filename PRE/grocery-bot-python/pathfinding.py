"""Pathfinding - optimized with flat arrays and tuples."""
from __future__ import annotations
from collections import deque

from types_ import (
    FLOOR, WALL, SHELF, DROPOFF,
    UP, DOWN, LEFT, RIGHT,
    DIR_DX, DIR_DY,
    UNREACHABLE,
)

# Reusable direction offsets
_DDX = (0, 0, -1, 1)
_DDY = (-1, 1, 0, 0)
_DIRS = (UP, DOWN, LEFT, RIGHT)
_ADJ_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def bfs_dist_map(grid, w, h, sx, sy):
    """BFS distance map from (sx,sy). Returns flat list[int] of size w*h."""
    size = w * h
    dm = [UNREACHABLE] * size

    if sx < 0 or sy < 0 or sx >= w or sy >= h:
        return dm
    idx = sy * w + sx
    cell = grid[idx]
    if cell == WALL or cell == SHELF:
        return dm

    dm[idx] = 0
    queue = deque()
    queue.append(idx)

    while queue:
        ci = queue.popleft()
        cy, cx = divmod(ci, w)
        # Optimized: inline divmod result
        cy = ci // w
        cx = ci - cy * w
        cd = dm[ci]
        nd = cd + 1
        for i in range(4):
            nx = cx + _DDX[i]
            ny = cy + _DDY[i]
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            ni = ny * w + nx
            if dm[ni] != UNREACHABLE:
                continue
            c = grid[ni]
            if c == WALL or c == SHELF:
                continue
            dm[ni] = nd
            queue.append(ni)

    return dm


def bfs_path(grid, w, h, sx, sy, tx, ty, bot_id, bot_positions, bot_count):
    """BFS with first-step collision avoidance. Returns (dist, first_dir or -1)."""
    if sx == tx and sy == ty:
        return (0, -1)

    size = w * h
    # Use bytearray as visited (faster than set for small grids)
    visited = bytearray(size)
    si = sy * w + sx
    visited[si] = 1

    # Queue: flat list of (index, dist, first_dir)
    queue = deque()
    queue.append((si, 0, -1))

    while queue:
        ci, cd, cfd = queue.popleft()
        cy = ci // w
        cx = ci - cy * w
        for i in range(4):
            nx = cx + _DDX[i]
            ny = cy + _DDY[i]
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            ni = ny * w + nx
            if visited[ni]:
                continue

            cell = grid[ni]
            is_target = (nx == tx and ny == ty)
            if not is_target and (cell == WALL or cell == SHELF):
                continue

            # First-step collision avoidance
            if cd == 0:
                blocked = False
                for bi in range(bot_count):
                    if bi == bot_id:
                        continue
                    bx, by = bot_positions[bi]
                    if bx == nx and by == ny:
                        blocked = True
                        break
                if blocked:
                    continue

            visited[ni] = 1
            first = cfd if cfd >= 0 else _DIRS[i]
            if is_target:
                return (cd + 1, first)
            queue.append((ni, cd + 1, first))

    # Fallback: greedy direction
    return (UNREACHABLE, _safe_greedy(grid, w, h, sx, sy, tx, ty))


def _safe_greedy(grid, w, h, fx, fy, tx, ty):
    """Greedy direction toward target."""
    ddx = tx - fx
    ddy = ty - fy

    if abs(ddx) > abs(ddy):
        ordered = (RIGHT, DOWN, UP, LEFT) if ddx > 0 else (LEFT, DOWN, UP, RIGHT)
    elif ddy > 0:
        ordered = (DOWN, RIGHT, LEFT, UP)
    elif ddy < 0:
        ordered = (UP, RIGHT, LEFT, DOWN)
    else:
        ordered = (RIGHT, DOWN, LEFT, UP)

    for d in ordered:
        nx = fx + DIR_DX[d]
        ny = fy + DIR_DY[d]
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            continue
        cell = grid[ny * w + nx]
        if cell == FLOOR or cell == DROPOFF:
            return d
    return -1


def find_best_adj(grid, w, h, ix, iy, dm):
    """Find best adjacent floor cell to item at (ix,iy) using distance map dm.
    Returns (ax, ay) or None."""
    best = None
    best_d = UNREACHABLE

    for ox, oy in _ADJ_OFFSETS:
        nx, ny = ix + ox, iy + oy
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            continue
        cell = grid[ny * w + nx]
        if cell == FLOOR or cell == DROPOFF:
            d = dm[ny * w + nx]
            if d < best_d:
                best_d = d
                best = (nx, ny)

    return best
