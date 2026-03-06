"""BFS pathfinding for the grocery store grid."""

from collections import deque


# Cache shelf positions — they never change during a game.
# Populated on round 0 when all shelves have items.
_shelf_cache = set()


def build_blocked_set(state, exclude_bot_id=None):
    """Build a set of blocked positions (walls + shelves + other bots).

    Shelves are NOT in grid.walls but are impassable.
    We track all item positions we've ever seen as permanent shelves.
    """
    global _shelf_cache
    blocked = set()

    # Walls
    for w in state["grid"]["walls"]:
        blocked.add((w[0], w[1]))

    # Shelves — items sit on shelves. Even after pickup, the shelf stays.
    # Accumulate all item positions into the cache.
    for item in state["items"]:
        _shelf_cache.add((item["position"][0], item["position"][1]))
    blocked.update(_shelf_cache)

    # Other bot positions (not ourselves)
    for bot in state["bots"]:
        if bot["id"] != exclude_bot_id:
            blocked.add((bot["position"][0], bot["position"][1]))

    return blocked


def reset_shelf_cache():
    """Reset shelf cache between games."""
    global _shelf_cache
    _shelf_cache = set()


def bfs(start, goal, blocked, width, height):
    """BFS from start to goal, avoiding blocked cells.

    Returns list of positions from start to goal (inclusive),
    or None if no path exists.
    """
    sx, sy = start
    gx, gy = goal

    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    visited = {(sx, sy)}
    queue = deque()
    queue.append(((sx, sy), [(sx, sy)]))

    while queue:
        (cx, cy), path = queue.popleft()

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in blocked:
                continue
            if (nx, ny) in visited:
                continue

            new_path = path + [(nx, ny)]

            if (nx, ny) == (gx, gy):
                return new_path

            visited.add((nx, ny))
            queue.append(((nx, ny), new_path))

    return None


def bfs_adjacent(start, target, blocked, width, height):
    """BFS to any cell adjacent to target (for picking up from shelves).

    Shelves are not walkable, so we pathfind to a neighboring walkable cell.
    Returns (path, adjacent_pos) or (None, None).
    """
    tx, ty = target
    adjacent_cells = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ax, ay = tx + dx, ty + dy
        if 0 <= ax < width and 0 <= ay < height and (ax, ay) not in blocked:
            adjacent_cells.append((ax, ay))

    if not adjacent_cells:
        return None, None

    # If we're already adjacent, return immediately
    sx, sy = start
    if (sx, sy) in adjacent_cells:
        return [(sx, sy)], (sx, sy)

    # BFS to nearest adjacent cell
    best_path = None
    best_pos = None
    for adj in adjacent_cells:
        path = bfs(start, adj, blocked, width, height)
        if path and (best_path is None or len(path) < len(best_path)):
            best_path = path
            best_pos = adj

    return best_path, best_pos


def path_to_action(current, next_pos):
    """Convert a step in a path to a move action string."""
    cx, cy = current
    nx, ny = next_pos
    dx, dy = nx - cx, ny - cy

    if dx == 1:
        return "move_right"
    if dx == -1:
        return "move_left"
    if dy == 1:
        return "move_down"
    if dy == -1:
        return "move_up"

    return "wait"
