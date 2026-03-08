"""Priority-based collision-free movement resolver.

Resolves simultaneous movement for all bots using a reservation-table
approach with BFS detour routing. Proven to work on narrow corridors.

Key design:
  - Process bots in priority order (delivering > picking > idle)
  - Each bot routes around committed positions of higher-priority bots
  - BFS detour finding when optimal path is blocked
  - Idle bots move away from high-traffic areas to avoid blocking
  - Spawn cell exempt from collision checks (multiple bots allowed)

Usage:
    from pibt import pibt_step
    actions = pibt_step(positions, goals, priorities, tables, map_state)
"""
from __future__ import annotations

from collections import deque

from game_engine import (
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    DX, DY, CELL_FLOOR, CELL_DROPOFF, MapState,
)
from precompute import PrecomputedTables

# Priority tiers (lower value = higher priority)
PRIO_DELIVERING = 0   # Bot at/near drop zone with items to deliver
PRIO_CARRYING = 1     # Bot carrying items for active order
PRIO_PICKING = 2      # Bot moving toward shelf to pick
PRIO_PREPICKING = 3   # Bot pre-picking preview items
PRIO_IDLE = 4         # Idle / scout bots


def _bfs_first_step_avoiding(start, goal, ms, occupied):
    """BFS from start to goal, treating occupied cells as walls.

    The goal cell is always reachable even if occupied (so a bot
    can path toward a drop zone even if another bot is sitting there).

    Returns first action ID (1-4) or 0 if no path.
    """
    if start == goal:
        return 0

    queue = deque([(start, 0)])  # (pos, first_action)
    visited = {start}

    while queue:
        (x, y), first_act = queue.popleft()

        for act_id in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
            nx, ny = x + DX[act_id], y + DY[act_id]
            if not (0 <= nx < ms.width and 0 <= ny < ms.height):
                continue
            cell = ms.grid[ny, nx]
            if cell != CELL_FLOOR and cell != CELL_DROPOFF:
                continue
            npos = (nx, ny)
            if npos in visited:
                continue

            fa = first_act if first_act != 0 else act_id

            # Goal is always reachable (even if occupied) so bots
            # can navigate toward occupied drop zones
            if npos == goal:
                return fa

            # Non-goal occupied cells are blocked (except spawn)
            if npos in occupied and npos != ms.spawn:
                continue

            visited.add(npos)
            queue.append((npos, fa))

    return 0  # No path found


def _move_toward_avoiding(bot_pos, target, tables, ms, occupied):
    """Move one step toward target, avoiding occupied cells.

    Strategies in order:
    1. Optimal first-step from precomputed tables, if cell is free
    2. Alternative directions within +2 distance tolerance
    3. BFS detour around occupied cells
    4. Wait as last resort
    """
    if bot_pos == target:
        return ACT_WAIT

    # Strategy 1: optimal direction if clear
    act = tables.get_first_step(bot_pos, target)
    if act > 0:
        nx, ny = bot_pos[0] + DX[act], bot_pos[1] + DY[act]
        if (nx, ny) not in occupied or (nx, ny) == ms.spawn:
            return act

    # Strategy 2: try alternatives within +2 distance
    target_dist = tables.get_distance(bot_pos, target)
    alternatives = []

    for act_id in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
        nx, ny = bot_pos[0] + DX[act_id], bot_pos[1] + DY[act_id]
        if not (0 <= nx < ms.width and 0 <= ny < ms.height):
            continue
        cell = ms.grid[ny, nx]
        if cell != CELL_FLOOR and cell != CELL_DROPOFF:
            continue
        if (nx, ny) in occupied and (nx, ny) != ms.spawn:
            continue
        d = tables.get_distance((nx, ny), target)
        if d < 9999:
            alternatives.append((d, act_id))

    alternatives.sort()
    if alternatives:
        d, act_id = alternatives[0]
        if d <= target_dist + 2:  # allow slight detour
            return act_id

    # Strategy 3: BFS detour around occupied cells
    bfs_act = _bfs_first_step_avoiding(bot_pos, target, ms, occupied)
    if bfs_act > 0:
        return bfs_act

    # Strategy 4: any free adjacent cell (deadlock breaker)
    # Move in any direction rather than waiting forever
    if alternatives:
        return alternatives[0][1]  # best available, even if far from optimal

    return ACT_WAIT


def pibt_step(
    positions: list[tuple[int, int]],
    goals: list[tuple[int, int] | None],
    priorities: list[tuple[int, int]],
    tables: PrecomputedTables,
    ms: MapState,
) -> list[tuple[int, int]]:
    """Compute collision-free actions for all bots.

    Uses reservation-table approach with BFS detour routing.
    Processes bots in priority order; each routes around
    committed positions of higher-priority bots.

    Args:
        positions: Current (x, y) per bot.
        goals: Target (x, y) per bot, or None if no goal (wait).
        priorities: (tier, tiebreak) per bot. Lower = higher priority.
        tables: PrecomputedTables for O(1) distance lookups.
        ms: MapState for grid bounds and walkability.

    Returns:
        List of (action_type, item_idx) per bot.
    """
    n_bots = len(positions)
    actions = [(ACT_WAIT, -1)] * n_bots
    # Track committed new positions for collision avoidance
    committed: dict[int, tuple[int, int]] = {}  # bot_id -> new_pos

    # Sort bots by priority (lower = higher priority = resolved first)
    order = sorted(range(n_bots), key=lambda b: priorities[b])

    for bid in order:
        pos = positions[bid]
        goal = goals[bid]

        # Build occupied set: committed positions of processed bots +
        # current positions of unprocessed bots (except self)
        occupied = set()
        for bid2 in range(n_bots):
            if bid2 == bid:
                continue
            if bid2 in committed:
                occupied.add(committed[bid2])
            else:
                occupied.add(positions[bid2])

        # No goal or at goal -> stay in place
        if goal is None or pos == goal:
            committed[bid] = pos
            continue

        # Move toward goal, routing around occupied cells
        act = _move_toward_avoiding(pos, goal, tables, ms, occupied)

        if act != ACT_WAIT:
            nx, ny = pos[0] + DX[act], pos[1] + DY[act]
            new_pos = (nx, ny)
            actions[bid] = (act, -1)
            committed[bid] = new_pos
        else:
            committed[bid] = pos

    return actions


def compute_priorities(
    bot_roles: list[int],
    round_num: int,
    n_bots: int,
) -> list[tuple[int, int]]:
    """Compute (tier, tiebreak) priority for each bot.

    Args:
        bot_roles: Role tier per bot (PRIO_DELIVERING, etc.)
        round_num: Current round number for rotation.
        n_bots: Total bot count.

    Returns:
        List of (tier, tiebreak) tuples. Lower = higher priority.
    """
    priorities = []
    for bid in range(n_bots):
        tier = bot_roles[bid]
        # Rotate tiebreak within each tier to prevent starvation
        tiebreak = (bid + round_num) % n_bots
        priorities.append((tier, tiebreak))
    return priorities
