"""Enhanced PIBT pathfinder with push chains and follow convoys.

Extends NightmarePathfinderV2 with:
1. Push mechanism: high-priority bots force lower-priority bots aside
2. Follow chains: same-direction bots form convoys in narrow aisles
3. Longer lookahead (12 steps vs 5) with 5-step reservation
4. Numpy-based reservation table for O(1) lookup

Drop-in replacement for V2 — same plan_all() interface.
"""
from __future__ import annotations

import heapq

import numpy as np

from game_engine import (
    MapState, CELL_FLOOR, CELL_DROPOFF,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    DX, DY,
)
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


def build_walkable(ms: MapState) -> set[tuple[int, int]]:
    w = set()
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                w.add((x, y))
    return w


class ReservationTable:
    """Fast numpy-based 3D reservation table: [T, H, W]."""

    def __init__(self, width: int, height: int, max_t: int):
        self.width = width
        self.height = height
        self.max_t = max_t
        # -1 = free, >=0 = bot_id that reserved it
        self.table = np.full((max_t, height, width), -1, dtype=np.int16)
        # Edge reservations for swap prevention: (from_x, from_y, to_x, to_y, t)
        self._edges: set[tuple[int, int, int, int, int]] = set()

    def reserve(self, x: int, y: int, t: int, bid: int):
        if 0 <= t < self.max_t and 0 <= x < self.width and 0 <= y < self.height:
            self.table[t, y, x] = bid

    def reserve_edge(self, fx: int, fy: int, tx: int, ty: int, t: int):
        self._edges.add((fx, fy, tx, ty, t))

    def is_reserved(self, x: int, y: int, t: int) -> bool:
        if 0 <= t < self.max_t and 0 <= x < self.width and 0 <= y < self.height:
            return self.table[t, y, x] >= 0
        return t >= self.max_t  # Treat beyond-window as reserved

    def reserved_by(self, x: int, y: int, t: int) -> int:
        """Return bot ID that reserved (x,y,t), or -1 if free."""
        if 0 <= t < self.max_t and 0 <= x < self.width and 0 <= y < self.height:
            return int(self.table[t, y, x])
        return -1

    def has_swap_conflict(self, fx: int, fy: int, tx: int, ty: int, t: int) -> bool:
        return (tx, ty, fx, fy, t) in self._edges


class NightmarePathfinderV3:
    """Enhanced PIBT with push chains and follow convoys."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 traffic: TrafficRules, congestion: CongestionMap,
                 lookahead: int = 12, reserve_steps: int = 5):
        self.ms = ms
        self.walkable = build_walkable(ms)
        self.tables = tables
        self.traffic = traffic
        self.congestion = congestion
        self.spawn = ms.spawn
        self.drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
        self.lookahead = lookahead
        self.reserve_steps = reserve_steps

        # Precompute neighbor lists
        self._neighbors: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for pos in self.walkable:
            nbrs = []
            for act in MOVES:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in self.walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

        # Detect narrow aisle columns for convoy formation
        self._narrow_aisles: set[int] = set()
        self._detect_narrow_aisles()

    def _detect_narrow_aisles(self):
        """Identify narrow vertical aisle columns (walkable with shelves on sides)."""
        corridor_ys = {1, self.ms.height // 2, self.ms.height - 3}
        for x in range(1, self.ms.width - 1):
            is_aisle = True
            walkable_count = 0
            for y in range(2, self.ms.height - 2):
                if y in corridor_ys:
                    continue
                if (x, y) in self.walkable:
                    walkable_count += 1
                else:
                    is_aisle = False
                    break
            if is_aisle and walkable_count >= 3:
                self._narrow_aisles.add(x)

    def plan_all(self, bot_positions: dict[int, tuple[int, int]],
                 goals: dict[int, tuple[int, int]],
                 urgency_order: list[int],
                 goal_types: dict[int, str] | None = None) -> dict[int, int]:
        """Plan actions using enhanced PIBT with push/follow.

        Same interface as V2 for drop-in replacement.
        """
        max_t = self.lookahead + 2
        res = ReservationTable(self.ms.width, self.ms.height, max_t)
        actions: dict[int, int] = {}
        gt = goal_types or {}

        # Track planned paths for convoy detection
        planned_paths: dict[int, list[tuple[int, int]]] = {}
        # Track which bots have been planned
        planned_bots: set[int] = set()

        for bid in urgency_order:
            pos = bot_positions.get(bid)
            goal = goals.get(bid)

            if pos is None:
                actions[bid] = ACT_WAIT
                continue

            if goal is None or pos == goal:
                # At goal — reserve position
                if pos != self.spawn:
                    for t in range(min(self.reserve_steps, max_t)):
                        res.reserve(pos[0], pos[1], t, bid)
                actions[bid] = ACT_WAIT
                planned_bots.add(bid)
                continue

            # Try push if blocked by lower-priority bot
            first_action, path = self._plan_with_push(
                bid, pos, goal, res, urgency_order, bot_positions,
                actions, planned_bots, planned_paths)

            actions[bid] = first_action
            planned_bots.add(bid)

            # Reserve the planned path
            if path:
                planned_paths[bid] = path
                for t in range(min(len(path), self.reserve_steps)):
                    p = path[t]
                    if p != self.spawn:
                        res.reserve(p[0], p[1], t, bid)
                    if t > 0:
                        prev = path[t - 1]
                        res.reserve_edge(prev[0], prev[1], p[0], p[1], t)
            else:
                if pos != self.spawn:
                    res.reserve(pos[0], pos[1], 0, bid)
                    res.reserve(pos[0], pos[1], 1, bid)

        return actions

    def _plan_with_push(self, bid: int, pos: tuple[int, int],
                        goal: tuple[int, int],
                        res: ReservationTable,
                        urgency_order: list[int],
                        bot_positions: dict[int, tuple[int, int]],
                        actions: dict[int, int],
                        planned_bots: set[int],
                        planned_paths: dict[int, list[tuple[int, int]]],
                        ) -> tuple[int, list[tuple[int, int]]]:
        """Plan with push: if best path is blocked by lower-priority bot, push it.

        Returns (first_action, path).
        """
        # First try standard A*
        first_action, path = self._windowed_astar(pos, goal, res)

        if first_action != ACT_WAIT or pos == goal:
            return first_action, path

        # Path is blocked. Try push mechanism.
        # Check each neighbor to find blocking bot
        urgency_rank = {b: i for i, b in enumerate(urgency_order)}
        my_rank = urgency_rank.get(bid, 999)

        for act, dest in self._neighbors.get(pos, []):
            if dest == self.spawn:
                continue

            # Check if dest is reserved at t=1
            blocking_bid = res.reserved_by(dest[0], dest[1], 1)
            if blocking_bid < 0:
                continue

            # Only push lower-priority (higher rank number) bots
            blocker_rank = urgency_rank.get(blocking_bid, 999)
            if blocker_rank <= my_rank:
                continue  # Can't push higher priority

            # Check if blocking bot has already been planned
            if blocking_bid not in planned_bots:
                continue  # Will be planned later, can't push yet

            # Try to push the blocker away
            blocker_pos = bot_positions.get(blocking_bid)
            if blocker_pos is None:
                continue

            # Find an alternative cell for the blocker
            pushed = False
            for push_act, push_dest in self._neighbors.get(blocker_pos, []):
                if push_dest == pos:
                    continue  # Don't push into our position
                if push_dest == self.spawn:
                    pushed = True  # Spawn is always free
                    break
                if not res.is_reserved(push_dest[0], push_dest[1], 1):
                    pushed = True
                    # Update blocker's action
                    actions[blocking_bid] = push_act
                    # Update reservations
                    res.reserve(push_dest[0], push_dest[1], 1, blocking_bid)
                    # Clear blocker's old reservation at t=1
                    if 0 <= 1 < res.max_t:
                        res.table[1, blocker_pos[1], blocker_pos[0]] = -1
                    break

            if pushed:
                # Now our desired cell should be free, re-plan
                return self._windowed_astar(pos, goal, res)

        # Push failed, try follow chain
        return self._try_follow(bid, pos, goal, res, planned_paths, urgency_order)

    def _try_follow(self, bid: int, pos: tuple[int, int],
                    goal: tuple[int, int],
                    res: ReservationTable,
                    planned_paths: dict[int, list[tuple[int, int]]],
                    urgency_order: list[int],
                    ) -> tuple[int, list[tuple[int, int]]]:
        """Try to follow a same-direction bot in a narrow aisle (convoy)."""
        if pos[0] not in self._narrow_aisles:
            return ACT_WAIT, [pos]

        # Check if there's a higher-priority bot moving in same direction ahead
        my_dir_y = 1 if goal[1] > pos[1] else (-1 if goal[1] < pos[1] else 0)
        if my_dir_y == 0:
            return ACT_WAIT, [pos]

        # Look for a bot ahead of us in the same column moving the same direction
        ahead_pos = (pos[0], pos[1] + my_dir_y)
        if ahead_pos not in self.walkable:
            return ACT_WAIT, [pos]

        # Check if the cell one step ahead of the bot ahead will be free at t=2
        # (meaning the ahead bot is moving away)
        two_ahead = (pos[0], pos[1] + 2 * my_dir_y)
        if two_ahead in self.walkable:
            # Check if ahead cell becomes free at t=1 (bot moved to two_ahead)
            for leader_bid, leader_path in planned_paths.items():
                if len(leader_path) >= 2:
                    if leader_path[0] == ahead_pos and leader_path[1] == two_ahead:
                        # Leader is moving away! Follow into their vacated spot
                        move_act = ACT_MOVE_DOWN if my_dir_y > 0 else ACT_MOVE_UP
                        if not res.is_reserved(ahead_pos[0], ahead_pos[1], 1):
                            return move_act, [pos, ahead_pos]

        return ACT_WAIT, [pos]

    def _windowed_astar(self, start: tuple[int, int], goal: tuple[int, int],
                        res: ReservationTable,
                        ) -> tuple[int, list[tuple[int, int]]]:
        """Windowed A* in spacetime, using numpy reservation table.

        Returns (first_action, path).
        """
        max_t = self.lookahead
        # (f, g, t, x, y)
        open_list: list[tuple[float, float, int, int, int]] = []
        came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
        g_score: dict[tuple[int, int, int], float] = {}

        start_state = (start[0], start[1], 0)
        h0 = self.tables.get_distance(start, goal)
        heapq.heappush(open_list, (h0, 0.0, 0, start[0], start[1]))
        came_from[start_state] = None
        g_score[start_state] = 0.0

        best_h = h0
        best_state = start_state

        while open_list:
            f, g, t, x, y = heapq.heappop(open_list)
            pos = (x, y)
            state = (x, y, t)

            if g > g_score.get(state, float('inf')):
                continue

            if pos == goal:
                return self._extract(state, came_from)

            h = self.tables.get_distance(pos, goal)
            if h < best_h or (h == best_h and g < g_score.get(best_state, float('inf'))):
                best_h = h
                best_state = state

            if t >= max_t:
                continue

            next_t = t + 1

            # Wait (cost 1.5 to discourage waiting)
            self._try_expand(pos, pos, 1.5, next_t, x, y,
                             goal, res, open_list, g_score, came_from, state, g)

            # Moves (cost 1.0)
            for act, dest in self._neighbors.get(pos, []):
                # Add congestion cost
                cong_cost = self.congestion.get_penalty(dest) * 0.15
                self._try_expand(pos, dest, 1.0 + cong_cost, next_t, x, y,
                                 goal, res, open_list, g_score, came_from, state, g)

        return self._extract(best_state, came_from)

    def _try_expand(self, from_pos, dest, step_cost, next_t, x, y,
                    goal, res: ReservationTable, open_list, g_score,
                    came_from, parent_state, parent_g):
        next_state = (dest[0], dest[1], next_t)

        # Collision check via numpy table
        if dest != self.spawn and res.is_reserved(dest[0], dest[1], next_t):
            return
        # Swap check
        if from_pos != dest and res.has_swap_conflict(x, y, dest[0], dest[1], next_t):
            return

        new_g = parent_g + step_cost
        if next_state in g_score and g_score[next_state] <= new_g:
            return

        g_score[next_state] = new_g
        h = self.tables.get_distance(dest, goal)
        heapq.heappush(open_list, (new_g + h, new_g, next_t,
                                    dest[0], dest[1]))
        came_from[next_state] = parent_state

    def _extract(self, state, came_from):
        """Extract first action and full path."""
        path = []
        s = state
        while s is not None:
            path.append((s[0], s[1]))
            s = came_from.get(s)
        path.reverse()

        if len(path) <= 1:
            return ACT_WAIT, path

        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        if dx == 0 and dy == 0:
            return ACT_WAIT, path
        for act in MOVES:
            if DX[act] == dx and DY[act] == dy:
                return act, path
        return ACT_WAIT, path
