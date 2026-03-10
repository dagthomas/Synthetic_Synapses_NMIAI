"""Multi-step prioritized planning pathfinder for nightmare mode.

Uses windowed spacetime A*: each bot plans up to K steps ahead in spacetime,
avoiding cells reserved by higher-priority bots. When the goal can't be
reached within K steps, picks the best partial path (closest to goal).

Drop-in replacement for NightmarePathfinder — same plan_all() interface.
"""
from __future__ import annotations

import heapq
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


class NightmarePathfinderV2:
    """Prioritized planning with windowed spacetime A*."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 traffic: TrafficRules, congestion: CongestionMap,
                 lookahead: int = 5):
        self.ms = ms
        self.walkable = build_walkable(ms)
        self.tables = tables
        self.traffic = traffic
        self.congestion = congestion
        self.spawn = ms.spawn
        self.drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
        self.lookahead = lookahead
        # Precompute neighbor lists
        self._neighbors: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for pos in self.walkable:
            nbrs = []
            for act in MOVES:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in self.walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

    def plan_all(self, bot_positions: dict[int, tuple[int, int]],
                 goals: dict[int, tuple[int, int]],
                 urgency_order: list[int],
                 goal_types: dict[int, str] | None = None) -> dict[int, int]:
        """Plan actions using prioritized windowed spacetime A*."""
        # Reservation table: (x, y, t) reserved by higher-priority bot
        reserved: set[tuple[int, int, int]] = set()
        # Edge table for swap prevention: (from_x, from_y, to_x, to_y, t)
        edges: set[tuple[int, int, int, int, int]] = set()
        actions: dict[int, int] = {}
        gt = goal_types or {}

        for bid in urgency_order:
            pos = bot_positions.get(bid)
            goal = goals.get(bid)

            if pos is None:
                actions[bid] = ACT_WAIT
                continue

            if goal is None or pos == goal:
                # At goal — reserve position at t=0..1
                if pos != self.spawn:
                    reserved.add((pos[0], pos[1], 0))
                    reserved.add((pos[0], pos[1], 1))
                actions[bid] = ACT_WAIT
                continue

            # Windowed spacetime A*
            first_action, path = self._windowed_astar(
                pos, goal, reserved, edges)

            actions[bid] = first_action

            # Reserve the planned path (first 3 steps)
            if path:
                for t in range(min(len(path), 3)):
                    p = path[t]
                    if p != self.spawn:
                        reserved.add((p[0], p[1], t))
                    if t > 0:
                        prev = path[t - 1]
                        edges.add((prev[0], prev[1], p[0], p[1], t))
            else:
                # Failed — reserve current pos
                if pos != self.spawn:
                    reserved.add((pos[0], pos[1], 0))
                    reserved.add((pos[0], pos[1], 1))

        return actions

    def _windowed_astar(self, start: tuple[int, int], goal: tuple[int, int],
                         reserved: set[tuple[int, int, int]],
                         edges: set[tuple[int, int, int, int, int]],
                         ) -> tuple[int, list[tuple[int, int]]]:
        """Windowed A*: find best partial path toward goal within lookahead.

        Returns (first_action, path) where path is a list of positions.
        If goal is reachable within window, returns optimal path.
        Otherwise returns path to the explored node closest to goal.
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

        # Track best node by BFS distance to goal (for partial path)
        best_h = h0
        best_state = start_state

        while open_list:
            f, g, t, x, y = heapq.heappop(open_list)
            pos = (x, y)
            state = (x, y, t)

            # Skip if we've already found a better path to this state
            if g > g_score.get(state, float('inf')):
                continue

            # Check if goal reached
            if pos == goal:
                return self._extract(state, came_from)

            # Track best partial endpoint
            h = self.tables.get_distance(pos, goal)
            if h < best_h or (h == best_h and g < g_score.get(best_state, float('inf'))):
                best_h = h
                best_state = state

            if t >= max_t:
                continue

            next_t = t + 1

            # Expand: wait + 4 moves
            # Wait costs 1.5 (discourage waiting, prefer movement)
            self._try_expand(pos, pos, 1.5, next_t, x, y,
                             goal, reserved, edges, open_list, g_score,
                             came_from, state, g)

            # Moves cost 1.0
            for act, dest in self._neighbors.get(pos, []):
                self._try_expand(pos, dest, 1.0, next_t, x, y,
                                 goal, reserved, edges, open_list, g_score,
                                 came_from, state, g)

        # Goal not reached — use best partial path
        return self._extract(best_state, came_from)

    def _try_expand(self, from_pos, dest, step_cost, next_t, x, y,
                     goal, reserved, edges, open_list, g_score,
                     came_from, parent_state, parent_g):
        next_state = (dest[0], dest[1], next_t)

        # Collision check
        if dest != self.spawn and next_state in reserved:
            return
        # Swap check
        if from_pos != dest and (dest[0], dest[1], x, y, next_t) in edges:
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
        """Extract first action and full path from came_from chain."""
        path = []
        s = state
        while s is not None:
            path.append((s[0], s[1]))
            s = came_from.get(s)
        path.reverse()

        if len(path) <= 1:
            return ACT_WAIT, path

        # First action
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        if dx == 0 and dy == 0:
            return ACT_WAIT, path
        for act in MOVES:
            if DX[act] == dx and DY[act] == dy:
                return act, path
        return ACT_WAIT, path
