"""PIBT-style priority-based pathfinding for nightmare mode.

Process bots in urgency order (deliverers first). Each bot picks its
best move toward its goal. If the cell is already claimed by a
higher-priority bot, it tries alternatives. Swap detection prevents
head-on deadlocks.
"""
from __future__ import annotations

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


class NightmarePathfinder:
    """Priority-based pathfinder with swap detection."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 traffic: TrafficRules, congestion: CongestionMap):
        self.ms = ms
        self.walkable = build_walkable(ms)
        self.tables = tables
        self.traffic = traffic
        self.congestion = congestion
        self.spawn = ms.spawn
        self.drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

    def plan_all(self, bot_positions: dict[int, tuple[int, int]],
                 goals: dict[int, tuple[int, int]],
                 urgency_order: list[int],
                 goal_types: dict[int, str] | None = None) -> dict[int, int]:
        """Plan actions in priority order with swap detection."""
        claims: dict[tuple[int, int], int] = {}
        actions: dict[int, int] = {}
        planned_dest: dict[int, tuple[int, int]] = {}
        gt = goal_types or {}

        for bid in urgency_order:
            pos = bot_positions.get(bid)
            goal = goals.get(bid)

            if pos is None:
                actions[bid] = ACT_WAIT
                continue

            if goal is None or pos == goal:
                if pos not in claims or pos == self.spawn:
                    claims[pos] = bid
                actions[bid] = ACT_WAIT
                planned_dest[bid] = pos
                continue

            candidates = self._rank_moves(bid, pos, goal)

            assigned = False
            for act, dest in candidates:
                if dest == self.spawn:
                    claims[dest] = bid
                    actions[bid] = act
                    planned_dest[bid] = dest
                    assigned = True
                    break
                if dest not in claims:
                    # Swap detection
                    swap = False
                    for ob, op in bot_positions.items():
                        if op == dest and ob != bid and ob in planned_dest:
                            if planned_dest[ob] == pos:
                                swap = True
                                break
                    if not swap:
                        claims[dest] = bid
                        actions[bid] = act
                        planned_dest[bid] = dest
                        assigned = True
                        break

            if not assigned:
                if pos not in claims or pos == self.spawn:
                    claims[pos] = bid
                actions[bid] = ACT_WAIT
                planned_dest[bid] = pos

        return actions

    def _rank_moves(self, bid: int, pos: tuple[int, int],
                    goal: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        """Rank moves: optimal BFS first-step first, then alternatives by distance."""
        optimal_act = self.tables.get_first_step(pos, goal)

        candidates = []
        for act in MOVES:
            nx, ny = pos[0] + DX[act], pos[1] + DY[act]
            dest = (nx, ny)
            if dest not in self.walkable:
                continue
            d = self.tables.get_distance(dest, goal)
            traffic_pen = self.traffic.move_penalty(pos, dest)
            cong_pen = self.congestion.get_penalty(dest) * 2.0
            optimal_bonus = -0.5 if act == optimal_act else 0.0
            score = d + traffic_pen + cong_pen + optimal_bonus
            candidates.append((score, act, dest))

        candidates.sort()
        return [(act, dest) for _, act, dest in candidates]
