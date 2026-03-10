"""Recursive PIBT pathfinding for nightmare mode.

Hybrid approach: incremental claiming (like flat planner) with recursive
push chains (from PIBT). Corridor awareness and priority-based tiebreaking
for 20-bot coordination in narrow aisles.
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
    """Recursive PIBT pathfinder with corridor awareness."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 traffic: TrafficRules, congestion: CongestionMap):
        self.ms = ms
        self.walkable = build_walkable(ms)
        self.tables = tables
        self.traffic = traffic
        self.congestion = congestion
        self.spawn = ms.spawn
        self.drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

        # Corridor rows (horizontal passages)
        self._corridor_ys = {1, ms.height // 2, ms.height - 3}

        # Narrow aisle detection
        self._narrow_aisles: set[int] = set()
        self._detect_narrow_aisles()

        # Precompute neighbor lists for each walkable cell
        self._neighbors: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for pos in self.walkable:
            nbrs = []
            for act in MOVES:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in self.walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

    def _detect_narrow_aisles(self):
        """Identify narrow vertical aisle columns (walkable with shelves on sides)."""
        for x in range(1, self.ms.width - 1):
            is_aisle = True
            walkable_count = 0
            for y in range(2, self.ms.height - 2):
                if y in self._corridor_ys:
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
                 goal_types: dict[int, str] | None = None,
                 round_number: int = 0) -> dict[int, int]:
        """Plan actions: incremental claiming + recursive push chains."""
        claims: dict[tuple[int, int], int] = {}
        actions: dict[int, int] = {}
        planned_dest: dict[int, tuple[int, int]] = {}
        gt = goal_types or {}

        # Build priority rank from urgency_order
        priority_rank: dict[int, int] = {}
        for rank, bid in enumerate(urgency_order):
            priority_rank[bid] = rank

        max_depth = 4  # Limit push chain depth (practical: 1-3 deep)

        for bid in urgency_order:
            pos = bot_positions.get(bid)
            goal = goals.get(bid)

            if pos is None:
                actions[bid] = ACT_WAIT
                continue

            # Already planned (was pushed by a higher-priority bot)
            if bid in actions:
                continue

            if goal is None or pos == goal:
                if pos not in claims or pos == self.spawn:
                    claims[pos] = bid
                actions[bid] = ACT_WAIT
                planned_dest[bid] = pos
                continue

            candidates = self._rank_moves(bid, pos, goal, round_number)

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
                    for ob, od in planned_dest.items():
                        if ob != bid and od == pos and bot_positions.get(ob) == dest:
                            swap = True
                            break
                    if not swap:
                        claims[dest] = bid
                        actions[bid] = act
                        planned_dest[bid] = dest
                        assigned = True
                        break
                else:
                    # Dest claimed — try recursive push
                    blocker = claims[dest]
                    pushed = self._try_push(
                        blocker, dest, pos, 1, max_depth,
                        claims, actions, planned_dest,
                        bot_positions, priority_rank, goals, gt,
                        round_number)
                    if pushed:
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

    def _try_push(self, blocker: int, blocker_pos: tuple[int, int],
                  pusher_pos: tuple[int, int],
                  depth: int, max_depth: int,
                  claims: dict[tuple[int, int], int],
                  actions: dict[int, int],
                  planned_dest: dict[int, tuple[int, int]],
                  bot_positions: dict[int, tuple[int, int]],
                  priority_rank: dict[int, int],
                  goals: dict[int, tuple[int, int]],
                  goal_types: dict[int, str],
                  round_number: int) -> bool:
        """Try to recursively push a blocker out of its position.

        Returns True if blocker successfully moved and freed its cell.
        """
        if depth > max_depth:
            return False

        # Only push bots that are waiting at their current position
        if blocker not in actions:
            # Unplanned bot — plan it now via recursive push
            actual_pos = bot_positions.get(blocker)
            if actual_pos != blocker_pos:
                return False
        elif actions[blocker] != ACT_WAIT:
            # Bot is moving somewhere — can't push
            return False
        elif bot_positions.get(blocker) != blocker_pos:
            # Bot isn't actually at this position
            return False
        else:
            # Bot is waiting — check if pushable type
            bgt = goal_types.get(blocker, 'park')
            if blocker_pos != self.spawn and bgt not in ('park', 'flee', 'stage'):
                return False

        blocker_goal = goals.get(blocker, self.spawn)
        alt_moves = self._rank_moves(blocker, blocker_pos, blocker_goal, round_number)

        for alt_act, alt_dest in alt_moves:
            # Don't push back to where the pusher came from
            if alt_dest == pusher_pos:
                continue
            if alt_dest == self.spawn:
                # Spawn always available
                claims[alt_dest] = blocker
                actions[blocker] = alt_act
                planned_dest[blocker] = alt_dest
                # Free old position
                if blocker_pos in claims and claims[blocker_pos] == blocker:
                    del claims[blocker_pos]
                return True
            if alt_dest not in claims:
                # Swap detection
                swap = False
                for ob, od in planned_dest.items():
                    if ob != blocker and od == blocker_pos and bot_positions.get(ob) == alt_dest:
                        swap = True
                        break
                if swap:
                    continue
                claims[alt_dest] = blocker
                actions[blocker] = alt_act
                planned_dest[blocker] = alt_dest
                if blocker_pos in claims and claims[blocker_pos] == blocker:
                    del claims[blocker_pos]
                return True
            else:
                # Recursive: try to push the next blocker
                next_blocker = claims[alt_dest]
                if next_blocker == blocker:
                    continue
                pushed = self._try_push(
                    next_blocker, alt_dest, blocker_pos,
                    depth + 1, max_depth,
                    claims, actions, planned_dest,
                    bot_positions, priority_rank, goals, goal_types,
                    round_number)
                if pushed:
                    claims[alt_dest] = blocker
                    actions[blocker] = alt_act
                    planned_dest[blocker] = alt_dest
                    if blocker_pos in claims and claims[blocker_pos] == blocker:
                        del claims[blocker_pos]
                    return True

        return False

    def _rank_moves(self, bid: int, pos: tuple[int, int],
                    goal: tuple[int, int],
                    round_number: int = 0) -> list[tuple[int, tuple[int, int]]]:
        """Rank moves: optimal BFS first-step first, then alternatives by distance.

        Adds corridor penalty for narrow aisle cells.
        """
        optimal_act = self.tables.get_first_step(pos, goal)

        candidates = []
        for act, dest in self._neighbors.get(pos, []):
            nx, ny = dest
            d = self.tables.get_distance(dest, goal)
            traffic_pen = self.traffic.move_penalty(pos, dest)
            cong_pen = self.congestion.get_penalty(dest) * 2.0
            optimal_bonus = -0.5 if act == optimal_act else 0.0
            # Corridor penalty: discourage entering narrow aisle non-corridor cells
            corridor_pen = 0.0
            if nx in self._narrow_aisles and ny not in self._corridor_ys:
                corridor_pen = 0.1
            score = d + traffic_pen + cong_pen + optimal_bonus + corridor_pen
            candidates.append((score, act, dest))

        candidates.sort()
        return [(act, dest) for _, act, dest in candidates]
