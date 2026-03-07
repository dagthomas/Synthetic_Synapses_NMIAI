"""Traffic management for nightmare mode: one-way aisles + congestion heatmap."""
from __future__ import annotations

from game_engine import MapState, CELL_FLOOR, CELL_DROPOFF


class TrafficRules:
    """Assigns one-way directions to narrow vertical aisles to prevent deadlocks."""

    def __init__(self, ms: MapState):
        self.aisle_direction: dict[tuple[int, int], str] = {}
        # Corridor rows (horizontal passageways)
        corridor_ys = {1, ms.height // 2, ms.height - 3}
        # Detect narrow vertical aisle columns between shelf pairs
        # Walkable columns that have shelves on both sides
        seg_idx = 0
        for x in range(1, ms.width - 1):
            # Check if this column is a narrow aisle: walkable with shelves on left or right
            is_aisle_col = True
            walkable_count = 0
            for y in range(2, ms.height - 2):
                if y in corridor_ys:
                    continue
                cell = ms.grid[y, x]
                if cell in (CELL_FLOOR, CELL_DROPOFF):
                    walkable_count += 1
                else:
                    is_aisle_col = False
                    break
            if not is_aisle_col or walkable_count < 3:
                continue
            # Check shelves on either side
            has_shelf_neighbor = False
            for y in range(2, ms.height - 2):
                if y in corridor_ys:
                    continue
                for dx in [-1, 1]:
                    nx = x + dx
                    if 0 <= nx < ms.width and ms.grid[y, nx] not in (CELL_FLOOR, CELL_DROPOFF):
                        has_shelf_neighbor = True
                        break
                if has_shelf_neighbor:
                    break
            if not has_shelf_neighbor:
                continue
            # Assign direction: alternating by segment index
            direction = 'down' if seg_idx % 2 == 0 else 'up'
            for y in range(2, ms.height - 2):
                if y not in corridor_ys:
                    self.aisle_direction[(x, y)] = direction
            seg_idx += 1

    def move_penalty(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> float:
        """Returns additional cost for wrong-way aisle travel.

        Currently disabled — one-way penalties cause bots to take longer routes
        that don't pay off in reduced collisions. Keeping the class for potential
        future use with different penalty values.
        """
        return 0.0


class CongestionMap:
    """Decaying heatmap of bot positions for congestion-aware pathfinding."""

    def __init__(self):
        self.heat: dict[tuple[int, int], float] = {}

    def update(self, bot_positions: list[tuple[int, int]]):
        """Decay existing heat by 0.7, add 1.0 at each bot position."""
        new_heat = {}
        for pos, h in self.heat.items():
            v = h * 0.7
            if v > 0.1:
                new_heat[pos] = v
        for pos in bot_positions:
            new_heat[pos] = new_heat.get(pos, 0.0) + 1.0
        self.heat = new_heat

    def get_penalty(self, pos: tuple[int, int]) -> float:
        return self.heat.get(pos, 0.0)
