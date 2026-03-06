"""Pre-computed all-pairs distance matrix for the grocery store grid.

Computes BFS from every walkable cell once at game start.
All future distance queries are O(1) lookups.
"""

from collections import deque


class DistanceMatrix:
    """Pre-computed distances between all walkable cells."""

    def __init__(self, state):
        self.width = state["grid"]["width"]
        self.height = state["grid"]["height"]
        self.drop_off = tuple(state["drop_off"])

        # Build the set of impassable cells (walls + shelves)
        self._blocked = set()
        for w in state["grid"]["walls"]:
            self._blocked.add((w[0], w[1]))
        for item in state["items"]:
            self._blocked.add((item["position"][0], item["position"][1]))

        # Find all walkable cells
        self._walkable = set()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self._blocked:
                    self._walkable.add((x, y))

        # Pre-compute BFS from every walkable cell
        self._dist = {}  # (from_pos, to_pos) -> distance
        self._from = {}  # from_pos -> {to_pos: distance}
        for cell in self._walkable:
            dmap = self._bfs(cell)
            self._from[cell] = dmap
            for target, d in dmap.items():
                self._dist[(cell, target)] = d

        # Pre-compute adjacent walkable cells for each shelf/item position
        self._shelf_adjacent = {}
        for item in state["items"]:
            ipos = (item["position"][0], item["position"][1])
            if ipos not in self._shelf_adjacent:
                adjs = []
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    adj = (ipos[0] + dx, ipos[1] + dy)
                    if adj in self._walkable:
                        adjs.append(adj)
                self._shelf_adjacent[ipos] = adjs

    def _bfs(self, start):
        """BFS from start to all reachable cells."""
        dist = {start: 0}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            d = dist[(cx, cy)]
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in self._walkable and (nx, ny) not in dist:
                    dist[(nx, ny)] = d + 1
                    q.append((nx, ny))
        return dist

    def dist(self, a, b):
        """Distance between two walkable cells. Returns 999 if unreachable."""
        return self._dist.get((a, b), 999)

    def dist_to_dropoff(self, pos):
        """Distance from pos to the drop-off zone."""
        return self.dist(pos, self.drop_off)

    def adjacent_cells(self, shelf_pos):
        """Walkable cells adjacent to a shelf position."""
        return self._shelf_adjacent.get(shelf_pos, [])

    def best_adjacent(self, from_pos, shelf_pos):
        """Find the adjacent cell to shelf_pos closest to from_pos.

        Returns (adjacent_cell, distance) or (None, 999).
        """
        best_adj = None
        best_d = 999
        for adj in self.adjacent_cells(shelf_pos):
            d = self.dist(from_pos, adj)
            if d < best_d:
                best_d = d
                best_adj = adj
        return best_adj, best_d

    def next_step(self, from_pos, to_pos):
        """Return the adjacent cell to move to on the shortest path from_pos→to_pos.

        Uses gradient descent on pre-computed distances: O(1) per call.
        Returns from_pos if already there, None if unreachable.
        """
        if from_pos == to_pos:
            return from_pos
        to_map = self._from.get(to_pos)
        if to_map is None or from_pos not in to_map:
            return None
        best = None
        best_d = to_map[from_pos]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nb = (from_pos[0] + dx, from_pos[1] + dy)
            if nb in to_map and to_map[nb] < best_d:
                best_d = to_map[nb]
                best = nb
        return best

    def trip_cost(self, from_pos, shelf_pos):
        """Total cost: from_pos → best adjacent to shelf → drop_off.

        This is the "trip cost" for picking up an item from a shelf.
        """
        best_adj, d_to_item = self.best_adjacent(from_pos, shelf_pos)
        if best_adj is None:
            return 999
        d_to_drop = self.dist_to_dropoff(best_adj)
        return d_to_item + 1 + d_to_drop  # +1 for pickup action
