"""Order-agnostic route table: precompute optimal routes for ALL possible orders.

Given a captured map (items, walls, dropoff), enumerates every possible order
combination and finds the optimal item assignment + pickup route for each.

At runtime, the solver looks up the active order → gets instant optimal route.

Usage:
    python route_table.py <difficulty>            # Train from captured map
    python route_table.py <difficulty> --info      # Show route table stats
"""
import json
import os
import sys
import time
import numpy as np
from collections import deque
from itertools import combinations_with_replacement, permutations, product

from game_engine import build_map_from_capture, CELL_FLOOR, CELL_DROPOFF
from configs import CONFIGS, INV_CAP
from solution_store import load_capture, _dir
from precompute import PrecomputedTables

ROUTE_TABLE_FILE = 'route_table.json'


class RouteTable:
    """Precomputed route table for a specific map."""

    def __init__(self, capture_data):
        self.ms = build_map_from_capture(capture_data)
        self.difficulty = capture_data.get('difficulty', 'easy')
        cfg = CONFIGS[self.difficulty]
        self.order_sizes = range(cfg['order_size'][0], cfg['order_size'][1] + 1)

        # Use precomputed tables for walkable set and distances
        tables = PrecomputedTables.get(self.ms)
        self.walkable = tables.walkable
        self.pos_to_idx = tables.pos_to_idx
        self.n_cells = tables.n_cells
        self.dropoff = tuple(self.ms.drop_off)
        self.dropoff_idx = self.pos_to_idx[self.dropoff]
        self._precomputed_tables = tables

        # Item info: type -> list of {id, pos, adj_cells [(pos, idx)]}
        self.item_types = sorted(set(item['type'] for item in self.ms.items))
        self.items_by_type = {}
        for item in self.ms.items:
            t = item['type']
            if t not in self.items_by_type:
                self.items_by_type[t] = []
            idx_in_ms = next(i for i, it in enumerate(self.ms.items) if it['id'] == item['id'])
            adj_cells = []
            for adj_pos in self.ms.item_adjacencies.get(idx_in_ms, []):
                if adj_pos in self.pos_to_idx:
                    adj_cells.append((adj_pos, self.pos_to_idx[adj_pos]))
            self.items_by_type[t].append({
                'id': item['id'],
                'pos': item['position'],
                'adj': adj_cells,
            })

        # Use precomputed all-pairs shortest paths
        self.dist = self._precomputed_tables.dist_matrix

        # Route cache: order_key -> list of routes (best first)
        self.routes = {}

    def _compute_all_pairs(self):
        """BFS from every walkable cell. Unused — kept as fallback."""
        n = self.n_cells
        dist = np.full((n, n), 9999, dtype=np.int16)
        walkable_set = set(self.walkable)

        for start_idx, start_pos in enumerate(self.walkable):
            dist[start_idx, start_idx] = 0
            queue = deque([(start_pos, 0)])
            visited = {start_pos}
            while queue:
                (x, y), d = queue.popleft()
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    npos = (x + dx, y + dy)
                    if npos in walkable_set and npos not in visited:
                        visited.add(npos)
                        j = self.pos_to_idx[npos]
                        dist[start_idx, j] = d + 1
                        queue.append((npos, d + 1))
        return dist

    def _tsp_cost(self, start_idx, waypoint_idxs, end_idx):
        """Exact TSP: start → all waypoints → end. Returns (cost, best_perm)."""
        if not waypoint_idxs:
            return int(self.dist[start_idx, end_idx]), ()

        best_cost = 99999
        best_perm = None
        for perm in permutations(range(len(waypoint_idxs))):
            cost = int(self.dist[start_idx, waypoint_idxs[perm[0]]])
            for i in range(len(perm) - 1):
                cost += int(self.dist[waypoint_idxs[perm[i]], waypoint_idxs[perm[i+1]]])
            cost += int(self.dist[waypoint_idxs[perm[-1]], end_idx])
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        return best_cost, best_perm

    def _solve_order(self, order_items):
        """Find optimal item assignment + route for a specific order.

        Returns list of routes (trips), each trip is:
        [(item_id, adj_pos, adj_idx), ...] in pickup order
        Plus total_cost.
        """
        n_items = len(order_items)

        # Group by type with counts
        type_counts = {}
        for t in order_items:
            type_counts[t] = type_counts.get(t, 0) + 1

        # For each type, get candidate items with their best adj cells
        candidates_per_type = {}
        for t in type_counts:
            if t not in self.items_by_type:
                return None  # impossible order
            items = self.items_by_type[t]
            # Pre-sort by distance to dropoff (best adj cell)
            scored = []
            for item in items:
                if not item['adj']:
                    continue
                best_adj = min(item['adj'], key=lambda a: self.dist[self.dropoff_idx, a[1]])
                scored.append((item, best_adj))
            scored.sort(key=lambda x: self.dist[self.dropoff_idx, x[1][1]])
            candidates_per_type[t] = scored

        # Build candidate list: for each slot in the order, which items could fill it
        # Limit to top-K candidates per type to keep enumeration tractable
        K = 4  # top 4 closest items per type
        slots = []
        for t in order_items:
            cands = candidates_per_type.get(t, [])[:K]
            if not cands:
                return None
            slots.append(cands)

        # Enumerate item selections, solve TSP for each
        best_total_cost = 99999
        best_plan = None

        for selection in product(*slots):
            # Items can be reused (shelves never deplete), so allow duplicates
            # Split into trips of INV_CAP items
            adj_idxs = [s[1][1] for s in selection]  # adj cell indices
            adj_poses = [s[1][0] for s in selection]  # adj cell positions
            items_info = [(s[0]['id'], s[1][0], s[1][1]) for s in selection]

            # For orders <= INV_CAP: single trip
            if n_items <= INV_CAP:
                cost, perm = self._tsp_cost(self.dropoff_idx, adj_idxs, self.dropoff_idx)
                if cost < best_total_cost:
                    best_total_cost = cost
                    best_plan = [[items_info[p] for p in perm]]
            else:
                # Multiple trips needed. Try all ways to split into trips.
                # For size 4 with INV_CAP=3: first trip picks 3, second picks 1
                # Try all C(n,3) splits
                from itertools import combinations
                for first_trip_indices in combinations(range(n_items), INV_CAP):
                    second_trip_indices = [i for i in range(n_items) if i not in first_trip_indices]

                    first_adj = [adj_idxs[i] for i in first_trip_indices]
                    second_adj = [adj_idxs[i] for i in second_trip_indices]

                    cost1, perm1 = self._tsp_cost(self.dropoff_idx, first_adj, self.dropoff_idx)
                    cost2, perm2 = self._tsp_cost(self.dropoff_idx, second_adj, self.dropoff_idx)
                    total = cost1 + cost2

                    if total < best_total_cost:
                        best_total_cost = total
                        trip1 = [items_info[first_trip_indices[p]] for p in perm1]
                        trip2 = [items_info[second_trip_indices[p]] for p in perm2]
                        best_plan = [trip1, trip2]

        if best_plan is None:
            return None

        return {
            'cost': best_total_cost,
            'trips': [
                [{'item_id': iid, 'adj_pos': list(apos), 'adj_idx': aidx}
                 for iid, apos, aidx in trip]
                for trip in best_plan
            ],
        }

    def train(self, verbose=True):
        """Enumerate all possible orders and precompute optimal routes."""
        t0 = time.time()
        n_orders = 0
        n_solved = 0

        for size in self.order_sizes:
            for combo in combinations_with_replacement(self.item_types, size):
                order_key = ','.join(combo)
                order_items = list(combo)

                result = self._solve_order(order_items)
                if result:
                    self.routes[order_key] = result
                    n_solved += 1
                n_orders += 1

        elapsed = time.time() - t0
        if verbose:
            print(f"Trained {n_solved}/{n_orders} orders in {elapsed:.2f}s", file=sys.stderr)
            print(f"  Types: {self.item_types}", file=sys.stderr)
            print(f"  Order sizes: {list(self.order_sizes)}", file=sys.stderr)
            print(f"  Walkable cells: {self.n_cells}", file=sys.stderr)
            print(f"  Items: {sum(len(v) for v in self.items_by_type.values())}", file=sys.stderr)
            # Show a few example routes
            for key in list(self.routes.keys())[:3]:
                r = self.routes[key]
                print(f"  [{key}] cost={r['cost']} trips={len(r['trips'])}", file=sys.stderr)

    def get_route(self, order_items, start_pos=None):
        """Look up precomputed route for an order.

        If start_pos is given, re-solves TSP from that position (using
        precomputed item assignment but optimal ordering from start_pos).
        """
        key = ','.join(sorted(t.lower() for t in order_items))
        route = self.routes.get(key)
        if not route:
            return None

        if start_pos is None or tuple(start_pos) == self.dropoff:
            return route

        # Re-solve TSP from current position for better ordering
        start_idx = self.pos_to_idx.get(tuple(start_pos))
        if start_idx is None:
            return route  # Unknown position, use precomputed

        reoptimized_trips = []
        current_idx = start_idx
        for trip in route['trips']:
            adj_idxs = [wp['adj_idx'] for wp in trip]
            items_info = [(wp['item_id'], tuple(wp['adj_pos']), wp['adj_idx']) for wp in trip]

            _, perm = self._tsp_cost(current_idx, adj_idxs, self.dropoff_idx)
            reordered = [items_info[p] for p in perm] if perm else items_info

            reoptimized_trips.append([
                {'item_id': iid, 'adj_pos': list(apos), 'adj_idx': aidx}
                for iid, apos, aidx in reordered
            ])
            current_idx = self.dropoff_idx  # After delivery, back at dropoff

        return {'cost': route['cost'], 'trips': reoptimized_trips}

    def save(self, difficulty):
        """Save route table to solutions/<difficulty>/route_table.json."""
        d = _dir(difficulty)
        path = os.path.join(d, ROUTE_TABLE_FILE)

        data = {
            'difficulty': difficulty,
            'n_cells': self.n_cells,
            'n_orders': len(self.routes),
            'item_types': self.item_types,
            'dropoff': list(self.dropoff),
            'routes': self.routes,
            # Save distance matrix for runtime re-optimization
            'pos_to_idx': {f"{x},{y}": idx for (x, y), idx in self.pos_to_idx.items()},
            'dist_matrix': self.dist.tolist(),
        }

        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Route table saved: {path} ({len(self.routes)} routes)", file=sys.stderr)
        return path

    @classmethod
    def load(cls, difficulty):
        """Load a saved route table. Returns None if not found."""
        d = _dir(difficulty)
        path = os.path.join(d, ROUTE_TABLE_FILE)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            data = json.load(f)

        # Reconstruct minimal RouteTable for runtime use
        rt = object.__new__(cls)
        rt.difficulty = data['difficulty']
        rt.item_types = data['item_types']
        rt.dropoff = tuple(data['dropoff'])
        rt.n_cells = data['n_cells']
        rt.routes = data['routes']

        # Rebuild pos_to_idx and dist matrix
        rt.pos_to_idx = {}
        for key, idx in data['pos_to_idx'].items():
            x, y = map(int, key.split(','))
            rt.pos_to_idx[(x, y)] = idx
        rt.dropoff_idx = rt.pos_to_idx[rt.dropoff]

        rt.dist = np.array(data['dist_matrix'], dtype=np.int16)
        rt.walkable = [None] * rt.n_cells
        for pos, idx in rt.pos_to_idx.items():
            rt.walkable[idx] = pos

        return rt

    def lookup_next_action(self, bot_pos, order_items, delivered, inventory, trip_state):
        """High-level: given current state, return next action + updated trip_state.

        trip_state = {'trip_idx': int, 'wp_idx': int, 'phase': 'pick'|'deliver'}
        Returns (action_name, item_id_or_none, new_trip_state)
        """
        if trip_state is None:
            trip_state = {'trip_idx': 0, 'wp_idx': 0, 'phase': 'pick'}

        # Get remaining items needed
        remaining = list(order_items)
        for d in delivered:
            if d in remaining:
                remaining.remove(d)
        for inv_item in inventory:
            if inv_item in remaining:
                remaining.remove(inv_item)

        if not remaining and not inventory:
            return 'wait', None, trip_state

        # If we have items and nothing more to pick → deliver
        if inventory and not remaining:
            return self._navigate_action(bot_pos, self.dropoff), None, \
                   {'trip_idx': trip_state['trip_idx'], 'wp_idx': 0, 'phase': 'deliver'}

        # Get route
        route = self.get_route(order_items, bot_pos)
        if not route or not route['trips']:
            return 'wait', None, trip_state

        trip_idx = trip_state.get('trip_idx', 0)
        wp_idx = trip_state.get('wp_idx', 0)
        phase = trip_state.get('phase', 'pick')

        # Deliver phase: go to dropoff
        if phase == 'deliver':
            if tuple(bot_pos) == self.dropoff:
                if inventory:
                    return 'drop_off', None, \
                           {'trip_idx': trip_idx + 1, 'wp_idx': 0, 'phase': 'pick'}
                else:
                    return 'wait', None, {'trip_idx': trip_idx + 1, 'wp_idx': 0, 'phase': 'pick'}
            return self._navigate_action(bot_pos, self.dropoff), None, trip_state

        # Pick phase: navigate to waypoints
        if trip_idx >= len(route['trips']):
            # All trips done
            if inventory:
                return self._navigate_action(bot_pos, self.dropoff), None, \
                       {'trip_idx': trip_idx, 'wp_idx': 0, 'phase': 'deliver'}
            return 'wait', None, trip_state

        trip = route['trips'][trip_idx]
        if wp_idx >= len(trip):
            # All waypoints in this trip visited → deliver
            if inventory:
                return self._navigate_action(bot_pos, self.dropoff), None, \
                       {'trip_idx': trip_idx, 'wp_idx': 0, 'phase': 'deliver'}
            # Trip empty, next trip
            return 'wait', None, {'trip_idx': trip_idx + 1, 'wp_idx': 0, 'phase': 'pick'}

        wp = trip[wp_idx]
        target_pos = tuple(wp['adj_pos'])
        item_id = wp['item_id']

        # Check if adjacent to target item → pick up
        bx, by = bot_pos
        for item_info_list in (self.items_by_type.values() if hasattr(self, 'items_by_type') else []):
            pass  # items_by_type not available in loaded tables

        # Simple adjacency check: are we next to the item's shelf?
        if target_pos == tuple(bot_pos):
            # At the adj cell → pick up
            return 'pick_up', item_id, \
                   {'trip_idx': trip_idx, 'wp_idx': wp_idx + 1, 'phase': 'pick'}

        # Navigate to the waypoint
        return self._navigate_action(bot_pos, target_pos), None, trip_state

    def _navigate_action(self, from_pos, to_pos):
        """Get move action toward target using precomputed distances."""
        fx, fy = from_pos
        tx, ty = to_pos
        if (fx, fy) == (tx, ty):
            return 'wait'

        from_idx = self.pos_to_idx.get((fx, fy))
        to_idx = self.pos_to_idx.get((tx, ty))
        if from_idx is None or to_idx is None:
            return 'wait'

        # Try all 4 moves, pick the one that gets us closest
        best_act = 'wait'
        best_dist = self.dist[from_idx, to_idx]

        for act, (dx, dy) in [('move_up', (0, -1)), ('move_down', (0, 1)),
                               ('move_left', (-1, 0)), ('move_right', (1, 0))]:
            nx, ny = fx + dx, fy + dy
            n_idx = self.pos_to_idx.get((nx, ny))
            if n_idx is not None:
                d = self.dist[n_idx, to_idx]
                if d < best_dist:
                    best_dist = d
                    best_act = act

        return best_act


def train_from_capture(difficulty, verbose=True):
    """Train route table from captured map data."""
    capture = load_capture(difficulty)
    if not capture:
        print(f"No capture found for {difficulty}. Play a game with --save-capture first.",
              file=sys.stderr)
        return None

    rt = RouteTable(capture)
    rt.train(verbose=verbose)
    rt.save(difficulty)
    return rt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Route table training')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--info', action='store_true', help='Show route table info')
    args = parser.parse_args()

    if args.info:
        rt = RouteTable.load(args.difficulty)
        if rt:
            print(f"Route table: {args.difficulty}")
            print(f"  Orders: {len(rt.routes)}")
            print(f"  Types: {rt.item_types}")
            print(f"  Cells: {rt.n_cells}")
            costs = [r['cost'] for r in rt.routes.values()]
            print(f"  Avg route cost: {sum(costs)/len(costs):.1f}")
            print(f"  Min/Max cost: {min(costs)}/{max(costs)}")
        else:
            print(f"No route table found for {args.difficulty}")
    else:
        train_from_capture(args.difficulty)
