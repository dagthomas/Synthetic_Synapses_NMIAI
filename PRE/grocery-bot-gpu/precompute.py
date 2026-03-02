"""GPU-accelerated precomputed lookup tables for all-pairs shortest paths.

Uses matrix-multiply BFS on CUDA to compute all-pairs shortest paths for the
entire map in <5ms. Results are cached to disk (.npz) so subsequent loads are <1ms.

Tables provided:
- dist_matrix[from_idx, to_idx]     — int16, all-pairs shortest path distances
- next_step_matrix[from_idx, to_idx] — int8, first action (1-4) from→to
- dist_to_type[type_id, y, x]       — int16, distance to nearest item of type
- step_to_type[type_id, y, x]       — int8, first action toward nearest of type
- dist_to_dropoff[y, x]             — int16, distance to dropoff
- step_to_dropoff[y, x]             — int8, first action toward dropoff

Usage:
    from precompute import PrecomputedTables
    tables = PrecomputedTables.get(map_state)
    d = tables.get_distance((3, 2), (7, 5))       # O(1) lookup
    a = tables.get_first_step((3, 2), (7, 5))      # O(1) lookup
    dist_maps = tables.as_dist_maps()               # backward-compatible dict

CLI:
    python precompute.py                    # Precompute all 4 difficulties
    python precompute.py easy               # Precompute one difficulty
    python precompute.py --info             # Show cache stats
"""
import hashlib
import os
import sys
import time
import numpy as np

from game_engine import CELL_FLOOR, CELL_DROPOFF, MapState

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# Module-level cache: grid_hash -> PrecomputedTables instance
_tables_cache = {}


class PrecomputedTables:
    """Precomputed all-pairs shortest paths and navigation lookup tables."""

    def __init__(self, walkable, pos_to_idx, dist_matrix, next_step_matrix,
                 grid_shape, grid_hash):
        self.walkable = walkable           # list of (x, y) tuples, indexed by cell idx
        self.pos_to_idx = pos_to_idx       # dict: (x, y) -> cell idx
        self.n_cells = len(walkable)
        self.dist_matrix = dist_matrix     # [N, N] int16
        self.next_step_matrix = next_step_matrix  # [N, N] int8
        self.grid_shape = grid_shape       # (H, W)
        self.grid_hash = grid_hash

        # Lazily computed
        self._dist_maps = None
        self._gpu_tensors = None

        # Item/type tables (set by _compute_item_tables)
        self.dist_to_type = None     # [num_types, H, W] int16
        self.step_to_type = None     # [num_types, H, W] int8
        self.dist_to_dropoff = None  # [H, W] int16
        self.step_to_dropoff = None  # [H, W] int8

    @classmethod
    def get(cls, map_state):
        """Get or compute PrecomputedTables for a map state. Caches in memory + disk."""
        grid_hash = cls._compute_hash(map_state)

        # Memory cache
        if grid_hash in _tables_cache:
            tables = _tables_cache[grid_hash]
            # Recompute item tables if map_state has items (they may differ)
            if map_state.num_items > 0 and tables.dist_to_type is None:
                tables._compute_item_tables(map_state)
            return tables

        # Disk cache
        cache_path = os.path.join(CACHE_DIR, f'tables_{grid_hash}.npz')
        if os.path.exists(cache_path):
            tables = cls._load_from_cache(cache_path, grid_hash)
            if tables is not None:
                if map_state.num_items > 0:
                    tables._compute_item_tables(map_state)
                _tables_cache[grid_hash] = tables
                return tables

        # Compute from scratch
        tables = cls._compute(map_state, grid_hash)
        if map_state.num_items > 0:
            tables._compute_item_tables(map_state)
        _tables_cache[grid_hash] = tables
        return tables

    @staticmethod
    def _compute_hash(map_state):
        """Compute cache key from grid layout."""
        h = hashlib.md5()
        h.update(map_state.grid.tobytes())
        return h.hexdigest()[:12]

    @classmethod
    def _compute(cls, map_state, grid_hash):
        """Compute all-pairs shortest paths using GPU matrix-multiply BFS."""
        t0 = time.time()
        grid = map_state.grid
        H, W = grid.shape

        # Build walkable cell list and index mapping
        walkable = []
        pos_to_idx = {}
        for y in range(H):
            for x in range(W):
                if grid[y, x] == CELL_FLOOR or grid[y, x] == CELL_DROPOFF:
                    pos_to_idx[(x, y)] = len(walkable)
                    walkable.append((x, y))

        N = len(walkable)

        # Try GPU, fall back to CPU
        try:
            import torch
            if torch.cuda.is_available():
                dist_matrix, next_step_matrix = cls._gpu_bfs(
                    walkable, pos_to_idx, N, H, W)
            else:
                dist_matrix, next_step_matrix = cls._cpu_bfs(
                    walkable, pos_to_idx, N, grid)
        except ImportError:
            dist_matrix, next_step_matrix = cls._cpu_bfs(
                walkable, pos_to_idx, N, grid)

        tables = cls(walkable, pos_to_idx, dist_matrix, next_step_matrix,
                     (H, W), grid_hash)

        # Save to disk cache
        tables._save_to_cache()

        dt = time.time() - t0
        print(f"  PrecomputedTables: {N} cells, computed in {dt*1000:.1f}ms",
              file=sys.stderr)
        return tables

    @classmethod
    def _gpu_bfs(cls, walkable, pos_to_idx, N, H, W):
        """GPU matrix-multiply BFS for all-pairs shortest paths."""
        import torch

        device = 'cuda'

        # Build adjacency matrix A: [N, N] where A[i,j] = 1 if i,j are neighbors
        adj_i = []
        adj_j = []
        for idx, (x, y) in enumerate(walkable):
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                nidx = pos_to_idx.get((nx, ny))
                if nidx is not None:
                    adj_i.append(idx)
                    adj_j.append(nidx)

        # Build sparse adjacency and convert to dense on GPU
        A = torch.zeros((N, N), dtype=torch.float32, device=device)
        if adj_i:
            A[torch.tensor(adj_i, device=device),
              torch.tensor(adj_j, device=device)] = 1.0

        # Also build neighbor index table for next-step derivation
        # neighbors[i, d] = index of neighbor of cell i in direction d, or -1
        # directions: 0=up(0,-1), 1=down(0,1), 2=left(-1,0), 3=right(1,0)
        neighbors_np = np.full((N, 4), -1, dtype=np.int32)
        for idx, (x, y) in enumerate(walkable):
            for d, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
                nidx = pos_to_idx.get((x + dx, y + dy))
                if nidx is not None:
                    neighbors_np[idx, d] = nidx

        # Parallel BFS via matrix multiply
        dist = torch.full((N, N), 9999, dtype=torch.int16, device=device)
        # Set diagonal = 0 (distance from cell to itself)
        diag_idx = torch.arange(N, device=device)
        dist[diag_idx, diag_idx] = 0

        # frontier[i, j] = 1.0 means "cell j is on the frontier for source i"
        frontier = torch.eye(N, dtype=torch.float32, device=device)
        unvisited = dist == 9999  # [N, N] bool

        max_diameter = H + W  # upper bound on BFS depth
        for d in range(1, max_diameter):
            # Expand frontier: frontier @ A gives cells reachable from frontier
            expanded = torch.mm(frontier, A)  # [N, N]
            # Threshold to bool: any positive value means reachable
            expanded_bool = expanded > 0.0
            # Only mark unvisited cells
            new_cells = expanded_bool & unvisited
            if not new_cells.any():
                break
            dist[new_cells] = d
            unvisited = unvisited & ~new_cells
            frontier = new_cells.float()

        # Derive next-step matrix from dist + neighbor table
        # For each (src, tgt): look at src's neighbors, pick the one with min dist to tgt
        neighbors = torch.tensor(neighbors_np, dtype=torch.long, device=device)  # [N, 4]

        # Gather neighbor distances: for each src, get dist[neighbor[src,d], tgt]
        # dist is [N, N]. We need dist[neighbors[src, d], tgt] for all src, d, tgt
        # neighbors: [N, 4] -> clamp invalid (-1) to 0, will mask later
        valid_mask = neighbors >= 0  # [N, 4]
        nbr_clamped = neighbors.clamp(min=0)  # [N, 4]

        # Gather: nbr_dist[src, d, tgt] = dist[nbr_clamped[src, d], tgt]
        # Use advanced indexing: dist[nbr_clamped] gives [N, 4, N]
        nbr_dist = dist[nbr_clamped]  # [N, 4, N] int16

        # Mask invalid neighbors with large distance
        # valid_mask is [N, 4], expand to [N, 4, N]
        nbr_dist[~valid_mask.unsqueeze(-1).expand_as(nbr_dist)] = 9999

        # For each (src, tgt), find direction with minimum neighbor distance
        # next_step_matrix[src, tgt] = argmin direction + 1 (action IDs 1-4)
        # argmin gives 0-3, we add 1 to get action IDs 1-4
        min_dir = nbr_dist.argmin(dim=1)  # [N, N] values 0-3
        min_dist_val = nbr_dist.amin(dim=1)  # [N, N]

        next_step = (min_dir + 1).to(torch.int8)  # [N, N]
        # Where no valid neighbor leads closer (unreachable or at target), set to 0
        next_step[min_dist_val >= 9999] = 0
        # At target (diagonal), set to 0
        next_step[diag_idx, diag_idx] = 0

        dist_np = dist.cpu().numpy()
        next_step_np = next_step.cpu().numpy()

        return dist_np, next_step_np

    @classmethod
    def _cpu_bfs(cls, walkable, pos_to_idx, N, grid):
        """Fallback CPU BFS for all-pairs shortest paths."""
        from collections import deque

        H, W = grid.shape
        dist_matrix = np.full((N, N), 9999, dtype=np.int16)
        next_step_matrix = np.zeros((N, N), dtype=np.int8)

        walkable_set = set(pos_to_idx.keys())
        # Directions: action 1=up(0,-1), 2=down(0,1), 3=left(-1,0), 4=right(1,0)
        dirs = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]

        for src_idx, (sx, sy) in enumerate(walkable):
            dist_matrix[src_idx, src_idx] = 0
            # BFS from source
            parent_dir = np.zeros(N, dtype=np.int8)  # direction from parent
            visited = np.zeros(N, dtype=bool)
            visited[src_idx] = True
            queue = deque([src_idx])

            while queue:
                cidx = queue.popleft()
                cx, cy = walkable[cidx]
                d = dist_matrix[src_idx, cidx]
                for dx, dy, act in dirs:
                    nx, ny = cx + dx, cy + dy
                    nidx = pos_to_idx.get((nx, ny))
                    if nidx is not None and not visited[nidx]:
                        visited[nidx] = True
                        dist_matrix[src_idx, nidx] = d + 1
                        # The first step from src toward nidx:
                        # if cidx == src_idx, the first step IS this direction
                        # otherwise, inherit from parent
                        if cidx == src_idx:
                            parent_dir[nidx] = act
                        else:
                            parent_dir[nidx] = parent_dir[cidx]
                        queue.append(nidx)

            next_step_matrix[src_idx] = parent_dir

        return dist_matrix, next_step_matrix

    def _compute_item_tables(self, map_state):
        """Compute item/type distance and step tables from dist_matrix."""
        H, W = self.grid_shape
        N = self.n_cells
        num_types = map_state.num_types

        # Precompute src coordinate arrays for vectorized scatter
        src_ys = np.array([self.walkable[i][1] for i in range(N)], dtype=np.intp)
        src_xs = np.array([self.walkable[i][0] for i in range(N)], dtype=np.intp)

        # dist_to_type[t, y, x] = min distance from (x,y) to nearest adj cell of any item of type t
        dist_to_type = np.full((num_types, H, W), 9999, dtype=np.int16)
        step_to_type = np.zeros((num_types, H, W), dtype=np.int8)

        # For each item, find its adj cells and update type tables (vectorized per adj cell)
        for item_idx in range(map_state.num_items):
            type_id = int(map_state.item_types[item_idx])
            adj_cells = map_state.item_adjacencies.get(item_idx, [])
            for ax, ay in adj_cells:
                adj_idx = self.pos_to_idx.get((ax, ay))
                if adj_idx is None:
                    continue
                # dist_matrix[:, adj_idx] = distance from each walkable cell to this adj cell
                dists = self.dist_matrix[:, adj_idx]  # [N] int16
                steps = self.next_step_matrix[:, adj_idx]  # [N] int8
                # Update where this adj cell is closer
                current = dist_to_type[type_id, src_ys, src_xs]  # [N]
                better = dists < current
                dist_to_type[type_id, src_ys[better], src_xs[better]] = dists[better]
                step_to_type[type_id, src_ys[better], src_xs[better]] = steps[better]

        self.dist_to_type = dist_to_type
        self.step_to_type = step_to_type

        # Dropoff tables (vectorized)
        drop_x, drop_y = map_state.drop_off
        drop_idx = self.pos_to_idx.get((drop_x, drop_y))
        if drop_idx is not None:
            dist_to_dropoff = np.full((H, W), 9999, dtype=np.int16)
            step_to_dropoff = np.zeros((H, W), dtype=np.int8)
            dist_to_dropoff[src_ys, src_xs] = self.dist_matrix[:, drop_idx]
            step_to_dropoff[src_ys, src_xs] = self.next_step_matrix[:, drop_idx]
            self.dist_to_dropoff = dist_to_dropoff
            self.step_to_dropoff = step_to_dropoff

    def _save_to_cache(self):
        """Save tables to disk cache."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f'tables_{self.grid_hash}.npz')
        # Save walkable positions as flat array
        walkable_arr = np.array(self.walkable, dtype=np.int16)  # [N, 2]
        np.savez_compressed(path,
                            dist_matrix=self.dist_matrix,
                            next_step_matrix=self.next_step_matrix,
                            walkable=walkable_arr,
                            grid_shape=np.array(self.grid_shape, dtype=np.int32))
        size = os.path.getsize(path)
        print(f"  Cache saved: {path} ({size/1024:.1f} KB)", file=sys.stderr)

    @classmethod
    def _load_from_cache(cls, path, grid_hash):
        """Load tables from disk cache."""
        try:
            t0 = time.time()
            data = np.load(path)
            walkable_arr = data['walkable']
            walkable = [(int(x), int(y)) for x, y in walkable_arr]
            pos_to_idx = {pos: i for i, pos in enumerate(walkable)}
            grid_shape = tuple(data['grid_shape'])

            tables = cls(walkable, pos_to_idx,
                         data['dist_matrix'], data['next_step_matrix'],
                         grid_shape, grid_hash)
            dt = time.time() - t0
            print(f"  Cache loaded: {path} ({dt*1000:.1f}ms, {len(walkable)} cells)",
                  file=sys.stderr)
            return tables
        except Exception as e:
            print(f"  Cache load failed: {e}", file=sys.stderr)
            return None

    # === Public API ===

    def get_distance(self, src, tgt):
        """O(1) distance lookup. src/tgt are (x, y) tuples."""
        si = self.pos_to_idx.get(src)
        ti = self.pos_to_idx.get(tgt)
        if si is None or ti is None:
            return 9999
        return int(self.dist_matrix[si, ti])

    def get_first_step(self, src, tgt):
        """O(1) first-step action lookup. Returns action ID 1-4, or 0 if at target."""
        if src == tgt:
            return 0
        si = self.pos_to_idx.get(src)
        ti = self.pos_to_idx.get(tgt)
        if si is None or ti is None:
            return 0
        return int(self.next_step_matrix[si, ti])

    def get_nearest_item_cell(self, src, item_idx, map_state):
        """Find nearest walkable cell adjacent to item_idx from src. Returns (x, y, dist) or None."""
        adj = map_state.item_adjacencies.get(item_idx, [])
        si = self.pos_to_idx.get(src)
        if si is None:
            return None
        best = None
        best_d = 9999
        for (cx, cy) in adj:
            ti = self.pos_to_idx.get((cx, cy))
            if ti is not None:
                d = int(self.dist_matrix[si, ti])
                if d < best_d:
                    best_d = d
                    best = (cx, cy, d)
        return best

    def as_dist_maps(self):
        """Return backward-compatible dict: {(x,y) -> dist_map[H,W] int16}.

        Lazily computed and cached. The dist_map values use -1 for unreachable
        (matching original BFS format).
        """
        if self._dist_maps is not None:
            return self._dist_maps

        H, W = self.grid_shape
        N = self.n_cells

        # Precompute target coordinate arrays for vectorized scatter
        tgt_ys = np.array([self.walkable[i][1] for i in range(N)], dtype=np.intp)
        tgt_xs = np.array([self.walkable[i][0] for i in range(N)], dtype=np.intp)

        dist_maps = {}
        for src_idx in range(N):
            dm = np.full((H, W), -1, dtype=np.int16)
            row = self.dist_matrix[src_idx]  # [N] int16
            valid = row < 9999
            dm[tgt_ys[valid], tgt_xs[valid]] = row[valid]
            dist_maps[self.walkable[src_idx]] = dm
        self._dist_maps = dist_maps
        return dist_maps

    def to_gpu_tensors(self, device='cuda'):
        """Return all tables as CUDA tensors for gpu_beam_search.

        Returns dict with:
        - dist_to_dropoff: [H, W] int16
        - step_to_dropoff: [H, W] int8  (first_step_to_dropoff)
        - dist_to_type: [num_types, H, W] int16
        - step_to_type: [num_types, H, W] int8  (first_step_to_type)
        """
        import torch

        result = {}
        if self.dist_to_dropoff is not None:
            result['dist_to_dropoff'] = torch.tensor(
                self.dist_to_dropoff, dtype=torch.int16, device=device)
            result['step_to_dropoff'] = torch.tensor(
                self.step_to_dropoff, dtype=torch.int8, device=device)
        if self.dist_to_type is not None:
            result['dist_to_type'] = torch.tensor(
                self.dist_to_type, dtype=torch.int16, device=device)
            result['step_to_type'] = torch.tensor(
                self.step_to_type, dtype=torch.int8, device=device)
        return result


def precompute_for_difficulty(difficulty, verbose=True):
    """Precompute tables for a specific difficulty using captured map data."""
    from solution_store import load_capture
    from game_engine import build_map_from_capture

    capture = load_capture(difficulty)
    if not capture:
        if verbose:
            print(f"  No capture for {difficulty}, skipping", file=sys.stderr)
        return None

    ms = build_map_from_capture(capture)
    if verbose:
        print(f"\n=== {difficulty.upper()} ({ms.width}x{ms.height}) ===", file=sys.stderr)

    tables = PrecomputedTables.get(ms)

    if verbose:
        N = tables.n_cells
        dist_kb = tables.dist_matrix.nbytes / 1024
        step_kb = tables.next_step_matrix.nbytes / 1024
        cache_path = os.path.join(CACHE_DIR, f'tables_{tables.grid_hash}.npz')
        cache_kb = os.path.getsize(cache_path) / 1024 if os.path.exists(cache_path) else 0
        print(f"  Cells: {N}", file=sys.stderr)
        print(f"  dist_matrix: {tables.dist_matrix.shape} = {dist_kb:.1f} KB", file=sys.stderr)
        print(f"  next_step:   {tables.next_step_matrix.shape} = {step_kb:.1f} KB", file=sys.stderr)
        print(f"  Cache file:  {cache_kb:.1f} KB", file=sys.stderr)
        if tables.dist_to_type is not None:
            print(f"  dist_to_type: {tables.dist_to_type.shape}", file=sys.stderr)
            print(f"  num_types: {ms.num_types}, num_items: {ms.num_items}", file=sys.stderr)

    return tables


def show_info():
    """Show cache stats for all difficulties."""
    print("=== PrecomputedTables Cache ===")
    if not os.path.exists(CACHE_DIR):
        print("  No cache directory")
        return

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npz')]
    if not files:
        print("  No cached tables")
        return

    total_size = 0
    for f in sorted(files):
        path = os.path.join(CACHE_DIR, f)
        size = os.path.getsize(path)
        total_size += size
        data = np.load(path)
        N = len(data['walkable'])
        gs = tuple(data['grid_shape'])
        print(f"  {f}: {N} cells, grid {gs[1]}x{gs[0]}, {size/1024:.1f} KB")

    print(f"  Total: {len(files)} files, {total_size/1024:.1f} KB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Precompute lookup tables')
    parser.add_argument('difficulty', nargs='?', default=None,
                        choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--info', action='store_true', help='Show cache stats')
    args = parser.parse_args()

    if args.info:
        show_info()
    elif args.difficulty:
        precompute_for_difficulty(args.difficulty)
    else:
        for diff in ['easy', 'medium', 'hard', 'expert']:
            precompute_for_difficulty(diff)
