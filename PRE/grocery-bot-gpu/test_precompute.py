"""Tests for precompute.py -- BFS shortest paths and navigation tables.

All tests run on CPU only (no CUDA required). GPU-specific paths are
either tested via the CPU fallback or skipped.
"""
import os
import numpy as np
import pytest

from game_engine import (
    MapState, CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
    build_map,
)
from precompute import PrecomputedTables, _enum_multisets, _tables_cache


# ---------------------------------------------------------------------------
# Helpers: build tiny test maps
# ---------------------------------------------------------------------------

def _make_5x5_open_map():
    """Create a 5x5 map with walls on the border, floor inside, dropoff at (1,3).

    Layout (0=floor, 1=wall, 3=dropoff):
        W W W W W
        W . . . W
        W . . . W
        W D . . W
        W W W W W

    Walkable cells: (1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3),(2,3),(3,3)
    Dropoff: (1,3)
    """
    ms = MapState()
    ms.width = 5
    ms.height = 5
    ms.grid = np.zeros((5, 5), dtype=np.int8)
    # Border walls
    for x in range(5):
        ms.grid[0, x] = CELL_WALL
        ms.grid[4, x] = CELL_WALL
    for y in range(5):
        ms.grid[y, 0] = CELL_WALL
        ms.grid[y, 4] = CELL_WALL
    # Dropoff at (1,3)
    ms.grid[3, 1] = CELL_DROPOFF
    ms.drop_off = (1, 3)
    ms.spawn = (3, 3)
    # No items
    ms.items = []
    ms.item_positions = np.zeros((0, 2), dtype=np.int16)
    ms.item_types = np.zeros(0, dtype=np.int8)
    ms.item_type_names = []
    ms.type_name_to_id = {}
    ms.num_types = 0
    ms.num_items = 0
    ms.item_adjacencies = {}
    return ms


def _make_corridor_map():
    """Create a 7x3 corridor map:

        W W W W W W W
        W . . . . . W
        W W W W W W W

    Walkable cells: (1,1) through (5,1) -- a straight horizontal corridor.
    Dropoff at (1,1).
    """
    ms = MapState()
    ms.width = 7
    ms.height = 3
    ms.grid = np.ones((3, 7), dtype=np.int8)  # all walls
    for x in range(1, 6):
        ms.grid[1, x] = CELL_FLOOR
    ms.grid[1, 1] = CELL_DROPOFF
    ms.drop_off = (1, 1)
    ms.spawn = (5, 1)
    ms.items = []
    ms.item_positions = np.zeros((0, 2), dtype=np.int16)
    ms.item_types = np.zeros(0, dtype=np.int8)
    ms.item_type_names = []
    ms.type_name_to_id = {}
    ms.num_types = 0
    ms.num_items = 0
    ms.item_adjacencies = {}
    return ms


def _make_map_with_items():
    """Create a 7x5 map with a shelf and items for testing item_adjacencies.

    Layout:
        W W W W W W W
        W . S . . . W
        W . . . . . W
        W D . . . . W
        W W W W W W W

    Shelf at (2,1) with one item of type 0.
    Dropoff at (1,3).
    """
    ms = MapState()
    ms.width = 7
    ms.height = 5
    ms.grid = np.zeros((5, 7), dtype=np.int8)
    # Border walls
    for x in range(7):
        ms.grid[0, x] = CELL_WALL
        ms.grid[4, x] = CELL_WALL
    for y in range(5):
        ms.grid[y, 0] = CELL_WALL
        ms.grid[y, 6] = CELL_WALL
    # Shelf at (2,1)
    ms.grid[1, 2] = CELL_SHELF
    # Dropoff at (1,3)
    ms.grid[3, 1] = CELL_DROPOFF
    ms.drop_off = (1, 3)
    ms.spawn = (5, 3)

    # One item on the shelf
    ms.items = [{
        'id': 'item_0',
        'type': 'milk',
        'type_id': 0,
        'position': (2, 1),
    }]
    ms.item_positions = np.array([[2, 1]], dtype=np.int16)
    ms.item_types = np.array([0], dtype=np.int8)
    ms.item_type_names = ['milk']
    ms.type_name_to_id = {'milk': 0}
    ms.num_types = 1
    ms.num_items = 1
    # Adjacencies: walkable cells next to (2,1)
    adj = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = 2 + dx, 1 + dy
        if 0 <= nx < 7 and 0 <= ny < 5:
            cell = ms.grid[ny, nx]
            if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                adj.append((nx, ny))
    ms.item_adjacencies = {0: adj}
    return ms


# ---------------------------------------------------------------------------
# Fixture: clear module-level cache between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_tables_cache():
    """Ensure module-level cache is clear before each test."""
    _tables_cache.clear()
    yield
    _tables_cache.clear()


# ---------------------------------------------------------------------------
# CPU BFS tests
# ---------------------------------------------------------------------------

class TestCPUBFS:
    """Test _cpu_bfs via PrecomputedTables.get() on small maps (no GPU needed)."""

    def test_5x5_cell_count(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # 3x3 interior = 9 walkable cells
        assert tables.n_cells == 9

    def test_5x5_distance_self(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # Distance from any cell to itself is 0
        for pos in tables.walkable:
            assert tables.get_distance(pos, pos) == 0

    def test_5x5_adjacent_distance_one(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # (1,1) and (2,1) are adjacent -> distance 1
        assert tables.get_distance((1, 1), (2, 1)) == 1
        assert tables.get_distance((2, 1), (1, 1)) == 1

    def test_5x5_corner_to_corner(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # (1,1) to (3,3): Manhattan distance = 4 (open grid, no obstacles)
        assert tables.get_distance((1, 1), (3, 3)) == 4

    def test_5x5_symmetry(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        for a in tables.walkable:
            for b in tables.walkable:
                assert tables.get_distance(a, b) == tables.get_distance(b, a)

    def test_corridor_distances(self):
        ms = _make_corridor_map()
        tables = PrecomputedTables.get(ms)
        # 5 cells in a line: (1,1) through (5,1)
        assert tables.n_cells == 5
        assert tables.get_distance((1, 1), (5, 1)) == 4
        assert tables.get_distance((2, 1), (4, 1)) == 2

    def test_invalid_cell_returns_sentinel(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # Wall cell (0,0) is not walkable
        assert tables.get_distance((0, 0), (1, 1)) == 9999
        assert tables.get_distance((1, 1), (0, 0)) == 9999

    def test_first_step_at_target_is_zero(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert tables.get_first_step((2, 2), (2, 2)) == 0

    def test_first_step_returns_valid_action(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        # Valid actions are 1-4
        step = tables.get_first_step((1, 1), (3, 3))
        assert step in (1, 2, 3, 4)

    def test_first_step_invalid_cell_returns_zero(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert tables.get_first_step((0, 0), (1, 1)) == 0


# ---------------------------------------------------------------------------
# PrecomputedTables.get() caching and types
# ---------------------------------------------------------------------------

class TestPrecomputedTablesAPI:
    def test_get_returns_precomputed_tables(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert isinstance(tables, PrecomputedTables)

    def test_dist_matrix_type(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert isinstance(tables.dist_matrix, np.ndarray)
        assert tables.dist_matrix.dtype == np.int16

    def test_next_step_matrix_type(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert isinstance(tables.next_step_matrix, np.ndarray)
        assert tables.next_step_matrix.dtype == np.int8

    def test_matrix_shapes_match(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        N = tables.n_cells
        assert tables.dist_matrix.shape == (N, N)
        assert tables.next_step_matrix.shape == (N, N)

    def test_walkable_list(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert isinstance(tables.walkable, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in tables.walkable)

    def test_pos_to_idx_consistent(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        for idx, pos in enumerate(tables.walkable):
            assert tables.pos_to_idx[pos] == idx

    def test_grid_shape(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        assert tables.grid_shape == (5, 5)

    def test_memory_cache_returns_same_object(self):
        ms = _make_5x5_open_map()
        t1 = PrecomputedTables.get(ms)
        t2 = PrecomputedTables.get(ms)
        assert t1 is t2


# ---------------------------------------------------------------------------
# as_dist_maps
# ---------------------------------------------------------------------------

class TestAsDistMaps:
    def test_returns_dict(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        dm = tables.as_dist_maps()
        assert isinstance(dm, dict)

    def test_keys_are_walkable_cells(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        dm = tables.as_dist_maps()
        assert set(dm.keys()) == set(tables.walkable)

    def test_dist_map_shape(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        dm = tables.as_dist_maps()
        H, W = tables.grid_shape
        for key, arr in dm.items():
            assert arr.shape == (H, W)

    def test_dist_map_self_is_zero(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        dm = tables.as_dist_maps()
        for (x, y), arr in dm.items():
            assert arr[y, x] == 0

    def test_wall_cells_negative_one(self):
        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)
        dm = tables.as_dist_maps()
        src = tables.walkable[0]
        arr = dm[src]
        # Wall at (0,0) should be -1 (unreachable)
        assert arr[0, 0] == -1


# ---------------------------------------------------------------------------
# Item adjacencies and item tables
# ---------------------------------------------------------------------------

class TestItemTables:
    def test_item_adjacencies_computation(self):
        ms = _make_map_with_items()
        # Shelf at (2,1). Adjacent walkable cells:
        # (1,1)=floor, (3,1)=floor, (2,2)=floor. (2,0) is wall.
        adj = ms.item_adjacencies[0]
        adj_set = set(adj)
        assert (1, 1) in adj_set, "Left neighbor of shelf should be walkable"
        assert (3, 1) in adj_set, "Right neighbor of shelf should be walkable"
        assert (2, 2) in adj_set, "Below shelf should be walkable"
        assert (2, 0) not in adj_set, "Above shelf is wall, should not be in adj"

    def test_dist_to_type_computed(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        assert tables.dist_to_type is not None
        assert tables.dist_to_type.shape[0] == 1  # 1 type
        H, W = tables.grid_shape
        assert tables.dist_to_type.shape == (1, H, W)

    def test_step_to_type_computed(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        assert tables.step_to_type is not None
        assert tables.step_to_type.shape == tables.dist_to_type.shape

    def test_dist_to_dropoff_computed(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        assert tables.dist_to_dropoff is not None
        H, W = tables.grid_shape
        assert tables.dist_to_dropoff.shape == (H, W)

    def test_dist_to_type_at_adjacent_cell(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        # (1,1) is adjacent to the shelf -> dist_to_type[0, 1, 1] = 0
        assert tables.dist_to_type[0, 1, 1] == 0

    def test_dist_to_dropoff_at_dropoff(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        dx, dy = ms.drop_off
        assert tables.dist_to_dropoff[dy, dx] == 0


# ---------------------------------------------------------------------------
# get_nearest_item_cell
# ---------------------------------------------------------------------------

class TestGetNearestItemCell:
    def test_returns_tuple_with_distance(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        result = tables.get_nearest_item_cell((1, 1), 0, ms)
        assert result is not None
        x, y, d = result
        assert d >= 0

    def test_from_adjacent_cell_distance_zero(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        # (1,1) is adjacent to item 0
        result = tables.get_nearest_item_cell((1, 1), 0, ms)
        assert result is not None
        assert result[2] == 0  # already at adj cell

    def test_from_far_cell(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        # (5,3) is far from item 0
        result = tables.get_nearest_item_cell((5, 3), 0, ms)
        assert result is not None
        assert result[2] > 0

    def test_invalid_source_returns_none(self):
        ms = _make_map_with_items()
        tables = PrecomputedTables.get(ms)
        result = tables.get_nearest_item_cell((0, 0), 0, ms)
        assert result is None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class TestDiskCache:
    def test_save_and_load_cache(self, tmp_path, monkeypatch):
        """Test that tables can be saved to and loaded from disk cache."""
        monkeypatch.setattr('precompute.CACHE_DIR', str(tmp_path))

        ms = _make_5x5_open_map()
        tables = PrecomputedTables.get(ms)

        # Verify cache file was created
        cache_files = list(tmp_path.glob('tables_*.npz'))
        assert len(cache_files) == 1

        # Clear memory cache and reload from disk
        _tables_cache.clear()
        tables2 = PrecomputedTables.get(ms)

        assert tables2.n_cells == tables.n_cells
        np.testing.assert_array_equal(tables2.dist_matrix, tables.dist_matrix)
        np.testing.assert_array_equal(tables2.next_step_matrix, tables.next_step_matrix)


# ---------------------------------------------------------------------------
# _enum_multisets helper
# ---------------------------------------------------------------------------

class TestEnumMultisets:
    def test_single_type_max_size_1(self):
        combos, idx = _enum_multisets(1, 1)
        assert combos == [(0,)]
        assert len(idx) == 1

    def test_two_types_max_size_2(self):
        combos, idx = _enum_multisets(2, 2)
        # Size 1: (0,), (1,)
        # Size 2: (0,0), (0,1), (1,1)
        assert (0,) in combos
        assert (1,) in combos
        assert (0, 0) in combos
        assert (0, 1) in combos
        assert (1, 1) in combos
        assert len(combos) == 5

    def test_index_map_consistent(self):
        combos, idx = _enum_multisets(3, 3)
        for i, combo in enumerate(combos):
            assert idx[combo] == i

    def test_all_sorted(self):
        combos, _ = _enum_multisets(4, 3)
        for combo in combos:
            assert combo == tuple(sorted(combo))


# ---------------------------------------------------------------------------
# Integration: real map via build_map
# ---------------------------------------------------------------------------

class TestWithRealMap:
    def test_easy_map_computes(self):
        ms = build_map('easy')
        tables = PrecomputedTables.get(ms)
        assert tables.n_cells > 0
        # All distances should be >= 0 (no negative distances)
        assert (tables.dist_matrix >= 0).all()

    def test_easy_spawn_to_dropoff(self):
        ms = build_map('easy')
        tables = PrecomputedTables.get(ms)
        d = tables.get_distance(ms.spawn, ms.drop_off)
        # Spawn and dropoff should be reachable
        assert 0 < d < 9999

    def test_easy_dropoff_distance_is_reasonable(self):
        ms = build_map('easy')
        tables = PrecomputedTables.get(ms)
        d = tables.get_distance(ms.spawn, ms.drop_off)
        # Easy map is 12x10, max possible distance = 10+8=18, should be less
        assert d <= ms.width + ms.height


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
