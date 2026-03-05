"""Tests for game_engine.py — core game simulator.

Covers: map building, game init, step logic, order delivery, inventory management.
"""
import numpy as np
import pytest

from game_engine import (
    build_map, init_game, step, generate_all_orders,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, CELL_WALL, CELL_SHELF, CELL_FLOOR, CELL_DROPOFF,
    INV_CAP, MAX_ROUNDS, Order, GameState, MapState,
)
from configs import CONFIGS, parse_seeds, detect_difficulty


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

class TestBuildMap:
    def test_easy_dimensions(self):
        ms = build_map('easy')
        assert ms.width == 12
        assert ms.height == 10

    def test_hard_dimensions(self):
        ms = build_map('hard')
        assert ms.width == 22
        assert ms.height == 14

    def test_drop_off_and_spawn(self):
        ms = build_map('easy')
        assert ms.drop_off == (1, ms.height - 2)
        assert ms.spawn == (ms.width - 2, ms.height - 2)
        assert ms.grid[ms.drop_off[1], ms.drop_off[0]] == CELL_DROPOFF

    def test_border_walls(self):
        ms = build_map('medium')
        for x in range(ms.width):
            assert ms.grid[0, x] == CELL_WALL, f"Top wall missing at x={x}"
            assert ms.grid[ms.height - 1, x] == CELL_WALL, f"Bottom wall missing at x={x}"
        for y in range(ms.height):
            assert ms.grid[y, 0] == CELL_WALL, f"Left wall missing at y={y}"
            assert ms.grid[y, ms.width - 1] == CELL_WALL, f"Right wall missing at y={y}"

    def test_items_on_shelves(self):
        ms = build_map('easy')
        for i in range(ms.num_items):
            x, y = ms.item_positions[i]
            assert ms.grid[y, x] == CELL_SHELF, f"Item {i} not on shelf: ({x},{y})"

    def test_item_types_bounded(self):
        for diff in CONFIGS:
            ms = build_map(diff)
            assert ms.num_types == CONFIGS[diff]['types']
            for i in range(ms.num_items):
                assert 0 <= ms.items[i]['type_id'] < ms.num_types


# ---------------------------------------------------------------------------
# Game init
# ---------------------------------------------------------------------------

class TestInitGame:
    def test_init_returns_state_and_orders(self):
        gs, orders = init_game(42, 'easy')
        assert isinstance(gs, GameState)
        assert len(orders) > 0

    def test_bot_count_matches_config(self):
        for diff, cfg in CONFIGS.items():
            gs, _ = init_game(42, diff)
            assert gs.bot_positions.shape[0] == cfg['bots']

    def test_initial_score_zero(self):
        gs, _ = init_game(42, 'medium')
        assert gs.score == 0
        assert gs.round == 0
        assert gs.items_delivered == 0

    def test_bots_start_at_spawn(self):
        gs, _ = init_game(42, 'easy')
        ms = gs.map_state
        for b in range(gs.bot_positions.shape[0]):
            assert gs.bot_positions[b, 0] == ms.spawn[0]
            assert gs.bot_positions[b, 1] == ms.spawn[1]

    def test_deterministic_seed(self):
        gs1, o1 = init_game(123, 'medium')
        gs2, o2 = init_game(123, 'medium')
        assert len(o1) == len(o2)
        for a, b in zip(o1, o2):
            assert list(a.required) == list(b.required)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    def test_needs(self):
        o = Order(0, [1, 2, 3])
        assert o.needs() == [1, 2, 3]
        o.deliver_type(2)
        assert 2 not in o.needs()

    def test_deliver_type(self):
        o = Order(0, [1, 1, 2])
        assert o.deliver_type(1) is True
        assert o.deliver_type(1) is True
        assert o.deliver_type(1) is False  # no more type-1 needed
        assert o.deliver_type(2) is True
        assert o.is_complete()

    def test_copy_independent(self):
        o = Order(0, [1, 2])
        o2 = o.copy()
        o2.deliver_type(1)
        assert o.needs() == [1, 2]  # original unchanged

    def test_needs_type(self):
        o = Order(0, [3, 5, 3])
        assert o.needs_type(3) is True
        assert o.needs_type(5) is True
        assert o.needs_type(0) is False


# ---------------------------------------------------------------------------
# Step logic
# ---------------------------------------------------------------------------

class TestStep:
    def test_wait_does_nothing(self):
        gs, orders = init_game(42, 'easy')
        pos_before = gs.bot_positions.copy()
        step(gs, [(ACT_WAIT, -1)], orders)
        np.testing.assert_array_equal(gs.bot_positions, pos_before)
        assert gs.round == 1

    def test_move_changes_position(self):
        gs, orders = init_game(42, 'easy')
        ms = gs.map_state
        x, y = gs.bot_positions[0]
        # Move left (should work from spawn which is near right edge)
        step(gs, [(ACT_MOVE_LEFT, -1)], orders)
        new_x = gs.bot_positions[0, 0]
        # Either moved left or was blocked
        assert new_x <= x

    def test_wall_blocks_movement(self):
        gs, orders = init_game(42, 'easy')
        ms = gs.map_state
        # Move bot into top-left corner area (near walls)
        gs.bot_positions[0] = [1, 1]  # near corner
        step(gs, [(ACT_MOVE_UP, -1)], orders)
        # Should be blocked by wall at y=0
        assert gs.bot_positions[0, 1] == 1

    def test_round_increments(self):
        gs, orders = init_game(42, 'easy')
        for i in range(5):
            step(gs, [(ACT_WAIT, -1)], orders)
        assert gs.round == 5


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_parse_seeds_range(self):
        assert parse_seeds('7001-7003') == [7001, 7002, 7003]

    def test_parse_seeds_count(self):
        result = parse_seeds('3')
        assert len(result) == 3
        assert result[0] == 7001

    def test_parse_seeds_comma(self):
        assert parse_seeds('42,100') == [42, 100]

    def test_detect_difficulty_by_bots(self):
        assert detect_difficulty(1) == 'easy'
        assert detect_difficulty(3) == 'medium'
        assert detect_difficulty(5) == 'hard'
        assert detect_difficulty(10) == 'expert'

    def test_detect_difficulty_by_dims(self):
        assert detect_difficulty(1, width=12, height=10) == 'easy'
        assert detect_difficulty(10, width=28, height=18) == 'expert'
        assert detect_difficulty(1, width=99, height=99) is None


# ---------------------------------------------------------------------------
# Full game simulation
# ---------------------------------------------------------------------------

class TestFullGame:
    def test_all_wait_scores_zero(self):
        """300 rounds of waiting should score 0."""
        gs, orders = init_game(42, 'easy')
        for _ in range(MAX_ROUNDS):
            step(gs, [(ACT_WAIT, -1)], orders)
        assert gs.score == 0
        assert gs.round == MAX_ROUNDS

    def test_medium_all_wait(self):
        """Multi-bot all-wait also scores 0."""
        gs, orders = init_game(42, 'medium')
        num_bots = gs.bot_positions.shape[0]
        for _ in range(MAX_ROUNDS):
            step(gs, [(ACT_WAIT, -1)] * num_bots, orders)
        assert gs.score == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
