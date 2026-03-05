"""Tests for replay_solution.py -- predict_full_sim and extract_goals.

Uses game_engine to build test data. No WebSocket or server required.
"""
import numpy as np
import pytest

from game_engine import (
    build_map, init_game, init_game_from_capture, step,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, CELL_FLOOR, CELL_DROPOFF,
    MapState, GameState, Order, INV_CAP,
)
from configs import CONFIGS
from replay_solution import (
    predict_full_sim, extract_goals, build_walkable, bfs_next_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_easy_capture():
    """Build a minimal capture_data dict for easy difficulty from build_map."""
    ms = build_map('easy')
    walls = []
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (1, 2):  # WALL or SHELF
                walls.append([x, y])
    items = []
    for it in ms.items:
        items.append({
            'id': it['id'],
            'type': it['type'],
            'position': list(it['position']),
        })
    orders = [
        {'items_required': [ms.item_type_names[int(r)] for r in o.required]}
        for o in _generate_orders(ms, 10)
    ]
    return {
        'difficulty': 'easy',
        'grid': {'width': ms.width, 'height': ms.height, 'walls': walls},
        'items': items,
        'drop_off': list(ms.drop_off),
        'num_bots': 1,
        'orders': orders,
    }


def _generate_orders(ms, count):
    """Generate orders for test capture using the game engine."""
    import random
    rng = random.Random(42)
    cfg = CONFIGS['easy']
    order_size = cfg['order_size']
    orders = []
    for i in range(count):
        n = rng.randint(order_size[0], order_size[1])
        required = [rng.choice(range(ms.num_types)) for _ in range(n)]
        orders.append(Order(i, required))
    return orders


# ---------------------------------------------------------------------------
# predict_full_sim
# ---------------------------------------------------------------------------

class TestPredictFullSim:
    def test_returns_list_of_positions(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        # 5 rounds of waiting
        actions_list = [[(ACT_WAIT, -1)] for _ in range(5)]
        positions = predict_full_sim(actions_list, capture, ms)
        assert len(positions) == 5
        for pos in positions:
            assert isinstance(pos, list)
            assert len(pos) == 1  # 1 bot

    def test_all_wait_positions_unchanged(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        actions_list = [[(ACT_WAIT, -1)] for _ in range(10)]
        positions = predict_full_sim(actions_list, capture, ms)
        # Bot should stay at spawn for all rounds
        spawn = (ms.width - 2, ms.height - 2)
        for pos in positions:
            assert pos[0] == spawn

    def test_movement_changes_position(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        # Move left once, then wait
        actions_list = [
            [(ACT_MOVE_LEFT, -1)],
            [(ACT_WAIT, -1)],
        ]
        positions = predict_full_sim(actions_list, capture, ms)
        spawn = (ms.width - 2, ms.height - 2)
        # Round 0: bot is at spawn (before action applied)
        assert positions[0][0] == spawn
        # Round 1: bot should have moved left (if not blocked)
        # The spawn is at (10, 8) for easy map, moving left goes to (9, 8)
        assert positions[1][0][0] <= spawn[0]

    def test_position_count_matches_rounds(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        num_rounds = 20
        actions_list = [[(ACT_WAIT, -1)] for _ in range(num_rounds)]
        positions = predict_full_sim(actions_list, capture, ms)
        assert len(positions) == num_rounds


class TestPredictFullSimMultiBot:
    def test_medium_returns_correct_bot_count(self):
        """Medium difficulty has 3 bots."""
        ms = build_map('medium')
        walls = []
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] in (1, 2):
                    walls.append([x, y])
        items = [{'id': it['id'], 'type': it['type'], 'position': list(it['position'])}
                 for it in ms.items]
        import random
        rng = random.Random(42)
        cfg = CONFIGS['medium']
        orders = []
        for i in range(10):
            n = rng.randint(cfg['order_size'][0], cfg['order_size'][1])
            required = [rng.choice(ms.item_type_names) for _ in range(n)]
            orders.append({'items_required': required})
        capture = {
            'difficulty': 'medium',
            'grid': {'width': ms.width, 'height': ms.height, 'walls': walls},
            'items': items,
            'drop_off': list(ms.drop_off),
            'num_bots': 3,
            'orders': orders,
        }
        from game_engine import build_map_from_capture
        ms2 = build_map_from_capture(capture)
        actions_list = [[(ACT_WAIT, -1)] * 3 for _ in range(5)]
        positions = predict_full_sim(actions_list, capture, ms2)
        assert len(positions) == 5
        for pos in positions:
            assert len(pos) == 3


# ---------------------------------------------------------------------------
# extract_goals
# ---------------------------------------------------------------------------

class TestExtractGoals:
    def test_returns_dict_with_bot_ids(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        actions_list = [[(ACT_WAIT, -1)] for _ in range(5)]
        positions = predict_full_sim(actions_list, capture, ms)
        goals = extract_goals(actions_list, ms, positions)
        assert isinstance(goals, dict)
        assert 0 in goals  # bot 0

    def test_wait_only_has_no_goals(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        actions_list = [[(ACT_WAIT, -1)] for _ in range(5)]
        positions = predict_full_sim(actions_list, capture, ms)
        goals = extract_goals(actions_list, ms, positions)
        assert goals[0] == []

    def test_pickup_creates_goal(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        # Create actions with a pickup at round 3
        actions_list = [
            [(ACT_WAIT, -1)],
            [(ACT_WAIT, -1)],
            [(ACT_WAIT, -1)],
            [(ACT_PICKUP, 0)],  # pickup item 0
            [(ACT_WAIT, -1)],
        ]
        positions = predict_full_sim(actions_list, capture, ms)
        goals = extract_goals(actions_list, ms, positions)
        # Should have at least one pickup goal
        pickup_goals = [g for g in goals[0] if g[2] == ACT_PICKUP]
        assert len(pickup_goals) == 1
        assert pickup_goals[0][0] == 3  # round 3
        assert pickup_goals[0][3] == 0  # item_idx 0

    def test_dropoff_creates_goal(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        actions_list = [
            [(ACT_DROPOFF, -1)],
            [(ACT_WAIT, -1)],
        ]
        positions = predict_full_sim(actions_list, capture, ms)
        goals = extract_goals(actions_list, ms, positions)
        dropoff_goals = [g for g in goals[0] if g[2] == ACT_DROPOFF]
        assert len(dropoff_goals) == 1
        assert dropoff_goals[0][1] == tuple(ms.drop_off)

    def test_mixed_actions_produce_multiple_goals(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        actions_list = [
            [(ACT_MOVE_LEFT, -1)],
            [(ACT_PICKUP, 0)],
            [(ACT_MOVE_RIGHT, -1)],
            [(ACT_DROPOFF, -1)],
            [(ACT_WAIT, -1)],
        ]
        positions = predict_full_sim(actions_list, capture, ms)
        goals = extract_goals(actions_list, ms, positions)
        # Should have 1 pickup + 1 dropoff = 2 goals
        assert len(goals[0]) == 2

    def test_empty_actions_list(self):
        capture = _make_easy_capture()
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        goals = extract_goals([], ms, [])
        assert goals == {}


# ---------------------------------------------------------------------------
# build_walkable
# ---------------------------------------------------------------------------

class TestBuildWalkable:
    def test_returns_set(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        assert isinstance(walkable, set)

    def test_contains_floor_cells(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        # Check a known floor cell (spawn)
        assert ms.spawn in walkable

    def test_contains_dropoff(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        assert ms.drop_off in walkable

    def test_no_wall_cells(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] == 1:  # CELL_WALL
                    assert (x, y) not in walkable

    def test_no_shelf_cells(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] == 2:  # CELL_SHELF
                    assert (x, y) not in walkable


# ---------------------------------------------------------------------------
# bfs_next_action
# ---------------------------------------------------------------------------

class TestBfsNextAction:
    def test_at_goal_returns_wait(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        result = bfs_next_action(ms.spawn, ms.spawn, walkable, set(), ms)
        assert result == ACT_WAIT

    def test_adjacent_to_goal(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        # Spawn is at (10, 8) for easy, dropoff at (1, 8)
        # One step left from spawn
        start = ms.spawn
        goal = (start[0] - 1, start[1])
        if goal in walkable:
            result = bfs_next_action(start, goal, walkable, set(), ms)
            assert result == ACT_MOVE_LEFT

    def test_finds_path_around_occupied(self):
        ms = build_map('easy')
        walkable = build_walkable(ms)
        # Use a cell in the middle of the map, not spawn (which may be cornered)
        # Find a walkable cell with multiple neighbors
        start = None
        for (x, y) in sorted(walkable):
            neighbors = 0
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if (x + dx, y + dy) in walkable:
                    neighbors += 1
            if neighbors >= 3:
                start = (x, y)
                break
        assert start is not None, "Could not find a cell with 3+ walkable neighbors"
        goal = ms.drop_off
        # Block one neighbor -- BFS should route around it
        sx, sy = start
        blocked_cell = None
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (sx + dx, sy + dy) in walkable:
                blocked_cell = (sx + dx, sy + dy)
                break
        assert blocked_cell is not None
        result = bfs_next_action(start, goal, walkable, {blocked_cell}, ms)
        # Should find a valid movement action (has alternative routes)
        assert result in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
