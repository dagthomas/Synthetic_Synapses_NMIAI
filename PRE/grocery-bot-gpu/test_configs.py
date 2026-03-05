"""Tests for configs.py -- difficulty configurations and seed parsing."""
import pytest

from configs import (
    CONFIGS, ALL_TYPES, MAX_ROUNDS, INV_CAP, MAX_BOTS, MAX_ITEMS,
    MAX_ORDERS, MAX_ORDER_SIZE, DIFF_IDX,
    parse_seeds, detect_difficulty,
    _DIFFICULTY_DIMS, _BOTS_TO_DIFF,
)


# ---------------------------------------------------------------------------
# CONFIGS dict
# ---------------------------------------------------------------------------

class TestCONFIGS:
    def test_has_four_difficulties(self):
        assert set(CONFIGS.keys()) == {'easy', 'medium', 'hard', 'expert'}

    def test_each_config_has_required_keys(self):
        required = {'w', 'h', 'bots', 'aisles', 'types', 'order_size'}
        for diff, cfg in CONFIGS.items():
            assert required.issubset(cfg.keys()), f"{diff} missing keys: {required - cfg.keys()}"

    def test_easy_config_values(self):
        c = CONFIGS['easy']
        assert c['w'] == 12
        assert c['h'] == 10
        assert c['bots'] == 1
        assert c['aisles'] == 2
        assert c['types'] == 4
        assert c['order_size'] == (3, 4)

    def test_medium_config_values(self):
        c = CONFIGS['medium']
        assert c['w'] == 16
        assert c['h'] == 12
        assert c['bots'] == 3
        assert c['types'] == 8

    def test_hard_config_values(self):
        c = CONFIGS['hard']
        assert c['w'] == 22
        assert c['h'] == 14
        assert c['bots'] == 5
        assert c['types'] == 12

    def test_expert_config_values(self):
        c = CONFIGS['expert']
        assert c['w'] == 28
        assert c['h'] == 18
        assert c['bots'] == 10
        assert c['types'] == 16

    def test_dimensions_increase_with_difficulty(self):
        diffs = ['easy', 'medium', 'hard', 'expert']
        for i in range(len(diffs) - 1):
            a = CONFIGS[diffs[i]]
            b = CONFIGS[diffs[i + 1]]
            assert b['w'] > a['w'], f"{diffs[i+1]}.w should be > {diffs[i]}.w"
            assert b['h'] > a['h'], f"{diffs[i+1]}.h should be > {diffs[i]}.h"
            assert b['bots'] >= a['bots']
            assert b['types'] >= a['types']

    def test_order_size_is_tuple_of_two(self):
        for diff, cfg in CONFIGS.items():
            os = cfg['order_size']
            assert isinstance(os, tuple) and len(os) == 2, f"{diff}.order_size not a 2-tuple"
            assert os[0] <= os[1], f"{diff}.order_size min > max"


# ---------------------------------------------------------------------------
# ALL_TYPES
# ---------------------------------------------------------------------------

class TestALL_TYPES:
    def test_has_sixteen_items(self):
        assert len(ALL_TYPES) == 16

    def test_all_strings(self):
        for t in ALL_TYPES:
            assert isinstance(t, str)

    def test_no_duplicates(self):
        assert len(ALL_TYPES) == len(set(ALL_TYPES))

    def test_known_items_present(self):
        for expected in ['milk', 'bread', 'eggs', 'butter']:
            assert expected in ALL_TYPES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_rounds(self):
        assert MAX_ROUNDS == 300

    def test_inv_cap(self):
        assert INV_CAP == 3

    def test_max_bots(self):
        assert MAX_BOTS == 10

    def test_max_items(self):
        assert MAX_ITEMS == 200

    def test_max_orders(self):
        assert MAX_ORDERS == 100

    def test_max_order_size(self):
        assert MAX_ORDER_SIZE == 6


# ---------------------------------------------------------------------------
# DIFF_IDX
# ---------------------------------------------------------------------------

class TestDIFF_IDX:
    def test_has_four_entries(self):
        assert len(DIFF_IDX) == 4

    def test_values_are_sequential(self):
        assert DIFF_IDX == {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}

    def test_keys_match_configs(self):
        assert set(DIFF_IDX.keys()) == set(CONFIGS.keys())


# ---------------------------------------------------------------------------
# parse_seeds
# ---------------------------------------------------------------------------

class TestParseSeeds:
    def test_range_format(self):
        result = parse_seeds('7001-7003')
        assert result == [7001, 7002, 7003]

    def test_range_large_end(self):
        """Both endpoints >= 100: standard range."""
        result = parse_seeds('100-105')
        assert result == [100, 101, 102, 103, 104, 105]

    def test_range_small_end_means_count(self):
        """If end < 100, it's treated as start + count - 1."""
        result = parse_seeds('7001-3')
        assert result == [7001, 7002, 7003]

    def test_comma_two_seeds(self):
        result = parse_seeds('42,7001')
        assert result == [42, 7001]

    def test_comma_multiple(self):
        result = parse_seeds('1,2,3,4')
        assert result == [1, 2, 3, 4]

    def test_comma_with_spaces(self):
        result = parse_seeds('42, 100, 200')
        assert result == [42, 100, 200]

    def test_count_small_number(self):
        """A small number < 100 is treated as count from 7001."""
        result = parse_seeds('3')
        assert result == [7001, 7002, 7003]

    def test_count_one(self):
        result = parse_seeds('1')
        assert result == [7001]

    def test_single_large_seed(self):
        """A number >= 100 is treated as a single seed."""
        result = parse_seeds('7001')
        assert result == [7001]

    def test_single_seed_200(self):
        result = parse_seeds('200')
        assert result == [200]

    def test_returns_list_of_ints(self):
        for spec in ['7001-7003', '42,100', '3', '7001']:
            result = parse_seeds(spec)
            assert isinstance(result, list)
            assert all(isinstance(s, int) for s in result)


# ---------------------------------------------------------------------------
# detect_difficulty
# ---------------------------------------------------------------------------

class TestDetectDifficulty:
    def test_by_bot_count_easy(self):
        assert detect_difficulty(1) == 'easy'

    def test_by_bot_count_medium(self):
        assert detect_difficulty(3) == 'medium'

    def test_by_bot_count_hard(self):
        assert detect_difficulty(5) == 'hard'

    def test_by_bot_count_expert(self):
        assert detect_difficulty(10) == 'expert'

    def test_unknown_bot_count_returns_none(self):
        assert detect_difficulty(7) is None
        assert detect_difficulty(0) is None

    def test_with_correct_dimensions(self):
        assert detect_difficulty(1, width=12, height=10) == 'easy'
        assert detect_difficulty(3, width=16, height=12) == 'medium'
        assert detect_difficulty(5, width=22, height=14) == 'hard'
        assert detect_difficulty(10, width=28, height=18) == 'expert'

    def test_with_wrong_dimensions_returns_none(self):
        """Correct bot count but wrong dimensions -> None."""
        assert detect_difficulty(1, width=99, height=99) is None
        assert detect_difficulty(10, width=12, height=10) is None

    def test_partial_dimensions_ignored(self):
        """If only width or only height is given (other is None), falls back to bot count."""
        assert detect_difficulty(1, width=12) == 'easy'
        assert detect_difficulty(3, height=12) == 'medium'

    def test_reverse_lookup_dicts_consistent(self):
        """Internal lookup dicts match CONFIGS."""
        for name, cfg in CONFIGS.items():
            key = (cfg['w'], cfg['h'], cfg['bots'])
            assert _DIFFICULTY_DIMS[key] == name
            assert _BOTS_TO_DIFF[cfg['bots']] == name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
