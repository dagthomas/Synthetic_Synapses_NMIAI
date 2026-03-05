"""Tests for solution_store.py — score-safe solution storage."""
import json
import os
import pytest

from solution_store import save_solution, load_solution, load_meta, save_capture


class TestSaveAndLoad:
    def test_save_and_load_solution(self, tmp_path, monkeypatch):
        monkeypatch.setattr('solution_store.SOLUTIONS_DIR', str(tmp_path))
        os.makedirs(str(tmp_path / 'easy'), exist_ok=True)

        actions = [[(0, -1), (1, -1)] for _ in range(300)]
        save_solution('easy', 42, actions, force=True)

        sol = load_solution('easy')
        assert sol is not None
        meta = load_meta('easy')
        assert meta is not None
        assert meta['score'] == 42

    def test_never_overwrites_better(self, tmp_path, monkeypatch):
        monkeypatch.setattr('solution_store.SOLUTIONS_DIR', str(tmp_path))
        os.makedirs(str(tmp_path / 'hard'), exist_ok=True)

        actions = [[(0, -1)] * 5 for _ in range(300)]
        save_solution('hard', 100, actions, force=True)
        # Second save with lower score should NOT overwrite
        saved = save_solution('hard', 50, actions)
        assert saved is False

        meta = load_meta('hard')
        assert meta['score'] == 100  # Better score preserved


class TestSaveCapture:
    def test_save_capture_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr('solution_store.SOLUTIONS_DIR', str(tmp_path))
        os.makedirs(str(tmp_path / 'easy'), exist_ok=True)

        capture = {
            'difficulty': 'easy',
            'grid': {'width': 12, 'height': 10},
            'items': [],
            'drop_off': [1, 8],
            'num_bots': 1,
            'orders': [{'items_required': ['milk', 'bread']}],
        }
        path = save_capture('easy', capture)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['difficulty'] == 'easy'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
