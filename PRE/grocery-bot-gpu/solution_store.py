"""Solution storage helpers for the Learn & Replay workflow.

Stores per-difficulty solutions in solutions/<difficulty>/:
  - best.json: Action sequence [[action, item_idx], ...] per round
  - capture.json: Full game capture (grid, items, orders, drop_off) for re-optimization
  - meta.json: Score, date, seed, difficulty, timestamps, optimization count
"""
import json
import os
import time
from datetime import datetime, timezone

SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solutions')
DIFFICULTIES = ['easy', 'medium', 'hard', 'expert']


def _dir(difficulty):
    d = os.path.join(SOLUTIONS_DIR, difficulty)
    os.makedirs(d, exist_ok=True)
    return d


def save_capture(difficulty, capture_data):
    """Save capture data for re-optimization."""
    path = os.path.join(_dir(difficulty), 'capture.json')
    with open(path, 'w') as f:
        json.dump(capture_data, f)
    return path


def load_capture(difficulty):
    """Load capture data. Returns None if not found."""
    path = os.path.join(_dir(difficulty), 'capture.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _capture_hash(difficulty):
    """Get a hash of the current capture file for consistency checking."""
    import hashlib
    path = os.path.join(_dir(difficulty), 'capture.json')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def save_solution(difficulty, score, actions, seed=0, force=False):
    """Save solution if it beats existing best (or no existing solution).

    Args:
        force: Always save (used when capture just changed, e.g., new game)

    Returns True if saved (new best), False if existing is better.
    """
    d = _dir(difficulty)
    meta_path = os.path.join(d, 'meta.json')
    best_path = os.path.join(d, 'best.json')

    existing_meta = load_meta(difficulty)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    cap_hash = _capture_hash(difficulty)

    if not force:
        # NEVER overwrite a better score (regardless of capture hash or date)
        if existing_meta and existing_meta.get('score', 0) >= score:
            return False

    # Save actions
    serializable = [[(int(a), int(i)) for a, i in round_actions] for round_actions in actions]
    with open(best_path, 'w') as f:
        json.dump(serializable, f)

    # Save/update meta
    now = datetime.now(timezone.utc).isoformat()
    # Preserve created_at and optimization count only if same capture
    same_capture = existing_meta and existing_meta.get('capture_hash') == cap_hash and existing_meta.get('date') == today
    meta = {
        'score': score,
        'date': today,
        'seed': seed,
        'difficulty': difficulty,
        'num_bots': len(actions[0]) if actions and actions[0] else 0,
        'num_rounds': len(actions),
        'capture_hash': cap_hash,
        'created_at': existing_meta.get('created_at', now) if same_capture else now,
        'updated_at': now,
        'optimizations_run': existing_meta.get('optimizations_run', 0) if same_capture else 0,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return True


def load_solution(difficulty):
    """Load best action sequence. Returns None if not found."""
    path = os.path.join(_dir(difficulty), 'best.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [[(a, i) for a, i in round_actions] for round_actions in data]


def load_meta(difficulty):
    """Load solution metadata. Returns None if not found."""
    path = os.path.join(_dir(difficulty), 'meta.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def increment_optimizations(difficulty):
    """Increment the optimization counter in meta.json."""
    meta = load_meta(difficulty)
    if meta:
        meta['optimizations_run'] = meta.get('optimizations_run', 0) + 1
        meta['updated_at'] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(_dir(difficulty), 'meta.json')
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)


def get_all_solutions():
    """Get meta for all difficulties. Returns dict {difficulty: meta_or_None}."""
    return {d: load_meta(d) for d in DIFFICULTIES}
