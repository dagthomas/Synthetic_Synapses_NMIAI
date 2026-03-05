"""Solution storage helpers for the Learn & Replay workflow.

Stores per-difficulty solutions in solutions/<difficulty>/:
  - best.json: Action sequence [[action, item_idx], ...] per round
  - capture.json: Full game capture (grid, items, orders, drop_off) for re-optimization
  - meta.json: Score, date, seed, difficulty, timestamps, optimization count

Error contract:
  - All load_* functions return Optional values (None when file missing or corrupt).
  - save_* functions raise on I/O errors (let caller handle).
  - Never overwrites a better score unless force=True.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from game_engine import CaptureData

SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solutions')
ORDER_LISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'order_lists')
DIFFICULTIES = ['easy', 'medium', 'hard', 'expert']


def _dir(difficulty):
    d = os.path.join(SOLUTIONS_DIR, difficulty)
    os.makedirs(d, exist_ok=True)
    return d


def _clear_solution_files(difficulty: str) -> None:
    """Clear best.json and meta.json (but NOT capture.json) for a difficulty.

    Called when merge_capture detects stale data — the old solution is
    incompatible with the new capture (different seed/map/types).
    """
    d = _dir(difficulty)
    for fname in ['best.json', 'meta.json']:
        fpath = os.path.join(d, fname)
        if os.path.exists(fpath):
            os.remove(fpath)


def save_capture(difficulty: str, capture_data: CaptureData) -> str:
    """Save capture data for re-optimization. Returns the file path."""
    path = os.path.join(_dir(difficulty), 'capture.json')
    with open(path, 'w') as f:
        json.dump(capture_data, f)
    return path


def _save_order_list(difficulty: str, orders: list) -> None:
    """Auto-save orders to order_lists/<difficulty>_orders.json."""
    if not orders:
        return
    os.makedirs(ORDER_LISTS_DIR, exist_ok=True)
    path = os.path.join(ORDER_LISTS_DIR, f'{difficulty}_orders.json')
    data = {
        'difficulty': difficulty,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'total_orders': len(orders),
        'orders': [
            {'index': i, 'items_required': o.get('items_required', [])}
            for i, o in enumerate(orders)
        ],
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    import sys
    print(f"  Auto-saved {len(orders)} orders to {path}", file=sys.stderr)


def merge_capture(difficulty: str, new_capture: CaptureData) -> tuple[CaptureData, int, int]:
    """Merge new capture with existing one, keeping ALL orders by position.

    Orders are sequential and deterministic per seed -- the longer list is
    always the more complete one.  Map data (grid, items, drop_off) is
    always taken from the new capture (latest game state).

    If the existing capture's orders reference item types not present on
    the new map, the existing capture is discarded (server changed types).

    Returns:
        (merged_capture, num_new_orders, total_orders) tuple.
    """
    existing = load_capture(difficulty)
    if existing is None:
        save_capture(difficulty, new_capture)
        n = len(new_capture.get('orders', []))
        _save_order_list(difficulty, new_capture.get('orders', []))
        return new_capture, n, n

    # Build set of valid item types from the NEW map
    new_types = {item['type'] for item in new_capture.get('items', [])}

    # Check if existing orders reference types that don't exist on the new map
    existing_orders = existing.get('orders', [])
    stale = False
    for order in existing_orders:
        for item_name in order.get('items_required', []):
            if item_name not in new_types:
                stale = True
                break
        if stale:
            break

    if stale:
        import sys
        print(f"  WARNING: Existing capture has stale item types (map changed), discarding old data",
              file=sys.stderr)
        _clear_solution_files(difficulty)
        save_capture(difficulty, new_capture)
        n = len(new_capture.get('orders', []))
        _save_order_list(difficulty, new_capture.get('orders', []))
        return new_capture, n, n

    new_orders = new_capture.get('orders', [])

    # Check if first order matches (same seed = same order sequence)
    if (existing_orders and new_orders
            and existing_orders[0].get('items_required') != new_orders[0].get('items_required')):
        import sys
        print(f"  WARNING: First order mismatch (seed changed), discarding old data",
              file=sys.stderr)
        _clear_solution_files(difficulty)
        save_capture(difficulty, new_capture)
        n = len(new_orders)
        _save_order_list(difficulty, new_orders)
        return new_capture, n, n

    # Positional merge: orders are sequential per seed.
    # Take the longer list (more orders discovered = played further).
    if len(new_orders) > len(existing_orders):
        merged_orders = list(new_orders)
        num_new = len(new_orders) - len(existing_orders)
    else:
        merged_orders = list(existing_orders)
        num_new = 0

    # Use new capture's map data (always fresh) but merged orders
    merged = dict(new_capture)
    merged['orders'] = merged_orders

    save_capture(difficulty, merged)
    _save_order_list(difficulty, merged_orders)
    return merged, num_new, len(merged_orders)


def load_capture(difficulty: str) -> CaptureData | None:
    """Load capture data. Returns None if file not found."""
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
        return hashlib.md5(f.read()).hexdigest()[:12]  # nosec B324


def save_solution(difficulty: str, score: int, actions: List[List[Tuple[int, int]]],
                   seed: int = 0, force: bool = False) -> bool:
    """Save solution if it beats existing best (or no existing solution).

    Args:
        difficulty: Game difficulty level.
        score: Achieved score.
        actions: Per-round action list, each is [(act, item)] * num_bots.
        seed: Game seed (0 if unknown).
        force: Always save (used when capture just changed, e.g., new game).

    Returns:
        True if saved (new best), False if existing is better.
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


def load_solution(difficulty: str) -> Optional[List[List[Tuple[int, int]]]]:
    """Load best action sequence. Returns None if file not found."""
    path = os.path.join(_dir(difficulty), 'best.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [[(a, i) for a, i in round_actions] for round_actions in data]


def load_meta(difficulty: str) -> Optional[Dict[str, Any]]:
    """Load solution metadata.

    Returns:
        Parsed meta dict, or None if file not found or corrupt (JSONDecodeError/IOError).
        Callers should always handle the None case.
    """
    path = os.path.join(_dir(difficulty), 'meta.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def increment_optimizations(difficulty: str) -> None:
    """Increment the optimization counter in meta.json. No-op if no meta exists."""
    meta = load_meta(difficulty)
    if meta:
        meta['optimizations_run'] = meta.get('optimizations_run', 0) + 1
        meta['updated_at'] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(_dir(difficulty), 'meta.json')
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)


def clear_solutions(difficulty: Optional[str] = None) -> None:
    """Clear solution files (best.json, capture.json, meta.json) for a difficulty or all.

    Used at the start of a new pipeline run since a new token = new game = old data invalid.
    """
    diffs = [difficulty] if difficulty else DIFFICULTIES
    for d in diffs:
        dirpath = _dir(d)
        for fname in ['best.json', 'capture.json', 'meta.json']:
            fpath = os.path.join(dirpath, fname)
            if os.path.exists(fpath):
                os.remove(fpath)


def get_all_solutions() -> Dict[str, Optional[Dict[str, Any]]]:
    """Get meta for all difficulties. Returns dict {difficulty: meta_or_None}."""
    return {d: load_meta(d) for d in DIFFICULTIES}
