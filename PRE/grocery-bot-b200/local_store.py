"""File-based solution and capture storage (no PostgreSQL dependency).

Drop-in replacement for solution_store.py — same API, stores to local JSON files.
Data directory: grocery-bot-b200/data/<difficulty>/<date>/

Files per difficulty/date:
  capture.json      — Map data + accumulated orders
  solution.json     — Best action sequence
  meta.json         — Solution metadata (score, timestamps, etc.)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DIFFICULTIES = ['easy', 'medium', 'hard', 'expert', 'nightmare']


def _today() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _data_path(difficulty: str, date: str | None = None) -> str:
    date = date or _today()
    path = os.path.join(DATA_DIR, difficulty, date)
    os.makedirs(path, exist_ok=True)
    return path


def _capture_hash_from_data(capture_data: dict) -> str:
    raw = json.dumps(capture_data, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:12]


# ── Capture operations ──

def save_capture(difficulty: str, capture_data: dict, date: str | None = None) -> str:
    path = os.path.join(_data_path(difficulty, date), 'capture.json')
    with open(path, 'w') as f:
        json.dump(capture_data, f)
    return _capture_hash_from_data(capture_data)


def load_capture(difficulty: str, date: str | None = None) -> dict | None:
    path = os.path.join(_data_path(difficulty, date), 'capture.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def merge_capture(difficulty: str, new_capture: dict,
                  date: str | None = None) -> tuple[dict, int, int]:
    existing = load_capture(difficulty, date=date)
    if existing is None:
        save_capture(difficulty, new_capture, date=date)
        n = len(new_capture.get('orders', []))
        return new_capture, n, n

    new_types = {item['type'] for item in new_capture.get('items', [])}
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
        print(f"  WARNING: Stale capture types, discarding old data", file=sys.stderr)
        _clear_solution(difficulty, date=date)
        save_capture(difficulty, new_capture, date=date)
        n = len(new_capture.get('orders', []))
        return new_capture, n, n

    new_orders = new_capture.get('orders', [])
    if (existing_orders and new_orders
            and existing_orders[0].get('items_required') != new_orders[0].get('items_required')):
        print(f"  WARNING: First order mismatch, discarding old data", file=sys.stderr)
        _clear_solution(difficulty, date=date)
        save_capture(difficulty, new_capture, date=date)
        n = len(new_orders)
        return new_capture, n, n

    if len(new_orders) > len(existing_orders):
        merged_orders = list(new_orders)
        num_new = len(new_orders) - len(existing_orders)
    else:
        merged_orders = list(existing_orders)
        num_new = 0

    merged = dict(new_capture)
    merged['orders'] = merged_orders
    save_capture(difficulty, merged, date=date)
    return merged, num_new, len(merged_orders)


# ── Solution operations ──

def _clear_solution(difficulty: str, date: str | None = None) -> None:
    for fname in ['solution.json', 'meta.json']:
        path = os.path.join(_data_path(difficulty, date), fname)
        if os.path.exists(path):
            os.remove(path)


def save_solution(difficulty: str, score: int, actions: list,
                  seed: int = 0, force: bool = False,
                  date: str | None = None) -> bool:
    existing_meta = load_meta(difficulty, date=date)
    if not force and existing_meta and existing_meta.get('score', 0) >= score:
        return False

    date = date or _today()
    cap_hash = None
    cap = load_capture(difficulty, date=date)
    if cap:
        cap_hash = _capture_hash_from_data(cap)

    now = datetime.now(timezone.utc).isoformat()
    same_capture = (existing_meta and existing_meta.get('capture_hash') == cap_hash)
    created_at = existing_meta.get('created_at', now) if same_capture else now
    opt_count = existing_meta.get('optimizations_run', 0) if same_capture else 0

    serializable = [[(int(a), int(i)) for a, i in round_actions]
                    for round_actions in actions]
    num_bots = len(actions[0]) if actions and actions[0] else 0

    base = _data_path(difficulty, date)

    with open(os.path.join(base, 'solution.json'), 'w') as f:
        json.dump(serializable, f)

    meta = {
        'score': score,
        'seed': seed,
        'difficulty': difficulty,
        'date': date,
        'num_bots': num_bots,
        'num_rounds': len(actions),
        'capture_hash': cap_hash,
        'optimizations_run': opt_count,
        'created_at': created_at,
        'updated_at': now,
    }
    with open(os.path.join(base, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return True


def load_solution(difficulty: str, date: str | None = None) -> list | None:
    path = os.path.join(_data_path(difficulty, date), 'solution.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return [[(a, i) for a, i in round_actions] for round_actions in data]
    return None


def load_meta(difficulty: str, date: str | None = None) -> dict | None:
    path = os.path.join(_data_path(difficulty, date), 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def increment_optimizations(difficulty: str, date: str | None = None) -> None:
    meta = load_meta(difficulty, date=date)
    if meta:
        meta['optimizations_run'] = meta.get('optimizations_run', 0) + 1
        meta['updated_at'] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(_data_path(difficulty, date), 'meta.json')
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)


def clear_solutions(difficulty: str | None = None, date: str | None = None) -> None:
    diffs = [difficulty] if difficulty else DIFFICULTIES
    for d in diffs:
        _clear_solution(d, date=date)
        cap_path = os.path.join(_data_path(d, date), 'capture.json')
        if os.path.exists(cap_path):
            os.remove(cap_path)


def get_all_solutions(date: str | None = None) -> dict:
    return {d: load_meta(d, date=date) for d in DIFFICULTIES}


# ── DP Plan operations ──

def save_dp_plan(difficulty: str, plan_data: dict, date: str | None = None) -> None:
    path = os.path.join(_data_path(difficulty, date), 'dp_plan.json')
    with open(path, 'w') as f:
        json.dump(plan_data, f)


def load_dp_plan(difficulty: str, date: str | None = None) -> dict | None:
    path = os.path.join(_data_path(difficulty, date), 'dp_plan.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Compatibility shim ──
# ensure_schema() is a no-op for file-based store
def ensure_schema():
    pass


if __name__ == '__main__':
    print(f"Data directory: {DATA_DIR}")
    print(f"Current state:")
    for d, meta in get_all_solutions().items():
        if meta:
            print(f"  {d}: score={meta['score']}, date={meta['date']}")
        else:
            print(f"  {d}: no solution")
