"""Solution storage helpers — PostgreSQL backend.

Stores per-difficulty data in PostgreSQL (localhost:5433/grocery_bot):
  - captures: Map data + accumulated orders for GPU solver
  - gpu_solutions: Best action sequences per difficulty
  - order_sequences: Order lists per difficulty/seed

Error contract:
  - All load_* functions return Optional values (None when not found).
  - save_* functions raise on DB errors (let caller handle).
  - Never overwrites a better score unless force=True.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json

from game_engine import CaptureData

DB_URL = os.environ.get(
    "GROCERY_DB_URL",
    "postgres://grocery:grocery123@localhost:5433/grocery_bot",
)

# Legacy file paths (for migration only)
SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solutions')
ORDER_LISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'order_lists')
DIFFICULTIES = ['easy', 'medium', 'hard', 'expert']


def _conn():
    """Get a DB connection."""
    return psycopg2.connect(DB_URL)


def _capture_hash_from_data(capture_data: dict) -> str:
    """Compute hash of capture data for consistency checking."""
    raw = json.dumps(capture_data, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:12]  # nosec B324


# ── Capture operations ──

def save_capture(difficulty: str, capture_data: CaptureData) -> str:
    """Save capture data for re-optimization. Returns capture hash."""
    num_orders = len(capture_data.get('orders', []))
    cap_hash = _capture_hash_from_data(capture_data)
    now = datetime.now(timezone.utc)

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO captures (difficulty, capture_data, num_orders, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (difficulty) DO UPDATE SET
                    capture_data = EXCLUDED.capture_data,
                    num_orders = EXCLUDED.num_orders,
                    updated_at = EXCLUDED.updated_at
            """, (difficulty, Json(capture_data), num_orders, now, now))
        conn.commit()

    # Also save order list
    _save_order_list(difficulty, capture_data.get('orders', []))
    return cap_hash


def load_capture(difficulty: str) -> CaptureData | None:
    """Load capture data. Returns None if not found."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT capture_data FROM captures WHERE difficulty = %s",
                    (difficulty,))
                row = cur.fetchone()
                if row:
                    return row[0]
    except Exception as e:
        print(f"  DB load_capture error: {e}", file=sys.stderr)
    return None


def _capture_hash(difficulty: str) -> Optional[str]:
    """Get hash of current capture data."""
    cap = load_capture(difficulty)
    if cap is None:
        return None
    return _capture_hash_from_data(cap)


def _save_order_list(difficulty: str, orders: list) -> None:
    """Save orders to order_sequences table."""
    if not orders:
        return

    # Try to get map_seed from capture
    map_seed = 0
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT map_seed FROM captures WHERE difficulty = %s",
                    (difficulty,))
                row = cur.fetchone()
                if row and row[0]:
                    map_seed = row[0]
    except Exception:
        pass

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    order_data = [
        {'index': i, 'items_required': o.get('items_required', [])}
        for i, o in enumerate(orders)
    ]

    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO order_sequences (difficulty, map_seed, orders, total_orders, date, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (difficulty, map_seed) DO UPDATE SET
                        orders = EXCLUDED.orders,
                        total_orders = EXCLUDED.total_orders,
                        date = EXCLUDED.date,
                        updated_at = NOW()
                """, (difficulty, map_seed, Json(order_data), len(orders), today))
            conn.commit()
        print(f"  Saved {len(orders)} orders to DB ({difficulty}, seed={map_seed})",
              file=sys.stderr)
    except Exception as e:
        print(f"  DB save_order_list error: {e}", file=sys.stderr)


def merge_capture(difficulty: str, new_capture: CaptureData) -> tuple[CaptureData, int, int]:
    """Merge new capture with existing one, keeping ALL orders by position.

    Orders are sequential and deterministic per seed -- the longer list is
    always the more complete one. Map data is always taken from new capture.

    Returns:
        (merged_capture, num_new_orders, total_orders) tuple.
    """
    existing = load_capture(difficulty)
    if existing is None:
        save_capture(difficulty, new_capture)
        n = len(new_capture.get('orders', []))
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
        print(f"  WARNING: Existing capture has stale item types (map changed), discarding old data",
              file=sys.stderr)
        _clear_solution(difficulty)
        save_capture(difficulty, new_capture)
        n = len(new_capture.get('orders', []))
        return new_capture, n, n

    new_orders = new_capture.get('orders', [])

    # Check if first order matches (same seed = same order sequence)
    if (existing_orders and new_orders
            and existing_orders[0].get('items_required') != new_orders[0].get('items_required')):
        print(f"  WARNING: First order mismatch (seed changed), discarding old data",
              file=sys.stderr)
        _clear_solution(difficulty)
        save_capture(difficulty, new_capture)
        n = len(new_orders)
        return new_capture, n, n

    # Positional merge: take the longer list
    if len(new_orders) > len(existing_orders):
        merged_orders = list(new_orders)
        num_new = len(new_orders) - len(existing_orders)
    else:
        merged_orders = list(existing_orders)
        num_new = 0

    # Use new capture's map data but merged orders
    merged = dict(new_capture)
    merged['orders'] = merged_orders

    save_capture(difficulty, merged)
    return merged, num_new, len(merged_orders)


# ── Solution operations ──

def _clear_solution(difficulty: str) -> None:
    """Clear solution for a difficulty (when capture becomes stale)."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM gpu_solutions WHERE difficulty = %s", (difficulty,))
            conn.commit()
    except Exception as e:
        print(f"  DB _clear_solution error: {e}", file=sys.stderr)


def save_solution(difficulty: str, score: int, actions: List[List[Tuple[int, int]]],
                   seed: int = 0, force: bool = False) -> bool:
    """Save solution if it beats existing best.

    Returns True if saved (new best), False if existing is better.
    """
    cap_hash = _capture_hash(difficulty)
    now = datetime.now(timezone.utc)
    today = now.strftime('%Y-%m-%d')

    existing_meta = load_meta(difficulty)

    if not force:
        if existing_meta and existing_meta.get('score', 0) >= score:
            return False

    # Serialize actions
    serializable = [[(int(a), int(i)) for a, i in round_actions] for round_actions in actions]
    num_bots = len(actions[0]) if actions and actions[0] else 0

    # Preserve created_at and optimization count if same capture
    same_capture = (existing_meta
                    and existing_meta.get('capture_hash') == cap_hash
                    and existing_meta.get('date') == today)
    created_at = (existing_meta.get('created_at', now.isoformat())
                  if same_capture and existing_meta else now.isoformat())
    opt_count = (existing_meta.get('optimizations_run', 0)
                 if same_capture and existing_meta else 0)

    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO gpu_solutions
                        (difficulty, map_seed, score, actions, num_bots, num_rounds,
                         capture_hash, optimizations_run, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (difficulty) DO UPDATE SET
                        map_seed = EXCLUDED.map_seed,
                        score = EXCLUDED.score,
                        actions = EXCLUDED.actions,
                        num_bots = EXCLUDED.num_bots,
                        num_rounds = EXCLUDED.num_rounds,
                        capture_hash = EXCLUDED.capture_hash,
                        optimizations_run = EXCLUDED.optimizations_run,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at
                """, (difficulty, seed, score, Json(serializable), num_bots,
                      len(actions), cap_hash, opt_count, created_at, now))
            conn.commit()
    except Exception as e:
        print(f"  DB save_solution error: {e}", file=sys.stderr)
        return False

    return True


def load_solution(difficulty: str) -> Optional[List[List[Tuple[int, int]]]]:
    """Load best action sequence. Returns None if not found."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT actions FROM gpu_solutions WHERE difficulty = %s",
                    (difficulty,))
                row = cur.fetchone()
                if row:
                    return [[(a, i) for a, i in round_actions] for round_actions in row[0]]
    except Exception as e:
        print(f"  DB load_solution error: {e}", file=sys.stderr)
    return None


def load_meta(difficulty: str) -> Optional[Dict[str, Any]]:
    """Load solution metadata. Returns None if not found."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT score, map_seed, num_bots, num_rounds, capture_hash,
                           optimizations_run, created_at, updated_at
                    FROM gpu_solutions WHERE difficulty = %s
                """, (difficulty,))
                row = cur.fetchone()
                if row:
                    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    return {
                        'score': row[0],
                        'seed': row[1] or 0,
                        'difficulty': difficulty,
                        'num_bots': row[2],
                        'num_rounds': row[3],
                        'capture_hash': row[4],
                        'optimizations_run': row[5] or 0,
                        'date': row[6].strftime('%Y-%m-%d') if row[6] else today,
                        'created_at': row[6].isoformat() if row[6] else None,
                        'updated_at': row[7].isoformat() if row[7] else None,
                    }
    except Exception as e:
        print(f"  DB load_meta error: {e}", file=sys.stderr)
    return None


def increment_optimizations(difficulty: str) -> None:
    """Increment the optimization counter. No-op if no solution exists."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE gpu_solutions
                    SET optimizations_run = optimizations_run + 1,
                        updated_at = NOW()
                    WHERE difficulty = %s
                """, (difficulty,))
            conn.commit()
    except Exception as e:
        print(f"  DB increment_optimizations error: {e}", file=sys.stderr)


def clear_solutions(difficulty: Optional[str] = None) -> None:
    """Clear solutions and captures for a difficulty or all."""
    diffs = [difficulty] if difficulty else DIFFICULTIES
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                for d in diffs:
                    cur.execute("DELETE FROM gpu_solutions WHERE difficulty = %s", (d,))
                    cur.execute("DELETE FROM captures WHERE difficulty = %s", (d,))
            conn.commit()
    except Exception as e:
        print(f"  DB clear_solutions error: {e}", file=sys.stderr)


def get_all_solutions() -> Dict[str, Optional[Dict[str, Any]]]:
    """Get meta for all difficulties."""
    return {d: load_meta(d) for d in DIFFICULTIES}


# ── Migration helper ──

def migrate_files_to_db() -> None:
    """One-time migration: import existing file-based data into PostgreSQL."""
    for diff in DIFFICULTIES:
        # Migrate capture
        cap_path = os.path.join(SOLUTIONS_DIR, diff, 'capture.json')
        if os.path.exists(cap_path):
            with open(cap_path) as f:
                cap = json.load(f)
            save_capture(diff, cap)
            print(f"  Migrated {diff} capture ({len(cap.get('orders', []))} orders)")

        # Migrate solution
        best_path = os.path.join(SOLUTIONS_DIR, diff, 'best.json')
        meta_path = os.path.join(SOLUTIONS_DIR, diff, 'meta.json')
        if os.path.exists(best_path) and os.path.exists(meta_path):
            with open(best_path) as f:
                actions_raw = json.load(f)
            with open(meta_path) as f:
                meta = json.load(f)
            actions = [[(a, i) for a, i in round_actions] for round_actions in actions_raw]
            save_solution(diff, meta.get('score', 0), actions,
                          seed=meta.get('seed', 0), force=True)
            print(f"  Migrated {diff} solution (score={meta.get('score', 0)})")

        # Migrate order list
        ol_path = os.path.join(ORDER_LISTS_DIR, f'{diff}_orders.json')
        if os.path.exists(ol_path):
            with open(ol_path) as f:
                ol = json.load(f)
            orders = ol.get('orders', [])
            if orders:
                # Convert to capture-style format
                cap_orders = [{'items_required': o['items_required']} for o in orders]
                _save_order_list(diff, cap_orders)
                print(f"  Migrated {diff} order list ({len(orders)} orders)")


if __name__ == '__main__':
    print("Migrating file-based data to PostgreSQL...")
    migrate_files_to_db()
    print("\nCurrent state:")
    for d, meta in get_all_solutions().items():
        if meta:
            print(f"  {d}: score={meta['score']}, orders=?")
        else:
            print(f"  {d}: no solution")
