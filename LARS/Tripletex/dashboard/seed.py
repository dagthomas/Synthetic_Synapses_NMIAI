"""Export/import dashboard data as JSON for sharing via git.

Usage:
    python dashboard/seed.py export          # DB → seed_data.json
    python dashboard/seed.py import          # seed_data.json → DB (merges, skips existing)
    python dashboard/seed.py import --reset  # Wipe DB first, then import
"""

import json
import os
import sqlite3
import sys
from contextlib import closing

DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(DIR, "dashboard.db")
SEED_PATH = os.path.join(DIR, "seed_data.json")

TABLES = ["eval_runs", "solve_logs"]


def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def export_data(path=SEED_PATH):
    """Export all dashboard tables to JSON."""
    data = {}
    with closing(_get_conn()) as conn:
        for table in TABLES:
            try:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                data[table] = [dict(r) for r in rows]
            except sqlite3.OperationalError:
                data[table] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    total = sum(len(v) for v in data.values())
    print(f"Exported {total} rows to {os.path.basename(path)}")
    for table, rows in data.items():
        print(f"  {table}: {len(rows)} rows")


def import_data(path=SEED_PATH, reset=False):
    """Import JSON seed data into dashboard DB."""
    if not os.path.exists(path):
        print(f"No seed file found at {os.path.basename(path)}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure DB schema exists
    from db import init_db
    init_db()

    with closing(_get_conn()) as conn:
        if reset:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table}")
            print("Cleared existing data.")

        imported = 0
        skipped = 0
        for table in TABLES:
            rows = data.get(table, [])
            if not rows:
                continue

            cols = [k for k in rows[0].keys()]
            placeholders = ", ".join("?" * len(cols))
            col_names = ", ".join(cols)

            for row in rows:
                vals = [row.get(c) for c in cols]
                try:
                    conn.execute(
                        f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})",
                        vals,
                    )
                    if conn.total_changes:
                        imported += 1
                    else:
                        skipped += 1
                except sqlite3.IntegrityError:
                    skipped += 1

        conn.commit()

    total = sum(len(data.get(t, [])) for t in TABLES)
    print(f"Imported {total} rows from {os.path.basename(path)}")
    if reset:
        print("  (full reset — all rows replaced)")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "import"):
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "export":
        export_data()
    elif cmd == "import":
        reset = "--reset" in sys.argv
        import_data(reset=reset)
