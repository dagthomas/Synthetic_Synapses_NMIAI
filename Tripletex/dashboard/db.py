"""SQLite database for eval dashboard — schema + queries."""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "dashboard.db")

# Whitelist of columns allowed in update_run
_UPDATE_COLUMNS = frozenset({
    "status", "api_calls", "api_errors", "elapsed_seconds",
    "correctness", "base_score", "efficiency_bonus", "final_score",
    "max_possible", "checks_json", "error_message",
})


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_conn()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        # Migrate: add tool_calls_json column if missing
        try:
            conn.execute("ALTER TABLE solve_logs ADD COLUMN tool_calls_json TEXT")
        except Exception:
            pass  # Column already exists
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL,
            tier INTEGER,
            language TEXT,
            prompt TEXT,
            expected_json TEXT,
            status TEXT DEFAULT 'pending',
            agent_url TEXT,
            api_calls INTEGER DEFAULT 0,
            api_errors INTEGER DEFAULT 0,
            elapsed_seconds REAL DEFAULT 0,
            correctness REAL,
            base_score REAL,
            efficiency_bonus REAL,
            final_score REAL,
            max_possible REAL,
            checks_json TEXT,
            error_message TEXT,
            created_at TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS solve_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            prompt TEXT,
            files_json TEXT,
            base_url TEXT,
            api_calls INTEGER DEFAULT 0,
            api_errors INTEGER DEFAULT 0,
            elapsed_seconds REAL DEFAULT 0,
            agent_response TEXT,
            tool_calls_json TEXT,
            created_at TEXT
        );
        """)
        conn.commit()


# ── eval_runs CRUD ──────────────────────────────────────────────────

def create_run(*, task_name, tier, language, prompt, expected_json, agent_url):
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """INSERT INTO eval_runs
               (task_name, tier, language, prompt, expected_json, status, agent_url, created_at)
               VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
            (task_name, tier, language, prompt, expected_json, agent_url, _now()),
        )
        conn.commit()
        return cur.lastrowid


def update_run(run_id, **kwargs):
    with closing(get_conn()) as conn:
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k not in _UPDATE_COLUMNS:
                raise ValueError(f"Invalid column: {k}")
            sets.append(f"{k} = ?")
            vals.append(v)
        if "status" in kwargs and kwargs["status"] in ("completed", "failed"):
            sets.append("completed_at = ?")
            vals.append(_now())
        vals.append(run_id)
        conn.execute(f"UPDATE eval_runs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()


def delete_run(run_id):
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
        conn.commit()


def delete_runs(run_ids: list[int]):
    with closing(get_conn()) as conn:
        placeholders = ",".join("?" * len(run_ids))
        conn.execute(f"DELETE FROM eval_runs WHERE id IN ({placeholders})", run_ids)
        conn.commit()


def get_run(run_id):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_runs(*, task="", status="", language="", limit=100):
    with closing(get_conn()) as conn:
        sql = "SELECT * FROM eval_runs WHERE 1=1"
        params = []
        if task:
            sql += " AND task_name = ?"
            params.append(task)
        if status:
            sql += " AND status = ?"
            params.append(status)
        if language:
            sql += " AND language = ?"
            params.append(language)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def fail_stale_runs(max_age_minutes=30):
    """Mark runs stuck in 'running' for too long as failed."""
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """UPDATE eval_runs SET status = 'failed',
                      error_message = 'Timed out (stuck in running)',
                      completed_at = ?
               WHERE status = 'running'
                 AND created_at < datetime('now', ? || ' minutes')""",
            (_now(), f"-{max_age_minutes}"),
        )
        conn.commit()
        return cur.rowcount


def get_stats():
    with closing(get_conn()) as conn:
        rows = conn.execute("""
            SELECT task_name, tier, language,
                   COUNT(*) as run_count,
                   AVG(final_score) as avg_score,
                   AVG(correctness) as avg_correctness,
                   AVG(elapsed_seconds) as avg_elapsed,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
            FROM eval_runs
            WHERE status IN ('completed', 'failed')
            GROUP BY task_name, tier, language
            ORDER BY task_name, language
        """).fetchall()
        return [dict(r) for r in rows]


# ── solve_logs CRUD ─────────────────────────────────────────────────

def create_solve_log(*, request_id, prompt, files_json, base_url,
                     api_calls=0, api_errors=0, elapsed_seconds=0,
                     agent_response="", tool_calls_json=""):
    with closing(get_conn()) as conn:
        conn.execute(
            """INSERT INTO solve_logs
               (request_id, prompt, files_json, base_url,
                api_calls, api_errors, elapsed_seconds, agent_response,
                tool_calls_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, prompt, files_json, base_url,
             api_calls, api_errors, elapsed_seconds, agent_response,
             tool_calls_json, _now()),
        )
        conn.commit()


def get_solve_logs(limit=100):
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM solve_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
