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
        # Migrate: add columns if missing (solve_logs)
        for col, ctype in [("tool_calls_json", "TEXT"), ("api_log_json", "TEXT"), ("task_type", "TEXT"), ("tool_count", "INTEGER"), ("source", "TEXT"), ("classification_level", "TEXT"), ("llm_task_type", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE solve_logs ADD COLUMN {col} {ctype}")
            except Exception:
                pass  # Column already exists
        # Migrate: add source column to eval_runs
        try:
            conn.execute("ALTER TABLE eval_runs ADD COLUMN source TEXT DEFAULT 'simulator'")
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
            source TEXT DEFAULT 'simulator',
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
            api_log_json TEXT,
            created_at TEXT,
            task_type TEXT,
            tool_count INTEGER,
            source TEXT
        );
        """)
        # auto_test_results table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS auto_test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            solve_log_id INTEGER,
            task_type TEXT,
            prompt TEXT,
            expected_fields TEXT,
            checks_json TEXT,
            total_points INTEGER DEFAULT 0,
            max_points INTEGER DEFAULT 0,
            correctness REAL DEFAULT 0,
            intent_passed INTEGER DEFAULT 0,
            intent_reasoning TEXT,
            issues TEXT,
            api_calls INTEGER DEFAULT 0,
            api_errors INTEGER DEFAULT 0,
            created_at TEXT
        )
        """)
        # Score tracking tables
        conn.execute("""
        CREATE TABLE IF NOT EXISTS score_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_score REAL DEFAULT 0,
            rank INTEGER,
            tasks_attempted INTEGER DEFAULT 0,
            total_submissions INTEGER DEFAULT 0,
            raw_json TEXT,
            created_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS score_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            task_number INTEGER NOT NULL,
            checks_passed INTEGER DEFAULT 0,
            checks_total INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            mapped_task_type TEXT,
            submission_time TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES score_snapshots(id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS task_number_map (
            task_number INTEGER PRIMARY KEY,
            task_type TEXT NOT NULL,
            confidence TEXT DEFAULT 'manual',
            updated_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS task_manual_checks (
            task_name TEXT PRIMARY KEY,
            checks_passed INTEGER DEFAULT 0,
            checks_total INTEGER DEFAULT 0,
            updated_at TEXT
        )
        """)
        conn.commit()


# ── eval_runs CRUD ──────────────────────────────────────────────────

def create_run(*, task_name, tier, language, prompt, expected_json, agent_url, source="simulator"):
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """INSERT INTO eval_runs
               (task_name, tier, language, prompt, expected_json, status, agent_url, source, created_at)
               VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
            (task_name, tier, language, prompt, expected_json, agent_url, source, _now()),
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


def get_runs(*, task="", status="", language="", source="", limit=100):
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
        if source:
            sql += " AND source = ?"
            params.append(source)
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
                     agent_response="", tool_calls_json="", api_log_json="",
                     task_type="", tool_count=0, source="", classification_level=""):
    with closing(get_conn()) as conn:
        conn.execute(
            """INSERT INTO solve_logs
               (request_id, prompt, files_json, base_url,
                api_calls, api_errors, elapsed_seconds, agent_response,
                tool_calls_json, api_log_json, task_type, tool_count, source,
                classification_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, prompt, files_json, base_url,
             api_calls, api_errors, elapsed_seconds, agent_response,
             tool_calls_json, api_log_json, task_type or "", tool_count, source or "",
             classification_level or "", _now()),
        )
        conn.commit()


def update_solve_log_llm_task_type(request_id: str, llm_task_type: str):
    """Set the LLM-based independent classification for a solve log."""
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE solve_logs SET llm_task_type = ? WHERE request_id = ?",
            (llm_task_type, request_id),
        )
        conn.commit()


def delete_all_solve_logs():
    with closing(get_conn()) as conn:
        cur = conn.execute("DELETE FROM solve_logs")
        conn.commit()
        return cur.rowcount


def get_live_task_stats():
    """Aggregate solve_log stats per task_type for live competition runs."""
    with closing(get_conn()) as conn:
        rows = conn.execute("""
            SELECT task_type,
                   COUNT(*) as run_count,
                   AVG(api_calls) as avg_api_calls,
                   AVG(api_errors) as avg_api_errors,
                   AVG(elapsed_seconds) as avg_elapsed,
                   MIN(api_calls) as min_api_calls,
                   MAX(api_calls) as max_api_calls,
                   MAX(created_at) as last_run
            FROM solve_logs
            WHERE source = 'competition'
              AND task_type IS NOT NULL
              AND task_type != ''
            GROUP BY task_type
        """).fetchall()
        return [dict(r) for r in rows]


def get_sample_prompts_by_task():
    """Return the most recent sample prompt per task_type from competition runs."""
    with closing(get_conn()) as conn:
        rows = conn.execute("""
            SELECT task_type, prompt, created_at
            FROM solve_logs s1
            WHERE source = 'competition'
              AND task_type IS NOT NULL AND task_type != ''
              AND prompt IS NOT NULL AND prompt != ''
              AND created_at = (
                  SELECT MAX(s2.created_at) FROM solve_logs s2
                  WHERE s2.task_type = s1.task_type AND s2.source = 'competition'
              )
        """).fetchall()
        return {r["task_type"]: r["prompt"][:500] for r in rows}


def get_classification_stats():
    """Aggregate classification level stats from solve_logs."""
    with closing(get_conn()) as conn:
        rows = conn.execute("""
            SELECT classification_level,
                   COUNT(*) as count,
                   GROUP_CONCAT(DISTINCT task_type) as task_types
            FROM solve_logs
            WHERE classification_level IS NOT NULL AND classification_level != ''
            GROUP BY classification_level
        """).fetchall()
        total = sum(r["count"] for r in rows) if rows else 0
        return {
            "total": total,
            "levels": [
                {
                    "level": r["classification_level"],
                    "count": r["count"],
                    "pct": round(r["count"] / total * 100, 1) if total else 0,
                    "task_types": (r["task_types"] or "").split(","),
                }
                for r in rows
            ],
        }


def get_solve_logs_as_runs(*, source="", limit=100):
    """Return solve_logs mapped to EvalRun-compatible dicts for merged results view.

    IDs are offset by 1_000_000 to avoid collision with eval_runs.
    """
    with closing(get_conn()) as conn:
        sql = "SELECT * FROM solve_logs WHERE 1=1"
        params = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            result.append({
                "id": 1_000_000 + (r["id"] or 0),
                "task_name": r["task_type"] or "(unknown)",
                "tier": 0,
                "language": "-",
                "prompt": r["prompt"] or "",
                "expected_json": None,
                "status": "completed",
                "agent_url": r["base_url"] or "",
                "api_calls": r["api_calls"] or 0,
                "api_errors": r["api_errors"] or 0,
                "elapsed_seconds": r["elapsed_seconds"] or 0,
                "correctness": None,
                "base_score": None,
                "efficiency_bonus": None,
                "final_score": None,
                "max_possible": None,
                "checks_json": None,
                "error_message": None,
                "source": r["source"] or "competition",
                "created_at": r["created_at"],
                "completed_at": r["created_at"],
            })
        return result


def get_solve_logs(limit=100, source=""):
    with closing(get_conn()) as conn:
        if source:
            rows = conn.execute(
                "SELECT * FROM solve_logs WHERE source = ? ORDER BY id DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM solve_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_solve_log_by_id(log_id: int) -> dict | None:
    """Fetch a single solve_log by its ID."""
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM solve_logs WHERE id = ?", (log_id,)).fetchone()
        if not row:
            return None
        return dict(row)


# ── auto_test_results CRUD ─────────────────────────────────────────

def init_auto_test_table():
    """Create auto_test_results table if not exists."""
    with closing(get_conn()) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS auto_test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            solve_log_id INTEGER,
            task_type TEXT,
            prompt TEXT,
            expected_fields TEXT,
            checks_json TEXT,
            total_points INTEGER DEFAULT 0,
            max_points INTEGER DEFAULT 0,
            correctness REAL DEFAULT 0,
            intent_passed INTEGER DEFAULT 0,
            intent_reasoning TEXT,
            issues TEXT,
            api_calls INTEGER DEFAULT 0,
            api_errors INTEGER DEFAULT 0,
            created_at TEXT
        )
        """)
        conn.commit()


def create_auto_test_result(*, solve_log_id, task_type, prompt, expected_fields,
                            checks_json, total_points, max_points, correctness,
                            intent_passed, intent_reasoning, issues,
                            api_calls=0, api_errors=0):
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """INSERT INTO auto_test_results
               (solve_log_id, task_type, prompt, expected_fields, checks_json,
                total_points, max_points, correctness, intent_passed,
                intent_reasoning, issues, api_calls, api_errors, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (solve_log_id, task_type, prompt, expected_fields, checks_json,
             total_points, max_points, correctness, int(intent_passed),
             intent_reasoning, issues, api_calls, api_errors, _now()),
        )
        conn.commit()
        return cur.lastrowid


def get_auto_test_results(limit=200):
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM auto_test_results ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_auto_test_result_by_log_id(solve_log_id: int) -> dict | None:
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM auto_test_results WHERE solve_log_id = ? ORDER BY id DESC LIMIT 1",
            (solve_log_id,),
        ).fetchone()
        return dict(row) if row else None


def delete_auto_test_result(result_id: int):
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM auto_test_results WHERE id = ?", (result_id,))
        conn.commit()


def delete_all_auto_test_results():
    with closing(get_conn()) as conn:
        cur = conn.execute("DELETE FROM auto_test_results")
        conn.commit()
        return cur.rowcount


# ── score tracking CRUD ────────────────────────────────────────────

def create_score_snapshot(total_score, rank, tasks_attempted, total_submissions,
                          raw_json, tasks: list[dict]) -> int:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """INSERT INTO score_snapshots
               (total_score, rank, tasks_attempted, total_submissions, raw_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (total_score, rank, tasks_attempted, total_submissions, raw_json, _now()),
        )
        snapshot_id = cur.lastrowid
        for t in tasks:
            mapped = t.get("mapped_task_type") or _lookup_task_mapping(conn, t.get("task_number"))
            conn.execute(
                """INSERT INTO score_tasks
                   (snapshot_id, task_number, checks_passed, checks_total, score,
                    mapped_task_type, submission_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, t.get("task_number"), t.get("checks_passed", 0),
                 t.get("checks_total", 0), t.get("score", 0),
                 mapped, t.get("submission_time")),
            )
        conn.commit()
        return snapshot_id


def _lookup_task_mapping(conn, task_number) -> str | None:
    if task_number is None:
        return None
    row = conn.execute(
        "SELECT task_type FROM task_number_map WHERE task_number = ?",
        (task_number,),
    ).fetchone()
    return row["task_type"] if row else None


def get_latest_snapshot() -> dict | None:
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM score_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        snap = dict(row)
        tasks = conn.execute(
            "SELECT * FROM score_tasks WHERE snapshot_id = ? ORDER BY task_number",
            (snap["id"],),
        ).fetchall()
        snap["tasks"] = [dict(t) for t in tasks]
        # Enrich with current mappings
        mappings = _get_all_mappings(conn)
        for t in snap["tasks"]:
            if not t.get("mapped_task_type") and t["task_number"] in mappings:
                t["mapped_task_type"] = mappings[t["task_number"]]
        return snap


def get_snapshot_pair() -> tuple[dict | None, dict | None]:
    """Return (latest, previous) snapshots with their tasks for diffing."""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM score_snapshots ORDER BY id DESC LIMIT 2"
        ).fetchall()
        if not rows:
            return None, None
        mappings = _get_all_mappings(conn)

        def _enrich(snap_row):
            snap = dict(snap_row)
            tasks = conn.execute(
                "SELECT * FROM score_tasks WHERE snapshot_id = ? ORDER BY task_number",
                (snap["id"],),
            ).fetchall()
            snap["tasks"] = [dict(t) for t in tasks]
            for t in snap["tasks"]:
                if not t.get("mapped_task_type") and t["task_number"] in mappings:
                    t["mapped_task_type"] = mappings[t["task_number"]]
            return snap

        latest = _enrich(rows[0])
        previous = _enrich(rows[1]) if len(rows) > 1 else None
        return latest, previous


def get_score_history(limit=50) -> list[dict]:
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM score_snapshots ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def _get_all_mappings(conn) -> dict[int, str]:
    rows = conn.execute("SELECT task_number, task_type FROM task_number_map").fetchall()
    return {r["task_number"]: r["task_type"] for r in rows}


def get_task_number_map() -> dict[int, dict]:
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT * FROM task_number_map ORDER BY task_number").fetchall()
        return {r["task_number"]: dict(r) for r in rows}


def set_task_number_mapping(task_number: int, task_type: str, confidence: str = "manual"):
    with closing(get_conn()) as conn:
        conn.execute(
            """INSERT INTO task_number_map (task_number, task_type, confidence, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(task_number) DO UPDATE SET
                 task_type = excluded.task_type,
                 confidence = excluded.confidence,
                 updated_at = excluded.updated_at""",
            (task_number, task_type, confidence, _now()),
        )
        conn.commit()


def get_all_manual_checks() -> dict[str, dict]:
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT * FROM task_manual_checks ORDER BY task_name").fetchall()
        return {r["task_name"]: dict(r) for r in rows}


def set_manual_checks(task_name: str, checks_passed: int, checks_total: int):
    with closing(get_conn()) as conn:
        conn.execute(
            """INSERT INTO task_manual_checks (task_name, checks_passed, checks_total, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(task_name) DO UPDATE SET
                 checks_passed = excluded.checks_passed,
                 checks_total = excluded.checks_total,
                 updated_at = excluded.updated_at""",
            (task_name, checks_passed, checks_total, _now()),
        )
        conn.commit()


def get_solve_logs_near_time(iso_time: str, window_seconds: int = 30) -> list[dict]:
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """SELECT * FROM solve_logs
               WHERE source = 'competition'
                 AND created_at BETWEEN datetime(?, ? || ' seconds')
                                     AND datetime(?, '+' || ? || ' seconds')
               ORDER BY created_at""",
            (iso_time, f"-{window_seconds}", iso_time, str(window_seconds)),
        ).fetchall()
        return [dict(r) for r in rows]
