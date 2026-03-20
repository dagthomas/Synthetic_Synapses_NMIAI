"""Tripletex Eval Dashboard — FastAPI server."""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from glob import glob

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure parent dir on path for sim/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from dashboard import db
from dashboard.runner import run_eval
from dashboard.tool_tester import run_all_tool_tests
from sim.task_definitions import ALL_TASKS, LANGUAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PAYLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "payloads")

# Keep references to background tasks so they don't get GC'd
_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    yield


app = FastAPI(title="Tripletex Eval Dashboard", lifespan=lifespan)


# ── Serve SPA ───────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── API: Tasks ──────────────────────────────────────────────────────

@app.get("/api/tasks")
def list_tasks():
    tasks = []
    for name, td in ALL_TASKS.items():
        tasks.append({
            "name": name,
            "tier": td.tier,
            "description": td.description,
            "baseline_calls": td.baseline_calls,
            "field_count": len(td.field_checks),
            "max_points": sum(fc.points for fc in td.field_checks),
        })
    return tasks


@app.get("/api/languages")
def list_languages():
    return LANGUAGES


# ── API: Runs ───────────────────────────────────────────────────────

@app.get("/api/runs")
def list_runs(task: str = "", status: str = "", language: str = "",
              limit: int = Query(100, le=500)):
    return db.get_runs(task=task, status=status, language=language, limit=limit)


@app.get("/api/eval/status/{run_id}")
def run_status(run_id: int):
    row = db.get_run(run_id)
    if not row:
        return JSONResponse({"error": "Not found"}, 404)
    return row


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: int):
    db.delete_run(run_id)
    return {"ok": True}


class DeleteRunsRequest(BaseModel):
    run_ids: list[int]


@app.post("/api/runs/delete")
def delete_runs(req: DeleteRunsRequest):
    if not req.run_ids:
        return {"ok": True, "deleted": 0}
    db.delete_runs(req.run_ids)
    return {"ok": True, "deleted": len(req.run_ids)}


# ── API: Run Eval ───────────────────────────────────────────────────

class EvalRequest(BaseModel):
    task_name: str
    language: str = "no"
    count: int = Field(1, ge=1, le=20)


class BatchRequest(BaseModel):
    task_names: list[str]
    languages: list[str] = ["no"]
    count_per_combo: int = Field(1, ge=1, le=10)


def _get_credentials():
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    agent_url = os.environ.get("EVAL_AGENT_URL", "http://localhost:8000")
    return base_url, token, agent_url


def _track_task(task: asyncio.Task):
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


@app.post("/api/eval/run")
async def start_eval(req: EvalRequest):
    base_url, token, agent_url = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )
    if req.task_name not in ALL_TASKS:
        return JSONResponse({"error": f"Unknown task: {req.task_name}"}, 400)

    for _ in range(req.count):
        _track_task(asyncio.create_task(run_eval(
            req.task_name, req.language, agent_url, base_url, token
        )))
    return {"message": f"Started {req.count} eval(s) for {req.task_name}/{req.language}"}


@app.post("/api/eval/batch")
async def start_batch(req: BatchRequest):
    base_url, token, agent_url = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )

    total = 0
    skipped = []
    for task_name in req.task_names:
        if task_name not in ALL_TASKS:
            skipped.append(task_name)
            continue
        for language in req.languages:
            for _ in range(req.count_per_combo):
                _track_task(asyncio.create_task(run_eval(
                    task_name, language, agent_url, base_url, token
                )))
                total += 1

    result = {"message": f"Started {total} eval(s)", "total": total}
    if skipped:
        result["skipped"] = skipped
    return result


# ── API: Stats ──────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    return db.get_stats()


# ── API: Payloads (replay) ──────────────────────────────────────────

@app.get("/api/payloads")
def list_payloads(limit: int = Query(50, le=200)):
    """List saved payload files."""
    if not os.path.isdir(PAYLOADS_DIR):
        return []
    files = sorted(glob(os.path.join(PAYLOADS_DIR, "*.json")), reverse=True)[:limit]
    result = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            fname = os.path.basename(f)
            result.append({
                "filename": fname,
                "prompt": data.get("prompt", ""),
                "files": [fi.get("filename", "") if isinstance(fi, dict) else fi
                          for fi in data.get("files", [])],
                "base_url": data.get("tripletex_credentials", {}).get("base_url", ""),
            })
        except Exception:
            continue
    return result


class ReplayRequest(BaseModel):
    filenames: list[str] = []  # empty = replay all
    timeout: int = Field(180, ge=30, le=600)


@app.post("/api/replay")
async def replay_payloads(req: ReplayRequest):
    """Replay saved payloads through /solve-debug with live credentials."""
    import requests as http_requests

    base_url, token, agent_url = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )

    if not os.path.isdir(PAYLOADS_DIR):
        return JSONResponse({"error": "No payloads directory"}, status_code=404)

    # Resolve filenames
    if req.filenames:
        paths = [os.path.join(PAYLOADS_DIR, f) for f in req.filenames
                 if os.path.isfile(os.path.join(PAYLOADS_DIR, f))]
    else:
        paths = sorted(glob(os.path.join(PAYLOADS_DIR, "*.json")))

    if not paths:
        return JSONResponse({"error": "No payload files found"}, status_code=404)

    results = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            results.append({"filename": os.path.basename(p), "error": str(e)})
            continue

        # Override credentials with live token
        payload["tripletex_credentials"] = {
            "base_url": base_url,
            "session_token": token,
        }

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("AGENT_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = http_requests.post(
                f"{agent_url}/solve-debug",
                json=payload, headers=headers, timeout=req.timeout,
            )
            body = resp.json() if resp.status_code == 200 else {"error": resp.text[:500]}
        except Exception as e:
            body = {"error": str(e)}

        tool_calls = body.get("tool_calls", [])
        failed_tools = [tc for tc in tool_calls
                        if tc.get("result", {}).get("ok") is False]

        results.append({
            "filename": os.path.basename(p),
            "prompt": payload.get("prompt", ""),
            "status": "FAIL" if failed_tools or body.get("error") else "OK",
            "tool_calls": tool_calls,
            "api_calls": body.get("api_calls", 0),
            "api_errors": body.get("api_errors", 0),
            "api_log": body.get("api_log", []),
            "agent_response": body.get("agent_response", ""),
            "elapsed": body.get("elapsed", 0),
            "error": body.get("error", ""),
        })

    return results


# ── API: Test Tools ────────────────────────────────────────────────

@app.post("/api/test-tools")
async def test_tools():
    """Run all tool tests against the Tripletex sandbox."""
    base_url, token, _ = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_all_tool_tests, base_url, token)
    return result


# ── API: Solve Logs ─────────────────────────────────────────────────

@app.get("/api/logs")
def list_logs(limit: int = Query(50, le=200)):
    """Return solve logs from DB + payload files."""
    # DB logs
    db_logs = db.get_solve_logs(limit=limit)

    # Also scan payloads/ directory for any not yet in DB
    file_logs = []
    if os.path.isdir(PAYLOADS_DIR):
        files = sorted(glob(os.path.join(PAYLOADS_DIR, "*.json")), reverse=True)[:limit]
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                fname = os.path.basename(f)
                file_logs.append({
                    "id": None,
                    "request_id": fname.split("_")[-1].replace(".json", ""),
                    "prompt": data.get("prompt", ""),
                    "files_json": json.dumps(data.get("files", [])),
                    "base_url": data.get("tripletex_credentials", {}).get("base_url", ""),
                    "api_calls": 0,
                    "api_errors": 0,
                    "elapsed_seconds": 0,
                    "agent_response": "",
                    "created_at": fname[:15].replace("_", " ") if len(fname) > 15 else "",
                    "source": "file",
                })
            except Exception:
                continue

    # Merge: DB logs first, then file-based (deduplicate by request_id)
    seen_ids = {l["request_id"] for l in db_logs if l.get("request_id")}
    merged = db_logs + [fl for fl in file_logs if fl["request_id"] not in seen_ids]
    return merged[:limit]


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Tripletex Eval Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
