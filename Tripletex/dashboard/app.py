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
from starlette.responses import StreamingResponse

# Ensure parent dir on path for sim/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from dashboard import db
from dashboard.runner import run_eval
from dashboard.tool_tester import run_all_tool_tests, stream_tool_tests, get_tool_catalog
from dashboard import sandbox as sandbox_mod
from sim.task_definitions import ALL_TASKS, LANGUAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DIST_DIR = os.path.join(STATIC_DIR, "dist")
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
    dist_index = os.path.join(DIST_DIR, "index.html")
    if os.path.isfile(dist_index):
        return FileResponse(dist_index)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount dist assets first (React build output)
if os.path.isdir(os.path.join(DIST_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="dist-assets")

# Keep old static mount as fallback
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


@app.post("/api/runs/cleanup-stale")
def cleanup_stale_runs():
    """Mark runs stuck in 'running' for >10 min as failed."""
    count = db.fail_stale_runs(max_age_minutes=10)
    return {"ok": True, "cleaned": count}


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
    import time as _time
    import httpx

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

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("AGENT_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results = []
    async with httpx.AsyncClient(timeout=req.timeout) as http:
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception as e:
                results.append({"filename": os.path.basename(p), "error": str(e)})
                continue

            # Override credentials with live session token from env
            payload["tripletex_credentials"] = {
                "base_url": base_url,
                "session_token": token,
            }

            t0 = _time.time()
            http_status = 0
            try:
                resp = await http.post(
                    f"{agent_url}/solve-debug",
                    json=payload, headers=headers,
                )
                elapsed = _time.time() - t0
                http_status = resp.status_code
                content_type = resp.headers.get("content-type", "")
                if "application/json" not in content_type:
                    body = {"error": f"Agent returned {content_type or 'unknown content-type'} (not JSON). Is the agent running at {agent_url}?"}
                elif resp.status_code == 200:
                    body = resp.json()
                else:
                    body = {"error": resp.text[:500]}
            except httpx.ConnectError:
                elapsed = _time.time() - t0
                body = {"error": f"Cannot connect to agent at {agent_url}. Is the agent server running?"}
            except Exception as e:
                elapsed = _time.time() - t0
                body = {"error": str(e)}

            tool_calls = body.get("tool_calls", [])
            failed_tools = [tc for tc in tool_calls
                            if tc.get("result", {}).get("ok") is False]

            results.append({
                "filename": os.path.basename(p),
                "prompt": payload.get("prompt", ""),
                "status": "FAIL" if failed_tools or body.get("error") else "OK",
                "http_status": http_status,
                "elapsed": round(elapsed, 2),
                "error": body.get("error", ""),
                "tool_calls": tool_calls,
                "api_calls": body.get("api_calls", 0),
                "api_errors": body.get("api_errors", 0),
                "agent_response": body.get("agent_response", ""),
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


@app.get("/api/test-tools/stream")
async def test_tools_stream():
    """SSE stream: yields each tool test result as it completes."""
    base_url, token, _ = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )

    def _generate():
        for result in stream_tool_tests(base_url, token):
            yield f"data: {json.dumps(result)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── API: Tool Catalog ──────────────────────────────────────────────

_tool_catalog_cache = None

@app.get("/api/tools")
def list_tools():
    """Return tool catalog: name, module, docstring, params — no credentials needed."""
    global _tool_catalog_cache
    if _tool_catalog_cache is None:
        _tool_catalog_cache = get_tool_catalog()
    return _tool_catalog_cache


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


# ── API: Sandbox ───────────────────────────────────────────────────

def _get_sandbox_client():
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not base_url or not token:
        return None
    from tripletex_client import TripletexClient
    return TripletexClient(base_url, token)


@app.get("/api/sandbox/health")
async def sandbox_health():
    client = _get_sandbox_client()
    if not client:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )
    result = await asyncio.to_thread(sandbox_mod.check_health, client)
    return result


class SeedRequest(BaseModel):
    types: list[str] = ["all"]
    clean: bool = False


@app.post("/api/sandbox/seed")
async def sandbox_seed(req: SeedRequest):
    client = _get_sandbox_client()
    if not client:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )
    result = await asyncio.to_thread(
        sandbox_mod.seed_entities, client, req.types, req.clean
    )
    return result


@app.post("/api/sandbox/clean")
async def sandbox_clean():
    client = _get_sandbox_client()
    if not client:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )
    result = await asyncio.to_thread(sandbox_mod.clean_entities, client)
    return result


# ── API: Coverage ──────────────────────────────────────────────────

_coverage_cache = None

@app.get("/api/coverage")
def api_coverage():
    """Return API coverage mapping: dynamically scans tools/ directory."""
    global _coverage_cache
    if _coverage_cache is None:
        _coverage_cache = _build_coverage()
    return _coverage_cache


@app.post("/api/coverage/refresh")
def refresh_coverage():
    """Force refresh the coverage cache."""
    global _coverage_cache
    _coverage_cache = None
    _coverage_cache = _build_coverage()
    return {"ok": True, "categories": len(_coverage_cache)}


def _build_coverage():
    """Dynamically scan tools/ and map to Tripletex API endpoints."""
    import importlib, inspect, re

    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")

    # 1. Scan all tool modules → collect tool names + docstrings + source
    tool_info = {}  # {tool_name: {module, docstring, endpoints}}
    for py_file in sorted(glob(os.path.join(tools_dir, "*.py"))):
        mod_name = os.path.basename(py_file).replace(".py", "")
        if mod_name.startswith("_"):
            continue
        try:
            source = open(py_file, "r", encoding="utf-8").read()
        except Exception:
            continue

        # Find all nested function definitions (indented = tool closures)
        # Handles multi-line params and -> return type annotations
        for match in re.finditer(
            r'    def\s+(\w+)\s*\(.*?\)(?:\s*->\s*\w+)?\s*:\s*(?:"""(.*?)""")?',
            source, re.DOTALL
        ):
            fname = match.group(1)
            doc = (match.group(2) or "").strip().split("\n")[0]  # first line
            # Skip internal/helper functions
            if fname.startswith("_") or fname.startswith("build_"):
                continue
            tool_info[fname] = {
                "module": mod_name,
                "docstring": doc,
                "endpoints": _infer_endpoints(fname, source, match.start(), mod_name),
            }

    # 2. Map tool names → API categories
    # Category is derived from module name
    MODULE_TO_CATEGORY = {
        "employees": "EMPLOYEE", "employee_extras": "EMPLOYEE",
        "employment": "EMPLOYEE",
        "customers": "CUSTOMER", "contacts": "CONTACT",
        "products": "PRODUCT", "departments": "DEPARTMENT",
        "projects": "PROJECT", "order": "ORDER", "invoicing": "INVOICE",
        "bank": "BANK", "ledger": "LEDGER", "supplier": "SUPPLIER",
        "supplier_invoice": "SUPPLIERINVOICE",
        "travel": "TRAVELEXPENSE", "travel_extras": "TRAVELEXPENSE",
        "address": "DELIVERYADDRESS", "balance": "BALANCE",
        "company": "COMPANY", "common": "GENERIC/UTILITY",
        "files": "GENERIC/UTILITY",
        "activity": "ACTIVITY", "division": "DIVISION",
        "timesheet": "TIMESHEET", "salary": "SALARY",
        "year_end": "YEAREND", "incoming_invoice": "INCOMINGINVOICE",
    }

    # Some tools from invoicing.py belong to ORDER, not INVOICE
    ORDER_TOOLS = {"create_order", "search_orders", "update_order", "delete_order",
                   "send_invoice"}  # send_invoice is in invoicing.py but maps to invoice

    # 3. Build per-category data
    categories = {}
    for tool_name, info in tool_info.items():
        cat = MODULE_TO_CATEGORY.get(info["module"], info["module"].upper())
        # Split invoicing tools: order-related go to ORDER
        if info["module"] == "invoicing" and tool_name in ORDER_TOOLS:
            cat = "ORDER"
        if cat not in categories:
            categories[cat] = {"tools": [], "covered": [], "modules": set()}
        categories[cat]["tools"].append(tool_name)
        categories[cat]["modules"].add(info["module"])
        for ep in info["endpoints"]:
            if ep not in categories[cat]["covered"]:
                categories[cat]["covered"].append(ep)

    # 4. Load endpoint counts from reference file (if available)
    endpoints_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "tripletex_all_endpoints.txt")
    all_endpoints = _parse_endpoint_file(endpoints_file) if os.path.isfile(endpoints_file) else {}

    # 5. Build final output
    result = []
    seen_cats = set()
    for cat_name in sorted(categories.keys()):
        cat = categories[cat_name]
        seen_cats.add(cat_name)
        # Match to reference endpoints
        ref_key = _match_ref_category(cat_name, all_endpoints)
        ref_endpoints = all_endpoints.get(ref_key, []) if ref_key else []
        uncovered = [ep for ep in ref_endpoints if ep not in cat["covered"]]
        result.append({
            "category": cat_name,
            "endpoint_count": max(len(ref_endpoints), len(cat["covered"])),
            "tools": sorted(cat["tools"]),
            "covered": sorted(cat["covered"]),
            "uncovered": uncovered,
        })

    # Add uncovered reference categories
    for ref_cat, endpoints in sorted(all_endpoints.items()):
        if not _match_any(ref_cat, seen_cats):
            result.append({
                "category": ref_cat,
                "endpoint_count": len(endpoints),
                "tools": [],
                "covered": [],
                "uncovered": endpoints,
            })

    # Sort: covered categories first, then uncovered by size
    result.sort(key=lambda c: (-len(c["tools"]), -len(c["covered"]), c["category"]))
    return result


def _infer_endpoints(func_name: str, source: str, func_start: int, module: str) -> list:
    """Infer which API endpoints a tool function covers from its source code."""
    import re
    # Get function body (up to next def or end)
    next_def = source.find("\ndef ", func_start + 1)
    body = source[func_start:next_def] if next_def > 0 else source[func_start:]

    endpoints = []
    # Look for client.post/get/put/delete calls
    for m in re.finditer(r'client\.(post|get|put|delete)\(\s*f?"(/[^"]+)"', body):
        method = m.group(1).upper()
        path = m.group(2)
        # Clean up f-string vars
        path = re.sub(r'\{[^}]+\}', '{id}', path)
        endpoints.append(f"{method} {path}")

    # Also check for string patterns like "/employee" in the body
    if not endpoints:
        for m in re.finditer(r'client\.(post|get|put|delete)\(\s*(["\']|f["\'])([^"\']+)', body):
            method = m.group(1).upper()
            path = m.group(3)
            path = re.sub(r'\{[^}]+\}', '{id}', path)
            endpoints.append(f"{method} {path}")

    return endpoints


def _parse_endpoint_file(filepath: str) -> dict:
    """Parse tripletex_all_endpoints.txt into {category: [endpoints]}."""
    import re
    categories = {}
    current_cat = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Category headers: "  CATEGORY (N endpoints)"
                cat_match = re.match(r'^([A-Z][A-Z_/\s]+?)\s*\(\d+\s*endpoint', line)
                if cat_match:
                    current_cat = cat_match.group(1).strip()
                    if current_cat not in categories:
                        categories[current_cat] = []
                    continue
                # Endpoint lines: "POST   /path" or "PUT    /path"
                ep_match = re.match(r'^(GET|POST|PUT|DELETE)\s+(/\S+)', line)
                if ep_match and current_cat:
                    categories[current_cat].append(f"{ep_match.group(1)} {ep_match.group(2)}")
    except Exception:
        pass
    return categories


def _match_ref_category(cat_name: str, ref_cats: dict) -> str | None:
    """Find matching reference category for a tool category."""
    if cat_name in ref_cats:
        return cat_name
    # GENERIC/UTILITY has no reference category
    if cat_name == "GENERIC/UTILITY":
        return None
    # Try case-insensitive match
    for ref in ref_cats:
        if ref.replace(" ", "").upper() == cat_name.replace(" ", "").upper():
            return ref
    return None


def _match_any(ref_cat: str, seen: set) -> bool:
    """Check if a reference category has already been covered."""
    if ref_cat in seen:
        return True
    # Reverse mapping
    for s in seen:
        mapped = _match_ref_category(s, {ref_cat: []})
        if mapped == ref_cat:
            return True
    return False


# ── SPA catch-all ─────────────────────────────────────────────────

@app.get("/{path:path}")
def spa_fallback(path: str):
    """Serve React SPA for any non-API route."""
    dist_index = os.path.join(DIST_DIR, "index.html")
    if os.path.isfile(dist_index):
        return FileResponse(dist_index)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Tripletex Eval Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
