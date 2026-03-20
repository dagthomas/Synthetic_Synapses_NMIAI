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
from dashboard.tool_tester import run_all_tool_tests, stream_tool_tests
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

@app.get("/api/coverage")
def api_coverage():
    """Return API coverage mapping: all Tripletex endpoints vs implemented tools."""
    return _API_COVERAGE


# Static coverage data — maps every Tripletex API category to our tools.
# "endpoints" lists all POST/PUT/DELETE endpoints in the category.
# "tools" lists our tools that cover endpoints in this category.
# "covered" lists specific endpoints our tools handle.
_API_COVERAGE = [
    {
        "category": "EMPLOYEE",
        "endpoint_count": 27,
        "tools": ["create_employee", "update_employee", "search_employees",
                  "create_employment", "search_employments"],
        "covered": [
            "POST /employee",
            "PUT /employee/{id}",
            "GET /employee (search)",
            "POST /employee/employment",
            "GET /employee/employment (search)",
        ],
        "uncovered": [
            "POST /employee/category", "POST /employee/category/list",
            "PUT /employee/category/list", "DELETE /employee/category/list",
            "PUT /employee/category/{id}", "DELETE /employee/category/{id}",
            "POST /employee/employment/details",
            "PUT /employee/employment/details/{id}",
            "POST /employee/employment/leaveOfAbsence",
            "POST /employee/employment/leaveOfAbsence/list",
            "PUT /employee/employment/leaveOfAbsence/{id}",
            "PUT /employee/employment/{id}",
            "PUT /employee/entitlement/:grantClientEntitlementsByTemplate",
            "PUT /employee/entitlement/:grantEntitlementsByTemplate",
            "POST /employee/hourlyCostAndRate",
            "PUT /employee/hourlyCostAndRate/{id}",
            "POST /employee/list",
            "POST /employee/nextOfKin", "PUT /employee/nextOfKin/{id}",
            "PUT /employee/preferences/:changeLanguage",
            "PUT /employee/preferences/list", "PUT /employee/preferences/{id}",
            "POST /employee/standardTime", "PUT /employee/standardTime/{id}",
        ],
    },
    {
        "category": "CUSTOMER",
        "endpoint_count": 7,
        "tools": ["create_customer", "update_customer", "search_customers"],
        "covered": [
            "POST /customer",
            "PUT /customer/{id}",
            "GET /customer (search)",
        ],
        "uncovered": [
            "POST /customer/category", "PUT /customer/category/{id}",
            "POST /customer/list", "PUT /customer/list",
            "DELETE /customer/{id}",
        ],
    },
    {
        "category": "CONTACT",
        "endpoint_count": 4,
        "tools": ["create_contact", "search_contacts"],
        "covered": [
            "POST /contact",
            "GET /contact (search)",
        ],
        "uncovered": [
            "POST /contact/list", "DELETE /contact/list",
            "PUT /contact/{id}",
        ],
    },
    {
        "category": "PRODUCT",
        "endpoint_count": 34,
        "tools": ["create_product", "search_products"],
        "covered": [
            "POST /product",
            "GET /product (search)",
        ],
        "uncovered": [
            "POST /product/group", "POST /product/group/list",
            "PUT /product/group/list", "DELETE /product/group/list",
            "PUT /product/group/{id}", "DELETE /product/group/{id}",
            "POST /product/groupRelation", "POST /product/groupRelation/list",
            "DELETE /product/groupRelation/list",
            "DELETE /product/groupRelation/{id}",
            "POST /product/inventoryLocation",
            "POST /product/inventoryLocation/list",
            "PUT /product/inventoryLocation/list",
            "PUT /product/inventoryLocation/{id}",
            "DELETE /product/inventoryLocation/{id}",
            "POST /product/list", "PUT /product/list",
            "PUT /product/logisticsSettings",
            "POST /product/supplierProduct",
            "POST /product/supplierProduct/list",
            "PUT /product/supplierProduct/list",
            "PUT /product/supplierProduct/{id}",
            "DELETE /product/supplierProduct/{id}",
            "POST /product/unit", "POST /product/unit/list",
            "PUT /product/unit/list", "PUT /product/unit/{id}",
            "DELETE /product/unit/{id}",
            "PUT /product/{id}", "DELETE /product/{id}",
            "POST /product/{id}/image", "DELETE /product/{id}/image",
        ],
    },
    {
        "category": "DEPARTMENT",
        "endpoint_count": 5,
        "tools": ["create_department"],
        "covered": [
            "POST /department",
        ],
        "uncovered": [
            "POST /department/list", "PUT /department/list",
            "PUT /department/{id}", "DELETE /department/{id}",
        ],
    },
    {
        "category": "PROJECT",
        "endpoint_count": 40,
        "tools": ["create_project"],
        "covered": [
            "POST /project",
        ],
        "uncovered": [
            "DELETE /project", "POST /project/category",
            "PUT /project/category/{id}",
            "POST /project/hourlyRates", "POST /project/hourlyRates/list",
            "PUT /project/hourlyRates/list",
            "DELETE /project/hourlyRates/list",
            "POST /project/hourlyRates/projectSpecificRates",
            "POST /project/hourlyRates/projectSpecificRates/list",
            "PUT /project/hourlyRates/projectSpecificRates/list",
            "DELETE /project/hourlyRates/projectSpecificRates/list",
            "PUT /project/hourlyRates/projectSpecificRates/{id}",
            "DELETE /project/hourlyRates/projectSpecificRates/{id}",
            "PUT /project/hourlyRates/{id}",
            "DELETE /project/hourlyRates/{id}",
            "POST /project/import",
            "POST /project/list", "PUT /project/list",
            "DELETE /project/list",
            "POST /project/orderline", "POST /project/orderline/list",
            "PUT /project/orderline/{id}", "DELETE /project/orderline/{id}",
            "POST /project/participant", "POST /project/participant/list",
            "DELETE /project/participant/list",
            "PUT /project/participant/{id}",
            "POST /project/projectActivity",
            "DELETE /project/projectActivity/list",
            "DELETE /project/projectActivity/{id}",
            "PUT /project/settings",
            "POST /project/subcontract",
            "PUT /project/subcontract/{id}",
            "DELETE /project/subcontract/{id}",
            "PUT /project/{id}", "DELETE /project/{id}",
        ],
    },
    {
        "category": "ORDER",
        "endpoint_count": 21,
        "tools": ["create_order"],
        "covered": [
            "POST /order",
        ],
        "uncovered": [
            "PUT /order/:invoiceMultipleOrders",
            "POST /order/list",
            "POST /order/orderGroup", "PUT /order/orderGroup",
            "DELETE /order/orderGroup/{id}",
            "POST /order/orderline", "POST /order/orderline/list",
            "PUT /order/orderline/{id}", "DELETE /order/orderline/{id}",
            "PUT /order/orderline/{id}/:pickLine",
            "PUT /order/orderline/{id}/:unpickLine",
            "PUT /order/sendInvoicePreview/{orderId}",
            "PUT /order/sendOrderConfirmation/{orderId}",
            "PUT /order/sendPackingNote/{orderId}",
            "PUT /order/{id}", "DELETE /order/{id}",
            "PUT /order/{id}/:approveSubscriptionInvoice",
            "PUT /order/{id}/:attach",
            "PUT /order/{id}/:invoice",
            "PUT /order/{id}/:unApproveSubscriptionInvoice",
        ],
    },
    {
        "category": "INVOICE",
        "endpoint_count": 6,
        "tools": ["create_invoice", "register_payment", "create_credit_note"],
        "covered": [
            "POST /invoice",
            "PUT /invoice/{id}/:payment",
            "PUT /invoice/{id}/:createCreditNote",
        ],
        "uncovered": [
            "POST /invoice/list",
            "PUT /invoice/{id}/:createReminder",
            "PUT /invoice/{id}/:send",
        ],
    },
    {
        "category": "BANK",
        "endpoint_count": 14,
        "tools": ["search_bank_accounts", "search_bank_reconciliations",
                  "get_last_bank_reconciliation",
                  "get_last_closed_bank_reconciliation",
                  "create_bank_reconciliation",
                  "adjust_bank_reconciliation",
                  "close_bank_reconciliation",
                  "delete_bank_reconciliation",
                  "get_bank_reconciliation_match_count",
                  "search_bank_statements"],
        "covered": [
            "GET /bank (search)",
            "GET /bank/reconciliation (search)",
            "GET /bank/reconciliation/>last",
            "GET /bank/reconciliation/>lastClosed",
            "POST /bank/reconciliation",
            "PUT /bank/reconciliation/{id}/:adjustment",
            "PUT /bank/reconciliation/{id}",
            "DELETE /bank/reconciliation/{id}",
            "GET /bank/reconciliation/match/count",
            "GET /bank/statement (search)",
        ],
        "uncovered": [
            "POST /bank/reconciliation/match",
            "PUT /bank/reconciliation/match/:suggest",
            "PUT /bank/reconciliation/match/{id}",
            "DELETE /bank/reconciliation/match/{id}",
            "POST /bank/reconciliation/matches/counter",
            "POST /bank/reconciliation/settings",
            "PUT /bank/reconciliation/settings/{id}",
            "POST /bank/statement/import",
            "DELETE /bank/statement/{id}",
        ],
    },
    {
        "category": "LEDGER",
        "endpoint_count": 39,
        "tools": ["get_ledger_accounts", "get_ledger_postings",
                  "create_voucher", "delete_voucher",
                  "search_voucher_types"],
        "covered": [
            "GET /ledger/account (search)",
            "GET /ledger/posting (search)",
            "POST /ledger/voucher",
            "DELETE /ledger/voucher/{id}",
            "GET /ledger/voucherType (search)",
        ],
        "uncovered": [
            "POST /ledger/account", "POST /ledger/account/list",
            "PUT /ledger/account/list", "DELETE /ledger/account/list",
            "PUT /ledger/account/{id}", "DELETE /ledger/account/{id}",
            "POST /ledger/accountingDimensionName",
            "PUT /ledger/accountingDimensionName/{id}",
            "DELETE /ledger/accountingDimensionName/{id}",
            "POST /ledger/accountingDimensionValue",
            "PUT /ledger/accountingDimensionValue/list",
            "DELETE /ledger/accountingDimensionValue/{id}",
            "POST /ledger/paymentTypeOut",
            "POST /ledger/paymentTypeOut/list",
            "PUT /ledger/paymentTypeOut/list",
            "PUT /ledger/paymentTypeOut/{id}",
            "DELETE /ledger/paymentTypeOut/{id}",
            "PUT /ledger/posting/:closePostings",
            "PUT /ledger/vatSettings",
            "PUT /ledger/vatType/createRelativeVatType",
            "PUT /ledger/voucher/historical/:closePostings",
            "PUT /ledger/voucher/historical/:reverseHistoricalVouchers",
            "POST /ledger/voucher/historical/employee",
            "POST /ledger/voucher/historical/historical",
            "POST /ledger/voucher/historical/{voucherId}/attachment",
            "POST /ledger/voucher/importDocument",
            "POST /ledger/voucher/importGbat10",
            "PUT /ledger/voucher/list",
            "POST /ledger/voucher/openingBalance",
            "DELETE /ledger/voucher/openingBalance",
            "PUT /ledger/voucher/{id}",
            "PUT /ledger/voucher/{id}/:reverse",
            "PUT /ledger/voucher/{id}/:sendToInbox",
            "PUT /ledger/voucher/{id}/:sendToLedger",
            "POST /ledger/voucher/{voucherId}/attachment",
            "DELETE /ledger/voucher/{voucherId}/attachment",
            "POST /ledger/voucher/{voucherId}/pdf/{fileName}",
        ],
    },
    {
        "category": "SUPPLIER",
        "endpoint_count": 5,
        "tools": ["create_supplier", "search_suppliers", "update_supplier"],
        "covered": [
            "POST /supplier",
            "GET /supplier (search)",
            "PUT /supplier/{id}",
        ],
        "uncovered": [
            "POST /supplier/list", "PUT /supplier/list",
            "DELETE /supplier/{id}",
        ],
    },
    {
        "category": "TRAVELEXPENSE",
        "endpoint_count": 38,
        "tools": ["create_travel_expense", "delete_travel_expense",
                  "search_travel_expenses"],
        "covered": [
            "POST /travelExpense",
            "DELETE /travelExpense/{id}",
            "GET /travelExpense (search)",
        ],
        "uncovered": [
            "PUT /travelExpense/:approve", "PUT /travelExpense/:copy",
            "PUT /travelExpense/:createVouchers",
            "PUT /travelExpense/:deliver",
            "PUT /travelExpense/:unapprove",
            "PUT /travelExpense/:undeliver",
            "POST /travelExpense/accommodationAllowance",
            "PUT /travelExpense/accommodationAllowance/{id}",
            "DELETE /travelExpense/accommodationAllowance/{id}",
            "POST /travelExpense/cost",
            "PUT /travelExpense/cost/list",
            "PUT /travelExpense/cost/{id}",
            "DELETE /travelExpense/cost/{id}",
            "POST /travelExpense/costParticipant",
            "POST /travelExpense/costParticipant/list",
            "DELETE /travelExpense/costParticipant/list",
            "DELETE /travelExpense/costParticipant/{id}",
            "POST /travelExpense/drivingStop",
            "DELETE /travelExpense/drivingStop/{id}",
            "POST /travelExpense/mileageAllowance",
            "PUT /travelExpense/mileageAllowance/{id}",
            "DELETE /travelExpense/mileageAllowance/{id}",
            "POST /travelExpense/passenger",
            "POST /travelExpense/passenger/list",
            "DELETE /travelExpense/passenger/list",
            "PUT /travelExpense/passenger/{id}",
            "DELETE /travelExpense/passenger/{id}",
            "POST /travelExpense/perDiemCompensation",
            "PUT /travelExpense/perDiemCompensation/{id}",
            "DELETE /travelExpense/perDiemCompensation/{id}",
            "PUT /travelExpense/{id}",
            "PUT /travelExpense/{id}/convert",
            "POST /travelExpense/{travelExpenseId}/attachment",
            "DELETE /travelExpense/{travelExpenseId}/attachment",
            "POST /travelExpense/{travelExpenseId}/attachment/list",
        ],
    },
    {
        "category": "DELIVERYADDRESS",
        "endpoint_count": 1,
        "tools": ["search_delivery_addresses", "update_delivery_address"],
        "covered": [
            "GET /deliveryAddress (search)",
            "PUT /deliveryAddress/{id}",
        ],
        "uncovered": [],
    },
    {
        "category": "BALANCE / REPORTING",
        "endpoint_count": 1,
        "tools": ["get_balance_sheet", "search_currencies", "get_company_info",
                  "get_year_end", "search_year_ends"],
        "covered": [
            "GET /balanceSheet",
            "GET /currency (search)",
            "GET /company/{id}",
            "GET /yearEnd/{id}",
            "GET /yearEnd (search)",
        ],
        "uncovered": [],
    },
    {
        "category": "GENERIC / UTILITY",
        "endpoint_count": 0,
        "tools": ["get_entity_by_id", "delete_entity", "extract_file_content"],
        "covered": [
            "GET /{entity_type}/{id} (any entity)",
            "DELETE /{entity_type}/{id} (any entity)",
            "Local file extraction (PDF/image)",
        ],
        "uncovered": [],
    },
    # ── Uncovered categories ──
    {
        "category": "ACCOUNTINGOFFICE",
        "endpoint_count": 3, "tools": [], "covered": [],
        "uncovered": [
            "PUT /accountingOffice/reconciliations/{id}/control/:controlReconciliation",
            "PUT /accountingOffice/reconciliations/{id}/control/:reconcile",
            "PUT /accountingOffice/reconciliations/{id}/control/:requestControl",
        ],
    },
    {
        "category": "ACTIVITY",
        "endpoint_count": 2, "tools": [], "covered": [],
        "uncovered": ["POST /activity", "POST /activity/list"],
    },
    {
        "category": "ASSET",
        "endpoint_count": 8, "tools": [], "covered": [],
        "uncovered": [
            "POST /asset", "DELETE /asset/deleteImport",
            "DELETE /asset/deleteStartingBalance",
            "POST /asset/duplicate/{id}", "POST /asset/list",
            "POST /asset/upload", "PUT /asset/{id}",
            "DELETE /asset/{id}",
        ],
    },
    {
        "category": "ATTESTATION",
        "endpoint_count": 1, "tools": [], "covered": [],
        "uncovered": ["PUT /attestation/:addApprover"],
    },
    {
        "category": "DIVISION",
        "endpoint_count": 4, "tools": [], "covered": [],
        "uncovered": [
            "POST /division", "POST /division/list",
            "PUT /division/list", "PUT /division/{id}",
        ],
    },
    {
        "category": "DOCUMENTARCHIVE",
        "endpoint_count": 10, "tools": [], "covered": [],
        "uncovered": [
            "POST /documentArchive/account/{id}",
            "POST /documentArchive/customer/{id}",
            "POST /documentArchive/employee/{id}",
            "POST /documentArchive/product/{id}",
            "POST /documentArchive/project/{id}",
            "POST /documentArchive/reception",
            "POST /documentArchive/supplier/{id}",
            "PUT /documentArchive/{id}",
            "DELETE /documentArchive/{id}",
        ],
    },
    {
        "category": "EVENT",
        "endpoint_count": 6, "tools": [], "covered": [],
        "uncovered": [
            "POST /event/subscription", "POST /event/subscription/list",
            "PUT /event/subscription/list", "DELETE /event/subscription/list",
            "PUT /event/subscription/{id}", "DELETE /event/subscription/{id}",
        ],
    },
    {
        "category": "INCOMINGINVOICE",
        "endpoint_count": 3, "tools": [], "covered": [],
        "uncovered": [
            "POST /incomingInvoice", "PUT /incomingInvoice/{id}",
            "POST /incomingInvoice/{id}/addPayment",
        ],
    },
    {
        "category": "INVENTORY",
        "endpoint_count": 16, "tools": [], "covered": [],
        "uncovered": [
            "POST /inventory", "POST /inventory/location",
            "POST /inventory/location/list", "PUT /inventory/location/list",
            "DELETE /inventory/location/list", "PUT /inventory/location/{id}",
            "DELETE /inventory/location/{id}",
            "POST /inventory/stocktaking",
            "POST /inventory/stocktaking/productline",
            "PUT /inventory/stocktaking/productline/{id}",
            "DELETE /inventory/stocktaking/productline/{id}",
            "PUT /inventory/stocktaking/{id}",
            "DELETE /inventory/stocktaking/{id}",
            "PUT /inventory/{id}", "DELETE /inventory/{id}",
        ],
    },
    {
        "category": "PURCHASEORDER",
        "endpoint_count": 40, "tools": [], "covered": [],
        "uncovered": [
            "POST /purchaseOrder", "POST /purchaseOrder/deviation",
            "POST /purchaseOrder/goodsReceipt",
            "POST /purchaseOrder/orderline",
            "PUT /purchaseOrder/{id}", "DELETE /purchaseOrder/{id}",
            "(+34 more endpoints)",
        ],
    },
    {
        "category": "SALARY",
        "endpoint_count": 24, "tools": [], "covered": [],
        "uncovered": [
            "POST /salary/transaction", "DELETE /salary/transaction/{id}",
            "PUT /salary/settings",
            "POST /salary/settings/holiday",
            "POST /salary/settings/pensionScheme",
            "(+19 more endpoints)",
        ],
    },
    {
        "category": "SUPPLIERINVOICE",
        "endpoint_count": 9, "tools": [], "covered": [],
        "uncovered": [
            "PUT /supplierInvoice/:approve",
            "PUT /supplierInvoice/:reject",
            "POST /supplierInvoice/{id}/:addPayment",
            "PUT /supplierInvoice/{id}/:changeDimension",
            "(+5 more endpoints)",
        ],
    },
    {
        "category": "TIMESHEET",
        "endpoint_count": 34, "tools": [], "covered": [],
        "uncovered": [
            "POST /timesheet/entry", "POST /timesheet/entry/list",
            "PUT /timesheet/entry/{id}", "DELETE /timesheet/entry/{id}",
            "POST /timesheet/allocated", "PUT /timesheet/allocated/{id}",
            "PUT /timesheet/timeClock/:start",
            "PUT /timesheet/timeClock/{id}/:stop",
            "(+26 more endpoints)",
        ],
    },
    {
        "category": "SAFT",
        "endpoint_count": 1, "tools": [], "covered": [],
        "uncovered": ["POST /saft/importSAFT"],
    },
    {
        "category": "TOKEN",
        "endpoint_count": 4, "tools": [], "covered": [],
        "uncovered": [
            "PUT /token/employee/:create",
            "POST /token/session/:create",
            "PUT /token/session/:create",
            "DELETE /token/session/{token}",
        ],
    },
    {
        "category": "YEAREND",
        "endpoint_count": 17, "tools": [], "covered": [],
        "uncovered": [
            "POST /yearEnd/penneo/casefiles",
            "POST /yearEnd/penneo/documents",
            "POST /yearEnd/researchAndDevelopment2024",
            "(+14 more endpoints)",
        ],
    },
    {
        "category": "COMPANY",
        "endpoint_count": 3,
        "tools": ["get_company_info"],
        "covered": ["GET /company/{id}"],
        "uncovered": [
            "PUT /company",
            "POST /company/salesmodules",
            "PUT /company/settings/altinn",
        ],
    },
    {
        "category": "OTHER (small)",
        "endpoint_count": 8, "tools": [], "covered": [],
        "uncovered": [
            "POST /balance/reconciliation/annual/context",
            "POST /userLicense/export",
            "POST /vatTermSizeSettings",
            "PUT /vatTermSizeSettings/{id}",
            "DELETE /vatTermSizeSettings/{id}",
            "POST /voucherMessage",
            "POST /voucherStatus",
            "PUT /subscription/cancel",
        ],
    },
]


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
