import asyncio
import base64
import json
import logging
import os
import queue as _queue_mod
import shutil
import threading
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from agent import create_agent
from config import AGENT_API_KEY, GOOGLE_API_KEY, MAX_AGENT_TURNS, DEFAULT_MODE, GEMINI_MODEL
from tripletex_client import TripletexClient
from tools import build_tools_dict
from tool_router import classify_task, classify_tasks, select_tools, extract_currency_info, llm_classify_task
from static_runner import run_static, has_pipeline

# Concurrency limiter — prevent Gemini API rate-limiting when competition
# sends many tasks at once.  Allow up to 10 agents running in parallel.
_AGENT_SEMAPHORE = asyncio.Semaphore(10)

# Hard timeout for agent execution (seconds).  Competition requires
# response within 300s; leave 30s margin for HTTP overhead.
_AGENT_TIMEOUT = 270

# Dashboard solve_log integration (lazy init on first use)
_has_dashboard = False
_dashboard_inited = False
create_solve_log = None
_update_llm_task_type = None

def _ensure_dashboard():
    global _has_dashboard, _dashboard_inited, create_solve_log, _update_llm_task_type
    if _dashboard_inited:
        return
    _dashboard_inited = True
    try:
        from dashboard.db import init_db, create_solve_log as _csl, update_solve_log_llm_task_type as _ult
        init_db()
        create_solve_log = _csl
        _update_llm_task_type = _ult
        _has_dashboard = True
    except Exception:
        _has_dashboard = False


def _run_llm_classify_bg(request_id: str, prompt: str):
    """Run LLM classification in background thread and save to DB."""
    try:
        llm_type = llm_classify_task(prompt)
        if llm_type and _update_llm_task_type:
            _update_llm_task_type(request_id, llm_type)
            log.debug(f"[{request_id[:8]}] LLM classify: {llm_type}")
    except Exception as e:
        log.debug(f"[{request_id[:8]}] LLM classify bg error: {e}")

# ── Live Event Broadcasting ────────────────────────────────────────
# Thread-safe pub/sub for SSE live activity streaming.
# Each SSE subscriber gets its own queue; events are broadcast to all.
_sse_subscribers: dict[str, _queue_mod.Queue] = {}
_sse_lock = threading.Lock()


def _emit_event(event: dict):
    """Broadcast an event to all SSE subscribers (thread-safe)."""
    event.setdefault("ts", datetime.now().isoformat())
    with _sse_lock:
        dead = []
        for sub_id, q in _sse_subscribers.items():
            try:
                q.put_nowait(event)
            except _queue_mod.Full:
                dead.append(sub_id)
        for d in dead:
            del _sse_subscribers[d]


# Set Google API key for ADK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "google", "grpc", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class _NoDocsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "GET /docs" not in msg and "GET /events" not in msg


logging.getLogger("uvicorn.access").addFilter(_NoDocsFilter())

security = HTTPBearer(auto_error=False)

app = FastAPI(title="Tripletex AI Agent")


# --- Request models for Swagger UI ---
class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str = "application/pdf"


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = []
    tripletex_credentials: TripletexCredentials

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
                    "files": [],
                    "tripletex_credentials": {
                        "base_url": "https://YOUR-SANDBOX.tripletex.dev/v2",
                        "session_token": "YOUR_TOKEN_HERE"
                    }
                }
            ]
        }
    }


@app.get("/events")
async def live_events():
    """SSE endpoint: streams live agent activity events to subscribers."""
    sub_id = str(uuid.uuid4())
    q = _queue_mod.Queue(maxsize=500)
    with _sse_lock:
        _sse_subscribers[sub_id] = q

    async def generate():
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    event = await loop.run_in_executor(
                        None, lambda: q.get(timeout=15)
                    )
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except _queue_mod.Empty:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            with _sse_lock:
                _sse_subscribers.pop(sub_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _run_agent(body: SolveRequest, save_payload: bool = False, source: str = "", mode: str = "") -> dict:
    """Shared agent execution logic. Returns full details dict."""
    import time as _time

    prompt = body.prompt
    files = body.files
    creds = body.tripletex_credentials

    # Per-request isolation
    request_id = str(uuid.uuid4())

    # Emit: request started
    _emit_event({"type": "request_start", "request_id": request_id,
                 "prompt": prompt, "files": [f.filename for f in files], "source": source})

    # Save payload only for live competition runs
    if save_payload:
        payloads_dir = os.path.join(os.path.dirname(__file__), "payloads")
        os.makedirs(payloads_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload_file = os.path.join(payloads_dir, f"{ts}_{request_id[:8]}.json")
        with open(payload_file, "w", encoding="utf-8") as pf:
            json.dump({
                "prompt": prompt,
                "files": [{"filename": f.filename, "mime_type": f.mime_type} for f in files],
                "tripletex_credentials": {"base_url": creds.base_url, "session_token": creds.session_token[:8] + "..."},
            }, pf, ensure_ascii=False, indent=2)
        log.info(f"Payload saved to {payload_file}")
    files_dir = os.path.join(os.environ.get("TEMP", "/tmp"), f"tripletex_{request_id}")

    # Decode attachments
    file_names = []
    if files:
        os.makedirs(files_dir, exist_ok=True)
        for f in files:
            data = base64.b64decode(f.content_base64)
            filepath = os.path.join(files_dir, f.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as fh:
                fh.write(data)
            file_names.append(f.filename)
            log.info(f"Saved attachment: {f.filename} ({len(data)} bytes)")

    # Build per-request client and tools (with live event callback for API calls)
    def _on_api_call(entry):
        _emit_event({"type": "api_call", "request_id": request_id,
                     "method": entry.get("method", ""), "url": entry.get("url", ""),
                     "status": entry.get("status", 0), "ok": entry.get("ok", False),
                     "elapsed": entry.get("elapsed", 0),
                     "error": entry.get("error", "")})

    client = TripletexClient(creds.base_url, creds.session_token, on_api_call=_on_api_call)

    # Pre-warm caches in thread pool to avoid blocking the event loop
    await asyncio.get_running_loop().run_in_executor(None, client._prewarm_caches)

    all_tools_dict = build_tools_dict(client, files_dir=files_dir if files else "")
    task_types = classify_tasks(prompt)
    task_type = task_types[0] if task_types else None  # primary type for backward compat
    tool_selection = select_tools(task_types, all_tools_dict, has_files=bool(files), prompt=prompt)
    tools = tool_selection.tools
    classification_level = tool_selection.classification_level
    missing_tools = tool_selection.missing_tools
    log.info(f"[{request_id[:8]}] classify: {task_types or '?'} ({classification_level}) | {len(tools)}/{len(all_tools_dict)} tools")
    if missing_tools:
        log.warning(f"[{request_id[:8]}] MISSING TOOLS for {task_types}: {missing_tools}")

    # Emit: classification result
    _emit_event({"type": "classify", "request_id": request_id,
                 "task_type": task_type or "", "task_types": task_types,
                 "classification_level": classification_level,
                 "tool_count": len(tools), "total_tools": len(all_tools_dict),
                 "missing_tools": missing_tools,
                 "tools": [t.__name__ if hasattr(t, '__name__') else str(t) for t in tools[:20]]})

    # ── Static pipeline shortcut ────────────────────────────────────
    effective_mode = mode or DEFAULT_MODE
    t_start = _time.time()
    if effective_mode == "static" and task_type and has_pipeline(task_type):
        try:
            _emit_event({"type": "static_start", "request_id": request_id,
                         "task_type": task_type})
            loop = asyncio.get_running_loop()
            static_result = await loop.run_in_executor(
                None, run_static, task_type, prompt, all_tools_dict, client,
                _emit_event, request_id, file_names,
            )
            elapsed = _time.time() - t_start
            static_result["elapsed"] = round(elapsed, 2)
            static_result["task_types"] = task_types
            static_result["mode"] = "static"
            log.info(f"[{request_id[:8]}] ── STATIC DONE ── {elapsed:.1f}s | {static_result.get('api_calls', 0)} API calls")
            _emit_event({"type": "request_done", "request_id": request_id,
                         "elapsed": round(elapsed, 2),
                         "api_calls": static_result.get("api_calls", 0),
                         "api_errors": static_result.get("api_errors", 0),
                         "response": static_result.get("agent_response", "")[:300],
                         "task_type": task_type, "task_types": task_types,
                         "mode": "static"})
            # Cleanup temp files
            if files_dir and os.path.exists(files_dir):
                shutil.rmtree(files_dir, ignore_errors=True)
            # Save to dashboard
            _ensure_dashboard()
            if _has_dashboard:
                try:
                    create_solve_log(
                        request_id=request_id, prompt=prompt,
                        files_json=json.dumps([f.filename for f in files]),
                        base_url=creds.base_url,
                        api_calls=static_result.get("api_calls", 0),
                        api_errors=static_result.get("api_errors", 0),
                        elapsed_seconds=elapsed,
                        agent_response=static_result.get("agent_response", ""),
                        tool_calls_json=json.dumps(static_result.get("tool_calls", []), default=str),
                        api_log_json=json.dumps(static_result.get("api_log", []), default=str),
                        task_type=task_type, tool_count=len(tools),
                        source=source, classification_level=classification_level,
                    )
                    # Fire-and-forget LLM classification in background
                    threading.Thread(target=_run_llm_classify_bg, args=(request_id, prompt), daemon=True).start()
                except Exception as e:
                    log.warning(f"Failed to save solve_log (static): {e}")
            return static_result
        except Exception as e:
            log.warning(f"[{request_id[:8]}] Static pipeline failed ({type(e).__name__}: {e}), falling back to agent")
            _emit_event({"type": "static_fallback", "request_id": request_id,
                         "error": str(e)})

    # Build agent and runner
    agent = create_agent(tools, task_types=task_types or None,
                         missing_tools=missing_tools or None)
    runner = InMemoryRunner(agent=agent, app_name="tripletex_agent")

    # Create session for this request
    session = await runner.session_service.create_session(
        app_name="tripletex_agent",
        user_id="tripletex",
    )
    session_id = session.id

    # Build user message
    user_text = prompt
    if file_names:
        user_text += f"\n\nAttached files: {', '.join(file_names)}"
        user_text += "\nUse extract_file_content to read file contents."

    # Inject pre-computed currency calculations for agio tasks
    if task_type == "invoice_with_payment":
        ci = extract_currency_info(prompt)
        if ci:
            diff_label = "agio (gain)" if ci["is_agio"] else "disagio (loss)"
            user_text += (
                f"\n\n[PRE-COMPUTED CURRENCY CALCULATION — use these exact values]"
                f"\nForeign amount: {ci['amount']} {ci['currency']}"
                f"\nInvoice NOK (old rate {ci['old_rate']}): {ci['amount']} × {ci['old_rate']} = {ci['invoice_nok']}"
                f"\nPayment NOK (new rate {ci['new_rate']}): {ci['amount']} × {ci['new_rate']} = {ci['payment_nok']}"
                f"\n{diff_label}: {abs(ci['diff'])}"
                f"\n→ register_payment(amount={ci['payment_nok']}, paidAmountCurrency={ci['amount']})"
                f"\n→ create_voucher: account 1500 amount={ci['diff']} (customerId!), "
                f"account {ci['agio_account']} amount={-ci['diff']}"
                f"\nDo NOT skip the create_voucher step!"
            )
            log.info(f"[{request_id[:8]}] currency note injected: {ci['currency']} {ci['amount']}, "
                     f"rates {ci['old_rate']}->{ci['new_rate']}, diff={ci['diff']}")

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_text)],
    )

    # Run agent with turn limit
    log.info(f"[{request_id[:8]}] ── START ── {prompt[:120]}{'…' if len(prompt)>120 else ''}")
    if file_names:
        log.info(f"[{request_id[:8]}]   files: {file_names}")

    final_text = ""
    turn_count = 0
    tool_calls = []
    t_start = _time.time()

    # Emit: agent starting
    _emit_event({"type": "agent_start", "request_id": request_id})

    try:
        async for event in runner.run_async(
            user_id="tripletex",
            session_id=session_id,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text = part.text
                        log.info(f"[{request_id[:8]}] agent: {part.text[:200]}{'…' if len(part.text)>200 else ''}")
                        _emit_event({"type": "text", "request_id": request_id,
                                     "text": part.text[:500]})
                    if hasattr(part, "function_call") and part.function_call:
                        turn_count += 1
                        fc = part.function_call
                        args_brief = {k: (str(v)[:60] + '…' if len(str(v)) > 60 else v) for k, v in (fc.args or {}).items()}
                        log.info(f"[{request_id[:8]}] tool #{turn_count}: {fc.name}({args_brief})")
                        tool_calls.append({
                            "tool": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                            "result": None,
                        })
                        _emit_event({"type": "tool_call", "request_id": request_id,
                                     "turn": turn_count, "tool": fc.name,
                                     "args": dict(fc.args) if fc.args else {}})
                        if turn_count >= MAX_AGENT_TURNS:
                            log.warning(f"[{request_id[:8]}] turn limit ({MAX_AGENT_TURNS})")
                            break
                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        resp_data = fr.response if fr.response else {}
                        is_err = isinstance(resp_data, dict) and resp_data.get("error")
                        if is_err:
                            log.warning(f"[{request_id[:8]}]   => {fr.name} ERROR: {str(resp_data.get('message', resp_data))[:200]}")
                        else:
                            log.info(f"[{request_id[:8]}]   => {fr.name} OK")
                        log.debug(f"[{request_id[:8]}]   => {fr.name} full: {resp_data}")
                        _emit_event({"type": "tool_result", "request_id": request_id,
                                     "turn": turn_count, "tool": fr.name,
                                     "ok": not is_err,
                                     "error": resp_data.get("message", "") if is_err else ""})
                        for tc in tool_calls:
                            if tc["tool"] == fr.name and tc["result"] is None:
                                tc["result"] = {
                                    "ok": not is_err,
                                    "data": str(resp_data),
                                }
                                if is_err:
                                    tc["result"]["error"] = resp_data.get("message", str(resp_data))
                                break
            if turn_count >= MAX_AGENT_TURNS:
                break
    except Exception as e:
        log.error(f"[{request_id[:8]}] error: {e}", exc_info=True)
        _emit_event({"type": "request_error", "request_id": request_id, "error": str(e)})

    elapsed = _time.time() - t_start
    log.info(f"[{request_id[:8]}] ── DONE ── {elapsed:.1f}s | {turn_count} turns | {client._call_count} API calls ({client._error_count} err) | {final_text[:100]}{'…' if len(final_text)>100 else ''}")

    # Emit: request done
    _emit_event({"type": "request_done", "request_id": request_id,
                 "elapsed": round(elapsed, 2), "api_calls": client._call_count,
                 "api_errors": client._error_count, "response": (final_text or "")[:300],
                 "task_type": task_type or "", "task_types": task_types,
                 "missing_tools": missing_tools, "turns": turn_count})

    # Cleanup temp files
    if files_dir and os.path.exists(files_dir):
        shutil.rmtree(files_dir, ignore_errors=True)

    # Save to dashboard solve_logs
    _ensure_dashboard()
    if _has_dashboard:
        try:
            create_solve_log(
                request_id=request_id,
                prompt=prompt,
                files_json=json.dumps([f.filename for f in files]),
                base_url=creds.base_url,
                api_calls=client._call_count,
                api_errors=client._error_count,
                elapsed_seconds=elapsed,
                agent_response=final_text or "",
                tool_calls_json=json.dumps(tool_calls, default=str),
                api_log_json=json.dumps(client._call_log, default=str),
                task_type=task_type or "",
                tool_count=len(tools),
                source=source,
                classification_level=classification_level,
            )
            # Fire-and-forget LLM classification in background
            threading.Thread(target=_run_llm_classify_bg, args=(request_id, prompt), daemon=True).start()
        except Exception as e:
            log.warning(f"Failed to save solve_log: {e}")

    result = {
        "agent_response": final_text or "",
        "tool_calls": tool_calls,
        "api_calls": client._call_count,
        "api_errors": client._error_count,
        "api_log": client._call_log,
        "elapsed": round(elapsed, 2),
        "task_types": task_types,
        "mode": "agent",
    }
    if missing_tools:
        result["missing_tools"] = missing_tools
    return result


@app.post("/solve")
async def solve(body: SolveRequest, mode: str = "",
                credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Competition endpoint — returns only {"status": "completed"}."""
    if AGENT_API_KEY:
        if not credentials or credentials.credentials != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _guarded():
        async with _AGENT_SEMAPHORE:
            await _run_agent(body, save_payload=True, source="competition", mode=mode)
    try:
        await asyncio.wait_for(_guarded(), timeout=_AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(f"SOLVE timeout ({_AGENT_TIMEOUT}s)")
    except Exception as e:
        log.error(f"SOLVE error: {e}", exc_info=True)
    return JSONResponse({"status": "completed"})


@app.post("/solve-debug")
async def solve_debug(body: SolveRequest, source: str = "debug", mode: str = "",
                      credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Debug endpoint — returns full tool call and API call details."""
    if AGENT_API_KEY:
        if not credentials or credentials.credentials != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _guarded():
        async with _AGENT_SEMAPHORE:
            return await _run_agent(body, source=source, mode=mode)
    try:
        details = await asyncio.wait_for(_guarded(), timeout=_AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(f"SOLVE-DEBUG timeout ({_AGENT_TIMEOUT}s)")
        details = {"agent_response": "TIMEOUT", "tool_calls": [], "api_calls": 0, "api_errors": 0, "api_log": [], "elapsed": _AGENT_TIMEOUT}
    except Exception as e:
        log.error(f"SOLVE-DEBUG error: {e}", exc_info=True)
        details = {"agent_response": f"ERROR: {e}", "tool_calls": [], "api_calls": 0, "api_errors": 0, "api_log": [], "elapsed": 0}
    return JSONResponse({"status": "completed", **details})


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    log.info(f"Server starting — default mode: {DEFAULT_MODE}, model: {GEMINI_MODEL}, max turns: {MAX_AGENT_TURNS}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
