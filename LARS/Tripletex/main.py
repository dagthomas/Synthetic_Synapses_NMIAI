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
from config import AGENT_API_KEY, GOOGLE_API_KEY, MAX_AGENT_TURNS
from tripletex_client import TripletexClient
from tools import build_tools_dict
from tool_router import classify_task, select_tools

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

def _ensure_dashboard():
    global _has_dashboard, _dashboard_inited, create_solve_log
    if _dashboard_inited:
        return
    _dashboard_inited = True
    try:
        from dashboard.db import init_db, create_solve_log as _csl
        init_db()
        create_solve_log = _csl
        _has_dashboard = True
    except Exception:
        _has_dashboard = False

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

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "google", "grpc", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

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
                        "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
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


async def _run_agent(body: SolveRequest, save_payload: bool = False, source: str = "") -> dict:
    """Shared agent execution logic. Returns full details dict."""
    import time as _time

    prompt = body.prompt
    files = body.files
    creds = body.tripletex_credentials

    # Per-request isolation
    request_id = str(uuid.uuid4())

    # Emit: request started
    _emit_event({"type": "request_start", "request_id": request_id,
                 "prompt": prompt[:300], "files": [f.filename for f in files], "source": source})

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
    task_type = classify_task(prompt)
    tool_selection = select_tools(task_type, all_tools_dict, has_files=bool(files), prompt=prompt)
    tools = tool_selection.tools
    classification_level = tool_selection.classification_level
    log.info(f"[REQ {request_id[:8]}] Task type: {task_type or 'UNKNOWN'}, classification: {classification_level}, tools: {len(tools)}/{len(all_tools_dict)}")

    # Emit: classification result
    _emit_event({"type": "classify", "request_id": request_id,
                 "task_type": task_type or "", "classification_level": classification_level,
                 "tool_count": len(tools), "total_tools": len(all_tools_dict),
                 "tools": [t.__name__ if hasattr(t, '__name__') else str(t) for t in tools[:20]]})

    # Build agent and runner
    agent = create_agent(tools, task_type=task_type)
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

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_text)],
    )

    # Run agent with turn limit
    log.info(f"{'='*60}")
    log.info(f"[REQ {request_id[:8]}] START")
    log.info(f"[REQ {request_id[:8]}] Prompt: {prompt}")
    log.info(f"[REQ {request_id[:8]}] Files: {file_names}")
    log.info(f"[REQ {request_id[:8]}] Base URL: {creds.base_url}")
    log.info(f"[REQ {request_id[:8]}] Session: {session_id}")
    log.info(f"{'='*60}")

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
                        log.info(f"[AGENT] Text: {part.text[:500]}")
                        _emit_event({"type": "text", "request_id": request_id,
                                     "text": part.text[:500]})
                    if hasattr(part, "function_call") and part.function_call:
                        turn_count += 1
                        fc = part.function_call
                        log.info(f"[AGENT] Tool call #{turn_count}: {fc.name}({fc.args})")
                        tool_calls.append({
                            "tool": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                            "result": None,
                        })
                        _emit_event({"type": "tool_call", "request_id": request_id,
                                     "turn": turn_count, "tool": fc.name,
                                     "args": dict(fc.args) if fc.args else {}})
                        if turn_count >= MAX_AGENT_TURNS:
                            log.warning(f"[AGENT] Turn limit reached ({MAX_AGENT_TURNS}), stopping")
                            break
                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        resp_data = fr.response if fr.response else {}
                        resp_str = str(resp_data)[:500]
                        log.info(f"[AGENT] Tool result: {fr.name} -> {resp_str}")
                        log.debug(f"[AGENT] Tool result FULL: {fr.name} -> {resp_data}")
                        is_err = isinstance(resp_data, dict) and resp_data.get("error")
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
        log.error(f"[AGENT] Error: {e}", exc_info=True)
        _emit_event({"type": "request_error", "request_id": request_id, "error": str(e)})

    elapsed = _time.time() - t_start
    log.info(f"{'='*60}")
    log.info(f"[REQ {request_id[:8]}] DONE in {elapsed:.1f}s")
    log.info(f"[REQ {request_id[:8]}] API calls: {client._call_count}, errors: {client._error_count}")
    log.info(f"[REQ {request_id[:8]}] Agent turns: {turn_count}")
    log.info(f"[REQ {request_id[:8]}] Final response: {final_text}")
    log.info(f"{'='*60}")

    # Emit: request done
    _emit_event({"type": "request_done", "request_id": request_id,
                 "elapsed": round(elapsed, 2), "api_calls": client._call_count,
                 "api_errors": client._error_count, "response": (final_text or "")[:300],
                 "task_type": task_type or "", "turns": turn_count})

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
        except Exception as e:
            log.warning(f"Failed to save solve_log: {e}")

    return {
        "agent_response": final_text or "",
        "tool_calls": tool_calls,
        "api_calls": client._call_count,
        "api_errors": client._error_count,
        "api_log": client._call_log,
        "elapsed": round(elapsed, 2),
    }


@app.post("/solve")
async def solve(body: SolveRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Competition endpoint — returns only {"status": "completed"}."""
    if AGENT_API_KEY:
        if not credentials or credentials.credentials != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _guarded():
        async with _AGENT_SEMAPHORE:
            await _run_agent(body, save_payload=True, source="competition")
    try:
        await asyncio.wait_for(_guarded(), timeout=_AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(f"[SOLVE] Agent timed out after {_AGENT_TIMEOUT}s")
    except Exception as e:
        log.error(f"[SOLVE] Agent error: {e}", exc_info=True)
    return JSONResponse({"status": "completed"})


@app.post("/solve-debug")
async def solve_debug(body: SolveRequest, source: str = "debug",
                      credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Debug endpoint — returns full tool call and API call details."""
    if AGENT_API_KEY:
        if not credentials or credentials.credentials != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _guarded():
        async with _AGENT_SEMAPHORE:
            return await _run_agent(body, source=source)
    try:
        details = await asyncio.wait_for(_guarded(), timeout=_AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(f"[SOLVE-DEBUG] Agent timed out after {_AGENT_TIMEOUT}s")
        details = {"agent_response": "TIMEOUT", "tool_calls": [], "api_calls": 0, "api_errors": 0, "api_log": [], "elapsed": _AGENT_TIMEOUT}
    except Exception as e:
        log.error(f"[SOLVE-DEBUG] Agent error: {e}", exc_info=True)
        details = {"agent_response": f"ERROR: {e}", "tool_calls": [], "api_calls": 0, "api_errors": 0, "api_log": [], "elapsed": 0}
    return JSONResponse({"status": "completed", **details})


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
