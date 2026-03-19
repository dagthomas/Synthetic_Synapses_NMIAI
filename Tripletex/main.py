import base64
import logging
import os
import shutil
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from agent import create_agent
from config import AGENT_API_KEY, GOOGLE_API_KEY, MAX_AGENT_TURNS
from tripletex_client import TripletexClient
from tools import build_all_tools

# Set Google API key for ADK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")


@app.post("/solve")
async def solve(request: Request):
    # Optional Bearer token auth
    if AGENT_API_KEY:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {AGENT_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    # Per-request isolation
    request_id = str(uuid.uuid4())
    files_dir = os.path.join(os.environ.get("TEMP", "/tmp"), f"tripletex_{request_id}")

    # Decode attachments
    file_names = []
    if files:
        os.makedirs(files_dir, exist_ok=True)
        for f in files:
            data = base64.b64decode(f["content_base64"])
            filepath = os.path.join(files_dir, f["filename"])
            with open(filepath, "wb") as fh:
                fh.write(data)
            file_names.append(f["filename"])
            log.info(f"Saved attachment: {f['filename']} ({len(data)} bytes)")

    # Build per-request client and tools
    client = TripletexClient(creds["base_url"], creds["session_token"])
    tools = build_all_tools(client, files_dir=files_dir if files else "")

    # Build agent and runner
    agent = create_agent(tools)
    runner = InMemoryRunner(agent=agent, app_name="tripletex_agent")

    # Build user message
    user_text = prompt
    if file_names:
        user_text += f"\n\nAttached files: {', '.join(file_names)}"
        user_text += "\nUse extract_file_content to read file contents."

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_text)],
    )

    # Pre-activate common modules (1 API call, prevents module-not-enabled errors)
    client.put("/company/modules", json={
        "moduleDepartment": True,
        "moduleProjectEconomy": True,
    })

    # Run agent with turn limit
    log.info(f"Running agent for request {request_id}")
    log.info(f"Prompt: {prompt[:200]}...")

    final_text = ""
    turn_count = 0
    try:
        async for event in runner.run_async(
            user_id="tripletex",
            session_id=request_id,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text = part.text
                    if hasattr(part, "function_call") and part.function_call:
                        turn_count += 1
                        log.info(f"Tool call #{turn_count}: {part.function_call.name}({part.function_call.args})")
                        if turn_count >= MAX_AGENT_TURNS:
                            log.warning(f"Turn limit reached ({MAX_AGENT_TURNS}), stopping agent")
                            break
            if turn_count >= MAX_AGENT_TURNS:
                break
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)

    log.info(f"Agent done. API calls: {client._call_count}, errors: {client._error_count}")
    log.info(f"Final response: {final_text[:200]}")

    # Cleanup temp files
    if files_dir and os.path.exists(files_dir):
        shutil.rmtree(files_dir, ignore_errors=True)

    return JSONResponse({"status": "completed"})
