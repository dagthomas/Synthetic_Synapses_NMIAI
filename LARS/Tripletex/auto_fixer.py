"""Auto-fixer: replay real competition prompts, analyze failures, suggest and apply code fixes.

Picks a random real prompt from solve_logs (competition source) for the given
task type, runs it through the agent, evaluates with LLM, then uses Gemini
to analyze failures and generate code fixes.

Usage:
    python auto_fixer.py --task create_employee          # run + fix one task
    python auto_fixer.py --task create_invoice --apply    # auto-apply fixes
    python auto_fixer.py --task create_employee --retries 3  # retry after fixing
    python auto_fixer.py --list                           # list available tasks
    python auto_fixer.py --pick                           # interactive task picker
    python auto_fixer.py --pick --apply --retries 2       # pick + auto-fix batch
    python auto_fixer.py --tasks create_employee create_invoice  # batch specific tasks
    python auto_fixer.py --failed --apply                 # re-run all previously failed
"""

import argparse
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tripletex_client import TripletexClient
from sim.task_definitions import ALL_TASKS
from dashboard.sandbox import ensure_sandbox_ready
from dashboard.db import get_conn
from tool_router import TASK_TOOL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("auto_fixer")

# ── Tool name → source file mapping ─────────────────────────────────

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")

_TOOL_FILE_MAP: dict[str, str] = {
    "create_employee": "employees.py", "update_employee": "employees.py", "search_employees": "employees.py",
    "create_customer": "customers.py", "update_customer": "customers.py", "search_customers": "customers.py",
    "delete_customer": "customers.py",
    "create_product": "products.py", "update_product": "products.py", "search_products": "products.py",
    "delete_product": "products.py",
    "create_department": "departments.py", "update_department": "departments.py",
    "search_departments": "departments.py", "delete_department": "departments.py",
    "create_supplier": "supplier.py", "update_supplier": "supplier.py", "search_suppliers": "supplier.py",
    "delete_supplier": "supplier.py",
    "create_contact": "contacts.py", "update_contact": "contacts.py", "search_contacts": "contacts.py",
    "delete_contact": "contacts.py",
    "create_order": "order.py", "create_invoice": "invoicing.py", "search_invoices": "invoicing.py",
    "register_payment": "invoicing.py", "create_credit_note": "invoicing.py",
    "create_travel_expense": "travel.py", "search_travel_expenses": "travel.py",
    "delete_travel_expense": "travel.py", "update_travel_expense": "travel.py",
    "create_travel_expense_cost": "travel_extras.py",
    "create_mileage_allowance": "travel_extras.py",
    "create_per_diem_compensation": "travel_extras.py",
    "create_project": "projects.py",
    "create_voucher": "ledger.py", "search_vouchers": "ledger.py", "reverse_voucher": "ledger.py",
    "get_ledger_accounts": "ledger.py", "create_opening_balance": "balance.py",
    "create_employment": "employment.py", "create_employment_details": "employment.py",
    "create_standard_time": "employment.py", "create_leave_of_absence": "employment.py",
    "create_incoming_invoice": "incoming_invoice.py",
    "search_bank_accounts": "bank.py",
    "extract_file_content": "files.py",
    "get_entity_by_id": "common.py", "delete_entity": "common.py",
    "search_salary_types": "salary.py", "create_salary_transaction": "salary.py",
    "search_year_ends": "year_end.py", "search_year_end_annexes": "year_end.py",
    "create_year_end_note": "year_end.py",
}


def _get_real_prompts(task_name: str, limit: int = 50) -> list[str]:
    """Get real competition prompts for a task type from solve_logs."""
    from contextlib import closing
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """SELECT prompt FROM solve_logs
               WHERE task_type = ? AND source = 'competition'
                 AND prompt IS NOT NULL AND prompt != ''
               ORDER BY id DESC LIMIT ?""",
            (task_name, limit),
        ).fetchall()
    return [r["prompt"] for r in rows]


def _get_all_real_task_types() -> list[str]:
    """Get all task types that have real competition logs."""
    from contextlib import closing
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """SELECT DISTINCT task_type FROM solve_logs
               WHERE source = 'competition'
                 AND task_type IS NOT NULL AND task_type != ''
                 AND prompt IS NOT NULL AND prompt != ''
               ORDER BY task_type""",
        ).fetchall()
    return [r["task_type"] for r in rows]


def _get_sandbox_client() -> TripletexClient:
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not base_url or not token:
        print("ERROR: Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env")
        sys.exit(1)
    return TripletexClient(base_url, token)


def _read_source_file(filename: str) -> str:
    """Read a tool source file."""
    path = os.path.join(TOOLS_DIR, filename)
    if not os.path.exists(path):
        # Try alternate names (supplier vs suppliers, etc.)
        for alt in [filename.replace(".py", "s.py"), filename.replace("s.py", ".py")]:
            alt_path = os.path.join(TOOLS_DIR, alt)
            if os.path.exists(alt_path):
                path = alt_path
                break
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"# File not found: {path}"


def _find_relevant_sources(tool_calls: list, task_name: str) -> dict[str, str]:
    """Find and read source files relevant to the failed task."""
    files_to_read = set()

    # From TASK_TOOL_MAP
    task_tools = TASK_TOOL_MAP.get(task_name, [])
    for tool_name in task_tools:
        if tool_name in _TOOL_FILE_MAP:
            files_to_read.add(_TOOL_FILE_MAP[tool_name])

    # From actual tool calls
    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        if tool_name in _TOOL_FILE_MAP:
            files_to_read.add(_TOOL_FILE_MAP[tool_name])

    sources = {}
    for filename in files_to_read:
        sources[filename] = _read_source_file(filename)

    # Always include agent.py for system instructions context
    agent_path = os.path.join(os.path.dirname(__file__), "agent.py")
    if os.path.exists(agent_path):
        with open(agent_path, "r", encoding="utf-8") as f:
            sources["agent.py"] = f.read()

    return sources


def run_eval_with_logs(task_name: str,
                       agent_url: str = "http://localhost:8000") -> dict:
    """Run a single eval using a real competition prompt and LLM-based evaluation.

    Pulls a random real prompt from solve_logs for the given task type,
    runs it through the agent, then uses LLM to evaluate success.

    Returns a dict with all diagnostic info:
        task_name, prompt, expected, agent_result, tool_calls, api_log,
        api_calls, api_errors, verification, score, elapsed
    """
    client = _get_sandbox_client()
    ensure_sandbox_ready(client)

    task_def = ALL_TASKS.get(task_name)

    # Get real prompt from competition logs
    real_prompts = _get_real_prompts(task_name)
    if not real_prompts:
        raise ValueError(f"No real competition prompts found for task '{task_name}'. "
                         f"Available: {', '.join(_get_all_real_task_types())}")

    import random as _rnd
    prompt = _rnd.choice(real_prompts)

    log.info(f"[AUTO-FIX] Task: {task_name} (real prompt)")
    log.info(f"[AUTO-FIX] Prompt: {prompt[:120]}...")

    # Call agent via /solve-debug
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": client.base_url,
            "session_token": client.auth[1],
        },
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("AGENT_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    try:
        resp = requests.post(
            f"{agent_url}/solve-debug", params={"source": "auto_fixer"},
            json=payload, headers=headers, timeout=300,
        )
        elapsed = time.time() - t0
        agent_result = resp.json() if resp.status_code == 200 else {"error": resp.text[:500]}
    except Exception as e:
        elapsed = time.time() - t0
        agent_result = {"error": str(e), "tool_calls": [], "api_log": []}

    tool_calls = agent_result.get("tool_calls", [])
    api_log = agent_result.get("api_log", [])
    api_calls = agent_result.get("api_calls", 0)
    api_errors = agent_result.get("api_errors", 0)
    agent_response = agent_result.get("agent_response", "")

    # LLM-based evaluation (no expected values needed)
    llm_eval = llm_evaluate_logs(prompt, tool_calls, api_log, agent_response)

    # Build verification-compatible dict from LLM evaluation
    passed = llm_eval["passed"]
    checks = [
        {"field": "_llm_eval", "passed": passed, "points": 10 if passed else 0,
         "max": 10, "detail": llm_eval["reasoning"]},
    ]
    for issue in llm_eval.get("issues", []):
        checks.append({"field": "_issue", "passed": False, "points": 0, "max": 0, "detail": issue})

    verification = {
        "total_points": 10 if passed else 0,
        "max_points": 10,
        "checks": checks,
    }

    tier = task_def.tier if task_def else 2
    score = {
        "correctness": 1.0 if passed else 0.0,
        "base_score": tier if passed else 0,
        "efficiency_bonus": 0,
        "final_score": tier if passed else 0,
        "max_possible": tier * 2,
    }

    return {
        "task_name": task_name,
        "task_def": task_def,
        "language": "-",
        "prompt": prompt,
        "expected": {},
        "agent_result": agent_result,
        "tool_calls": tool_calls,
        "api_log": api_log,
        "api_calls": api_calls,
        "api_errors": api_errors,
        "verification": verification,
        "score": score,
        "elapsed": elapsed,
    }


def build_error_report(result: dict) -> str:
    """Build a human/LLM-readable error report from eval results."""
    lines = []
    score = result["score"]
    verification = result["verification"]
    tool_calls = result["tool_calls"]
    api_log = result["api_log"]

    lines.append(f"=== EVAL REPORT: {result['task_name']} ({result['language']}) ===")
    lines.append(f"Correctness: {score['correctness']:.0%} ({verification['total_points']}/{verification['max_points']})")
    lines.append(f"Score: {score['final_score']}/{score['max_possible']}")
    lines.append(f"API calls: {result['api_calls']}, errors: {result['api_errors']}")

    # Break down write vs read calls (only writes count for efficiency)
    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    write_calls = [e for e in api_log if e.get("method", "").upper() in write_methods]
    read_calls = [e for e in api_log if e.get("method", "").upper() not in write_methods]
    write_errors = [e for e in write_calls if e.get("status", 200) >= 400]
    lines.append(f"Write calls (POST/PUT/DELETE/PATCH): {len(write_calls)} ({len(write_errors)} errors)")
    lines.append(f"Read calls (GET): {len(read_calls)} (free, don't count for efficiency)")

    # Optimal call count from TASK_INSTRUCTIONS
    _OPTIMAL_WRITES = {
        "create_employee": 1, "create_customer": 1, "create_product": 1,
        "create_department": 1, "create_supplier": 1, "create_contact": 2,
        "update_employee": 2, "update_customer": 2, "update_product": 2,
        "update_supplier": 2, "update_department": 2, "update_contact": 3,
        "create_invoice": 4, "create_multi_line_invoice": 6,
        "create_project": 3, "create_travel_expense": 2,
        "create_travel_expense_with_costs": 4, "invoice_with_payment": 5,
        "create_credit_note": 5, "create_employee_with_employment": 2,
        "create_supplier_invoice": 2, "project_invoice": 7,
        "delete_travel_expense": 1, "delete_customer": 1, "delete_supplier": 1,
        "delete_product": 1, "delete_department": 1, "delete_contact": 1,
        "delete_employee": 1, "create_ledger_voucher": 1,
        "reverse_voucher": 1, "reverse_payment": 1,
        "delete_invoice": 5, "create_opening_balance": 1,
        "create_dimension": 3, "bank_reconciliation": 2,
        "process_invoice_file": 4, "salary_with_bonus": 3,
    }
    optimal = _OPTIMAL_WRITES.get(result["task_name"])
    if optimal:
        delta = len(write_calls) - optimal
        if delta > 0:
            lines.append(f"EFFICIENCY WARNING: {len(write_calls)} writes vs {optimal} optimal (+{delta} extra)")
        else:
            lines.append(f"Write efficiency: {len(write_calls)}/{optimal} (on target)")

    lines.append(f"Elapsed: {result['elapsed']:.1f}s")
    lines.append("")

    # Prompt + expected
    lines.append(f"PROMPT: {result['prompt']}")
    lines.append(f"EXPECTED: {json.dumps(result['expected'], ensure_ascii=False)}")
    lines.append("")

    # Field checks
    lines.append("=== FIELD CHECKS ===")
    checks = verification.get("checks", [])
    failed_checks = []
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"  [{status}] {c['field']}: {c['detail']} ({c['points']}/{c['max']})")
        if not c["passed"]:
            failed_checks.append(c)
    lines.append("")

    # Tool calls with errors
    lines.append("=== TOOL CALLS ===")
    for i, tc in enumerate(tool_calls, 1):
        ok = tc.get("result", {}).get("ok", True) if tc.get("result") else "?"
        status = "OK" if ok else "ERROR"
        lines.append(f"  #{i} [{status}] {tc['tool']}({json.dumps(tc.get('args', {}), ensure_ascii=False)[:200]})")
        if tc.get("result") and not tc["result"].get("ok"):
            lines.append(f"       Error: {tc['result'].get('error', tc['result'].get('data', ''))[:300]}")
    lines.append("")

    # API log — only errors
    error_calls = [e for e in api_log if e.get("status", 200) >= 400]
    if error_calls:
        lines.append("=== API ERRORS ===")
        for e in error_calls:
            lines.append(f"  {e.get('method', '?')} {e.get('url', '?')} -> {e.get('status', '?')}")
            if e.get("response"):
                lines.append(f"       Response: {str(e['response'])[:300]}")
        lines.append("")

    # Agent final response
    agent_resp = result.get("agent_result", {}).get("agent_response", "")
    if agent_resp:
        lines.append(f"AGENT RESPONSE: {agent_resp[:500]}")
        lines.append("")

    return "\n".join(lines)


def get_fix_suggestions(error_report: str, sources: dict[str, str]) -> str:
    """Use Gemini to analyze failures and suggest code fixes.

    Returns a string with suggested fixes, including file paths and diffs.
    """
    from google import genai
    from config import GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Build source context
    source_context = ""
    for filename, content in sources.items():
        source_context += f"\n\n=== {filename} ===\n{content}"

    system_prompt = """\
You are an expert Python developer fixing bugs in a Tripletex API integration agent for a competition.

SCORING RULES (critical context):
- Correctness: field-by-field verification against Tripletex API. Must be 1.0 (perfect) to get efficiency bonus.
- Efficiency bonus (only applies at perfect correctness): based on WRITE calls (POST/PUT/DELETE/PATCH) vs best known solution. GET requests are FREE and don't count. Every 4xx error reduces the bonus.
- Max score = tier × 2 (e.g. Tier 2 task max = 4.0). Perfect correctness + optimal writes + zero errors = max score.
- Best score per task is kept forever — bad runs don't hurt, only improvements count.

PRIORITY ORDER for fixes:
1. CORRECTNESS FIRST: Fix field mismatches, missing steps, wrong values. A non-perfect run scores 0 efficiency bonus.
2. ELIMINATE 4xx ERRORS: Every 4xx error (400, 404, 422) on a write call reduces the efficiency bonus. Pre-validate inputs in tool code when possible.
3. MINIMIZE WRITE CALLS: Remove unnecessary POST/PUT/DELETE/PATCH calls. GET calls are free — reading data to avoid a failed write is always worth it.
4. NEVER add extra verification calls: Don't call GET after a successful create — the create response already has all data.

You are given:
1. An error report from running an eval task (includes write vs read call breakdown)
2. The source code of the relevant tool files

Analyze the error report to understand what went wrong:
- Field check FAILs: the tool created/updated the entity but a field value is wrong
- API errors (4xx): the tool sent an invalid request to the Tripletex API
- Tool call errors: the agent called a tool with wrong arguments
- Missing tool calls: the agent didn't call a necessary tool
- Extra write calls: unnecessary POST/PUT/DELETE beyond what the task requires

Return your fix as concrete code changes. For each change:
1. State the file to modify
2. Show the exact code to replace (OLD) and the new code (NEW)
3. Explain why

Format each fix as:
```
FILE: tools/example.py
OLD:
<exact lines to replace>
NEW:
<replacement lines>
REASON: <brief explanation>
```

Rules:
- Only suggest changes to tool files (tools/*.py) or agent.py
- Keep changes minimal — fix only what's broken
- Don't add unnecessary error handling or comments
- If the issue is in the agent instructions (wrong flow), suggest agent.py changes
- If the issue is a tool not sending the right fields, fix the tool
- If the fix requires a new parameter, update the function signature and docstring
- NEVER suggest adding GET verification calls after creates — this wastes turns (even though GETs are free, they waste agent turns)
- If a tool can pre-validate inputs (e.g. check postings balance) to avoid a 422, that's a good fix
"""

    user_msg = f"""\
{error_report}

=== SOURCE CODE ==={source_context}

Analyze the failures and suggest minimal code fixes.
If the eval passed perfectly (100% correctness) AND write efficiency is on target, say "No fixes needed."
If correctness is perfect but there are extra write calls or 4xx errors, suggest efficiency fixes (these affect the efficiency bonus).
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    return response.text


def parse_fixes(fix_text: str) -> list[dict]:
    """Parse fix suggestions into structured edits.

    Returns list of {file, old, new, reason}.
    """
    fixes = []
    # Match blocks: FILE: ...\nOLD:\n...\nNEW:\n...\nREASON: ...
    pattern = r'FILE:\s*(.+?)\s*\nOLD:\s*\n(.*?)\nNEW:\s*\n(.*?)\nREASON:\s*(.+?)(?=\n(?:FILE:|```)|$)'
    for m in re.finditer(pattern, fix_text, re.DOTALL):
        fixes.append({
            "file": m.group(1).strip(),
            "old": m.group(2).strip(),
            "new": m.group(3).strip(),
            "reason": m.group(4).strip(),
        })
    return fixes


def apply_fixes(fixes: list[dict], base_dir: str = "") -> list[dict]:
    """Apply parsed fixes to source files.

    Returns list of {file, applied, error} results.
    """
    if not base_dir:
        base_dir = os.path.dirname(__file__)

    results = []
    for fix in fixes:
        filepath = os.path.join(base_dir, fix["file"])
        if not os.path.exists(filepath):
            results.append({"file": fix["file"], "applied": False, "error": "File not found"})
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if fix["old"] not in content:
            results.append({"file": fix["file"], "applied": False,
                            "error": "OLD block not found in file (may have already been fixed)"})
            continue

        new_content = content.replace(fix["old"], fix["new"], 1)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        results.append({"file": fix["file"], "applied": True, "reason": fix["reason"]})
        log.info(f"[AUTO-FIX] Applied fix to {fix['file']}: {fix['reason']}")

    return results


def llm_evaluate_logs(prompt: str, tool_calls: list, api_log: list,
                      agent_response: str) -> dict:
    """Use Gemini to evaluate whether logs accomplished what the prompt asked.

    Returns {"passed": bool, "reasoning": str, "issues": list[str]}.
    """
    from google import genai
    from config import GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Build a structured summary of what happened
    tc_summary = []
    for i, tc in enumerate(tool_calls[:30], 1):
        res = tc.get("result") or {}
        ok = res.get("ok", "?")
        status = "OK" if ok else "ERROR"
        args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)[:300]
        result_str = ""
        if res:
            if res.get("error"):
                result_str = f" → Error: {res['error'][:200]}"
            elif res.get("data"):
                result_str = f" → {str(res['data'])[:200]}"
        tc_summary.append(f"  #{i} [{status}] {tc.get('tool', '?')}({args_str}){result_str}")

    api_summary = []
    for e in api_log[:50]:
        status = e.get("status", "?")
        api_summary.append(f"  {e.get('method', '?')} {status} {e.get('url', '?')}")

    context = f"""\
=== ORIGINAL PROMPT ===
{prompt}

=== TOOL CALLS ({len(tool_calls)} total) ===
{chr(10).join(tc_summary) if tc_summary else '  (none)'}

=== API CALLS ({len(api_log)} total) ===
{chr(10).join(api_summary) if api_summary else '  (none)'}

=== AGENT FINAL RESPONSE ===
{agent_response[:1000] if agent_response else '(empty)'}
"""

    system_prompt = """\
You are a strict QA evaluator for a Tripletex accounting API integration agent.

Given an original task prompt and the complete execution logs (tool calls, API calls, agent response), determine whether the agent CORRECTLY and COMPLETELY accomplished what the prompt asked.

Check for:
1. Did the agent create/update/delete the correct entity type?
2. Were all fields from the prompt set correctly (names, amounts, dates, etc.)?
3. Were all required sub-steps completed (e.g., for an invoice: customer + product + order + invoice)?
4. Did any tool calls or API calls fail that prevented completion?
5. Does the agent response confirm successful completion?

Be STRICT: if the prompt asked for specific values and you can't confirm they were set correctly from the logs, mark it as FAIL.

Respond in this exact JSON format (no markdown, no code fences):
{"passed": true/false, "reasoning": "2-3 sentence explanation", "issues": ["issue 1", "issue 2"]}

If passed, issues should be an empty list.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=context,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    )

    # Parse JSON response
    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"passed": False, "reasoning": f"LLM returned unparseable response: {text[:200]}", "issues": ["parse_error"]}

    return {
        "passed": bool(result.get("passed", False)),
        "reasoning": str(result.get("reasoning", "")),
        "issues": list(result.get("issues", [])),
    }


def run_and_fix(task_name: str,
                agent_url: str = "http://localhost:8000",
                max_attempts: int = 1, auto_apply: bool = False) -> dict:
    """Run eval, analyze failures, optionally fix and retry.

    Args:
        task_name: Task to evaluate (e.g. "create_employee").
        agent_url: Agent /solve-debug endpoint base URL.
        max_attempts: Max fix→retry cycles.
        auto_apply: If True, apply fixes without confirmation.

    Returns:
        Final result dict from run_eval_with_logs.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"  Attempt {attempt}/{max_attempts}: {task_name}")
        print(f"{'='*60}\n")

        # 1. Run eval
        result = run_eval_with_logs(task_name, agent_url)
        score = result["score"]

        # 2. Print summary
        print(f"\n  Correctness: {score['correctness']:.0%}")
        print(f"  Score: {score['final_score']}/{score['max_possible']}")
        print(f"  API calls: {result['api_calls']}, errors: {result['api_errors']}")

        checks = result["verification"].get("checks", [])
        failed = [c for c in checks if not c["passed"]]
        if failed:
            print(f"\n  Failed checks ({len(failed)}):")
            for c in failed:
                print(f"    FAIL: {c['field']} — {c['detail']}")

        # 3. If perfect, we're done
        if score["correctness"] == 1.0:
            print(f"\n  PASS — all fields correct!")
            return result

        # 4. If last attempt, just report
        if attempt == max_attempts:
            print(f"\n  Max attempts reached. Final score: {score['final_score']}")
            return result

        # 5. Build error report and get fixes
        print(f"\n  Analyzing failures...")
        report = build_error_report(result)
        sources = _find_relevant_sources(result["tool_calls"], task_name)
        fix_text = get_fix_suggestions(report, sources)

        print(f"\n{'='*60}")
        print("  SUGGESTED FIXES")
        print(f"{'='*60}")
        print(fix_text)

        # 6. Parse and apply fixes
        fixes = parse_fixes(fix_text)
        if not fixes:
            print("\n  No actionable fixes found. Stopping.")
            return result

        if auto_apply:
            apply_results = apply_fixes(fixes)
            for r in apply_results:
                status = "APPLIED" if r["applied"] else f"SKIPPED ({r.get('error', '')})"
                print(f"  {status}: {r['file']}")
        else:
            print(f"\n  {len(fixes)} fix(es) suggested. Use --apply to auto-apply.")
            return result

        print(f"\n  Retrying in 2s...")
        time.sleep(2)

    return result


def _load_last_results() -> dict[str, dict]:
    """Load last eval results from eval_results.json.

    Returns {task_name: {status, classifier_correct, lang, prompt_snippet}}.
    """
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if not os.path.exists(results_path):
        return {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    out = {}
    for r in data:
        name = r.get("task", "")
        out[name] = {
            "status": r.get("status", "?"),
            "classifier_ok": r.get("classifier", {}).get("correct", None),
            "lang": r.get("lang", "?"),
        }
    return out


def _print_task_table(last_results: dict[str, dict]) -> list[str]:
    """Print a numbered task table with status indicators. Returns ordered task names.

    Only shows tasks that have real competition prompts in solve_logs.
    """
    real_types = set(_get_all_real_task_types())
    task_names = [name for name in ALL_TASKS if name in real_types]
    if not task_names:
        print("\n  No tasks with real competition prompts found in solve_logs.")
        return []

    # Get prompt counts per task
    from contextlib import closing
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """SELECT task_type, COUNT(*) as cnt FROM solve_logs
               WHERE source = 'competition' AND task_type IS NOT NULL AND task_type != ''
                 AND prompt IS NOT NULL AND prompt != ''
               GROUP BY task_type""",
        ).fetchall()
    prompt_counts = {r["task_type"]: r["cnt"] for r in rows}

    # Column widths
    print(f"\n  Tasks with real competition prompts ({len(task_names)}/{len(ALL_TASKS)}):")
    hdr = f"  {'#':>3}  {'Status':<6} {'Cls':<4} {'Tier':<5} {'Task':<35} {'Logs':>4}  {'Description'}"
    print(f"\n{hdr}")
    print(f"  {'-'*3}  {'-'*6} {'-'*4} {'-'*5} {'-'*35} {'-'*4}  {'-'*30}")

    for i, name in enumerate(task_names, 1):
        td = ALL_TASKS[name]
        lr = last_results.get(name, {})

        # Status indicator
        st = lr.get("status", "-")
        if st == "PASS":
            status_str = " PASS"
        elif st == "FAIL":
            status_str = " FAIL"
        elif st == "SKIP":
            status_str = " SKIP"
        else:
            status_str = "  -  "

        # Classifier correctness
        cls_ok = lr.get("classifier_ok")
        if cls_ok is True:
            cls_str = " ok"
        elif cls_ok is False:
            cls_str = " X "
        else:
            cls_str = " - "

        tier_str = f"T{td.tier}"
        n_logs = prompt_counts.get(name, 0)
        print(f"  {i:>3}  {status_str:<6} {cls_str:<4} {tier_str:<5} {name:<35} {n_logs:>4}  {td.description}")

    return task_names


def _parse_selection(selection_str: str, max_idx: int) -> list[int]:
    """Parse user selection like '1,3,5-8,12' into a list of 0-based indices."""
    indices = []
    parts = selection_str.replace(" ", "").split(",")
    for part in parts:
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    if 1 <= i <= max_idx:
                        indices.append(i - 1)
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= max_idx:
                    indices.append(i - 1)
            except ValueError:
                pass
    # Deduplicate preserving order
    seen = set()
    result = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def interactive_pick(last_results: dict[str, dict]) -> list[str]:
    """Show task table and let user pick tasks interactively."""
    task_names = _print_task_table(last_results)

    task_set = set(task_names)
    n_pass = sum(1 for n, r in last_results.items() if n in task_set and r.get("status") == "PASS")
    n_fail = sum(1 for n, r in last_results.items() if n in task_set and r.get("status") == "FAIL")
    n_skip = sum(1 for n, r in last_results.items() if n in task_set and r.get("status") == "SKIP")
    n_none = len(task_set) - sum(1 for n in task_set if n in last_results)

    print(f"\n  Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_none} untested")
    print(f"\n  Select tasks to run:")
    print(f"    Numbers:  1,3,5-8     (specific tasks)")
    print(f"    Shortcuts: all | failed | skip | untested | pass")
    print(f"    Tiers:     t1 | t2 | t3")
    print(f"    Combine:   failed,t3  (all failed + all tier 3)")
    print(f"    Enter 'q' to quit\n")

    selection = input("  > ").strip().lower()
    if selection in ("q", "quit", "exit"):
        return []

    selected_indices = set()

    for token in selection.split(","):
        token = token.strip()
        if token == "all":
            selected_indices.update(range(len(task_names)))
        elif token == "failed":
            for i, name in enumerate(task_names):
                if last_results.get(name, {}).get("status") == "FAIL":
                    selected_indices.add(i)
        elif token == "pass":
            for i, name in enumerate(task_names):
                if last_results.get(name, {}).get("status") == "PASS":
                    selected_indices.add(i)
        elif token in ("skip", "skipped"):
            for i, name in enumerate(task_names):
                if last_results.get(name, {}).get("status") == "SKIP":
                    selected_indices.add(i)
        elif token == "untested":
            for i, name in enumerate(task_names):
                if name not in last_results:
                    selected_indices.add(i)
        elif token in ("t1", "tier1"):
            for i, name in enumerate(task_names):
                if ALL_TASKS[name].tier == 1:
                    selected_indices.add(i)
        elif token in ("t2", "tier2"):
            for i, name in enumerate(task_names):
                if ALL_TASKS[name].tier == 2:
                    selected_indices.add(i)
        elif token in ("t3", "tier3"):
            for i, name in enumerate(task_names):
                if ALL_TASKS[name].tier == 3:
                    selected_indices.add(i)
        else:
            # Try as number/range
            parsed = _parse_selection(token, len(task_names))
            selected_indices.update(parsed)

    selected = [task_names[i] for i in sorted(selected_indices)]

    if selected:
        print(f"\n  Selected {len(selected)} task(s):")
        for name in selected:
            td = ALL_TASKS[name]
            st = last_results.get(name, {}).get("status", "-")
            print(f"    {st:<6} T{td.tier}  {name}")
        print()
        confirm = input("  Proceed? [Y/n] ").strip().lower()
        if confirm and confirm not in ("y", "yes", "ja"):
            return []

    return selected


def run_batch_fix(task_names: list[str],
                  agent_url: str = "http://localhost:8000",
                  max_attempts: int = 1, auto_apply: bool = False,
                  report_only: bool = False) -> list[dict]:
    """Run eval+fix for multiple tasks, print summary at end."""
    results = []
    total = len(task_names)

    for idx, task_name in enumerate(task_names, 1):
        print(f"\n{'#'*60}")
        print(f"  [{idx}/{total}] {task_name}")
        print(f"{'#'*60}")

        try:
            if report_only:
                result = run_eval_with_logs(task_name, agent_url)
                report = build_error_report(result)
                print(report)
            else:
                result = run_and_fix(
                    task_name=task_name,
                    agent_url=agent_url,
                    max_attempts=max_attempts,
                    auto_apply=auto_apply,
                )
            results.append({
                "task": task_name,
                "correctness": result["score"]["correctness"],
                "score": result["score"]["final_score"],
                "max": result["score"]["max_possible"],
                "calls": result["api_calls"],
                "errors": result["api_errors"],
                "elapsed": result["elapsed"],
            })
        except Exception as e:
            log.error(f"Error running {task_name}: {e}")
            traceback.print_exc()
            results.append({
                "task": task_name,
                "correctness": 0,
                "score": 0,
                "max": 0,
                "calls": 0,
                "errors": 0,
                "elapsed": 0,
                "error": str(e),
            })

    # Print summary
    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY ({len(results)} tasks)")
    print(f"{'='*70}")
    print(f"  {'Task':<35} {'Correct':>8} {'Score':>10} {'Calls':>6} {'Err':>4} {'Time':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*6} {'-'*4} {'-'*6}")

    n_pass = 0
    n_fail = 0
    total_score = 0
    total_max = 0

    for r in results:
        if r.get("error"):
            print(f"  {r['task']:<35} {'ERROR':>8} {'':>10} {'':>6} {'':>4} {'':>6}  {r['error'][:40]}")
            n_fail += 1
            continue

        corr_str = f"{r['correctness']:.0%}"
        score_str = f"{r['score']}/{r['max']}"
        time_str = f"{r['elapsed']:.0f}s"
        status = "PASS" if r["correctness"] == 1.0 else "FAIL"

        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        total_score += r["score"]
        total_max += r["max"]

        print(f"  {r['task']:<35} {corr_str:>8} {score_str:>10} {r['calls']:>6} {r['errors']:>4} {time_str:>6}")

    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*6} {'-'*4} {'-'*6}")
    print(f"  {'TOTAL':<35} {n_pass}/{n_pass+n_fail}{'':>3} {total_score:.1f}/{total_max:.1f}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-fixer: run eval + fix code from logs")
    parser.add_argument("--task", "-t", help="Task type (e.g. create_employee)")
    parser.add_argument("--tasks", nargs="+", help="Multiple task types for batch run")
    parser.add_argument("--pick", "-p", action="store_true", help="Interactive task picker")
    parser.add_argument("--failed", action="store_true", help="Re-run all previously failed tasks")
    parser.add_argument("--agent-url", "-u", default="http://localhost:8000", help="Agent URL")
    parser.add_argument("--retries", "-r", type=int, default=1, help="Max fix+retry cycles")
    parser.add_argument("--apply", "-a", action="store_true", help="Auto-apply fixes without confirmation")
    parser.add_argument("--report-only", action="store_true", help="Run eval and print report only (no LLM fix)")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list:
        last_results = _load_last_results()
        _print_task_table(last_results)
        return

    # Interactive picker mode
    if args.pick:
        last_results = _load_last_results()
        selected = interactive_pick(last_results)
        if not selected:
            print("No tasks selected.")
            return
        run_batch_fix(
            selected,
            agent_url=args.agent_url,
            max_attempts=args.retries,
            auto_apply=args.apply,
            report_only=args.report_only,
        )
        return

    # Batch: --failed
    if args.failed:
        last_results = _load_last_results()
        real_types = set(_get_all_real_task_types())
        selected = [name for name, r in last_results.items()
                    if r.get("status") == "FAIL" and name in real_types]
        if not selected:
            print("No previously failed tasks with real competition prompts found.")
            return
        print(f"Re-running {len(selected)} failed task(s): {', '.join(selected)}")
        run_batch_fix(
            selected,
            agent_url=args.agent_url,
            max_attempts=args.retries,
            auto_apply=args.apply,
            report_only=args.report_only,
        )
        return

    # Batch: --tasks
    if args.tasks:
        real_types = set(_get_all_real_task_types())
        missing = [t for t in args.tasks if t not in real_types]
        if missing:
            print(f"ERROR: No real competition prompts for: {', '.join(missing)}")
            sys.exit(1)
        run_batch_fix(
            args.tasks,
            agent_url=args.agent_url,
            max_attempts=args.retries,
            auto_apply=args.apply,
            report_only=args.report_only,
        )
        return

    # Single task mode
    if not args.task:
        print("ERROR: --task, --tasks, --pick, or --failed required (use --list to see tasks)")
        sys.exit(1)

    real_types = set(_get_all_real_task_types())
    if args.task not in real_types:
        print(f"ERROR: No real competition prompts for '{args.task}'.")
        if real_types:
            print(f"  Available: {', '.join(sorted(real_types))}")
        sys.exit(1)

    if args.report_only:
        result = run_eval_with_logs(args.task, args.agent_url)
        report = build_error_report(result)
        print(report)
        return

    run_and_fix(
        task_name=args.task,
        agent_url=args.agent_url,
        max_attempts=args.retries,
        auto_apply=args.apply,
    )


if __name__ == "__main__":
    main()
