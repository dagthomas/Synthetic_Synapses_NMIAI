"""Auto-fixer: run eval tasks, analyze failures, suggest and apply code fixes.

Runs a tool/task through the eval pipeline, captures logs and errors,
then uses Gemini to analyze failures and generate code fixes.

Usage:
    python auto_fixer.py --task create_employee          # run + fix one task
    python auto_fixer.py --task create_invoice --apply    # auto-apply fixes
    python auto_fixer.py --task create_employee --retries 3  # retry after fixing
    python auto_fixer.py --list                           # list available tasks
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
from sim.task_definitions import ALL_TASKS, LANGUAGES
from sim.generator import generate_task
from sim.verifier import verify_task
from sim.scorer import calculate_score
from dashboard.sandbox import ensure_sandbox_ready
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


def run_eval_with_logs(task_name: str, language: str = "",
                       agent_url: str = "http://localhost:8000") -> dict:
    """Run a single eval and capture detailed logs.

    Returns a dict with all diagnostic info:
        task_name, prompt, expected, agent_result, tool_calls, api_log,
        api_calls, api_errors, verification, score, elapsed
    """
    client = _get_sandbox_client()
    ensure_sandbox_ready(client)

    task_def = ALL_TASKS[task_name]

    # Generate task
    generated = generate_task(task_def, language=language)
    prompt = generated["prompt"]
    expected = generated["expected"]
    lang = generated["language"]

    log.info(f"[AUTO-FIX] Task: {task_name} ({lang})")
    log.info(f"[AUTO-FIX] Prompt: {prompt[:120]}...")
    log.info(f"[AUTO-FIX] Expected: {json.dumps(expected, ensure_ascii=False)}")

    # Pre-create for deletion tasks
    pre_created_id = 0
    if task_def.pre_create:
        from simulator import pre_create_for_deletion
        pre_created_id = pre_create_for_deletion(client, task_def, expected)
        log.info(f"[AUTO-FIX] Pre-created entity ID: {pre_created_id}")

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

    # Wait for propagation, then verify
    time.sleep(1)
    verify_client = TripletexClient(client.base_url, client.auth[1])
    verification = verify_task(verify_client, task_def, expected, pre_created_id)

    score = calculate_score(
        verification["total_points"], verification["max_points"],
        task_def.tier, api_calls, api_errors, task_def.baseline_calls,
    )

    return {
        "task_name": task_name,
        "task_def": task_def,
        "language": lang,
        "prompt": prompt,
        "expected": expected,
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
You are an expert Python developer fixing bugs in a Tripletex API integration.

You are given:
1. An error report from running an eval task
2. The source code of the relevant tool files

Analyze the error report to understand what went wrong:
- Field check FAILs: the tool created/updated the entity but a field value is wrong
- API errors (4xx): the tool sent an invalid request to the Tripletex API
- Tool call errors: the agent called a tool with wrong arguments
- Missing tool calls: the agent didn't call a necessary tool

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
"""

    user_msg = f"""\
{error_report}

=== SOURCE CODE ==={source_context}

Analyze the failures and suggest minimal code fixes.
If the eval passed perfectly (100% correctness), say "No fixes needed."
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
        ok = tc.get("result", {}).get("ok", "?")
        status = "OK" if ok else "ERROR"
        args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)[:300]
        result_str = ""
        if tc.get("result"):
            if tc["result"].get("error"):
                result_str = f" → Error: {tc['result']['error'][:200]}"
            elif tc["result"].get("data"):
                result_str = f" → {str(tc['result']['data'])[:200]}"
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


def run_and_fix(task_name: str, language: str = "",
                agent_url: str = "http://localhost:8000",
                max_attempts: int = 1, auto_apply: bool = False) -> dict:
    """Run eval, analyze failures, optionally fix and retry.

    Args:
        task_name: Task to evaluate (e.g. "create_employee").
        language: Language code or "" for random.
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
        result = run_eval_with_logs(task_name, language, agent_url)
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


def main():
    parser = argparse.ArgumentParser(description="Auto-fixer: run eval + fix code from logs")
    parser.add_argument("--task", "-t", help="Task type (e.g. create_employee)")
    parser.add_argument("--lang", "-l", default="", help="Language code")
    parser.add_argument("--agent-url", "-u", default="http://localhost:8000", help="Agent URL")
    parser.add_argument("--retries", "-r", type=int, default=1, help="Max fix+retry cycles")
    parser.add_argument("--apply", "-a", action="store_true", help="Auto-apply fixes without confirmation")
    parser.add_argument("--report-only", action="store_true", help="Run eval and print report only (no LLM fix)")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable tasks:")
        for name, td in ALL_TASKS.items():
            print(f"  {name:<30} Tier {td.tier}  {td.description}")
        return

    if not args.task:
        print("ERROR: --task required (use --list to see available tasks)")
        sys.exit(1)

    if args.task not in ALL_TASKS:
        print(f"ERROR: Unknown task '{args.task}'. Use --list to see available.")
        sys.exit(1)

    if args.report_only:
        result = run_eval_with_logs(args.task, args.lang, args.agent_url)
        report = build_error_report(result)
        print(report)
        return

    run_and_fix(
        task_name=args.task,
        language=args.lang,
        agent_url=args.agent_url,
        max_attempts=args.retries,
        auto_apply=args.apply,
    )


if __name__ == "__main__":
    main()
