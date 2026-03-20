"""
End-to-end agent tests: send real prompts through /solve and verify via Tripletex API.
Outputs a detailed diagnostic report showing every tool call and API call per task.

Usage:
    python test_e2e.py                  # run tests, print report
    python test_e2e.py --report-file report.md  # also save markdown report
"""
import argparse
import json
import os
import sys
import time
import requests
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8003/solve-debug")
BASE_URL = os.environ["TRIPLETEX_BASE_URL"]
TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]
API_KEY = os.environ.get("AGENT_API_KEY", "")
AUTH = ("0", TOKEN)

ts = int(time.time())
today = date.today().isoformat()
tomorrow = (date.today() + timedelta(days=1)).isoformat()
results = []


def solve(prompt: str, timeout: int = 120) -> dict:
    """Send a prompt to the agent and return full response with tool details."""
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": BASE_URL,
            "session_token": TOKEN,
        },
    }
    t0 = time.time()
    resp = requests.post(AGENT_URL, json=payload, headers=headers, timeout=timeout)
    elapsed = time.time() - t0
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return {"status_code": resp.status_code, "elapsed": elapsed, "body": body}


def api_get(endpoint, params=None):
    """Direct GET to Tripletex API."""
    r = requests.get(f"{BASE_URL}{endpoint}", auth=AUTH, params=params or {"fields": "*"})
    return r.json()


def run_e2e(name, prompt, verify_fn, timeout=120):
    """Run an E2E test: send prompt, collect tool details, verify result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    resp = solve(prompt, timeout=timeout)
    elapsed = time.time() - t0
    body = resp["body"]

    tool_calls = body.get("tool_calls", [])
    api_log = body.get("api_log", [])
    api_calls = body.get("api_calls", 0)
    api_errors = body.get("api_errors", 0)
    agent_response = body.get("agent_response", "")

    print(f"  Agent: {elapsed:.1f}s | {len(tool_calls)} tools | {api_calls} API calls | {api_errors} errors")

    # Print tool calls
    for i, tc in enumerate(tool_calls):
        ok = tc.get("result", {}).get("ok", True) if tc.get("result") else "?"
        icon = "OK" if ok is True else "FAIL" if ok is False else "?"
        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in tc.get("args", {}).items())
        print(f"    [{icon}] {tc['tool']}({args_str})")
        if ok is False and tc.get("result", {}).get("error"):
            print(f"           ERROR: {tc['result']['error'][:200]}")

    # Print API errors
    for api in api_log:
        if not api.get("ok"):
            print(f"    [API FAIL] {api['method']} {api['url']} -> {api['status']}: {api.get('error', '')[:150]}")

    if resp["status_code"] != 200:
        print(f"  FAIL: HTTP {resp['status_code']}")
        results.append({
            "name": name, "status": "FAIL",
            "error": f"HTTP {resp['status_code']}",
            "time": elapsed, "prompt": prompt,
            "tool_calls": tool_calls, "api_log": api_log,
            "api_calls": api_calls, "api_errors": api_errors,
            "agent_response": agent_response,
        })
        return

    # Run verification
    try:
        ok, detail = verify_fn()
        status = "OK" if ok else "FAIL"
        print(f"  Verify: [{status}] {detail}")
        results.append({
            "name": name, "status": status,
            "error": "" if ok else detail,
            "time": elapsed, "prompt": prompt,
            "tool_calls": tool_calls, "api_log": api_log,
            "api_calls": api_calls, "api_errors": api_errors,
            "agent_response": agent_response,
        })
    except Exception as e:
        print(f"  Verify: [EXCP] {e}")
        results.append({
            "name": name, "status": "EXCEPTION",
            "error": str(e),
            "time": elapsed, "prompt": prompt,
            "tool_calls": tool_calls, "api_log": api_log,
            "api_calls": api_calls, "api_errors": api_errors,
            "agent_response": agent_response,
        })


# ============================================================
# TEST 1: Create employee
# ============================================================
emp_email = f"kari.e2e.{ts}@example.org"
run_e2e(
    "Create employee",
    f"Opprett en ansatt med navn Kari Nordmann og epostadresse {emp_email}. Hun skal være kontoadministrator.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} employees with email {emp_email}"
        ))(api_get("/employee", {"email": emp_email, "fields": "id,firstName,lastName,email"}))
    ),
)

# ============================================================
# TEST 2: Create customer
# ============================================================
cust_name = f"Fjordkraft E2E {ts}"
run_e2e(
    "Create customer",
    f"Opprett kunden {cust_name} med epost fjordkraft@example.org og organisasjonsnummer 998877665.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} customers matching '{cust_name}'"
        ))(api_get("/customer", {"name": cust_name, "fields": "id,name,email,organizationNumber"}))
    ),
)

# ============================================================
# TEST 3: Create product + invoice
# ============================================================
prod_name = f"Konsulenttime E2E {ts}"
run_e2e(
    "Create invoice",
    f"Opprett en faktura for kunden {cust_name}. Produktet heter '{prod_name}' og koster 1500 kr eks. mva. "
    f"Kunden bestiller 10 timer. Fakturadato er {today}, forfallsdato 2026-04-20.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} invoices dated {today}"
        ))(api_get("/invoice", {"fields": "id,invoiceDate", "count": 5,
                                "invoiceDateFrom": today, "invoiceDateTo": tomorrow}))
    ),
    timeout=180,
)

# ============================================================
# TEST 4: Create travel expense
# ============================================================
run_e2e(
    "Create travel expense",
    f"Opprett en reiseregning for Kari Nordmann ({emp_email}). "
    f"Turen heter 'Kundebesøk Bergen', avreise 2026-04-01, retur 2026-04-03.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} travel expenses"
        ))(api_get("/travelExpense", {"fields": "id,title,employee,travelDetails", "count": 5}))
    ),
    timeout=120,
)

# ============================================================
# TEST 5: Create voucher (ledger correction)
# ============================================================
run_e2e(
    "Create voucher",
    f"Opprett et korrigeringsbilag datert {today} med beskrivelse 'Korreksjon bankinnskudd'. "
    "Debet konto 1920 (Bankinnskudd) 5000 kr, kredit konto 7700 5000 kr.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} vouchers dated {today}"
        ))(api_get("/ledger/voucher", {"fields": "id,date,description", "count": 10,
                                        "dateFrom": today, "dateTo": tomorrow}))
    ),
    timeout=120,
)

# ============================================================
# TEST 6: Create supplier
# ============================================================
supp_name = f"Leverandør E2E {ts}"
run_e2e(
    "Create supplier",
    f"Opprett leverandøren '{supp_name}' med epost lev@example.org og telefon 99887766.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} suppliers matching '{supp_name}'"
        ))(api_get("/supplier", {"name": supp_name, "fields": "id,name,email,phoneNumber"}))
    ),
)

# ============================================================
# SUMMARY (terminal)
# ============================================================
print(f"\n{'='*60}")
print("E2E SUMMARY")
print(f"{'='*60}")

ok_count = sum(1 for r in results if r["status"] == "OK")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
excp_count = sum(1 for r in results if r["status"] == "EXCEPTION")
total = len(results)
total_time = sum(r["time"] for r in results)
total_api = sum(r.get("api_calls", 0) for r in results)
total_errors = sum(r.get("api_errors", 0) for r in results)

print(f"Total: {total} | OK: {ok_count} | FAIL: {fail_count} | EXCEPTION: {excp_count}")
print(f"Total time: {total_time:.0f}s | API calls: {total_api} | API errors: {total_errors}")
print()

for r in results:
    icon = " OK " if r["status"] == "OK" else "FAIL" if r["status"] == "FAIL" else "EXCP"
    tc_count = len(r.get("tool_calls", []))
    err_count = r.get("api_errors", 0)
    print(f"  [{icon}] {r['name']} ({r['time']:.0f}s, {tc_count} tools, {err_count} errs) {r['error']}")

print(f"\nPass rate: {ok_count}/{total} ({100*ok_count/total:.0f}%)" if total else "No tests run")


# ============================================================
# MARKDOWN REPORT (for Claude debugging)
# ============================================================
def generate_report() -> str:
    """Generate a markdown report with full tool/API details for debugging."""
    lines = []
    lines.append(f"# E2E Test Report — {today}")
    lines.append(f"")
    lines.append(f"**Pass rate: {ok_count}/{total} ({100*ok_count/total:.0f}%)**")
    lines.append(f"**Total: {total_time:.0f}s | API calls: {total_api} | API errors: {total_errors}**")
    lines.append("")

    # Quick summary table
    lines.append("| # | Test | Status | Time | Tools | API Calls | API Errors |")
    lines.append("|---|------|--------|------|-------|-----------|------------|")
    for i, r in enumerate(results, 1):
        st = r["status"]
        lines.append(f"| {i} | {r['name']} | {'PASS' if st == 'OK' else st} | {r['time']:.0f}s | {len(r.get('tool_calls', []))} | {r.get('api_calls', 0)} | {r.get('api_errors', 0)} |")
    lines.append("")

    # Detailed section for each test
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['name']} — {'PASS' if r['status'] == 'OK' else r['status']}")
        lines.append("")
        lines.append(f"**Prompt:** {r.get('prompt', 'N/A')}")
        lines.append("")

        tool_calls = r.get("tool_calls", [])
        if tool_calls:
            lines.append("**Tool calls:**")
            for j, tc in enumerate(tool_calls, 1):
                result = tc.get("result", {}) or {}
                ok = result.get("ok", "?")
                icon = "PASS" if ok is True else "FAIL" if ok is False else "?"
                args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                lines.append(f"{j}. `{tc['tool']}` [{icon}]")
                lines.append(f"   - Args: `{args_str}`")
                if ok is False:
                    lines.append(f"   - **Error: {result.get('error', 'unknown')}**")
                if result.get("data"):
                    data_preview = result["data"][:300]
                    lines.append(f"   - Response: `{data_preview}`")
            lines.append("")

        api_log = r.get("api_log", [])
        failed_apis = [a for a in api_log if not a.get("ok")]
        if failed_apis:
            lines.append("**Failed API calls:**")
            for a in failed_apis:
                lines.append(f"- `{a['method']} {a['url']}` -> **{a['status']}**: {a.get('error', '')}")
            lines.append("")

        if r.get("agent_response"):
            lines.append(f"**Agent response:** {r['agent_response'][:500]}")
            lines.append("")

        if r.get("error"):
            lines.append(f"**Verification error:** {r['error']}")
            lines.append("")

    return "\n".join(lines)


# Save report if requested
parser = argparse.ArgumentParser()
parser.add_argument("--report-file", type=str, default="", help="Save markdown report to file")
args, _ = parser.parse_known_args()

report = generate_report()

if args.report_file:
    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.report_file}")
else:
    # Always save to default location
    report_path = os.path.join(os.path.dirname(__file__), "e2e_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nDetailed report saved to {report_path}")
