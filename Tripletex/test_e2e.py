"""
End-to-end agent tests: send real prompts through /solve and verify via Tripletex API.
Requires the server running on port 8001.
"""
import json
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8003/solve")
BASE_URL = os.environ["TRIPLETEX_BASE_URL"]
TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]
API_KEY = os.environ.get("AGENT_API_KEY", "")
AUTH = ("0", TOKEN)

ts = int(time.time())
results = []


def solve(prompt: str, timeout: int = 120) -> dict:
    """Send a prompt to the agent and return response."""
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
    return {"status_code": resp.status_code, "elapsed": elapsed, "body": resp.text}


def api_get(endpoint, params=None):
    """Direct GET to Tripletex API."""
    r = requests.get(f"{BASE_URL}{endpoint}", auth=AUTH, params=params or {"fields": "*"})
    return r.json()


def run_e2e(name, prompt, verify_fn, timeout=120):
    """Run an E2E test: send prompt, verify result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"PROMPT: {prompt[:100]}...")
    print(f"{'='*60}")

    t0 = time.time()
    resp = solve(prompt, timeout=timeout)
    elapsed = time.time() - t0
    print(f"  Agent responded in {elapsed:.1f}s (HTTP {resp['status_code']})")

    if resp["status_code"] != 200:
        print(f"  FAIL: HTTP {resp['status_code']}")
        results.append({"name": name, "status": "FAIL", "error": f"HTTP {resp['status_code']}", "time": elapsed})
        return

    # Run verification
    try:
        ok, detail = verify_fn()
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {detail}")
        results.append({"name": name, "status": status, "error": "" if ok else detail, "time": elapsed})
    except Exception as e:
        print(f"  [EXCP] Verification error: {e}")
        results.append({"name": name, "status": "EXCEPTION", "error": str(e), "time": elapsed})


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
            + (f", userType={r['values'][0].get('userType')}" if r.get("values") else "")
        ))(api_get("/employee", {"email": emp_email, "fields": "id,firstName,lastName,email,userType"}))
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
    f"Kunden bestiller 10 timer. Fakturadato er 2026-03-20, forfallsdato 2026-04-20.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} invoices"
        ))(api_get("/invoice", {"fields": "id,invoiceDate,invoiceDueDate", "count": 5,
                                "invoiceDateFrom": "2026-03-20", "invoiceDateTo": "2026-03-21"}))
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
    "Opprett et korrigeringsbilag datert 2026-03-20 med beskrivelse 'Korreksjon bankinnskudd'. "
    "Debet konto 1920 (Bankinnskudd) 5000 kr, kredit konto 7700 5000 kr.",
    lambda: (
        (lambda r: (
            bool(r.get("values")),
            f"Found {len(r.get('values', []))} vouchers"
        ))(api_get("/ledger/voucher", {"fields": "id,date,description", "count": 10,
                                        "dateFrom": "2026-03-20", "dateTo": "2026-03-21"}))
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
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("E2E SUMMARY")
print(f"{'='*60}")

ok = sum(1 for r in results if r["status"] == "OK")
fail = sum(1 for r in results if r["status"] == "FAIL")
excp = sum(1 for r in results if r["status"] == "EXCEPTION")
total = len(results)
total_time = sum(r["time"] for r in results)

print(f"Total: {total} | OK: {ok} | FAIL: {fail} | EXCEPTION: {excp}")
print(f"Total time: {total_time:.0f}s")
print()

for r in results:
    icon = " OK " if r["status"] == "OK" else "FAIL" if r["status"] == "FAIL" else "EXCP"
    print(f"  [{icon}] {r['name']} ({r['time']:.0f}s) {r['error']}")

print(f"\nPass rate: {ok}/{total} ({100*ok/total:.0f}%)" if total else "No tests run")
