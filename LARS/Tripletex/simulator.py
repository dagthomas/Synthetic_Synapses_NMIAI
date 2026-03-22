"""Tripletex Submission Simulator.

Simulates the competition submission flow locally:
1. Generate a task prompt + expected answers (via Gemini)
2. POST to agent's /solve endpoint
3. Verify results against sandbox API
4. Score and display field breakdown
5. Clean up created entities

Usage:
    python simulator.py                              # random task, random language
    python simulator.py --task create_employee       # specific task
    python simulator.py --lang no                    # specific language
    python simulator.py --batch 5                    # run 5 random tasks
    python simulator.py --list                       # list available tasks
    python simulator.py --agent-url http://host:8000 # custom agent URL
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import requests
from dotenv import load_dotenv

# Add Tripletex dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tripletex_client import TripletexClient
from sim.task_definitions import ALL_TASKS, LANGUAGES
from sim.generator import generate_task
from sim.verifier import verify_task
from sim.scorer import calculate_score
from dashboard.sandbox import ensure_sandbox_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("simulator")


def get_sandbox_client() -> TripletexClient:
    """Create a client for the sandbox API."""
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not base_url or not token:
        print("ERROR: Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env")
        print("  TRIPLETEX_BASE_URL=https://YOUR-SANDBOX.tripletex.dev/v2")
        print("  TRIPLETEX_SESSION_TOKEN=your-token-here")
        sys.exit(1)
    return TripletexClient(base_url, token)


def pre_create_for_deletion(client: TripletexClient, task_def, expected: dict) -> int:
    """Pre-create an entity that the agent will need to delete/reverse."""
    if task_def.name == "delete_travel_expense":
        emp_result = client.get("/employee", params={"fields": "id", "count": 1})
        employees = emp_result.get("values", [])
        if not employees:
            dept_r = client.get("/department", params={"fields": "id", "count": 1})
            depts = dept_r.get("values", [])
            dept_id = depts[0]["id"] if depts else 0
            emp_body = {"firstName": "Temp", "lastName": "Ansatt",
                        "email": "temp@example.com", "userType": "STANDARD"}
            if dept_id:
                emp_body["department"] = {"id": dept_id}
            emp_result = client.post("/employee", json=emp_body)
            emp_id = emp_result.get("value", {}).get("id", 0)
        else:
            emp_id = employees[0]["id"]

        title = expected.get("title", "Test Reise")
        result = client.post("/travelExpense", json={
            "employee": {"id": emp_id},
            "title": title,
            "travelDetails": {
                "departureDate": "2026-03-01",
                "returnDate": "2026-03-03",
            },
        })
        if "error" in result:
            log.error(f"Failed to pre-create travel expense: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "delete_customer":
        name = expected.get("name", "Temp Slett AS")
        result = client.post("/customer", json={
            "name": name,
            "email": "slett@example.com",
            "isCustomer": True,
        })
        if "error" in result:
            log.error(f"Failed to pre-create customer: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "reverse_voucher":
        description = expected.get("description", "Test Bilag")
        # Look up account IDs first
        acct1920 = client.get("/ledger/account", params={"number": "1920", "fields": "id", "count": 1})
        acct3000 = client.get("/ledger/account", params={"number": "3000", "fields": "id", "count": 1})
        a1 = acct1920.get("values", [{}])[0].get("id", 0)
        a2 = acct3000.get("values", [{}])[0].get("id", 0)
        result = client.post("/ledger/voucher", json={
            "date": "2026-03-01",
            "description": description,
            "postings": [
                {"account": {"id": a1}, "amountGross": 1000, "amountGrossCurrency": 1000, "row": 1},
                {"account": {"id": a2}, "amountGross": -1000, "amountGrossCurrency": -1000, "row": 2},
            ],
        })
        if "error" in result:
            log.error(f"Failed to pre-create voucher: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "delete_supplier":
        name = expected.get("name", "Temp Leverandør AS")
        result = client.post("/supplier", json={
            "name": name,
            "email": "slett@example.com",
            "isSupplier": True,
        })
        if "error" in result:
            log.error(f"Failed to pre-create supplier: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "delete_product":
        name = expected.get("name", "Temp Produkt")
        result = client.post("/product", json={
            "name": name,
            "priceExcludingVatCurrency": 100,
        })
        if "error" in result:
            log.error(f"Failed to pre-create product: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "delete_department":
        name = expected.get("name", "Temp Avdeling")
        result = client.post("/department", json={"name": name})
        if "error" in result:
            log.error(f"Failed to pre-create department: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "delete_contact":
        first = expected.get("firstName", "Temp")
        last = expected.get("lastName", "Kontakt")
        # Need a customer first
        cust = client.post("/customer", json={"name": "Temp Kunde AS", "isCustomer": True, "email": "temp@example.com"})
        cust_id = cust.get("value", {}).get("id", 0)
        if not cust_id:
            log.error(f"Failed to pre-create customer for contact: {cust}")
            return 0
        result = client.post("/contact", json={
            "firstName": first,
            "lastName": last,
            "email": "kontakt@example.com",
            "customer": {"id": cust_id},
        })
        if "error" in result:
            log.error(f"Failed to pre-create contact: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    if task_def.name == "reverse_payment":
        import uuid as _uuid
        # Create customer + product + order + invoice + payment, return invoice_id
        cust_name = expected.get("customer_name", "Reversering Kunde AS")
        cust = client.post("/customer", json={
            "name": cust_name,
            "email": f"rev-{_uuid.uuid4().hex[:8]}@example.com",
            "isCustomer": True,
        })
        cust_id = cust.get("value", {}).get("id", 0)
        if not cust_id:
            log.error(f"Failed to pre-create customer for reverse_payment: {cust}")
            return 0
        prod = client.post("/product", json={
            "name": f"Reverseringsprodukt-{_uuid.uuid4().hex[:8]}",
            "priceExcludingVatCurrency": 1000,
        })
        prod_id = prod.get("value", {}).get("id", 0)
        if not prod_id:
            log.error(f"Failed to pre-create product for reverse_payment: {prod}")
            return 0
        order = client.post("/order", json={
            "customer": {"id": cust_id},
            "orderDate": "2026-03-01",
            "deliveryDate": "2026-03-01",
            "orderLines": [{"product": {"id": prod_id}, "count": 1}],
        })
        order_id = order.get("value", {}).get("id", 0)
        if not order_id:
            log.error(f"Failed to pre-create order for reverse_payment: {order}")
            return 0
        inv = client.post("/invoice", json={
            "invoiceDate": "2026-03-01",
            "invoiceDueDate": "2026-03-15",
            "orders": [{"id": order_id}],
        })
        inv_id = inv.get("value", {}).get("id", 0)
        if not inv_id:
            log.error(f"Failed to pre-create invoice for reverse_payment: {inv}")
            return 0
        # Register payment to make it fully paid (use PUT /:payment endpoint)
        inv_detail = client.get(f"/invoice/{inv_id}", params={"fields": "id,amount"})
        amount = float(inv_detail.get("value", {}).get("amount", 1000))
        # Resolve payment type
        pt_result = client.get("/invoice/paymentType", params={"fields": "id", "count": 1})
        pt_id = pt_result.get("values", [{}])[0].get("id", 0)
        pay = client.put(f"/invoice/{inv_id}/:payment", params={
            "paymentDate": "2026-03-02",
            "paymentTypeId": pt_id,
            "paidAmount": amount,
            "paidAmountCurrency": amount,
        })
        if "error" in pay:
            log.error(f"Failed to register payment for reverse_payment: {pay}")
            return 0
        return inv_id

    if task_def.name == "delete_employee":
        import uuid as _uuid
        first = expected.get("firstName", "Temp")
        last = expected.get("lastName", "Ansatt")
        dept_r = client.get("/department", params={"fields": "id", "count": 1})
        depts = dept_r.get("values", [])
        dept_id = depts[0]["id"] if depts else 0
        unique_email = f"del-{_uuid.uuid4().hex[:8]}@example.com"
        body = {"firstName": first, "lastName": last, "email": unique_email, "userType": "NO_ACCESS"}
        if dept_id:
            body["department"] = {"id": dept_id}
        result = client.post("/employee", json=body)
        if "error" in result:
            log.error(f"Failed to pre-create employee: {result}")
            return 0
        return result.get("value", {}).get("id", 0)

    return 0


def cleanup_entities(client: TripletexClient, verification_result: dict):
    """Best-effort cleanup of entities created during the test."""
    # Try entity_id from simple verification
    entity_id = verification_result.get("entity_id")
    entity_type = None

    if entity_id:
        # Infer type from context — not perfect but works for simple tasks
        log.info(f"Cleanup: would delete entity {entity_id} (skipping for safety)")
        return

    # Try entity_ids from complex verification
    entity_ids = verification_result.get("entity_ids", [])
    for etype, eid in entity_ids:
        log.info(f"Cleanup: attempting DELETE /{etype}/{eid}")
        result = client.delete(f"/{etype}/{eid}")
        if "error" in result:
            log.warning(f"Cleanup failed for {etype}/{eid}: {result.get('message', 'unknown')}")


def call_agent(agent_url: str, prompt: str, base_url: str, token: str) -> dict:
    """Send a task to the agent's /solve endpoint."""
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": base_url,
            "session_token": token,
        },
    }

    api_key = os.environ.get("AGENT_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    log.info(f"Calling agent at {agent_url}/solve-debug?source=eval")
    start = time.time()
    try:
        resp = requests.post(
            f"{agent_url}/solve-debug",
            params={"source": "eval"},
            json=payload,
            headers=headers,
            timeout=300,
        )
        elapsed = time.time() - start
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500] if resp.text else "(empty)"}
        return {
            "status_code": resp.status_code,
            "body": body,
            "elapsed": elapsed,
        }
    except requests.exceptions.ConnectionError:
        return {"status_code": 0, "body": {"error": "Connection refused"}, "elapsed": 0}
    except requests.exceptions.Timeout:
        return {"status_code": 0, "body": {"error": "Timeout (5min)"}, "elapsed": 300}


def print_results(task_name: str, tier: int, language: str, prompt: str,
                  verification: dict, score: dict, agent_result: dict,
                  api_calls: int, api_errors: int):
    """Pretty-print the simulation results."""
    lang_name = LANGUAGES.get(language, language)

    print("\n" + "=" * 70)
    print(f"  Task: {task_name} (Tier {tier}, {lang_name})")
    print("=" * 70)
    print(f"\n  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"\n  Agent: {agent_result.get('elapsed', 0):.1f}s, "
          f"HTTP {agent_result.get('status_code', '?')}")
    print(f"  API calls: {api_calls}, errors: {api_errors}")
    print()

    # Field breakdown
    checks = verification.get("checks", [])
    for c in checks:
        symbol = "OK" if c["passed"] else "FAIL"
        print(f"    [{symbol:4}] {c['field']:<25} {c['points']}/{c['max']}  ({c['detail']})")

    print()
    print(f"  Correctness: {verification['total_points']}/{verification['max_points']} "
          f"= {score['correctness']:.0%}")
    print(f"  Base score:  {score['base_score']} (x{score['tier_multiplier']} tier)")
    if score["efficiency_bonus"] > 0:
        print(f"  Efficiency:  +{score['efficiency_bonus']}")
    print(f"  Final score: {score['final_score']} / {score['max_possible']}")
    print("=" * 70)


def run_single(args, client: TripletexClient) -> dict:
    """Run a single simulated submission."""
    # Pick task
    if args.task:
        if args.task not in ALL_TASKS:
            print(f"Unknown task: {args.task}")
            print(f"Available: {', '.join(ALL_TASKS.keys())}")
            sys.exit(1)
        task_def = ALL_TASKS[args.task]
    else:
        task_def = random.choice(list(ALL_TASKS.values()))

    if task_def.sandbox_broken:
        log.warning(f"Skipping {task_def.name} — marked as sandbox_broken")
        return {"score": -1, "skipped": True, "task": task_def.name}

    # Generate task prompt + expected values
    log.info(f"Generating task: {task_def.name} (tier {task_def.tier})")
    generated = generate_task(task_def, language=args.lang)
    prompt = generated["prompt"]
    expected = generated["expected"]
    language = generated["language"]

    log.info(f"Generated prompt ({language}): {prompt[:100]}...")
    log.info(f"Expected values: {json.dumps(expected, ensure_ascii=False)}")

    # Pre-create for deletion tasks
    pre_created_id = 0
    if task_def.pre_create:
        pre_created_id = pre_create_for_deletion(client, task_def, expected)
        log.info(f"Pre-created entity ID: {pre_created_id}")

    # Reset client call counters
    client._call_count = 0
    client._error_count = 0

    # Call the agent
    agent_result = call_agent(
        args.agent_url,
        prompt,
        client.base_url,
        client.auth[1],  # session token
    )

    if agent_result["status_code"] != 200:
        log.error(f"Agent returned HTTP {agent_result['status_code']}: {agent_result['body']}")

    # Wait a moment for API propagation
    time.sleep(1)

    # Verify results
    log.info("Verifying results...")
    # Create a fresh client for verification (don't count these calls)
    verify_client = TripletexClient(client.base_url, client.auth[1])
    verification = verify_task(verify_client, task_def, expected, pre_created_id)

    # Get agent's API call stats (from the agent's perspective)
    # We use the original client's counters since pre_create also uses it
    # For accuracy, we'd need the agent to report its own stats
    # For now, estimate from the verify client
    agent_api_calls = max(task_def.baseline_calls, task_def.baseline_calls + 2)  # rough estimate
    agent_api_errors = 0

    # Calculate score
    score = calculate_score(
        verification["total_points"],
        verification["max_points"],
        task_def.tier,
        agent_api_calls,
        agent_api_errors,
        task_def.baseline_calls,
    )

    # Print results
    print_results(
        task_def.name, task_def.tier, language, prompt,
        verification, score, agent_result,
        agent_api_calls, agent_api_errors,
    )

    # Cleanup
    if not args.no_cleanup:
        cleanup_entities(verify_client, verification)

    return score


def main():
    parser = argparse.ArgumentParser(description="Tripletex Submission Simulator")
    parser.add_argument("--task", "-t", help="Task type (e.g. create_employee)")
    parser.add_argument("--lang", "-l", default="", help="Language code (no, en, es, pt, nn, de, fr)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Number of tasks to run")
    parser.add_argument("--agent-url", "-u", default="http://localhost:8000",
                        help="Agent endpoint URL (default: http://localhost:8000)")
    parser.add_argument("--list", action="store_true", help="List available task types")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup after verification")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate task only, don't call agent or verify")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable task types:")
        print("-" * 50)
        for name, td in ALL_TASKS.items():
            print(f"  {name:<25} Tier {td.tier}  {td.description}")
        print(f"\nLanguages: {', '.join(f'{k} ({v})' for k, v in LANGUAGES.items())}")
        return

    if args.dry_run:
        task_def = ALL_TASKS.get(args.task, random.choice(list(ALL_TASKS.values())))
        generated = generate_task(task_def, language=args.lang)
        print(f"\nTask: {task_def.name} (Tier {task_def.tier})")
        print(f"Language: {generated['language']}")
        print(f"\nPrompt:\n{generated['prompt']}")
        print(f"\nExpected:\n{json.dumps(generated['expected'], indent=2, ensure_ascii=False)}")
        return

    client = get_sandbox_client()

    # Ensure sandbox has required data (modules, departments, etc.)
    seed_result = ensure_sandbox_ready(client)
    if seed_result.get("status") == "connection_failed":
        print("ERROR: Cannot connect to sandbox!")
        sys.exit(1)

    # Run tasks
    scores = []
    for i in range(args.batch):
        if args.batch > 1:
            print(f"\n--- Task {i+1}/{args.batch} ---")
        score = run_single(args, client)
        scores.append(score)

    # Summary for batch runs
    if args.batch > 1:
        actual = [s for s in scores if not s.get("skipped")]
        skipped = [s for s in scores if s.get("skipped")]
        print("\n" + "=" * 70)
        print("  BATCH SUMMARY")
        print("=" * 70)
        if actual:
            total = sum(s["final_score"] for s in actual)
            avg = total / len(actual)
            perfect = sum(1 for s in actual if s["correctness"] == 1.0)
            print(f"  Tasks run:     {len(actual)}")
            print(f"  Perfect:       {perfect}/{len(actual)}")
            print(f"  Total score:   {total:.2f}")
            print(f"  Average score: {avg:.2f}")
        if skipped:
            print(f"  Skipped:       {len(skipped)} (sandbox broken)")
        print("=" * 70)


if __name__ == "__main__":
    main()
