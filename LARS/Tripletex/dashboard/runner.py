"""Eval runner — generates task, calls agent, verifies, scores."""

import asyncio
import json
import logging
import os
import sys
import time

# Ensure parent dir on path for sim/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import db

log = logging.getLogger(__name__)

# Lazy-init semaphore (must be created on the running event loop)
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(3)
    return _semaphore


async def run_eval(task_name: str, language: str, agent_url: str,
                   base_url: str, session_token: str) -> int:
    """Run a single eval (with concurrency limit). Returns run_id."""
    async with _get_semaphore():
        return await asyncio.to_thread(
            _run_eval_sync, task_name, language, agent_url, base_url, session_token
        )


def _run_eval_sync(task_name, language, agent_url, base_url, session_token) -> int:
    """Synchronous eval — runs in thread pool."""
    import requests
    from tripletex_client import TripletexClient
    from sim.task_definitions import ALL_TASKS
    from sim.generator import generate_task
    from sim.verifier import verify_task
    from sim.scorer import calculate_score
    from dashboard.sandbox import ensure_sandbox_ready

    task_def = ALL_TASKS[task_name]

    # 0. Ensure sandbox has required seed data (runs once per process)
    seed_client = TripletexClient(base_url, session_token)
    ensure_sandbox_ready(seed_client)

    # 1. Generate task prompt + expected values
    generated = generate_task(task_def, language=language)
    log.info(f"[EVAL] Generated {task_name}/{generated['language']}: {generated['prompt'][:80]}...")

    # 2. Create DB record (status=running)
    run_id = db.create_run(
        task_name=task_name,
        tier=task_def.tier,
        language=generated["language"],
        prompt=generated["prompt"],
        expected_json=json.dumps(generated["expected"], ensure_ascii=False),
        agent_url=agent_url,
    )

    try:
        # 3. Pre-create for deletion/reverse tasks
        pre_created_id = 0
        if task_def.pre_create:
            pre_client = TripletexClient(base_url, session_token)
            pre_created_id = _pre_create(pre_client, task_def, generated["expected"])
            log.info(f"[EVAL] Pre-created entity ID: {pre_created_id}")

        # 4. Call agent /solve
        payload = {
            "prompt": generated["prompt"],
            "files": [],
            "tripletex_credentials": {
                "base_url": base_url,
                "session_token": session_token,
            },
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("AGENT_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.time()
        resp = requests.post(
            f"{agent_url}/solve-debug", params={"source": "eval"},
            json=payload, headers=headers, timeout=300,
        )
        elapsed = time.time() - t0

        if resp.status_code != 200:
            db.update_run(
                run_id, status="failed",
                error_message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                elapsed_seconds=elapsed,
            )
            return run_id

        # 5. Wait for API propagation
        time.sleep(1)

        # 6. Verify
        verify_client = TripletexClient(base_url, session_token)
        verification = verify_task(verify_client, task_def, generated["expected"], pre_created_id)

        # 7. Score (estimate API calls — agent doesn't report back)
        api_calls = max(task_def.baseline_calls, task_def.baseline_calls + 2)
        api_errors = 0

        score = calculate_score(
            verification["total_points"],
            verification["max_points"],
            task_def.tier,
            api_calls,
            api_errors,
            task_def.baseline_calls,
        )

        # 8. Update DB
        db.update_run(
            run_id,
            status="completed",
            api_calls=api_calls,
            api_errors=api_errors,
            elapsed_seconds=elapsed,
            correctness=score["correctness"],
            base_score=score["base_score"],
            efficiency_bonus=score["efficiency_bonus"],
            final_score=score["final_score"],
            max_possible=score["max_possible"],
            checks_json=json.dumps(verification["checks"], ensure_ascii=False),
        )
        log.info(f"[EVAL] {task_name}/{generated['language']} -> "
                 f"score={score['final_score']}/{score['max_possible']} "
                 f"correctness={score['correctness']:.0%} in {elapsed:.1f}s")

    except Exception as e:
        log.error(f"[EVAL] Failed {task_name}: {e}", exc_info=True)
        db.update_run(run_id, status="failed", error_message=str(e))

    return run_id


def _pre_create(client, task_def, expected: dict) -> int:
    """Pre-create entity for deletion/reverse tasks."""
    if task_def.name == "delete_travel_expense":
        emp_result = client.get("/employee", params={"fields": "id", "count": 1})
        employees = emp_result.get("values", [])
        if not employees:
            emp_body = {"firstName": "Temp", "lastName": "Ansatt",
                        "email": "temp@example.com", "userType": "STANDARD"}
            emp_result = client.post("/employee", json=emp_body)
            emp_id = emp_result.get("value", {}).get("id", 0)
        else:
            emp_id = employees[0]["id"]
        title = expected.get("title", "Test Reise")
        result = client.post("/travelExpense", json={
            "employee": {"id": emp_id},
            "title": title,
            "travelDetails": {"departureDate": "2026-03-01", "returnDate": "2026-03-03"},
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "delete_customer":
        name = expected.get("name", "Temp Slett AS")
        result = client.post("/customer", json={
            "name": name, "email": "slett@example.com", "isCustomer": True,
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "delete_supplier":
        name = expected.get("name", "Temp Leverandør AS")
        result = client.post("/supplier", json={
            "name": name, "email": "slett@example.com", "isSupplier": True,
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "delete_product":
        name = expected.get("name", "Temp Produkt")
        result = client.post("/product", json={
            "name": name, "priceExcludingVatCurrency": 100,
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "delete_department":
        name = expected.get("name", "Temp Avdeling")
        result = client.post("/department", json={"name": name})
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "delete_contact":
        first = expected.get("firstName", "Temp")
        last = expected.get("lastName", "Kontakt")
        cust = client.post("/customer", json={
            "name": "Temp Kunde AS", "isCustomer": True, "email": "temp@example.com",
        })
        cust_id = cust.get("value", {}).get("id", 0)
        if not cust_id:
            return 0
        result = client.post("/contact", json={
            "firstName": first, "lastName": last,
            "email": "kontakt@example.com", "customer": {"id": cust_id},
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

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
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    if task_def.name == "reverse_voucher":
        description = expected.get("description", "Test Bilag")
        result = client.post("/ledger/voucher", json={
            "date": "2026-03-01",
            "description": description,
            "postings": [
                {"account": {"number": 1920}, "debitAmount": 1000, "creditAmount": 0},
                {"account": {"number": 3000}, "debitAmount": 0, "creditAmount": 1000},
            ],
        })
        return result.get("value", {}).get("id", 0) if "error" not in result else 0

    return 0
