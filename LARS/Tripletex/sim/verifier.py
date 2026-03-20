"""Field-by-field verification against the Tripletex sandbox API.

After the agent completes a task, we query the API to check each expected field.
"""

import logging

from tripletex_client import TripletexClient
from sim.task_definitions import TaskDef, FieldCheck

log = logging.getLogger(__name__)


def _normalize(value) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower()


def _normalize_phone(phone: str) -> str:
    """Normalize phone numbers for comparison (strip +47, spaces, dashes)."""
    return phone.replace("+47", "").replace(" ", "").replace("-", "").strip()


def _fields_match(actual, expected) -> bool:
    """Compare two field values with fuzzy matching."""
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        return _normalize(actual) == _normalize(expected)
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 0.01
        except (ValueError, TypeError):
            return False
    # Phone number fuzzy match (strip +47 prefix)
    a_str = _normalize(actual)
    e_str = _normalize(expected)
    if a_str == e_str:
        return True
    # Try phone normalization if it looks like a phone number
    if e_str.startswith("+47") or (e_str.isdigit() and len(e_str) == 8):
        return _normalize_phone(str(actual)) == _normalize_phone(str(expected))
    return False


def _search_entity(client: TripletexClient, entity_type: str, search_params: dict) -> list:
    """Search for entities matching the given params."""
    params = {"fields": "*", "count": 50}
    params.update(search_params)
    result = client.get(f"/{entity_type}", params=params)
    if "error" in result:
        log.warning(f"Search error for {entity_type}: {result}")
        return []
    return result.get("values", [])


def _make_check(field: str, points: int, max_pts: int, passed: bool, detail: str) -> dict:
    return {"field": field, "points": points, "max": max_pts, "passed": passed, "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# Simple entity creation (employee, customer, product, department, supplier)
# ═══════════════════════════════════════════════════════════════════════

def _best_match(entities: list, expected: dict, search_fields: list) -> dict | None:
    """Pick the entity that best matches expected values.

    When multiple entities match on search_fields, disambiguate using all
    expected fields (email, phone, etc.) to find the one the agent created.
    """
    if not entities:
        return None
    if len(entities) == 1:
        return entities[0]
    # Score each entity by how many expected fields match
    best, best_score = entities[0], 0
    for e in entities:
        score = 0
        for key, val in expected.items():
            if key.startswith("_"):
                continue
            # For update tasks: new_email -> email, new_phoneNumber -> phoneNumber
            api_key = key[4:] if key.startswith("new_") else key
            actual = e.get(api_key)
            if actual is not None and _fields_match(actual, val):
                score += 1
        if score > best_score:
            best, best_score = e, score
    return best


def verify_simple_entity(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a simple entity creation task."""
    search_params = {}
    for sf in task_def.search_fields:
        if sf in expected:
            search_params[sf] = expected[sf]

    entities = _search_entity(client, task_def.entity_type, search_params)

    checks = []
    entity = _best_match(entities, expected, task_def.search_fields)
    entity_id = entity["id"] if entity else None

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = entity is not None
            checks.append(_make_check(
                "Entity found", fc.points if passed else 0, fc.points, passed,
                f"Found {len(entities)} match(es)" if passed else "Not found",
            ))
            continue

        if fc.field not in expected:
            continue

        if entity is None:
            checks.append(_make_check(fc.field, 0, fc.points, False, "Entity not found"))
            continue

        actual_value = entity.get(fc.field)
        expected_value = expected[fc.field]
        passed = _fields_match(actual_value, expected_value)

        checks.append(_make_check(
            fc.field, fc.points if passed else 0, fc.points, passed,
            f"expected={expected_value}, actual={actual_value}",
        ))

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_id": entity_id, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Contact creation (linked to customer)
# ═══════════════════════════════════════════════════════════════════════

def verify_contact(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a contact creation task."""
    checks = []
    entity_ids = []

    # Check customer was created
    customer_name = expected.get("customer_name", "")
    customers = _search_entity(client, "customer", {"name": customer_name})
    customer = customers[0] if customers else None

    # Search for contact
    search_params = {}
    for sf in task_def.search_fields:
        if sf in expected:
            search_params[sf] = expected[sf]
    contacts = _search_entity(client, "contact", search_params)
    contact = contacts[0] if contacts else None

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = contact is not None
            checks.append(_make_check(
                "Contact found", fc.points if passed else 0, fc.points, passed,
                f"Found {len(contacts)} match(es)" if passed else "Not found",
            ))
            if contact:
                entity_ids.append(("contact", contact["id"]))
            continue

        if fc.field not in expected:
            continue

        if contact is None:
            checks.append(_make_check(fc.field, 0, fc.points, False, "Contact not found"))
            continue

        actual_value = contact.get(fc.field)
        expected_value = expected[fc.field]
        passed = _fields_match(actual_value, expected_value)
        checks.append(_make_check(
            fc.field, fc.points if passed else 0, fc.points, passed,
            f"expected={expected_value}, actual={actual_value}",
        ))

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Update tasks (employee, customer)
# ═══════════════════════════════════════════════════════════════════════

def verify_update(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify an update task (update_employee, update_customer)."""
    checks = []
    entity_ids = []

    search_params = {}
    for sf in task_def.search_fields:
        if sf in expected:
            search_params[sf] = expected[sf]

    entities = _search_entity(client, task_def.entity_type, search_params)
    entity = _best_match(entities, expected, task_def.search_fields)

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = entity is not None
            checks.append(_make_check(
                "Entity found", fc.points if passed else 0, fc.points, passed,
                f"Found {len(entities)} match(es)" if passed else "Not found",
            ))
            if entity:
                entity_ids.append((task_def.entity_type, entity["id"]))
            continue

        if fc.field not in expected:
            continue

        if entity is None:
            checks.append(_make_check(fc.field, 0, fc.points, False, "Entity not found"))
            continue

        # Map update fields: new_email -> email, new_phoneNumberMobile -> phoneNumberMobile
        api_field = fc.field
        if api_field.startswith("new_"):
            api_field = api_field[4:]

        actual_value = entity.get(api_field)
        expected_value = expected[fc.field]
        passed = _fields_match(actual_value, expected_value)
        checks.append(_make_check(
            fc.field, fc.points if passed else 0, fc.points, passed,
            f"expected={expected_value}, actual={actual_value}",
        ))

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Invoice tasks (create, multi-line, payment, credit note)
# ═══════════════════════════════════════════════════════════════════════

def verify_invoice(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify an invoice creation task (possibly with payment or credit note)."""
    checks = []
    entity_ids = []

    # 1. Check customer exists (use _best_match to avoid wrong match from sandbox data)
    customer_name = expected.get("customer_name", "")
    customers = _search_entity(client, "customer", {"name": customer_name})
    customer = _best_match(customers, {"name": customer_name}, ["name"])

    # 2. Find invoices — Tripletex requires date range params (and From != To!)
    inv_date = expected.get("invoice_date", "")
    if inv_date:
        # Use +-30 day window (Tripletex rejects same-day From==To)
        from datetime import date, timedelta
        d = date.fromisoformat(inv_date)
        inv_search_params = {
            "invoiceDateFrom": (d - timedelta(days=1)).isoformat(),
            "invoiceDateTo": (d + timedelta(days=30)).isoformat(),
        }
    else:
        inv_search_params = {
            "invoiceDateFrom": "2026-01-01",
            "invoiceDateTo": "2026-12-31",
        }
    invoices = _search_entity(client, "invoice", inv_search_params)
    matching_invoices = []
    if customer:
        for inv in invoices:
            inv_customer = inv.get("customer", {})
            if isinstance(inv_customer, dict) and inv_customer.get("id") == customer["id"]:
                matching_invoices.append(inv)
    invoice = matching_invoices[0] if matching_invoices else None

    # 3. For credit note tasks, find credit notes
    credit_note = None
    if task_def.name in ("create_credit_note", "delete_invoice"):
        # Credit notes are invoices with isCreditNote=True
        for inv in invoices:
            inv_customer = inv.get("customer", {})
            is_credit = inv.get("isCreditNote", False)
            if isinstance(inv_customer, dict) and inv_customer.get("id") == (customer["id"] if customer else -1):
                if is_credit:
                    credit_note = inv

    for fc in task_def.field_checks:
        if fc.field == "_customer_found":
            passed = customer is not None
            checks.append(_make_check(
                "Customer found", fc.points if passed else 0, fc.points, passed,
                f"'{customer_name}'" + (" found" if passed else " not found"),
            ))
            if customer:
                entity_ids.append(("customer", customer["id"]))
            continue

        if fc.field == "customer_name":
            passed = customer is not None and _fields_match(customer.get("name"), customer_name)
            checks.append(_make_check(
                "customer_name", fc.points if passed else 0, fc.points, passed,
                f"expected={customer_name}, actual={customer.get('name') if customer else 'N/A'}",
            ))
            continue

        if fc.field == "_invoice_found":
            passed = invoice is not None
            checks.append(_make_check(
                "Invoice found", fc.points if passed else 0, fc.points, passed,
                f"Found {len(matching_invoices)} invoice(s) for customer",
            ))
            if invoice:
                entity_ids.append(("invoice", invoice["id"]))
            continue

        if fc.field == "_credit_note_found":
            passed = credit_note is not None
            checks.append(_make_check(
                "Credit note found", fc.points if passed else 0, fc.points, passed,
                "Credit note exists" if passed else "No credit note found",
            ))
            continue

        if fc.field in ("invoice_date", "due_date") and fc.field in expected:
            api_field = "invoiceDate" if fc.field == "invoice_date" else "invoiceDueDate"
            if invoice:
                actual = invoice.get(api_field, "")
                passed = _fields_match(actual, expected[fc.field])
            else:
                actual = "N/A"
                passed = False
            checks.append(_make_check(
                fc.field, fc.points if passed else 0, fc.points, passed,
                f"expected={expected[fc.field]}, actual={actual}",
            ))
            continue

        if fc.field == "product_name" and fc.field in expected:
            passed = False
            actual = "N/A"
            if invoice:
                orders = invoice.get("orders", [])
                for order_ref in orders:
                    order_id = order_ref.get("id") if isinstance(order_ref, dict) else order_ref
                    order_data = client.get(f"/order/{order_id}", params={"fields": "id,orderLines(id,count,product(id,name))"})
                    if "error" not in order_data:
                        order_val = order_data.get("value", order_data)
                        for line in order_val.get("orderLines", []):
                            product = line.get("product", {})
                            if isinstance(product, dict):
                                pname = product.get("name", "")
                                if _fields_match(pname, expected["product_name"]):
                                    passed = True
                                    actual = pname
                                    break
            checks.append(_make_check(
                "product_name", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['product_name']}, actual={actual}",
            ))
            continue

        if fc.field == "product_count" and fc.field in expected:
            actual_count = 0
            if invoice:
                orders = invoice.get("orders", [])
                for order_ref in orders:
                    order_id = order_ref.get("id") if isinstance(order_ref, dict) else order_ref
                    order_data = client.get(f"/order/{order_id}", params={"fields": "id,orderLines(id,count,product(id,name))"})
                    if "error" not in order_data:
                        order_val = order_data.get("value", order_data)
                        actual_count = len(order_val.get("orderLines", []))
            passed = _fields_match(actual_count, expected["product_count"])
            checks.append(_make_check(
                "product_count", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['product_count']}, actual={actual_count}",
            ))
            continue

        if fc.field == "quantity" and fc.field in expected:
            passed = False
            actual = "N/A"
            if invoice:
                orders = invoice.get("orders", [])
                for order_ref in orders:
                    order_id = order_ref.get("id") if isinstance(order_ref, dict) else order_ref
                    order_data = client.get(f"/order/{order_id}", params={"fields": "id,orderLines(id,count,product(id,name))"})
                    if "error" not in order_data:
                        order_val = order_data.get("value", order_data)
                        for line in order_val.get("orderLines", []):
                            count = line.get("count", 0)
                            if _fields_match(count, expected["quantity"]):
                                passed = True
                                actual = count
                                break
            checks.append(_make_check(
                "quantity", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['quantity']}, actual={actual}",
            ))
            continue

        if fc.field == "_payment_found":
            passed = False
            detail = "N/A"
            if invoice:
                inv_detail = client.get(f"/invoice/{invoice['id']}", params={"fields": "*"})
                inv_val = inv_detail.get("value", inv_detail)
                amount_outstanding = inv_val.get("amountOutstanding", None)
                if amount_outstanding is not None and abs(float(amount_outstanding)) < 0.01:
                    passed = True
                    detail = "fully paid"
            checks.append(_make_check(
                "Payment registered", fc.points if passed else 0, fc.points, passed, detail,
            ))
            continue

        if fc.field == "payment_amount" and fc.field in expected:
            passed = False
            if invoice:
                inv_detail = client.get(f"/invoice/{invoice['id']}", params={"fields": "*"})
                inv_val = inv_detail.get("value", inv_detail)
                amount = inv_val.get("amount", 0)
                passed = _fields_match(amount, expected["payment_amount"])
            checks.append(_make_check(
                "payment_amount", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['payment_amount']}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Project creation
# ═══════════════════════════════════════════════════════════════════════

def verify_project(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a project creation task."""
    checks = []
    entity_ids = []

    project_name = expected.get("project_name", "")
    projects = _search_entity(client, "project", {"name": project_name})
    project = projects[0] if projects else None

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = project is not None
            checks.append(_make_check(
                "Project found", fc.points if passed else 0, fc.points, passed,
                f"'{project_name}'" + (" found" if passed else " not found"),
            ))
            if project:
                entity_ids.append(("project", project["id"]))
            continue

        if fc.field == "project_name":
            passed = project is not None and _fields_match(project.get("name"), project_name)
            checks.append(_make_check(
                "project_name", fc.points if passed else 0, fc.points, passed,
                f"expected={project_name}",
            ))
            continue

        if fc.field == "customer_name" and "customer_name" in expected:
            passed = False
            if project:
                cust = project.get("customer", {})
                if isinstance(cust, dict):
                    cust_name = cust.get("name", "")
                    # fields=* doesn't expand nested names — fetch customer if needed
                    if not cust_name and cust.get("id"):
                        cust_detail = client.get(f"/customer/{cust['id']}", params={"fields": "id,name"})
                        cust_name = cust_detail.get("value", {}).get("name", "")
                    passed = _fields_match(cust_name, expected["customer_name"])
            checks.append(_make_check(
                "customer_name", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['customer_name']}",
            ))
            continue

        if fc.field == "start_date" and "start_date" in expected:
            passed = False
            if project:
                passed = _fields_match(project.get("startDate"), expected["start_date"])
            checks.append(_make_check(
                "start_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['start_date']}",
            ))
            continue

        if fc.field == "description" and "description" in expected:
            passed = False
            if project:
                passed = _fields_match(project.get("description"), expected["description"])
            checks.append(_make_check(
                "description", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['description']}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Travel expense creation
# ═══════════════════════════════════════════════════════════════════════

def verify_travel_expense(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a travel expense creation task."""
    checks = []
    entity_ids = []

    expenses = _search_entity(client, "travelExpense", {})
    title = expected.get("title", "")
    matching = [e for e in expenses if _normalize(e.get("title", "")) == _normalize(title)]
    expense = matching[0] if matching else None

    emp_first = expected.get("employee_firstName", "")
    emp_last = expected.get("employee_lastName", "")
    employees = _search_entity(client, "employee", {"firstName": emp_first, "lastName": emp_last})
    employee = employees[0] if employees else None

    for fc in task_def.field_checks:
        if fc.field == "_employee_found":
            passed = employee is not None
            checks.append(_make_check(
                "Employee found", fc.points if passed else 0, fc.points, passed,
                f"'{emp_first} {emp_last}'" + (" found" if passed else " not found"),
            ))
            if employee:
                entity_ids.append(("employee", employee["id"]))
            continue

        if fc.field == "_found":
            passed = expense is not None
            checks.append(_make_check(
                "Travel expense found", fc.points if passed else 0, fc.points, passed,
                f"'{title}'" + (" found" if passed else " not found"),
            ))
            if expense:
                entity_ids.append(("travelExpense", expense["id"]))
            continue

        if fc.field == "title":
            passed = expense is not None and _fields_match(expense.get("title"), title)
            checks.append(_make_check(
                "title", fc.points if passed else 0, fc.points, passed,
                f"expected={title}",
            ))
            continue

        if fc.field == "departure_date" and "departure_date" in expected:
            passed = False
            if expense:
                # departureDate may be at top level or in travelDetails
                actual = expense.get("departureDate") or expense.get("travelDetails", {}).get("departureDate")
                passed = _fields_match(actual, expected["departure_date"])
            checks.append(_make_check(
                "departure_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['departure_date']}",
            ))
            continue

        if fc.field == "return_date" and "return_date" in expected:
            passed = False
            if expense:
                actual = expense.get("returnDate") or expense.get("travelDetails", {}).get("returnDate")
                passed = _fields_match(actual, expected["return_date"])
            checks.append(_make_check(
                "return_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['return_date']}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Employee with employment
# ═══════════════════════════════════════════════════════════════════════

def verify_employee_with_employment(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify employee creation with employment details."""
    checks = []
    entity_ids = []

    search_params = {}
    for sf in task_def.search_fields:
        if sf in expected:
            search_params[sf] = expected[sf]

    employees = _search_entity(client, "employee", search_params)
    employee = employees[0] if employees else None

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = employee is not None
            checks.append(_make_check(
                "Employee found", fc.points if passed else 0, fc.points, passed,
                f"Found {len(employees)} match(es)" if passed else "Not found",
            ))
            if employee:
                entity_ids.append(("employee", employee["id"]))
            continue

        if fc.field in ("firstName", "lastName", "email") and fc.field in expected:
            if employee is None:
                checks.append(_make_check(fc.field, 0, fc.points, False, "Employee not found"))
                continue
            actual = employee.get(fc.field)
            passed = _fields_match(actual, expected[fc.field])
            checks.append(_make_check(
                fc.field, fc.points if passed else 0, fc.points, passed,
                f"expected={expected[fc.field]}, actual={actual}",
            ))
            continue

        if fc.field == "start_date" and "start_date" in expected:
            passed = False
            if employee:
                # Check employments for this employee
                emp_id = employee["id"]
                emps = _search_entity(client, "employee/employment", {"employeeId": emp_id})
                for emp_detail in emps:
                    start = emp_detail.get("startDate", "")
                    if _fields_match(start, expected["start_date"]):
                        passed = True
                        break
            checks.append(_make_check(
                "start_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['start_date']}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Deletion tasks
# ═══════════════════════════════════════════════════════════════════════

def verify_deletion(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
    pre_created_id: int,
) -> dict:
    """Verify a deletion task (delete_travel_expense, delete_customer)."""
    checks = []

    result = client.get(f"/{task_def.entity_type}/{pre_created_id}", params={"fields": "id"})
    still_exists = "error" not in result

    for fc in task_def.field_checks:
        if fc.field == "_deleted":
            passed = not still_exists
            checks.append(_make_check(
                "Entity deleted", fc.points if passed else 0, fc.points, passed,
                "Successfully deleted" if passed else "Still exists",
            ))
            continue

        if fc.field in ("title", "name"):
            checks.append(_make_check(
                "Correct target", fc.points if not still_exists else 0, fc.points,
                not still_exists,
                f"{fc.field}={expected.get(fc.field, 'N/A')}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Ledger voucher tasks
# ═══════════════════════════════════════════════════════════════════════

def verify_ledger_voucher(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify ledger voucher creation (voucher, opening balance)."""
    checks = []
    entity_ids = []

    description = expected.get("description", "")
    date = expected.get("date", "")

    # Search vouchers by date range (Tripletex rejects dateFrom == dateTo)
    search_params = {}
    if date:
        from datetime import date as dt_date, timedelta
        d = dt_date.fromisoformat(date)
        search_params["dateFrom"] = (d - timedelta(days=1)).isoformat()
        search_params["dateTo"] = (d + timedelta(days=30)).isoformat()

    vouchers = _search_entity(client, "ledger/voucher", search_params)
    # Find matching by description
    matching = [v for v in vouchers if _normalize(v.get("description", "")) == _normalize(description)]
    voucher = matching[0] if matching else None

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = voucher is not None
            checks.append(_make_check(
                "Voucher found", fc.points if passed else 0, fc.points, passed,
                f"'{description}'" + (" found" if passed else " not found"),
            ))
            if voucher:
                entity_ids.append(("ledger/voucher", voucher["id"]))
            continue

        if fc.field == "description":
            passed = voucher is not None and _fields_match(voucher.get("description"), description)
            checks.append(_make_check(
                "description", fc.points if passed else 0, fc.points, passed,
                f"expected={description}",
            ))
            continue

        if fc.field == "date" and "date" in expected:
            passed = False
            if voucher:
                passed = _fields_match(voucher.get("date"), expected["date"])
            checks.append(_make_check(
                "date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['date']}",
            ))
            continue

        if fc.field == "amount" and "amount" in expected:
            passed = False
            if voucher:
                # Check postings for the expected amount
                voucher_detail = client.get(
                    f"/ledger/voucher/{voucher['id']}", params={"fields": "*"}
                )
                v = voucher_detail.get("value", voucher_detail)
                postings = v.get("postings", [])
                for p in postings:
                    debit = p.get("debitAmount", 0) or 0
                    credit = p.get("creditAmount", 0) or 0
                    if _fields_match(debit, expected["amount"]) or _fields_match(credit, expected["amount"]):
                        passed = True
                        break
            checks.append(_make_check(
                "amount", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['amount']}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Reverse voucher
# ═══════════════════════════════════════════════════════════════════════

def verify_reverse_voucher(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
    pre_created_id: int,
) -> dict:
    """Verify that a voucher was reversed."""
    checks = []

    description = expected.get("description", "")

    # Check if the original voucher has a reversal
    passed_reversed = False
    if pre_created_id:
        voucher_detail = client.get(
            f"/ledger/voucher/{pre_created_id}", params={"fields": "*"}
        )
        v = voucher_detail.get("value", voucher_detail)
        # Check for reversal indicator - reversed vouchers often have a reversedVoucherId
        reversed_id = v.get("reversedVoucherId", 0)
        if reversed_id:
            passed_reversed = True

    # Also check if a new reversing voucher exists with matching description
    if not passed_reversed:
        reverse_date = expected.get("reverse_date", "")
        if reverse_date:
            from datetime import date as dt_date, timedelta
            d = dt_date.fromisoformat(reverse_date)
            vouchers = _search_entity(client, "ledger/voucher", {
                "dateFrom": (d - timedelta(days=1)).isoformat(),
                "dateTo": (d + timedelta(days=30)).isoformat(),
            })
            # Look for a reversal voucher
            for v in vouchers:
                desc = v.get("description", "")
                if "tilbake" in _normalize(desc) or "revers" in _normalize(desc):
                    passed_reversed = True
                    break
                if v.get("id") != pre_created_id and _normalize(description) in _normalize(desc):
                    passed_reversed = True
                    break

    for fc in task_def.field_checks:
        if fc.field == "_reversed":
            checks.append(_make_check(
                "Voucher reversed", fc.points if passed_reversed else 0, fc.points,
                passed_reversed,
                "Reversal found" if passed_reversed else "No reversal found",
            ))
            continue

        if fc.field == "description":
            checks.append(_make_check(
                "description", fc.points if passed_reversed else 0, fc.points,
                passed_reversed,
                f"target={description}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "total_points": total, "max_points": max_pts}


# ═══════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════

def verify_task(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
    pre_created_id: int = 0,
) -> dict:
    """Route to the correct verification function based on task type."""
    name = task_def.name

    # Simple entity creation (Tier 1)
    if name in ("create_employee", "create_customer", "create_product",
                 "create_department", "create_supplier"):
        return verify_simple_entity(client, task_def, expected)

    # Contact creation (Tier 1)
    if name == "create_contact":
        return verify_contact(client, task_def, expected)

    # Update tasks (Tier 1)
    if name in ("update_employee", "update_customer"):
        return verify_update(client, task_def, expected)

    # Invoice tasks (Tier 2)
    if name in ("create_invoice", "create_multi_line_invoice",
                 "invoice_with_payment", "create_credit_note", "delete_invoice"):
        return verify_invoice(client, task_def, expected)

    # Project creation (Tier 2)
    if name == "create_project":
        return verify_project(client, task_def, expected)

    # Travel expense creation (Tier 2)
    if name == "create_travel_expense":
        return verify_travel_expense(client, task_def, expected)

    # Employee with employment (Tier 2)
    if name == "create_employee_with_employment":
        return verify_employee_with_employment(client, task_def, expected)

    # Deletion tasks (Tier 3)
    if name in ("delete_travel_expense", "delete_customer"):
        return verify_deletion(client, task_def, expected, pre_created_id)

    # Ledger voucher tasks (Tier 3)
    if name in ("create_ledger_voucher", "create_opening_balance"):
        return verify_ledger_voucher(client, task_def, expected)

    # Reverse voucher (Tier 3)
    if name == "reverse_voucher":
        return verify_reverse_voucher(client, task_def, expected, pre_created_id)

    log.warning(f"No verifier for task type: {name}")
    return {"checks": [], "total_points": 0, "max_points": 0}
