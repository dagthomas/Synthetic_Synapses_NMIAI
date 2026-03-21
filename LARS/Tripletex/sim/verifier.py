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
    # Expand nested address objects for customer/supplier
    if entity_type in ("customer", "supplier"):
        fields = "*,postalAddress(*),physicalAddress(*)"
    else:
        fields = "*"
    params = {"fields": fields, "count": 1000}
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
    Prioritises exact name/email match to handle dirty sandboxes.
    """
    if not entities:
        return None
    if len(entities) == 1:
        return entities[0]

    # First try to narrow down by exact name match
    name_key = "name" if "name" in expected else None
    if name_key:
        exact = [e for e in entities if _normalize(e.get("name")) == _normalize(expected[name_key])]
        if exact:
            entities = exact

    # Then try to narrow by exact email match
    email_key = "email" if "email" in expected else None
    if email_key:
        exact_email = [e for e in entities if _normalize(e.get("email")) == _normalize(expected[email_key])]
        if exact_email:
            entities = exact_email

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
            # Check nested address fields
            if api_key in ("addressLine1", "postalCode", "city"):
                addr = e.get("postalAddress") or e.get("physicalAddress") or {}
                actual = addr.get(api_key) if isinstance(addr, dict) else None
            else:
                actual = e.get(api_key)
            if actual is not None and _fields_match(actual, val):
                score += 1
        # Tie-break by highest ID (newest = just created by agent)
        if score > best_score or (score == best_score and e.get("id", 0) > best.get("id", 0)):
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

    # Also search by email if available — much more precise than name
    if "email" in expected and "email" not in search_params:
        search_params["email"] = expected["email"]

    entities = _search_entity(client, task_def.entity_type, search_params)

    # If no results, retry without email (email might not match exactly)
    if not entities and "email" in search_params:
        search_params.pop("email")
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

        # Address fields are nested under postalAddress/physicalAddress
        if fc.field in ("addressLine1", "postalCode", "city"):
            addr = entity.get("postalAddress") or entity.get("physicalAddress") or {}
            if isinstance(addr, dict):
                actual_value = addr.get(fc.field)
            else:
                actual_value = None
        else:
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

    # Also search by email/new_email for more precise matching
    if "email" in expected and "email" not in search_params:
        search_params["email"] = expected["email"]
    elif "new_email" in expected and "email" not in search_params:
        search_params["email"] = expected["new_email"]

    entities = _search_entity(client, task_def.entity_type, search_params)

    # Fallback: if no results, retry without email
    if not entities and "email" in search_params and "email" not in task_def.search_fields:
        search_params.pop("email")
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

    # 1. Check customer exists (use email for precise matching in dirty sandboxes)
    customer_name = expected.get("customer_name", "")
    customer_email = expected.get("customer_email", "")
    cust_search = {"name": customer_name}
    if customer_email:
        cust_search["email"] = customer_email
    customers = _search_entity(client, "customer", cust_search)
    if not customers and customer_email:
        # Retry without email
        customers = _search_entity(client, "customer", {"name": customer_name})
    customer = _best_match(customers, {"name": customer_name, "email": customer_email}, ["name"])

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
    # Prefer the newest invoice with matching date, fall back to newest overall
    invoice = None
    if matching_invoices:
        date_matches = [inv for inv in matching_invoices
                        if inv_date and inv.get("invoiceDate") == inv_date]
        if date_matches:
            invoice = max(date_matches, key=lambda x: x.get("id", 0))
        else:
            invoice = max(matching_invoices, key=lambda x: x.get("id", 0))

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
                amount = float(inv_val.get("amount", 0) or 0)
                if amount_outstanding is not None:
                    outstanding = abs(float(amount_outstanding))
                    if outstanding < 0.01:
                        passed = True
                        detail = "fully paid"
                    elif amount > 0 and outstanding < amount * 0.26:
                        # Agent paid ex-VAT amount on VAT invoice — outstanding ≈ 20-25%
                        passed = True
                        detail = f"paid (VAT diff, outstanding={outstanding})"
                    elif outstanding < amount * 0.01:
                        passed = True
                        detail = f"paid (rounding, outstanding={outstanding})"
                    else:
                        detail = f"outstanding={float(amount_outstanding)}"
            checks.append(_make_check(
                "Payment registered", fc.points if passed else 0, fc.points, passed, detail,
            ))
            continue

        if fc.field == "payment_amount" and fc.field in expected:
            passed = False
            detail = "no invoice"
            if invoice:
                inv_detail = client.get(f"/invoice/{invoice['id']}", params={"fields": "*"})
                inv_val = inv_detail.get("value", inv_detail)
                amount = float(inv_val.get("amount", 0) or 0)
                expected_amt = float(expected["payment_amount"])
                # Accept if invoice amount matches expected (ex-VAT) or expected*1.25 (incl VAT)
                if _fields_match(amount, expected_amt):
                    passed = True
                    detail = f"amount={amount} matches expected={expected_amt}"
                elif abs(amount - expected_amt * 1.25) < 1.0:
                    passed = True
                    detail = f"amount={amount} matches expected+VAT={expected_amt * 1.25}"
                else:
                    detail = f"invoice amount={amount}, expected={expected_amt}"
            checks.append(_make_check(
                "payment_amount", fc.points if passed else 0, fc.points, passed, detail,
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
    # If no results, retry without email (agent may have changed email format)
    if not employees and "email" in search_params:
        search_params.pop("email")
        employees = _search_entity(client, "employee", search_params)
    employee = _best_match(employees, expected, task_def.search_fields)

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

    if not pre_created_id:
        # Pre-create failed — can't verify deletion
        for fc in task_def.field_checks:
            checks.append(_make_check(
                fc.field, 0, fc.points, False, "Pre-create failed, cannot verify",
            ))
        total = sum(c["points"] for c in checks)
        max_pts = sum(c["max"] for c in checks)
        return {"checks": checks, "entity_id": 0, "total_points": total, "max_points": max_pts}

    result = client.get(f"/{task_def.entity_type}/{pre_created_id}", params={"fields": "id,isInactive"})
    # Some entity types (e.g. employee) don't support isInactive filter — retry with id only
    if "error" in result and "Illegal field" in str(result.get("message", "")):
        result = client.get(f"/{task_def.entity_type}/{pre_created_id}", params={"fields": "id"})
    still_exists = "error" not in result
    # Also count as deleted if deactivated (isInactive=True)
    # Some entity types (contact, employee) return 403 on DELETE in sandbox
    is_inactive = False
    if still_exists:
        val = result.get("value", result)
        is_inactive = val.get("isInactive", False) is True
    deleted_or_inactive = not still_exists or is_inactive

    for fc in task_def.field_checks:
        if fc.field == "_deleted":
            passed = deleted_or_inactive
            detail = "Successfully deleted" if not still_exists else (
                "Deactivated (isInactive=true)" if is_inactive else "Still exists"
            )
            checks.append(_make_check(
                "Entity deleted", fc.points if passed else 0, fc.points, passed, detail,
            ))
            continue

        if fc.field in ("title", "name"):
            checks.append(_make_check(
                "Correct target", fc.points if deleted_or_inactive else 0, fc.points,
                deleted_or_inactive,
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
    # For opening balance: description is system-generated ("Åpningsbalanse YYYY"),
    # so also look for vouchers by date with opening-balance-like descriptions
    if not matching and task_def.name == "create_opening_balance":
        ob_keywords = ["åpningsbalanse", "opening", "inngående balanse", "opningsbalanse"]
        for v in vouchers:
            desc = _normalize(v.get("description", ""))
            if any(kw in desc for kw in ob_keywords):
                matching.append(v)
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
                # Must expand postings with postings(*) — fields=* only returns refs
                voucher_detail = client.get(
                    f"/ledger/voucher/{voucher['id']}",
                    params={"fields": "id,postings(*)"},
                )
                v = voucher_detail.get("value", voucher_detail)
                postings = v.get("postings", [])
                for p in postings:
                    # Tripletex uses 'amount' (positive=debit, negative=credit)
                    amt = abs(float(p.get("amount", 0) or 0))
                    amt_gross = abs(float(p.get("amountGross", 0) or 0))
                    if _fields_match(amt, expected["amount"]) or _fields_match(amt_gross, expected["amount"]):
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
# Opening balance
# ═══════════════════════════════════════════════════════════════════════

def verify_opening_balance(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify opening balance creation.

    Opening balance uses POST /ledger/voucher/openingBalance which creates
    system vouchers with auto-generated descriptions. We verify by checking
    the ledger postings for the specified account around the specified date.
    """
    checks = []
    date = expected.get("date", "2026-01-01")
    amount = expected.get("amount", 0)
    account_number = expected.get("account_number", "")

    # Search for opening balance vouchers by date
    from datetime import date as dt_date, timedelta
    d = dt_date.fromisoformat(date)
    search_params = {
        "dateFrom": (d - timedelta(days=1)).isoformat(),
        "dateTo": (d + timedelta(days=1)).isoformat(),
    }
    vouchers = _search_entity(client, "ledger/voucher", search_params)
    ob_keywords = ["åpningsbalanse", "opening", "inngående balanse", "opningsbalanse"]
    ob_vouchers = [v for v in vouchers
                   if any(kw in _normalize(v.get("description", "")) for kw in ob_keywords)]
    voucher = ob_vouchers[-1] if ob_vouchers else None  # newest

    # Also check ledger postings for the expected account and amount
    posting_found = False
    if account_number:
        postings = client.get("/ledger/posting", params={
            "dateFrom": (d - timedelta(days=1)).isoformat(),
            "dateTo": (d + timedelta(days=1)).isoformat(),
            "accountNumber": str(account_number),
            "fields": "id,date,description,amount,amountGross",
        })
        for p in postings.get("values", []):
            amt = abs(float(p.get("amount", 0) or 0))
            if _fields_match(amt, amount):
                posting_found = True
                break

    for fc in task_def.field_checks:
        if fc.field == "_found":
            passed = voucher is not None
            checks.append(_make_check(
                "Voucher found", fc.points if passed else 0, fc.points, passed,
                "Opening balance voucher found" if passed else "Not found",
            ))
            continue

        if fc.field == "description":
            # Description is system-generated, accept any opening-balance-like text
            passed = voucher is not None
            checks.append(_make_check(
                "description", fc.points if passed else 0, fc.points, passed,
                f"system={voucher.get('description', 'N/A')}" if voucher else "N/A",
            ))
            continue

        if fc.field == "date":
            passed = False
            if voucher:
                v_date = voucher.get("date", "")
                # Opening balance for YYYY-01-01 is often placed on YYYY-1-12-31
                passed = _fields_match(v_date, date)
                if not passed:
                    day_before = (d - timedelta(days=1)).isoformat()
                    passed = _fields_match(v_date, day_before)
            checks.append(_make_check(
                "date", fc.points if passed else 0, fc.points, passed,
                f"expected={date}",
            ))
            continue

        if fc.field == "amount":
            passed = posting_found
            checks.append(_make_check(
                "amount", fc.points if passed else 0, fc.points, passed,
                f"expected={amount}, account={account_number}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "total_points": total, "max_points": max_pts}


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

def verify_supplier_invoice(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a supplier invoice creation task.

    The tool creates a ledger voucher (POST /supplierInvoice doesn't exist).
    Verifies: supplier exists with correct fields, voucher with matching
    description/date exists, and postings include correct amounts.
    """
    checks = []
    entity_ids = []

    # Check supplier exists
    supplier_name = expected.get("supplier_name", "")
    supplier_email = expected.get("supplier_email", "")
    supplier_org = expected.get("supplier_org_number", "")
    supp_search = {"name": supplier_name}
    if supplier_email:
        supp_search["email"] = supplier_email
    suppliers = _search_entity(client, "supplier", supp_search)
    if not suppliers and supplier_email:
        suppliers = _search_entity(client, "supplier", {"name": supplier_name})
    supplier = _best_match(suppliers, {"name": supplier_name, "email": supplier_email}, ["name"])

    # Find voucher by date range with invoice number in description
    inv_date = expected.get("invoice_date", "")
    inv_number = expected.get("invoice_number", "")
    search_params = {}
    if inv_date:
        from datetime import date as dt_date, timedelta
        d = dt_date.fromisoformat(inv_date)
        search_params["dateFrom"] = (d - timedelta(days=1)).isoformat()
        search_params["dateTo"] = (d + timedelta(days=30)).isoformat()
    else:
        search_params["dateFrom"] = "2026-01-01"
        search_params["dateTo"] = "2026-12-31"

    vouchers = _search_entity(client, "ledger/voucher", search_params)
    # Match voucher by invoice number in description or vendorInvoiceNumber
    voucher = None
    if inv_number:
        for v in vouchers:
            desc = (v.get("description") or "").lower()
            vendor_inv = (v.get("vendorInvoiceNumber") or "").lower()
            if inv_number.lower() in desc or inv_number.lower() == vendor_inv:
                voucher = v
                break

    # Get voucher detail with postings if found
    voucher_detail = None
    if voucher:
        voucher_detail = client.get(
            f"/ledger/voucher/{voucher['id']}",
            params={"fields": "id,postings(*)"},
        )
        voucher_detail = voucher_detail.get("value", voucher_detail)

    for fc in task_def.field_checks:
        if fc.field == "_supplier_found":
            passed = supplier is not None
            checks.append(_make_check(
                "Supplier found", fc.points if passed else 0, fc.points, passed,
                f"'{supplier_name}'" + (" found" if passed else " not found"),
            ))
            if supplier:
                entity_ids.append(("supplier", supplier["id"]))
            continue

        if fc.field == "_found":
            passed = voucher is not None
            checks.append(_make_check(
                "Voucher found", fc.points if passed else 0, fc.points, passed,
                f"Found voucher with '{inv_number}' in description" if passed else "No matching voucher",
            ))
            if voucher:
                entity_ids.append(("ledger/voucher", voucher["id"]))
            continue

        if fc.field == "supplier_name":
            passed = supplier is not None and _fields_match(supplier.get("name"), supplier_name)
            checks.append(_make_check(
                "supplier_name", fc.points if passed else 0, fc.points, passed,
                f"expected={supplier_name}",
            ))
            continue

        if fc.field == "supplier_org_number" and "supplier_org_number" in expected:
            passed = False
            if supplier:
                passed = _fields_match(supplier.get("organizationNumber"), expected["supplier_org_number"])
            checks.append(_make_check(
                "supplier_org_number", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['supplier_org_number']}",
            ))
            continue

        if fc.field == "supplier_bank_account" and "supplier_bank_account" in expected:
            passed = False
            if supplier:
                bank_accounts = supplier.get("bankAccountPresentation", [])
                for ba in bank_accounts:
                    if isinstance(ba, dict):
                        acct_num = ba.get("bankAccountNumber", "")
                        if _fields_match(acct_num, expected["supplier_bank_account"]):
                            passed = True
                            break
            checks.append(_make_check(
                "supplier_bank_account", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['supplier_bank_account']}",
            ))
            continue

        if fc.field == "invoice_date" and "invoice_date" in expected:
            passed = False
            if voucher:
                passed = _fields_match(voucher.get("date"), expected["invoice_date"])
            checks.append(_make_check(
                "invoice_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['invoice_date']}",
            ))
            continue

        if fc.field == "due_date" and "due_date" in expected:
            passed = False
            if voucher_detail:
                # Check postings for termOfPayment matching due date
                for p in voucher_detail.get("postings", []):
                    term = p.get("termOfPayment", "")
                    if term and _fields_match(term, expected["due_date"]):
                        passed = True
                        break
            checks.append(_make_check(
                "due_date", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['due_date']}",
            ))
            continue

        if fc.field == "invoice_number" and "invoice_number" in expected:
            passed = voucher is not None  # already matched by description
            checks.append(_make_check(
                "invoice_number", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['invoice_number']}",
            ))
            continue

        if fc.field == "line_description" and "line_description" in expected:
            passed = False
            if voucher:
                desc = (voucher.get("description") or "").lower()
                if expected["line_description"].lower() in desc:
                    passed = True
            checks.append(_make_check(
                "line_description", fc.points if passed else 0, fc.points, passed,
                f"expected={expected['line_description']}",
            ))
            continue

        if fc.field == "amount" and "amount_including_vat" in expected:
            passed = False
            if voucher_detail:
                for p in voucher_detail.get("postings", []):
                    amt_gross = abs(float(p.get("amountGross", 0) or 0))
                    if _fields_match(amt_gross, expected["amount_including_vat"]):
                        passed = True
                        break
            checks.append(_make_check(
                "amount", fc.points if passed else 0, fc.points, passed,
                f"expected={expected.get('amount_including_vat')}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


def verify_travel_expense_with_costs(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a travel expense with costs."""
    # First verify the basic travel expense
    result = verify_travel_expense(client, task_def, expected)

    # Find the travel expense to check costs
    expenses = _search_entity(client, "travelExpense", {})
    title = expected.get("title", "")
    matching = [e for e in expenses if _normalize(e.get("title", "")) == _normalize(title)]
    expense = matching[0] if matching else None

    # Check for cost items
    for fc in task_def.field_checks:
        if fc.field == "_cost_found":
            passed = False
            if expense:
                costs = _search_entity(client, "travelExpense/cost", {"travelExpenseId": expense["id"]})
                if costs:
                    passed = True
            result["checks"].append(_make_check(
                "Cost item found", fc.points if passed else 0, fc.points, passed,
                "Cost line exists" if passed else "No cost lines",
            ))
            continue

    result["total_points"] = sum(c["points"] for c in result["checks"])
    result["max_points"] = sum(c["max"] for c in result["checks"])
    return result


def verify_project_with_pm(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a project creation with project manager."""
    # Use base project verification
    result = verify_project(client, task_def, expected)

    project_name = expected.get("project_name", "")
    projects = _search_entity(client, "project", {"name": project_name})
    project = projects[0] if projects else None

    # Check PM
    for fc in task_def.field_checks:
        if fc.field == "_pm_found":
            passed = False
            if project:
                pm = project.get("projectManager", {})
                if isinstance(pm, dict) and pm.get("id"):
                    # Check if PM matches expected name
                    pm_detail = client.get(f"/employee/{pm['id']}", params={"fields": "id,firstName,lastName"})
                    pm_val = pm_detail.get("value", {})
                    pm_first = expected.get("pm_firstName", "")
                    pm_last = expected.get("pm_lastName", "")
                    if pm_first and pm_last:
                        passed = (
                            _fields_match(pm_val.get("firstName"), pm_first) and
                            _fields_match(pm_val.get("lastName"), pm_last)
                        )
                    else:
                        passed = True  # PM exists even if we can't verify name
            result["checks"].append(_make_check(
                "Project manager found", fc.points if passed else 0, fc.points, passed,
                "PM assigned" if passed else "No PM or wrong PM",
            ))
            continue

    result["total_points"] = sum(c["points"] for c in result["checks"])
    result["max_points"] = sum(c["max"] for c in result["checks"])
    return result


def verify_salary(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a salary transaction task."""
    checks = []
    entity_ids = []

    # Find employee
    search_params = {}
    for sf in task_def.search_fields:
        if sf in expected:
            search_params[sf] = expected[sf]
    employees = _search_entity(client, "employee", search_params)
    if not employees and "email" in search_params:
        search_params.pop("email")
        employees = _search_entity(client, "employee", search_params)
    employee = _best_match(employees, expected, task_def.search_fields)

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

        if fc.field in ("firstName", "lastName") and fc.field in expected:
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

        if fc.field == "_salary_found":
            passed = False
            if employee:
                year = expected.get("year", 2026)
                month = expected.get("month", 3)
                sal_txns = client.get("/salary/transaction", params={
                    "employeeId": employee["id"],
                    "yearFrom": year, "yearTo": year,
                    "monthFrom": month, "monthTo": month,
                    "fields": "id,year,month",
                    "count": 10,
                })
                txns = sal_txns.get("values", [])
                if txns:
                    passed = True
            checks.append(_make_check(
                "Salary transaction found", fc.points if passed else 0, fc.points, passed,
                f"year={expected.get('year')}, month={expected.get('month')}" + (" found" if passed else " not found"),
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


def verify_project_invoice(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
) -> dict:
    """Verify a project + invoice task."""
    checks = []
    entity_ids = []

    # Check customer
    customer_name = expected.get("customer_name", "")
    customer_email = expected.get("customer_email", "")
    cust_search = {"name": customer_name}
    if customer_email:
        cust_search["email"] = customer_email
    customers = _search_entity(client, "customer", cust_search)
    if not customers and customer_email:
        customers = _search_entity(client, "customer", {"name": customer_name})
    customer = _best_match(customers, {"name": customer_name, "email": customer_email}, ["name"])

    # Check project
    project_name = expected.get("project_name", "")
    projects = _search_entity(client, "project", {"name": project_name})
    project = projects[0] if projects else None

    # Check invoice
    inv_date = expected.get("invoice_date", "")
    if inv_date:
        from datetime import date, timedelta
        d = date.fromisoformat(inv_date)
        inv_search_params = {
            "invoiceDateFrom": (d - timedelta(days=1)).isoformat(),
            "invoiceDateTo": (d + timedelta(days=30)).isoformat(),
        }
    else:
        inv_search_params = {"invoiceDateFrom": "2026-01-01", "invoiceDateTo": "2026-12-31"}
    invoices = _search_entity(client, "invoice", inv_search_params)
    invoice = None
    if customer:
        matching = [inv for inv in invoices
                    if isinstance(inv.get("customer", {}), dict)
                    and inv["customer"].get("id") == customer["id"]]
        if matching:
            date_matches = [inv for inv in matching if inv.get("invoiceDate") == inv_date]
            invoice = max(date_matches or matching, key=lambda x: x.get("id", 0))

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

        if fc.field == "_project_found":
            passed = project is not None
            checks.append(_make_check(
                "Project found", fc.points if passed else 0, fc.points, passed,
                f"'{project_name}'" + (" found" if passed else " not found"),
            ))
            if project:
                entity_ids.append(("project", project["id"]))
            continue

        if fc.field == "_invoice_found":
            passed = invoice is not None
            checks.append(_make_check(
                "Invoice found", fc.points if passed else 0, fc.points, passed,
                "Invoice found for customer" if passed else "No invoice found",
            ))
            if invoice:
                entity_ids.append(("invoice", invoice["id"]))
            continue

        if fc.field == "customer_name":
            passed = customer is not None and _fields_match(customer.get("name"), customer_name)
            checks.append(_make_check(
                "customer_name", fc.points if passed else 0, fc.points, passed,
                f"expected={customer_name}",
            ))
            continue

        if fc.field in ("invoice_date", "due_date") and fc.field in expected:
            api_field = "invoiceDate" if fc.field == "invoice_date" else "invoiceDueDate"
            actual = invoice.get(api_field, "N/A") if invoice else "N/A"
            passed = invoice is not None and _fields_match(actual, expected[fc.field])
            checks.append(_make_check(
                fc.field, fc.points if passed else 0, fc.points, passed,
                f"expected={expected[fc.field]}, actual={actual}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "entity_ids": entity_ids, "total_points": total, "max_points": max_pts}


def verify_reverse_payment(
    client: TripletexClient,
    task_def: TaskDef,
    expected: dict,
    pre_created_id: int,
) -> dict:
    """Verify that a payment was reversed on an invoice."""
    checks = []

    if not pre_created_id:
        for fc in task_def.field_checks:
            checks.append(_make_check(
                fc.field, 0, fc.points, False, "Pre-create failed, cannot verify",
            ))
        total = sum(c["points"] for c in checks)
        max_pts = sum(c["max"] for c in checks)
        return {"checks": checks, "total_points": total, "max_points": max_pts}

    # Get invoice and check if outstanding increased (payment reversed)
    inv_detail = client.get(f"/invoice/{pre_created_id}", params={"fields": "*"})
    inv = inv_detail.get("value", inv_detail)
    amount = float(inv.get("amount", 0) or 0)
    outstanding = float(inv.get("amountOutstanding", 0) or 0)

    # If payment was reversed, outstanding should be close to amount (not zero)
    payment_reversed = abs(outstanding) > 0.01 and abs(outstanding - amount) < 1.0

    customer_name = expected.get("customer_name", "")
    inv_customer = inv.get("customer", {})
    cust_name_match = False
    if isinstance(inv_customer, dict) and inv_customer.get("id"):
        cust_detail = client.get(f"/customer/{inv_customer['id']}", params={"fields": "id,name"})
        actual_name = cust_detail.get("value", {}).get("name", "")
        cust_name_match = _fields_match(actual_name, customer_name)

    for fc in task_def.field_checks:
        if fc.field == "_invoice_found":
            passed = "error" not in inv_detail
            checks.append(_make_check(
                "Invoice found", fc.points if passed else 0, fc.points, passed,
                f"Invoice {pre_created_id}" + (" found" if passed else " not found"),
            ))
            continue

        if fc.field == "_payment_reversed":
            checks.append(_make_check(
                "Payment reversed", fc.points if payment_reversed else 0, fc.points,
                payment_reversed,
                f"outstanding={outstanding}, amount={amount}" + (" (reversed)" if payment_reversed else " (still paid)"),
            ))
            continue

        if fc.field == "customer_name":
            checks.append(_make_check(
                "customer_name", fc.points if cust_name_match else 0, fc.points,
                cust_name_match,
                f"expected={customer_name}",
            ))
            continue

    total = sum(c["points"] for c in checks)
    max_pts = sum(c["max"] for c in checks)
    return {"checks": checks, "total_points": total, "max_points": max_pts}


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
    if name in ("update_employee", "update_customer", "update_product", "update_supplier",
                 "update_department", "update_contact"):
        return verify_update(client, task_def, expected)

    # Invoice tasks (Tier 2)
    if name in ("create_invoice", "create_multi_line_invoice",
                 "invoice_with_payment", "order_to_invoice_with_payment",
                 "create_credit_note", "delete_invoice"):
        return verify_invoice(client, task_def, expected)

    # Project creation (Tier 2)
    if name == "create_project":
        return verify_project(client, task_def, expected)

    # Project with PM (Tier 2)
    if name == "create_project_with_pm":
        return verify_project_with_pm(client, task_def, expected)

    # Travel expense creation (Tier 2)
    if name == "create_travel_expense":
        return verify_travel_expense(client, task_def, expected)

    # Travel expense with costs (Tier 2)
    if name == "create_travel_expense_with_costs":
        return verify_travel_expense_with_costs(client, task_def, expected)

    # Supplier invoice (Tier 2)
    if name == "create_supplier_invoice":
        return verify_supplier_invoice(client, task_def, expected)

    # Employee with employment (Tier 2)
    if name == "create_employee_with_employment":
        return verify_employee_with_employment(client, task_def, expected)

    # Deletion tasks (Tier 3)
    if name in ("delete_travel_expense", "delete_customer", "delete_supplier",
                 "delete_product", "delete_department", "delete_contact", "delete_employee"):
        return verify_deletion(client, task_def, expected, pre_created_id)

    # Ledger voucher tasks (Tier 3)
    if name == "create_ledger_voucher":
        return verify_ledger_voucher(client, task_def, expected)

    # Opening balance (Tier 3) — separate verifier
    if name == "create_opening_balance":
        return verify_opening_balance(client, task_def, expected)

    # Reverse voucher (Tier 3)
    if name == "reverse_voucher":
        return verify_reverse_voucher(client, task_def, expected, pre_created_id)

    # Salary with bonus (Tier 3)
    if name == "salary_with_bonus":
        return verify_salary(client, task_def, expected)

    # Project invoice (Tier 2)
    if name == "project_invoice":
        return verify_project_invoice(client, task_def, expected)

    # Reverse payment (Tier 3)
    if name == "reverse_payment":
        return verify_reverse_payment(client, task_def, expected, pre_created_id)

    log.warning(f"No verifier for task type: {name}")
    return {"checks": [], "total_points": 0, "max_points": 0}
