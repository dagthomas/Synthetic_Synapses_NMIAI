"""Sandbox data curator — health check, seed, and clean operations.

Provides structured results for use by both the dashboard API and the
CLI (populate_sandbox.py).
"""

import logging

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)

# Entity types we track, in dependency order (seed top-down, clean bottom-up)
ENTITY_TYPES = [
    "department", "employee", "customer", "product",
    "contact", "project", "invoice", "travelExpense",
]

# Minimum count per type for sandbox to be considered "ready"
MIN_COUNTS = {t: 1 for t in ENTITY_TYPES}


# ── Health check ──────────────────────────────────────────────────

def check_health(client: TripletexClient) -> dict:
    """Check sandbox connectivity and entity inventory."""
    result = {
        "connected": False,
        "base_url": client.base_url,
        "entities": {},
        "modules": {"department": False, "projectEconomy": False},
        "bank_account_1920": False,
        "ready": False,
    }

    # Test connectivity
    try:
        test = client.get("/employee", params={"count": 0, "fields": "id"})
        if "error" in test:
            return result
        result["connected"] = True
    except Exception:
        return result

    # Extra params required by certain endpoints
    _EXTRA_PARAMS = {
        "invoice": {"invoiceDateFrom": "2000-01-01", "invoiceDateTo": "2099-12-31"},
    }

    # Count each entity type
    all_ok = True
    for entity_type in ENTITY_TYPES:
        try:
            params = {"count": 0, "fields": "id", **_EXTRA_PARAMS.get(entity_type, {})}
            r = client.get(f"/{entity_type}", params=params)
            count = r.get("fullResultSize", 0)
            ok = count >= MIN_COUNTS.get(entity_type, 1)
            if not ok:
                all_ok = False
            result["entities"][entity_type] = {"count": count, "ok": ok}
        except Exception:
            result["entities"][entity_type] = {"count": 0, "ok": False}
            all_ok = False

    # Check bank account on 1920
    try:
        r = client.get("/ledger/account", params={
            "number": "1920", "fields": "id,bankAccountNumber"
        })
        accounts = r.get("values", [])
        result["bank_account_1920"] = bool(
            accounts and accounts[0].get("bankAccountNumber")
        )
    except Exception:
        pass

    if not result["bank_account_1920"]:
        all_ok = False

    # Check modules (best-effort, may not be queryable)
    result["modules"]["department"] = result["entities"].get(
        "department", {}
    ).get("ok", False)
    result["modules"]["projectEconomy"] = result["entities"].get(
        "project", {}
    ).get("ok", False)

    result["ready"] = all_ok
    return result


# ── Seed functions ────────────────────────────────────────────────

def enable_modules(client: TripletexClient) -> dict:
    """Enable required modules. Returns {ok, error}."""
    try:
        client.put("/company/modules", json={
            "moduleDepartment": True,
            "moduleProjectEconomy": True,
        })
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def setup_bank_account(client: TripletexClient) -> dict:
    """Set bank account on ledger 1920. Returns {ok, already_set, error}."""
    try:
        r = client.get("/ledger/account", params={
            "number": "1920", "fields": "id,name,isBankAccount,bankAccountNumber"
        })
        accounts = r.get("values", [])
        if not accounts:
            return {"ok": False, "error": "Account 1920 not found"}

        acct = accounts[0]
        if acct.get("bankAccountNumber"):
            return {"ok": True, "already_set": True}

        client.put(f"/ledger/account/{acct['id']}", json={
            "id": acct["id"],
            "number": 1920,
            "name": acct["name"],
            "isBankAccount": True,
            "bankAccountNumber": "12345678903",
        })
        return {"ok": True, "already_set": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def seed_departments(client: TripletexClient) -> dict:
    """Create sample departments. Returns {created, ids, errors}."""
    departments = [
        {"name": "Salg", "departmentNumber": "10"},
        {"name": "IT", "departmentNumber": "20"},
        {"name": "Regnskap", "departmentNumber": "30"},
    ]
    return _seed_batch(client, "/department", departments)


def _get_or_create_base_dept(client: TripletexClient) -> int:
    """Get first existing department or create a default one."""
    r = client.get("/department", params={"fields": "id,name", "count": 1})
    existing = r.get("values", [])
    if existing:
        return existing[0]["id"]
    r = client.post("/department", json={
        "name": "Generell", "departmentNumber": "1"
    })
    return r.get("value", {}).get("id", 0)


def seed_employees(client: TripletexClient) -> dict:
    """Create sample employees. Returns {created, ids, errors}."""
    dept_id = _get_or_create_base_dept(client)
    employees = [
        {"firstName": "Ola", "lastName": "Nordmann",
         "email": "ola.nordmann@firma.no", "userType": "STANDARD",
         "department": {"id": dept_id}},
        {"firstName": "Kari", "lastName": "Hansen",
         "email": "kari.hansen@firma.no", "userType": "STANDARD",
         "department": {"id": dept_id}},
        {"firstName": "Erik", "lastName": "Johansen",
         "email": "erik.johansen@firma.no", "userType": "STANDARD",
         "phoneNumberMobile": "+47 91234567",
         "department": {"id": dept_id}},
        {"firstName": "Ingrid", "lastName": "Larsen",
         "email": "ingrid.larsen@firma.no", "userType": "STANDARD",
         "phoneNumberMobile": "+47 92345678",
         "department": {"id": dept_id}},
        {"firstName": "Magnus", "lastName": "Berg",
         "email": "magnus.berg@firma.no", "userType": "STANDARD",
         "department": {"id": dept_id}},
    ]
    return _seed_batch(client, "/employee", employees)


def seed_customers(client: TripletexClient) -> dict:
    """Create sample customers. Returns {created, ids, errors}."""
    customers = [
        {"name": "Nordvik Consulting AS", "email": "post@nordvik.no",
         "isCustomer": True},
        {"name": "Bergen IT Solutions AS", "email": "kontakt@bergenit.no",
         "isCustomer": True, "organizationNumber": "912345678"},
        {"name": "Tromsø Shipping DA", "email": "post@tromso-shipping.no",
         "isCustomer": True, "phoneNumber": "+47 77123456"},
        {"name": "Stavanger Olje AS", "email": "faktura@stavangerolje.no",
         "isCustomer": True, "organizationNumber": "923456789"},
        {"name": "Oslo Design Studio ANS", "email": "hei@oslodesign.no",
         "isCustomer": True},
        {"name": "Kristiansand Bygg AS", "email": "post@krsbygg.no",
         "isCustomer": True, "isSupplier": True},
    ]
    return _seed_batch(client, "/customer", customers)


def seed_products(client: TripletexClient) -> dict:
    """Create sample products. Returns {created, ids, errors}."""
    products = [
        {"name": "Konsulenttjenester",
         "priceExcludingVatCurrency": 1200.0},
        {"name": "Webdesign Pakke",
         "priceExcludingVatCurrency": 15000.0, "number": "PRD-001"},
        {"name": "IT Support Timer",
         "priceExcludingVatCurrency": 950.0, "number": "PRD-002"},
        {"name": "Prosjektledelse",
         "priceExcludingVatCurrency": 1500.0},
        {"name": "Programvareutvikling",
         "priceExcludingVatCurrency": 1800.0, "number": "PRD-003"},
        {"name": "Kontorrekvisita",
         "priceExcludingVatCurrency": 250.0},
    ]
    return _seed_batch(client, "/product", products)


def seed_contacts(client: TripletexClient) -> dict:
    """Create contact persons for existing customers. Returns {created, ids, errors}."""
    r = client.get("/customer", params={"fields": "id", "count": 2})
    customers = r.get("values", [])
    if len(customers) < 2:
        return {"created": 0, "ids": [], "errors": ["Need at least 2 customers"]}

    contacts = [
        {"firstName": "Per", "lastName": "Olsen", "email": "per@nordvik.no",
         "customer": {"id": customers[0]["id"]}},
        {"firstName": "Lise", "lastName": "Bakke", "email": "lise@bergenit.no",
         "customer": {"id": customers[1]["id"]}},
    ]
    return _seed_batch(client, "/contact", contacts)


def seed_projects(client: TripletexClient) -> dict:
    """Create sample projects. Returns {created, ids, errors}."""
    r = client.get("/customer", params={"fields": "id", "count": 2})
    customers = r.get("values", [])
    if len(customers) < 2:
        return {"created": 0, "ids": [], "errors": ["Need at least 2 customers"]}

    # Use account owner as project manager
    r = client.get("/employee", params={"fields": "id", "count": 1})
    employees = r.get("values", [])
    if not employees:
        return {"created": 0, "ids": [], "errors": ["Need at least 1 employee"]}
    manager_id = employees[0]["id"]

    projects = [
        {"name": "Nettside Redesign",
         "customer": {"id": customers[0]["id"]},
         "projectManager": {"id": manager_id},
         "startDate": "2026-03-01"},
        {"name": "ERP Implementering",
         "customer": {"id": customers[1]["id"]},
         "projectManager": {"id": manager_id},
         "startDate": "2026-04-01"},
    ]
    return _seed_batch(client, "/project", projects)


def seed_invoices(client: TripletexClient) -> dict:
    """Create sample invoices (with orders). Returns {created, ids, errors}."""
    r = client.get("/customer", params={"fields": "id", "count": 3})
    customers = r.get("values", [])
    r = client.get("/product", params={"fields": "id", "count": 3})
    products = r.get("values", [])

    if len(customers) < 3 or len(products) < 3:
        return {"created": 0, "ids": [],
                "errors": ["Need at least 3 customers and 3 products"]}

    specs = [
        {"cust": customers[0]["id"], "prod": products[0]["id"],
         "count": 10, "date": "2026-03-01", "due": "2026-03-15"},
        {"cust": customers[1]["id"], "prod": products[1]["id"],
         "count": 1, "date": "2026-03-05", "due": "2026-03-19"},
        {"cust": customers[2]["id"], "prod": products[2]["id"],
         "count": 5, "date": "2026-03-10", "due": "2026-03-24"},
    ]

    ids = []
    errors = []
    for s in specs:
        order_r = client.post("/order", json={
            "customer": {"id": s["cust"]},
            "orderDate": s["date"],
            "deliveryDate": s["date"],
            "orderLines": [
                {"product": {"id": s["prod"]}, "count": s["count"]}
            ],
        })
        if "error" in order_r:
            errors.append(f"Order failed: {order_r.get('message', '')}")
            continue

        order_id = order_r.get("value", {}).get("id", 0)
        inv_r = client.post("/invoice", json={
            "invoiceDate": s["date"],
            "invoiceDueDate": s["due"],
            "orders": [{"id": order_id}],
        })
        if "error" not in inv_r:
            ids.append(inv_r.get("value", {}).get("id", 0))
        else:
            errors.append(f"Invoice failed: {inv_r.get('message', '')}")

    return {"created": len(ids), "ids": ids, "errors": errors}


def seed_travel_expenses(client: TripletexClient) -> dict:
    """Create sample travel expenses. Returns {created, ids, errors}."""
    r = client.get("/employee", params={"fields": "id", "count": 2})
    employees = r.get("values", [])
    if len(employees) < 2:
        return {"created": 0, "ids": [],
                "errors": ["Need at least 2 employees"]}

    expenses = [
        {"employee": {"id": employees[0]["id"]},
         "title": "Kundebesøk Oslo",
         "travelDetails": {"departureDate": "2026-03-03",
                           "returnDate": "2026-03-04"}},
        {"employee": {"id": employees[1]["id"]},
         "title": "Konferanse Bergen",
         "travelDetails": {"departureDate": "2026-03-10",
                           "returnDate": "2026-03-12"}},
        {"employee": {"id": employees[0]["id"]},
         "title": "Messestand Trondheim",
         "travelDetails": {"departureDate": "2026-03-15",
                           "returnDate": "2026-03-17"}},
    ]
    return _seed_batch(client, "/travelExpense", expenses)


# ── Seed orchestration ────────────────────────────────────────────

_SEED_FNS = {
    "department": seed_departments,
    "employee": seed_employees,
    "customer": seed_customers,
    "product": seed_products,
    "contact": seed_contacts,
    "project": seed_projects,
    "invoice": seed_invoices,
    "travelExpense": seed_travel_expenses,
}

# Order matters: dependencies first
_SEED_ORDER = [
    "department", "employee", "customer", "product",
    "contact", "project", "invoice", "travelExpense",
]


def seed_entities(client: TripletexClient,
                  types: list[str] | None = None,
                  clean: bool = False) -> dict:
    """Seed sandbox entities. Returns structured results.

    Args:
        types: list of entity types, or None / ["all"] for everything
        clean: if True, clean all entities before seeding
    """
    if clean:
        clean_entities(client)

    # Enable modules and set up bank account first
    modules_result = enable_modules(client)
    bank_result = setup_bank_account(client)

    do_all = types is None or types == ["all"] or "all" in types
    order = _SEED_ORDER if do_all else [t for t in _SEED_ORDER if t in types]

    results = {}
    total_created = 0
    total_errors = 0

    for entity_type in order:
        fn = _SEED_FNS.get(entity_type)
        if not fn:
            continue
        log.info(f"Seeding {entity_type}...")
        r = fn(client)
        results[entity_type] = r
        total_created += r.get("created", 0)
        total_errors += len(r.get("errors", []))
        log.info(f"  {entity_type}: created={r['created']}, "
                 f"errors={len(r.get('errors', []))}")

    return {
        "results": results,
        "modules": modules_result,
        "bank_account": bank_result,
        "total_created": total_created,
        "total_errors": total_errors,
    }


# ── Clean ─────────────────────────────────────────────────────────

# Reverse dependency order for deletion
_CLEAN_ORDER = [
    "travelExpense", "invoice", "project", "contact",
    "product", "employee", "customer", "department",
]


def clean_entities(client: TripletexClient) -> dict:
    """Delete all sandbox entities. Returns structured results.

    Skips entity types that Tripletex doesn't allow deleting (employee,
    contact, product all return 403/422). For other types, stops trying
    after 2 consecutive failures to avoid pointless API calls.
    """
    # These types cannot be deleted via the Tripletex API
    _SKIP_DELETE = {"employee", "contact", "product"}

    results = {}
    total_deleted = 0

    for entity_type in _CLEAN_ORDER:
        if entity_type in _SKIP_DELETE:
            results[entity_type] = {
                "deleted": 0,
                "errors": [],
                "skipped": "Tripletex API does not allow deletion",
            }
            continue

        deleted = 0
        errors = []
        try:
            r = client.get(f"/{entity_type}",
                           params={"fields": "id", "count": 1000})
            entities = r.get("values", [])
        except Exception as e:
            errors.append(f"GET {entity_type}: {str(e)[:80]}")
            entities = []

        consecutive_fails = 0
        for e in entities:
            dr = client.delete(f"/{entity_type}/{e['id']}")
            if dr.get("error"):
                consecutive_fails += 1
                errors.append(
                    f"{entity_type}/{e['id']}: "
                    f"{dr.get('message', 'unknown')[:80]}"
                )
                # Stop after 2 consecutive failures — likely a permission issue
                if consecutive_fails >= 2:
                    remaining = len(entities) - deleted - consecutive_fails
                    if remaining > 0:
                        errors.append(
                            f"Stopped: {remaining} more skipped after "
                            f"{consecutive_fails} consecutive failures"
                        )
                    break
            else:
                deleted += 1
                consecutive_fails = 0

        results[entity_type] = {"deleted": deleted, "errors": errors}
        total_deleted += deleted
        if entities:
            log.info(f"Cleaned {entity_type}: {deleted}/{len(entities)} deleted")

    return {"results": results, "total_deleted": total_deleted}


# ── Helpers ───────────────────────────────────────────────────────

def _seed_batch(client: TripletexClient, endpoint: str,
                items: list[dict]) -> dict:
    """Create a batch of entities. Returns {created, ids, errors}."""
    ids = []
    errors = []
    for item in items:
        r = client.post(endpoint, json=item)
        if "error" not in r:
            eid = r.get("value", {}).get("id", 0)
            ids.append(eid)
        else:
            msg = r.get("message", "unknown error")
            errors.append(f"{endpoint}: {msg[:100]}")
    return {"created": len(ids), "ids": ids, "errors": errors}
