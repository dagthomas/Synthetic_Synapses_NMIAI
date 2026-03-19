"""Populate the Tripletex sandbox with realistic test data.

Creates a full set of entities so the agent and simulator have
data to work with (especially for update/delete/payment tasks).

Usage:
    python populate_sandbox.py          # populate everything
    python populate_sandbox.py --clean  # delete all created entities first
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tripletex_client import TripletexClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("populate")


def get_client() -> TripletexClient:
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not base_url or not token:
        print("Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env")
        sys.exit(1)
    return TripletexClient(base_url, token)


def enable_modules(c: TripletexClient):
    log.info("Enabling modules (may fail on sandbox — OK)...")
    c.put("/company/modules", json={
        "moduleDepartment": True,
        "moduleProjectEconomy": True,
    })


def setup_bank_account(c: TripletexClient):
    """Set a bank account on ledger account 1920 so invoices can be created."""
    log.info("Setting up bank account on ledger 1920...")
    result = c.get("/ledger/account", params={"number": "1920", "fields": "id,name,isBankAccount,bankAccountNumber"})
    accounts = result.get("values", [])
    if not accounts:
        log.warning("  Could not find account 1920")
        return
    acct = accounts[0]
    if acct.get("bankAccountNumber"):
        log.info(f"  Bank account already set: {acct['bankAccountNumber']}")
        return
    c.put(f"/ledger/account/{acct['id']}", json={
        "id": acct["id"],
        "number": 1920,
        "name": acct["name"],
        "isBankAccount": True,
        "bankAccountNumber": "12345678903",
    })
    log.info("  Bank account set on 1920")


def create_employees(c: TripletexClient, dept_id: int) -> list[int]:
    """Create sample employees. Requires a department ID."""
    employees = [
        {"firstName": "Ola", "lastName": "Nordmann", "email": "ola.nordmann@firma.no"},
        {"firstName": "Kari", "lastName": "Hansen", "email": "kari.hansen@firma.no"},
        {"firstName": "Erik", "lastName": "Johansen", "email": "erik.johansen@firma.no",
         "phoneNumberMobile": "+47 91234567"},
        {"firstName": "Ingrid", "lastName": "Larsen", "email": "ingrid.larsen@firma.no",
         "phoneNumberMobile": "+47 92345678"},
        {"firstName": "Magnus", "lastName": "Berg", "email": "magnus.berg@firma.no"},
    ]
    ids = []
    for emp in employees:
        emp["userType"] = "STANDARD"
        emp["department"] = {"id": dept_id}
        r = c.post("/employee", json=emp)
        if "error" not in r:
            eid = r.get("value", {}).get("id", 0)
            ids.append(eid)
            log.info(f"  Employee: {emp['firstName']} {emp['lastName']} (id={eid})")
        else:
            log.warning(f"  Failed: {emp['firstName']} - {r.get('message', '')}")
    return ids


def create_customers(c: TripletexClient) -> list[int]:
    """Create sample customers."""
    customers = [
        {"name": "Nordvik Consulting AS", "email": "post@nordvik.no", "isCustomer": True},
        {"name": "Bergen IT Solutions AS", "email": "kontakt@bergenit.no", "isCustomer": True,
         "organizationNumber": "912345678"},
        {"name": "Tromsø Shipping DA", "email": "post@tromso-shipping.no", "isCustomer": True,
         "phoneNumber": "+47 77123456"},
        {"name": "Stavanger Olje AS", "email": "faktura@stavangerolje.no", "isCustomer": True,
         "organizationNumber": "923456789"},
        {"name": "Oslo Design Studio ANS", "email": "hei@oslodesign.no", "isCustomer": True},
        {"name": "Kristiansand Bygg AS", "email": "post@krsbygg.no", "isCustomer": True,
         "isSupplier": True},
    ]
    ids = []
    for cust in customers:
        r = c.post("/customer", json=cust)
        if "error" not in r:
            cid = r.get("value", {}).get("id", 0)
            ids.append(cid)
            log.info(f"  Customer: {cust['name']} (id={cid})")
        else:
            log.warning(f"  Failed: {cust['name']} - {r.get('message', '')}")
    return ids


def create_products(c: TripletexClient) -> list[int]:
    """Create sample products."""
    products = [
        {"name": "Konsulenttjenester", "priceExcludingVatCurrency": 1200.0},
        {"name": "Webdesign Pakke", "priceExcludingVatCurrency": 15000.0, "number": "PRD-001"},
        {"name": "IT Support Timer", "priceExcludingVatCurrency": 950.0, "number": "PRD-002"},
        {"name": "Prosjektledelse", "priceExcludingVatCurrency": 1500.0},
        {"name": "Programvareutvikling", "priceExcludingVatCurrency": 1800.0, "number": "PRD-003"},
        {"name": "Kontorrekvisita", "priceExcludingVatCurrency": 250.0},
    ]
    ids = []
    for prod in products:
        r = c.post("/product", json=prod)
        if "error" not in r:
            pid = r.get("value", {}).get("id", 0)
            ids.append(pid)
            log.info(f"  Product: {prod['name']} @ {prod['priceExcludingVatCurrency']} NOK (id={pid})")
        else:
            log.warning(f"  Failed: {prod['name']} - {r.get('message', '')}")
    return ids


def create_departments(c: TripletexClient) -> list[int]:
    """Create sample departments."""
    departments = [
        {"name": "Salg", "departmentNumber": "10"},
        {"name": "IT", "departmentNumber": "20"},
        {"name": "Regnskap", "departmentNumber": "30"},
    ]
    ids = []
    for dept in departments:
        r = c.post("/department", json=dept)
        if "error" not in r:
            did = r.get("value", {}).get("id", 0)
            ids.append(did)
            log.info(f"  Department: {dept['name']} #{dept['departmentNumber']} (id={did})")
        else:
            log.warning(f"  Failed: {dept['name']} - {r.get('message', '')}")
    return ids


def create_invoices(c: TripletexClient, customer_ids: list[int], product_ids: list[int]) -> list[int]:
    """Create sample invoices (with orders)."""
    if len(customer_ids) < 2 or len(product_ids) < 2:
        log.warning("  Need customers and products to create invoices")
        return []

    invoices_spec = [
        {"customer_id": customer_ids[0], "product_id": product_ids[0], "count": 10,
         "date": "2026-03-01", "due": "2026-03-15"},
        {"customer_id": customer_ids[1], "product_id": product_ids[1], "count": 1,
         "date": "2026-03-05", "due": "2026-03-19"},
        {"customer_id": customer_ids[2], "product_id": product_ids[2], "count": 5,
         "date": "2026-03-10", "due": "2026-03-24"},
    ]

    ids = []
    for spec in invoices_spec:
        # Create order
        order_body = {
            "customer": {"id": spec["customer_id"]},
            "orderDate": spec["date"],
            "deliveryDate": spec["date"],
            "orderLines": [
                {"product": {"id": spec["product_id"]}, "count": spec["count"]}
            ],
        }
        order_r = c.post("/order", json=order_body)
        if "error" in order_r:
            log.warning(f"  Order failed: {order_r.get('message', '')}")
            continue
        order_id = order_r.get("value", {}).get("id", 0)

        # Create invoice
        inv_body = {
            "invoiceDate": spec["date"],
            "invoiceDueDate": spec["due"],
            "orders": [{"id": order_id}],
        }
        inv_r = c.post("/invoice", json=inv_body)
        if "error" not in inv_r:
            inv_id = inv_r.get("value", {}).get("id", 0)
            ids.append(inv_id)
            log.info(f"  Invoice: cust={spec['customer_id']} x{spec['count']} (id={inv_id})")
        else:
            log.warning(f"  Invoice failed: {inv_r.get('message', '')}")

    return ids


def create_travel_expenses(c: TripletexClient, employee_ids: list[int]) -> list[int]:
    """Create sample travel expenses."""
    if not employee_ids:
        log.warning("  Need employees to create travel expenses")
        return []

    expenses = [
        {"emp_id": employee_ids[0], "title": "Kundebesøk Oslo",
         "departure": "2026-03-03", "return": "2026-03-04"},
        {"emp_id": employee_ids[1], "title": "Konferanse Bergen",
         "departure": "2026-03-10", "return": "2026-03-12"},
        {"emp_id": employee_ids[0], "title": "Messestand Trondheim",
         "departure": "2026-03-15", "return": "2026-03-17"},
    ]

    ids = []
    for exp in expenses:
        r = c.post("/travelExpense", json={
            "employee": {"id": exp["emp_id"]},
            "title": exp["title"],
            "travelDetails": {
                "departureDate": exp["departure"],
                "returnDate": exp["return"],
            },
        })
        if "error" not in r:
            eid = r.get("value", {}).get("id", 0)
            ids.append(eid)
            log.info(f"  Travel: {exp['title']} (id={eid})")
        else:
            log.warning(f"  Failed: {exp['title']} - {r.get('message', '')}")
    return ids


def create_projects(c: TripletexClient, customer_ids: list[int], manager_id: int) -> list[int]:
    """Create sample projects. Requires customer IDs and a project manager (employee) ID."""
    if not customer_ids or not manager_id:
        log.warning("  Need customers and a manager to create projects")
        return []

    projects = [
        {"name": "Nettside Redesign", "customer_id": customer_ids[0],
         "startDate": "2026-03-01"},
        {"name": "ERP Implementering", "customer_id": customer_ids[1],
         "startDate": "2026-04-01"},
    ]

    ids = []
    for proj in projects:
        r = c.post("/project", json={
            "name": proj["name"],
            "customer": {"id": proj["customer_id"]},
            "projectManager": {"id": manager_id},
            "startDate": proj["startDate"],
        })
        if "error" not in r:
            pid = r.get("value", {}).get("id", 0)
            ids.append(pid)
            log.info(f"  Project: {proj['name']} (id={pid})")
        else:
            log.warning(f"  Failed: {proj['name']} - {r.get('message', '')}")
    return ids


def create_contacts(c: TripletexClient, customer_ids: list[int]) -> list[int]:
    """Create contact persons for customers."""
    if not customer_ids:
        return []

    contacts = [
        {"firstName": "Per", "lastName": "Olsen", "email": "per@nordvik.no",
         "customer_id": customer_ids[0]},
        {"firstName": "Lise", "lastName": "Bakke", "email": "lise@bergenit.no",
         "customer_id": customer_ids[1]},
    ]

    ids = []
    for cont in contacts:
        r = c.post("/contact", json={
            "firstName": cont["firstName"],
            "lastName": cont["lastName"],
            "email": cont["email"],
            "customer": {"id": cont["customer_id"]},
        })
        if "error" not in r:
            cid = r.get("value", {}).get("id", 0)
            ids.append(cid)
            log.info(f"  Contact: {cont['firstName']} {cont['lastName']} (id={cid})")
        else:
            log.warning(f"  Failed: {cont['firstName']} - {r.get('message', '')}")
    return ids


def populate_all(c: TripletexClient):
    """Populate the sandbox with a full set of test data."""
    print("\n=== Populating Tripletex Sandbox ===\n")

    enable_modules(c)
    setup_bank_account(c)

    # Get or create a base department (employees require one)
    dept_result = c.get("/department", params={"fields": "id,name", "count": 1})
    existing_depts = dept_result.get("values", [])
    if existing_depts:
        base_dept_id = existing_depts[0]["id"]
        log.info(f"Using existing department id={base_dept_id}")
    else:
        r = c.post("/department", json={"name": "Generell", "departmentNumber": "1"})
        base_dept_id = r.get("value", {}).get("id", 0)

    print("\n--- Departments ---")
    dept_ids = create_departments(c)

    print("\n--- Employees ---")
    emp_ids = create_employees(c, base_dept_id)

    print("\n--- Customers ---")
    cust_ids = create_customers(c)

    print("\n--- Products ---")
    prod_ids = create_products(c)

    print("\n--- Contacts ---")
    cont_ids = create_contacts(c, cust_ids)

    print("\n--- Projects ---")
    # Use account owner as project manager (they have PM permissions by default)
    all_emps = c.get("/employee", params={"fields": "id", "count": 1})
    owner_id = all_emps.get("values", [{}])[0].get("id", 0)
    proj_ids = create_projects(c, cust_ids, owner_id)

    print("\n--- Invoices ---")
    inv_ids = create_invoices(c, cust_ids, prod_ids)

    print("\n--- Travel Expenses ---")
    travel_ids = create_travel_expenses(c, emp_ids)

    print("\n=== Summary ===")
    print(f"  Employees:       {len(emp_ids)}")
    print(f"  Customers:       {len(cust_ids)}")
    print(f"  Products:        {len(prod_ids)}")
    print(f"  Departments:     {len(dept_ids)}")
    print(f"  Contacts:        {len(cont_ids)}")
    print(f"  Projects:        {len(proj_ids)}")
    print(f"  Invoices:        {len(inv_ids)}")
    print(f"  Travel Expenses: {len(travel_ids)}")
    total = (len(emp_ids) + len(cust_ids) + len(prod_ids) + len(dept_ids)
             + len(cont_ids) + len(proj_ids) + len(inv_ids) + len(travel_ids))
    print(f"  Total:           {total}")
    print(f"\n  API calls: {c._call_count}, errors: {c._error_count}")


def clean_sandbox(c: TripletexClient):
    """Delete all entities from the sandbox (best-effort)."""
    print("\n=== Cleaning Sandbox ===\n")

    # Delete in reverse dependency order
    for entity_type in ["travelExpense", "invoice", "project", "contact",
                         "department", "product", "customer", "employee"]:
        result = c.get(f"/{entity_type}", params={"fields": "id", "count": 1000})
        entities = result.get("values", [])
        if entities:
            log.info(f"Deleting {len(entities)} {entity_type}(s)...")
            for e in entities:
                r = c.delete(f"/{entity_type}/{e['id']}")
                if "error" in r:
                    log.warning(f"  Could not delete {entity_type}/{e['id']}: {r.get('message', '')[:60]}")
        else:
            log.info(f"No {entity_type} to delete")

    print(f"\n  API calls: {c._call_count}, errors: {c._error_count}")


def main():
    parser = argparse.ArgumentParser(description="Populate Tripletex Sandbox")
    parser.add_argument("--clean", action="store_true", help="Delete all entities first")
    args = parser.parse_args()

    c = get_client()

    # Test connectivity
    test = c.get("/employee", params={"count": 1, "fields": "id"})
    if "error" in test:
        print(f"Cannot connect to sandbox: {test}")
        sys.exit(1)

    if args.clean:
        clean_sandbox(c)

    populate_all(c)


if __name__ == "__main__":
    main()
