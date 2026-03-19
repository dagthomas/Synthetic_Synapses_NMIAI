"""
Comprehensive live API test for ALL Tripletex tools.
Tests every tool against the real sandbox API.

Usage:
    python test_all_tools.py
"""
import json
import os
import sys
import time
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tripletex_client import TripletexClient
from tools import build_all_tools

BASE_URL = os.environ["TRIPLETEX_BASE_URL"]
SESSION_TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]

client = TripletexClient(BASE_URL, SESSION_TOKEN)

# Build all tools as dict for easy access
from tools.employees import build_employee_tools
from tools.customers import build_customer_tools
from tools.products import build_product_tools
from tools.invoicing import build_invoicing_tools
from tools.travel import build_travel_tools
from tools.projects import build_project_tools
from tools.departments import build_department_tools
from tools.ledger import build_ledger_tools
from tools.contacts import build_contact_tools
from tools.employment import build_employment_tools
from tools.bank import build_bank_tools
from tools.supplier import build_supplier_tools
from tools.address import build_address_tools
from tools.balance import build_balance_tools
from tools.common import build_common_tools

employee_tools = build_employee_tools(client)
customer_tools = build_customer_tools(client)
product_tools = build_product_tools(client)
invoicing_tools = build_invoicing_tools(client)
travel_tools = build_travel_tools(client)
project_tools = build_project_tools(client)
department_tools = build_department_tools(client)
ledger_tools = build_ledger_tools(client)
contact_tools = build_contact_tools(client)
employment_tools = build_employment_tools(client)
bank_tools = build_bank_tools(client)
supplier_tools = build_supplier_tools(client)
address_tools = build_address_tools(client)
balance_tools = build_balance_tools(client)
common_tools = build_common_tools(client)

today = date.today().isoformat()
tomorrow = (date.today() + timedelta(days=1)).isoformat()
yesterday = (date.today() - timedelta(days=1)).isoformat()
month_ago = (date.today() - timedelta(days=30)).isoformat()

results = []

def run_test(test_name, func, *args, **kwargs):
    """Run a test and record result."""
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        is_error = isinstance(result, dict) and result.get("error")
        status = "FAIL" if is_error else "OK"
        error_msg = result.get("message", "") if is_error else ""
        status_code = result.get("status_code", "") if is_error else ""
        results.append({
            "name": test_name,
            "status": status,
            "time": elapsed,
            "error": f"[{status_code}] {error_msg}" if is_error else "",
            "result": result,
        })
        icon = "FAIL" if is_error else " OK "
        print(f"  [{icon}] {test_name} ({elapsed:.2f}s){f' -> {error_msg[:100]}' if is_error else ''}")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        results.append({
            "name": test_name,
            "status": "EXCEPTION",
            "time": elapsed,
            "error": str(e),
            "result": None,
        })
        print(f"  [EXCP] {test_name} ({elapsed:.2f}s) -> {str(e)[:120]}")
        return None


def get_id(result, key="value"):
    """Extract ID from API response."""
    if not result:
        return 0
    if isinstance(result, dict):
        val = result.get(key, result)
        if isinstance(val, dict):
            return val.get("id", 0)
        if isinstance(val, list) and val:
            return val[0].get("id", 0)
    return 0


def get_values(result):
    """Extract values list from API response."""
    if not result:
        return []
    return result.get("values", [])


print("=" * 70)
print("TRIPLETEX TOOL TEST SUITE")
print(f"Base URL: {BASE_URL}")
print(f"Date: {today}")
print("=" * 70)

# ============================================================
# 1. EMPLOYEES
# ============================================================
print("\n--- EMPLOYEES ---")

emp_result = run_test("create_employee", employee_tools["create_employee"],
    firstName="Test", lastName="Testesen", email=f"test.{int(time.time())}@example.org")
emp_id = get_id(emp_result)

run_test("search_employees", employee_tools["search_employees"],
    firstName="Test")

if emp_id:
    run_test("update_employee", employee_tools["update_employee"],
        employee_id=emp_id, firstName="TestUpdated")

# ============================================================
# 2. CUSTOMERS
# ============================================================
print("\n--- CUSTOMERS ---")

cust_result = run_test("create_customer", customer_tools["create_customer"],
    name="Test Kunde AS", email="kunde@example.org", organizationNumber="999888777")
cust_id = get_id(cust_result)

run_test("search_customers", customer_tools["search_customers"],
    name="Test Kunde")

if cust_id:
    run_test("update_customer", customer_tools["update_customer"],
        customer_id=cust_id, name="Test Kunde Updated AS")

# ============================================================
# 3. PRODUCTS
# ============================================================
print("\n--- PRODUCTS ---")

ts = int(time.time())
prod_result = run_test("create_product", product_tools["create_product"],
    name=f"Testprodukt {ts}", priceExcludingVatCurrency=100.0, priceIncludingVatCurrency=125.0)
prod_id = get_id(prod_result)

run_test("search_products", product_tools["search_products"],
    name="Testprodukt")

# ============================================================
# 4. INVOICING (order -> invoice -> payment -> credit note)
# ============================================================
print("\n--- INVOICING ---")

if cust_id and prod_id:
    order_lines = json.dumps([{"product_id": prod_id, "count": 2}])
    order_result = run_test("create_order", invoicing_tools["create_order"],
        customer_id=cust_id, deliveryDate=today, orderLines=order_lines)
    order_id = get_id(order_result)

    if order_id:
        inv_result = run_test("create_invoice", invoicing_tools["create_invoice"],
            invoiceDate=today, invoiceDueDate=tomorrow, order_id=order_id)
        inv_id = get_id(inv_result)

        if inv_id:
            run_test("register_payment", invoicing_tools["register_payment"],
                invoice_id=inv_id, amount=250.0, paymentDate=today)

            # Create another invoice for credit note test
            order_lines2 = json.dumps([{"product_id": prod_id, "count": 1}])
            order2_result = run_test("create_order (for credit)", invoicing_tools["create_order"],
                customer_id=cust_id, deliveryDate=today, orderLines=order_lines2)
            order2_id = get_id(order2_result)
            if order2_id:
                inv2_result = run_test("create_invoice (for credit)", invoicing_tools["create_invoice"],
                    invoiceDate=today, invoiceDueDate=tomorrow, order_id=order2_id)
                inv2_id = get_id(inv2_result)
                if inv2_id:
                    run_test("create_credit_note", invoicing_tools["create_credit_note"],
                        invoice_id=inv2_id, date=today)
else:
    print("  [SKIP] Invoicing tests (missing customer or product)")

# ============================================================
# 5. CONTACTS
# ============================================================
print("\n--- CONTACTS ---")

if cust_id:
    contact_result = run_test("create_contact", contact_tools["create_contact"],
        firstName="Kontakt", lastName="Person", email="kontakt@example.org", customer_id=cust_id)

run_test("search_contacts", contact_tools["search_contacts"],
    firstName="Kontakt")

# ============================================================
# 6. PROJECTS
# ============================================================
print("\n--- PROJECTS ---")

if cust_id:
    proj_result = run_test("create_project", project_tools["create_project"],
        name="Testprosjekt", customer_id=cust_id, startDate=today)
    proj_id = get_id(proj_result)
else:
    print("  [SKIP] create_project (no customer)")

# ============================================================
# 7. DEPARTMENTS
# ============================================================
print("\n--- DEPARTMENTS ---")

dept_result = run_test("create_department", department_tools["create_department"],
    name=f"Testavdeling {ts}", departmentNumber=f"T{ts % 10000}")

# ============================================================
# 8. EMPLOYMENT
# ============================================================
print("\n--- EMPLOYMENT ---")

if emp_id:
    empl_result = run_test("create_employment", employment_tools["create_employment"],
        employee_id=emp_id, startDate=today)

    run_test("search_employments", employment_tools["search_employments"],
        employee_id=emp_id)
else:
    print("  [SKIP] Employment tests (no employee)")

# ============================================================
# 9. TRAVEL EXPENSE
# ============================================================
print("\n--- TRAVEL EXPENSE ---")

if emp_id:
    travel_result = run_test("create_travel_expense", travel_tools["create_travel_expense"],
        employee_id=emp_id, title="Testtur Oslo", departureDate=today, returnDate=tomorrow,
        description="Testreise")
    travel_id = get_id(travel_result)

    run_test("search_travel_expenses", travel_tools["search_travel_expenses"],
        employee_id=emp_id)

    if travel_id:
        run_test("delete_travel_expense", travel_tools["delete_travel_expense"],
            travel_expense_id=travel_id)
else:
    print("  [SKIP] Travel tests (no employee)")

# ============================================================
# 10. LEDGER
# ============================================================
print("\n--- LEDGER ---")

run_test("get_ledger_accounts", ledger_tools["get_ledger_accounts"])
run_test("get_ledger_accounts (by number)", ledger_tools["get_ledger_accounts"], number="1920")

run_test("get_ledger_postings", ledger_tools["get_ledger_postings"],
    dateFrom=month_ago, dateTo=today)

voucher_result = run_test("create_voucher", ledger_tools["create_voucher"],
    date=today, description="Test korrigering",
    postings=json.dumps([
        {"accountNumber": 1920, "amount": 100},
        {"accountNumber": 7700, "amount": -100},
    ]))
voucher_id = get_id(voucher_result)

if voucher_id:
    run_test("delete_voucher", ledger_tools["delete_voucher"], voucher_id=voucher_id)

# ============================================================
# 11. BANK
# ============================================================
print("\n--- BANK ---")

bank_result = run_test("search_bank_accounts", bank_tools["search_bank_accounts"])
bank_accounts = get_values(bank_result)
bank_account_id = bank_accounts[0]["id"] if bank_accounts else 0

run_test("search_bank_reconciliations", bank_tools["search_bank_reconciliations"])

if bank_account_id:
    run_test("get_last_bank_reconciliation", bank_tools["get_last_bank_reconciliation"],
        accountId=bank_account_id)
    run_test("get_last_closed_bank_reconciliation", bank_tools["get_last_closed_bank_reconciliation"],
        accountId=bank_account_id)

run_test("search_bank_statements", bank_tools["search_bank_statements"])

# Bank reconciliation create requires a valid accounting period
# Tripletex doesn't expose /period endpoint; get period from year-end or reconciliation
year_end_result = balance_tools["search_year_ends"]()
year_ends = get_values(year_end_result)
period_id = 0
if year_ends:
    ye = year_ends[0]
    # Year-end has accountingPeriod reference
    ap = ye.get("accountingPeriod", {})
    period_id = ap.get("id", 0) if isinstance(ap, dict) else 0

if bank_account_id and period_id:
    recon_result = run_test("create_bank_reconciliation", bank_tools["create_bank_reconciliation"],
        accountId=bank_account_id, accountingPeriodId=period_id)
    recon_id = get_id(recon_result)

    if recon_id:
        run_test("get_bank_reconciliation_match_count", bank_tools["get_bank_reconciliation_match_count"],
            bankReconciliationId=recon_id)
        run_test("close_bank_reconciliation", bank_tools["close_bank_reconciliation"],
            reconciliationId=recon_id)
        # Try delete (may fail if closed)
        run_test("delete_bank_reconciliation", bank_tools["delete_bank_reconciliation"],
            reconciliationId=recon_id)

# Test adjust (needs an open reconciliation)
if bank_account_id and period_id:
    recon2_result = run_test("create_bank_reconciliation (for adjust)", bank_tools["create_bank_reconciliation"],
        accountId=bank_account_id, accountingPeriodId=period_id)
    recon2_id = get_id(recon2_result)
    if recon2_id:
        run_test("adjust_bank_reconciliation", bank_tools["adjust_bank_reconciliation"],
            reconciliationId=recon2_id,
            adjustments=json.dumps([{"amount": 50.0, "date": today, "description": "Test adjust"}]))

# ============================================================
# 12. SUPPLIER
# ============================================================
print("\n--- SUPPLIER ---")

supp_result = run_test("create_supplier", supplier_tools["create_supplier"],
    name="Test Leverandør AS", email="leverandor@example.org")
supp_id = get_id(supp_result)

run_test("search_suppliers", supplier_tools["search_suppliers"], name="Test Leverandør")

if supp_id:
    run_test("update_supplier", supplier_tools["update_supplier"],
        supplier_id=supp_id, name="Test Leverandør Updated AS")

# ============================================================
# 13. ADDRESS
# ============================================================
print("\n--- ADDRESS ---")

addr_result = run_test("search_delivery_addresses", address_tools["search_delivery_addresses"])
addresses = get_values(addr_result)
if addresses:
    addr_id = addresses[0]["id"]
    run_test("update_delivery_address", address_tools["update_delivery_address"],
        address_id=addr_id, addressLine1="Testgata 1", postalCode="0150", city="Oslo")
else:
    print("  [SKIP] update_delivery_address (no addresses found)")

# ============================================================
# 14. BALANCE / PERIODS / CURRENCY / COMPANY
# ============================================================
print("\n--- BALANCE & COMPANY ---")

run_test("get_balance_sheet", balance_tools["get_balance_sheet"],
    dateFrom=month_ago, dateTo=today)

run_test("search_voucher_types", balance_tools["search_voucher_types"])
run_test("search_year_ends", balance_tools["search_year_ends"])
run_test("search_currencies", balance_tools["search_currencies"])
run_test("search_currencies (NOK)", balance_tools["search_currencies"], code="NOK")
run_test("get_company_info", balance_tools["get_company_info"])

# ============================================================
# 15. COMMON (get_entity_by_id, delete_entity)
# ============================================================
print("\n--- COMMON ---")

if emp_id:
    run_test("get_entity_by_id (employee)", common_tools["get_entity_by_id"],
        entity_type="employee", entity_id=emp_id)
if cust_id:
    run_test("get_entity_by_id (customer)", common_tools["get_entity_by_id"],
        entity_type="customer", entity_id=cust_id)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

ok_count = sum(1 for r in results if r["status"] == "OK")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
excp_count = sum(1 for r in results if r["status"] == "EXCEPTION")
total = len(results)
total_time = sum(r["time"] for r in results)

print(f"Total: {total} tests | OK: {ok_count} | FAIL: {fail_count} | EXCEPTION: {excp_count}")
print(f"Total API time: {total_time:.1f}s | API calls: {client._call_count} | Errors: {client._error_count}")
print()

if fail_count or excp_count:
    print("FAILURES:")
    for r in results:
        if r["status"] in ("FAIL", "EXCEPTION"):
            print(f"  {r['status']:4s} | {r['name']}")
            print(f"       {r['error'][:200]}")
    print()

print(f"Pass rate: {ok_count}/{total} ({100*ok_count/total:.0f}%)")
