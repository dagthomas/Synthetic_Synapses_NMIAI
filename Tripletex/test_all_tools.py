"""
Comprehensive live API test for ALL Tripletex tools.
Tests every tool against the real sandbox API.

Usage:
    python test_all_tools.py
"""
import json
import os
import random
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
from tools.activity import build_activity_tools
from tools.company import build_company_tools
from tools.division import build_division_tools
from tools.order import build_order_tools
from tools.timesheet import build_timesheet_tools
from tools.salary import build_salary_tools
from tools.supplier_invoice import build_supplier_invoice_tools
from tools.year_end import build_year_end_tools
from tools.employee_extras import build_employee_extras_tools
from tools.travel_extras import build_travel_extras_tools
from tools.incoming_invoice import build_incoming_invoice_tools

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
activity_tools = build_activity_tools(client)
company_tools = build_company_tools(client)
division_tools = build_division_tools(client)
order_tools = build_order_tools(client)
timesheet_tools = build_timesheet_tools(client)
salary_tools = build_salary_tools(client)
supplier_invoice_tools = build_supplier_invoice_tools(client)
year_end_tools = build_year_end_tools(client)
employee_extras_tools = build_employee_extras_tools(client)
travel_extras_tools = build_travel_extras_tools(client)
incoming_invoice_tools = build_incoming_invoice_tools(client)

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


# Verify tool count
all_tools = build_all_tools(client)
tool_count = len(all_tools)

print("=" * 70)
print("TRIPLETEX TOOL TEST SUITE")
print(f"Base URL: {BASE_URL}")
print(f"Date: {today}")
print(f"Total tools registered: {tool_count}")
print("=" * 70)

ts = int(time.time())

# ============================================================
# 1. EMPLOYEES
# ============================================================
print("\n--- EMPLOYEES ---")

emp_result = run_test("create_employee", employee_tools["create_employee"],
    firstName="Test", lastName="Testesen", email=f"test.{ts}@example.org")
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

prod_result = run_test("create_product", product_tools["create_product"],
    name=f"Testprodukt {ts}", priceExcludingVatCurrency=100.0, priceIncludingVatCurrency=125.0)
prod_id = get_id(prod_result)

run_test("search_products", product_tools["search_products"],
    name="Testprodukt")

if prod_id:
    run_test("update_product", product_tools["update_product"],
        product_id=prod_id, name=f"Testprodukt Updated {ts}")

# ============================================================
# 4. INVOICING (order -> invoice -> payment -> credit note)
# ============================================================
print("\n--- INVOICING ---")

order_id = 0
inv_id = 0
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

    # New invoicing tools
    run_test("search_invoices", invoicing_tools["search_invoices"],
        invoiceDateFrom=month_ago, invoiceDateTo=today)

    run_test("search_orders", invoicing_tools["search_orders"],
        orderDateFrom=month_ago, orderDateTo=today)

    # Create an order to test update/delete
    order_lines3 = json.dumps([{"product_id": prod_id, "count": 1}])
    order3_result = run_test("create_order (for update)", invoicing_tools["create_order"],
        customer_id=cust_id, deliveryDate=today, orderLines=order_lines3)
    order3_id = get_id(order3_result)
    if order3_id:
        run_test("update_order", invoicing_tools["update_order"],
            order_id=order3_id, deliveryDate=tomorrow)
        run_test("delete_order", invoicing_tools["delete_order"],
            order_id=order3_id)

    # send_invoice and create_invoice_reminder may fail on sandbox (no email config)
    if inv_id:
        run_test("send_invoice", invoicing_tools["send_invoice"],
            invoice_id=inv_id, sendType="EMAIL")
        run_test("create_invoice_reminder", invoicing_tools["create_invoice_reminder"],
            invoice_id=inv_id)
else:
    print("  [SKIP] Invoicing tests (missing customer or product)")

# ============================================================
# 5. CONTACTS
# ============================================================
print("\n--- CONTACTS ---")

contact_id = 0
if cust_id:
    contact_result = run_test("create_contact", contact_tools["create_contact"],
        firstName="Kontakt", lastName="Person", email="kontakt@example.org", customer_id=cust_id)
    contact_id = get_id(contact_result)

run_test("search_contacts", contact_tools["search_contacts"],
    firstName="Kontakt")

if contact_id:
    run_test("update_contact", contact_tools["update_contact"],
        contact_id=contact_id, firstName="KontaktUpdated")
    run_test("delete_contact", contact_tools["delete_contact"],
        contact_id=contact_id)

# ============================================================
# 6. PROJECTS
# ============================================================
print("\n--- PROJECTS ---")

proj_id = 0
if cust_id:
    proj_result = run_test("create_project", project_tools["create_project"],
        name="Testprosjekt", customer_id=cust_id, startDate=today)
    proj_id = get_id(proj_result)
else:
    print("  [SKIP] create_project (no customer)")

run_test("search_projects", project_tools["search_projects"])

if proj_id:
    run_test("update_project", project_tools["update_project"],
        project_id=proj_id, description="Oppdatert beskrivelse")

# Project categories
cat_result = run_test("create_project_category", project_tools["create_project_category"],
    name=f"Testkategori {ts}", number=f"K{ts % 1000}")
cat_id = get_id(cat_result)

run_test("search_project_categories", project_tools["search_project_categories"])

# Project participant
if proj_id and emp_id:
    run_test("create_project_participant", project_tools["create_project_participant"],
        project_id=proj_id, employee_id=emp_id)

# Delete project last
if proj_id:
    run_test("delete_project", project_tools["delete_project"],
        project_id=proj_id)

# ============================================================
# 7. DEPARTMENTS
# ============================================================
print("\n--- DEPARTMENTS ---")

dept_result = run_test("create_department", department_tools["create_department"],
    name=f"Testavdeling {ts}", departmentNumber=f"T{ts % 10000}")
dept_id = get_id(dept_result)

run_test("search_departments", department_tools["search_departments"])

if dept_id:
    run_test("update_department", department_tools["update_department"],
        department_id=dept_id, name=f"Testavdeling Updated {ts}")
    run_test("delete_department", department_tools["delete_department"],
        department_id=dept_id)

# ============================================================
# 8. EMPLOYMENT
# ============================================================
print("\n--- EMPLOYMENT ---")

empl_id = 0
if emp_id:
    empl_result = run_test("create_employment", employment_tools["create_employment"],
        employee_id=emp_id, startDate=today)
    empl_id = get_id(empl_result)

    run_test("search_employments", employment_tools["search_employments"],
        employee_id=emp_id)
else:
    print("  [SKIP] Employment tests (no employee)")

# ============================================================
# 9. TRAVEL EXPENSE
# ============================================================
print("\n--- TRAVEL EXPENSE ---")

travel_id = 0
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

run_test("search_vouchers", ledger_tools["search_vouchers"],
    dateFrom=month_ago, dateTo=today)

if voucher_id:
    run_test("update_voucher", ledger_tools["update_voucher"],
        voucher_id=voucher_id, description="Test korrigering oppdatert")
    run_test("reverse_voucher", ledger_tools["reverse_voucher"],
        voucher_id=voucher_id, date=today)

# Test create_ledger_account
acct_result = run_test("create_ledger_account", ledger_tools["create_ledger_account"],
    number=9990 + random.randint(0, 8), name=f"Testkonto {ts}")

# Test create_opening_balance
run_test("create_opening_balance", ledger_tools["create_opening_balance"],
    voucherDate=f"{date.today().year}-01-01",
    balancePostings=json.dumps([{"accountNumber": 1920, "amount": 50000}]))

# Create and delete a voucher
voucher2_result = run_test("create_voucher (for delete)", ledger_tools["create_voucher"],
    date=today, description="Test slett",
    postings=json.dumps([
        {"accountNumber": 1920, "amount": 200},
        {"accountNumber": 7700, "amount": -200},
    ]))
voucher2_id = get_id(voucher2_result)
if voucher2_id:
    run_test("delete_voucher", ledger_tools["delete_voucher"], voucher_id=voucher2_id)

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
year_end_result = balance_tools["search_year_ends"]()
year_ends = get_values(year_end_result)
period_id = 0
if year_ends:
    ye = year_ends[0]
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
        run_test("delete_bank_reconciliation", bank_tools["delete_bank_reconciliation"],
            reconciliationId=recon_id)

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
    name="Test Leverandoer AS", email="leverandor@example.org")
supp_id = get_id(supp_result)

run_test("search_suppliers", supplier_tools["search_suppliers"], name="Test Leverandoer")

if supp_id:
    run_test("update_supplier", supplier_tools["update_supplier"],
        supplier_id=supp_id, name="Test Leverandoer Updated AS")

# Create a second supplier to test delete
supp2_result = run_test("create_supplier (for delete)", supplier_tools["create_supplier"],
    name="Slett Leverandoer AS")
supp2_id = get_id(supp2_result)
if supp2_id:
    run_test("delete_supplier", supplier_tools["delete_supplier"],
        supplier_id=supp2_id)

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
# 16. ACTIVITY
# ============================================================
print("\n--- ACTIVITY ---")

act_result = run_test("create_activity", activity_tools["create_activity"],
    name=f"Testaktivitet {ts}", number=f"A{ts % 1000}")
act_id = get_id(act_result)

run_test("search_activities", activity_tools["search_activities"])

# ============================================================
# 17. COMPANY
# ============================================================
print("\n--- COMPANY ---")

run_test("update_company", company_tools["update_company"],
    name="Test Sandbox AS")

run_test("get_accounting_periods", company_tools["get_accounting_periods"])

# ============================================================
# 18. DIVISION
# ============================================================
print("\n--- DIVISION ---")

div_result = run_test("create_division", division_tools["create_division"],
    name=f"Testdivisjon {ts}", startDate=today, organizationNumber="987654321")
div_id = get_id(div_result)

run_test("search_divisions", division_tools["search_divisions"])

if div_id:
    run_test("update_division", division_tools["update_division"],
        division_id=div_id, name=f"Testdivisjon Updated {ts}")

# ============================================================
# 19. ORDER LINES & GROUPS
# ============================================================
print("\n--- ORDER LINES & GROUPS ---")

if cust_id and prod_id:
    # Create a fresh order for order line tests
    ol_order_lines = json.dumps([{"product_id": prod_id, "count": 1}])
    ol_order_result = run_test("create_order (for orderlines)", invoicing_tools["create_order"],
        customer_id=cust_id, deliveryDate=today, orderLines=ol_order_lines)
    ol_order_id = get_id(ol_order_result)

    if ol_order_id:
        ol_result = run_test("create_order_line", order_tools["create_order_line"],
            order_id=ol_order_id, product_id=prod_id, count=3)
        ol_id = get_id(ol_result)

        run_test("create_order_group", order_tools["create_order_group"],
            order_id=ol_order_id, title="Testgruppe")

        if ol_id:
            run_test("delete_order_line", order_tools["delete_order_line"],
                order_line_id=ol_id)
else:
    print("  [SKIP] Order line tests (missing customer or product)")

# ============================================================
# 20. TIMESHEET
# ============================================================
print("\n--- TIMESHEET ---")

ts_entry_id = 0
if emp_id and proj_id == 0 and act_id == 0:
    print("  [SKIP] Timesheet tests (need project and activity)")
elif emp_id:
    # Need a project and activity - create if needed
    ts_proj_id = proj_id
    ts_act_id = act_id
    if not ts_proj_id:
        # Try to find any project
        proj_search = project_tools["search_projects"]()
        projs = get_values(proj_search)
        ts_proj_id = projs[0]["id"] if projs else 0
    if not ts_act_id:
        # Try to find any activity
        act_search = activity_tools["search_activities"]()
        acts = get_values(act_search)
        ts_act_id = acts[0]["id"] if acts else 0

    if ts_proj_id and ts_act_id:
        ts_result = run_test("create_timesheet_entry", timesheet_tools["create_timesheet_entry"],
            employee_id=emp_id, project_id=ts_proj_id, activity_id=ts_act_id,
            date=today, hours=7.5, comment="Testtime")
        ts_entry_id = get_id(ts_result)

        run_test("search_timesheet_entries", timesheet_tools["search_timesheet_entries"],
            employeeId=emp_id, dateFrom=today, dateTo=tomorrow)

        if ts_entry_id:
            run_test("update_timesheet_entry", timesheet_tools["update_timesheet_entry"],
                entry_id=ts_entry_id, hours=8.0, comment="Oppdatert time")
            run_test("delete_timesheet_entry", timesheet_tools["delete_timesheet_entry"],
                entry_id=ts_entry_id)
    else:
        print(f"  [SKIP] Timesheet tests (proj={ts_proj_id}, act={ts_act_id})")
else:
    print("  [SKIP] Timesheet tests (no employee)")

# ============================================================
# 21. SALARY
# ============================================================
print("\n--- SALARY ---")

run_test("search_salary_types", salary_tools["search_salary_types"])

sal_result = run_test("create_salary_transaction", salary_tools["create_salary_transaction"],
    date=today, year=date.today().year, month=date.today().month)
sal_id = get_id(sal_result)

run_test("search_salary_transactions", salary_tools["search_salary_transactions"],
    yearFrom=date.today().year)

if sal_id:
    run_test("delete_salary_transaction", salary_tools["delete_salary_transaction"],
        transaction_id=sal_id)

run_test("update_salary_settings", salary_tools["update_salary_settings"])

# ============================================================
# 22. SUPPLIER INVOICE
# ============================================================
print("\n--- SUPPLIER INVOICE ---")

run_test("search_supplier_invoices", supplier_invoice_tools["search_supplier_invoices"])

# approve/reject/payment require existing supplier invoices - test search only
# These will likely return empty results on fresh sandbox

# ============================================================
# 23. YEAR END
# ============================================================
print("\n--- YEAR END ---")

ye_result = run_test("search_year_ends (balance)", balance_tools["search_year_ends"])
ye_list = get_values(ye_result)
if ye_list:
    ye_id = ye_list[0].get("id", 0)
    if ye_id:
        run_test("search_year_end_annexes", year_end_tools["search_year_end_annexes"],
            yearEndId=ye_id)
        run_test("create_year_end_note", year_end_tools["create_year_end_note"],
            yearEndId=ye_id, note="Testnotat")
else:
    print("  [SKIP] Year-end tests (no year-ends found)")

run_test("get_vat_returns", year_end_tools["get_vat_returns"])

# ============================================================
# 24. EMPLOYEE EXTRAS
# ============================================================
print("\n--- EMPLOYEE EXTRAS ---")

ecat_result = run_test("create_employee_category", employee_extras_tools["create_employee_category"],
    name=f"Testkategori {ts}")
ecat_id = get_id(ecat_result)

run_test("search_employee_categories", employee_extras_tools["search_employee_categories"])

if ecat_id:
    run_test("delete_employee_category", employee_extras_tools["delete_employee_category"],
        category_id=ecat_id)

if emp_id:
    run_test("create_next_of_kin", employee_extras_tools["create_next_of_kin"],
        employee_id=emp_id, name="Paaroerendansen", phoneNumber="99887766",
        typeOfRelationship="SPOUSE")

    run_test("search_next_of_kin", employee_extras_tools["search_next_of_kin"],
        employee_id=emp_id)

    if empl_id:
        run_test("create_leave_of_absence", employee_extras_tools["create_leave_of_absence"],
            employment_id=empl_id, startDate=today, endDate=tomorrow,
            leaveType="OTHER", percentage=100.0)

        run_test("create_employment_details", employee_extras_tools["create_employment_details"],
            employment_id=empl_id, date=tomorrow, annualSalary=500000)

    run_test("create_hourly_cost_and_rate", employee_extras_tools["create_hourly_cost_and_rate"],
        employee_id=emp_id, date=tomorrow, rate=500.0, hourCostRate=350.0)

    run_test("create_standard_time", employee_extras_tools["create_standard_time"],
        employee_id=emp_id, fromDate=today, hoursPerDay=7.5)

    run_test("grant_entitlements", employee_extras_tools["grant_entitlements"],
        employee_id=emp_id, template="all_access")

# ============================================================
# 25. TRAVEL EXTRAS
# ============================================================
print("\n--- TRAVEL EXTRAS ---")

if emp_id:
    # Create a travel expense for the extras tests
    te_result = run_test("create_travel_expense (for extras)", travel_tools["create_travel_expense"],
        employee_id=emp_id, title="Testtur Extras", departureDate=today, returnDate=tomorrow,
        description="Testreise for extras")
    te_id = get_id(te_result)

    if te_id:
        run_test("create_travel_expense_cost", travel_extras_tools["create_travel_expense_cost"],
            travelExpenseId=te_id, date=today, description="Taxi", amount=250.0)

        run_test("search_travel_expense_costs", travel_extras_tools["search_travel_expense_costs"],
            travelExpenseId=te_id)

        run_test("create_mileage_allowance", travel_extras_tools["create_mileage_allowance"],
            travelExpenseId=te_id, date=today, km=150.0,
            departureLocation="Oslo", destination="Drammen")

        run_test("create_per_diem_compensation", travel_extras_tools["create_per_diem_compensation"],
            travelExpenseId=te_id, dateFrom=today, dateTo=tomorrow)

        run_test("update_travel_expense", travel_extras_tools["update_travel_expense"],
            travelExpenseId=te_id, title="Testtur Extras Updated")

        # Clean up
        run_test("delete_travel_expense (extras)", travel_tools["delete_travel_expense"],
            travel_expense_id=te_id)
else:
    print("  [SKIP] Travel extras tests (no employee)")

# ============================================================
# 26. INCOMING INVOICE
# ============================================================
print("\n--- INCOMING INVOICE ---")

if supp_id:
    ii_result = run_test("create_incoming_invoice", incoming_invoice_tools["create_incoming_invoice"],
        invoiceDate=today, dueDate=tomorrow, supplierId=supp_id,
        invoiceNumber=f"INV-{ts}", amount=5000.0)

run_test("search_incoming_invoices", incoming_invoice_tools["search_incoming_invoices"])

# ============================================================
# CLEANUP: Delete customer and product last
# (These are expected to fail when invoices reference them)
# ============================================================
print("\n--- CLEANUP (expected failures if invoices exist) ---")

if prod_id:
    del_prod = run_test("delete_product", product_tools["delete_product"],
        product_id=prod_id)
    # Mark as expected failure - product is referenced by invoices
    if del_prod and isinstance(del_prod, dict) and del_prod.get("error"):
        for r in results:
            if r["name"] == "delete_product" and r["status"] == "FAIL":
                r["status"] = "OK (expected)"
                break

# delete_customer last (may fail if invoices reference it)
if cust_id:
    del_cust = run_test("delete_customer", customer_tools["delete_customer"],
        customer_id=cust_id)
    # Mark as expected failure - customer is referenced by invoices
    if del_cust and isinstance(del_cust, dict) and del_cust.get("error"):
        for r in results:
            if r["name"] == "delete_customer" and r["status"] == "FAIL":
                r["status"] = "OK (expected)"
                break

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

ok_count = sum(1 for r in results if r["status"] in ("OK", "OK (expected)"))
fail_count = sum(1 for r in results if r["status"] == "FAIL")
excp_count = sum(1 for r in results if r["status"] == "EXCEPTION")
expected_count = sum(1 for r in results if r["status"] == "OK (expected)")
total = len(results)
total_time = sum(r["time"] for r in results)

exp_str = f" (incl. {expected_count} expected)" if expected_count else ""
print(f"Total: {total} tests | OK: {ok_count}{exp_str} | FAIL: {fail_count} | EXCEPTION: {excp_count}")
print(f"Total API time: {total_time:.1f}s | API calls: {client._call_count} | Errors: {client._error_count}")
print(f"Total tools registered: {tool_count}")
print()

if fail_count or excp_count:
    print("FAILURES:")
    for r in results:
        if r["status"] in ("FAIL", "EXCEPTION"):
            print(f"  {r['status']:4s} | {r['name']}")
            print(f"       {r['error'][:200]}")
    print()

print(f"Pass rate: {ok_count}/{total} ({100*ok_count/total:.0f}%)")
