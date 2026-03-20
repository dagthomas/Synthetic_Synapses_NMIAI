"""Run all Tripletex tools against the sandbox and return structured results."""

import inspect
import json
import time
from datetime import date, timedelta

from tripletex_client import TripletexClient
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


def _get_id(result, key="value"):
    if not result:
        return 0
    if isinstance(result, dict):
        val = result.get(key, result)
        if isinstance(val, dict):
            return val.get("id", 0)
        if isinstance(val, list) and val:
            return val[0].get("id", 0)
    return 0


def _get_values(result):
    if not result:
        return []
    return result.get("values", [])


def _run_tool(tool_name, func, *args, **kwargs):
    """Run a single tool and return result dict."""
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        is_error = isinstance(result, dict) and result.get("error")
        return {
            "tool": tool_name,
            "status": "FAIL" if is_error else "OK",
            "elapsed": round(elapsed, 3),
            "error": result.get("message", "")[:500] if is_error else "",
            "status_code": result.get("status_code", "") if is_error else "",
            "result": result,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "tool": tool_name,
            "status": "EXCEPTION",
            "elapsed": round(elapsed, 3),
            "error": str(e)[:500],
            "status_code": "",
            "result": None,
        }


def _safe_json(obj):
    """Convert result to JSON-safe dict, truncating very large values."""
    if obj is None:
        return None
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
        if len(s) > 20000:
            s = s[:20000] + "..."
        return json.loads(s) if not s.endswith("...") else {"_truncated": s}
    except Exception:
        return {"_raw": str(obj)[:5000]}


def _strip_result(r):
    """Convert result for streaming, keep full JSON."""
    out = dict(r)
    out["result_json"] = _safe_json(out.get("result"))
    del out["result"]
    return out


def _build_all_tool_sets(client):
    """Build all tool sets from all modules."""
    return {
        "employee": build_employee_tools(client),
        "customer": build_customer_tools(client),
        "product": build_product_tools(client),
        "invoicing": build_invoicing_tools(client),
        "travel": build_travel_tools(client),
        "project": build_project_tools(client),
        "department": build_department_tools(client),
        "ledger": build_ledger_tools(client),
        "contact": build_contact_tools(client),
        "employment": build_employment_tools(client),
        "bank": build_bank_tools(client),
        "supplier": build_supplier_tools(client),
        "address": build_address_tools(client),
        "balance": build_balance_tools(client),
        "common": build_common_tools(client),
        "activity": build_activity_tools(client),
        "company": build_company_tools(client),
        "division": build_division_tools(client),
        "order": build_order_tools(client),
        "timesheet": build_timesheet_tools(client),
        "salary": build_salary_tools(client),
        "supplier_invoice": build_supplier_invoice_tools(client),
        "year_end": build_year_end_tools(client),
        "employee_extras": build_employee_extras_tools(client),
        "travel_extras": build_travel_extras_tools(client),
        "incoming_invoice": build_incoming_invoice_tools(client),
    }



def _run_all_tests(tools, client):
    """Run all tool tests, yielding (raw_result, stripped_result) for each."""
    ts = int(time.time())
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    # ── 1. EMPLOYEES ──
    r = _run_tool("create_employee", tools["employee"]["create_employee"],
                  firstName="Test", lastName="Testesen",
                  email=f"test.{ts}@example.org")
    yield r, _strip_result(r)
    emp_id = _get_id(r["result"])

    r = _run_tool("search_employees", tools["employee"]["search_employees"],
                  firstName="Test")
    yield r, _strip_result(r)

    if emp_id:
        r = _run_tool("update_employee", tools["employee"]["update_employee"],
                      employee_id=emp_id, firstName="TestUpdated")
        yield r, _strip_result(r)

    # ── 2. CUSTOMERS ──
    r = _run_tool("create_customer", tools["customer"]["create_customer"],
                  name=f"Test Kunde {ts}", email="kunde@example.org")
    yield r, _strip_result(r)
    cust_id = _get_id(r["result"])

    r = _run_tool("search_customers", tools["customer"]["search_customers"],
                  name="Test Kunde")
    yield r, _strip_result(r)

    if cust_id:
        r = _run_tool("update_customer", tools["customer"]["update_customer"],
                      customer_id=cust_id, name=f"Test Kunde Updated {ts}")
        yield r, _strip_result(r)

        r = _run_tool("delete_customer", tools["customer"]["delete_customer"],
                      customer_id=cust_id)
        yield r, _strip_result(r)

    # Re-create customer (need it for later tests)
    r = _run_tool("create_customer (2)", tools["customer"]["create_customer"],
                  name=f"Test Kunde2 {ts}", email="kunde2@example.org")
    yield r, _strip_result(r)
    cust_id = _get_id(r["result"])

    # ── 3. PRODUCTS ──
    r = _run_tool("create_product", tools["product"]["create_product"],
                  name=f"Testprodukt {ts}", priceExcludingVatCurrency=100.0,
                  priceIncludingVatCurrency=125.0)
    yield r, _strip_result(r)
    prod_id = _get_id(r["result"])

    r = _run_tool("search_products", tools["product"]["search_products"],
                  name="Testprodukt")
    yield r, _strip_result(r)

    if prod_id:
        r = _run_tool("update_product", tools["product"]["update_product"],
                      product_id=prod_id, name=f"Testprodukt Updated {ts}")
        yield r, _strip_result(r)

    # ── 4. INVOICING ──
    order_id = 0
    inv_id = 0
    if cust_id and prod_id:
        order_lines = json.dumps([{"product_id": prod_id, "count": 2}])
        r = _run_tool("create_order", tools["invoicing"]["create_order"],
                      customer_id=cust_id, deliveryDate=today, orderLines=order_lines)
        yield r, _strip_result(r)
        order_id = _get_id(r["result"])

        if order_id:
            # search/update orders
            r = _run_tool("search_orders", tools["invoicing"]["search_orders"],
                          customerId=cust_id)
            yield r, _strip_result(r)

            r = _run_tool("update_order", tools["invoicing"]["update_order"],
                          order_id=order_id, deliveryDate=tomorrow)
            yield r, _strip_result(r)

            r = _run_tool("create_invoice", tools["invoicing"]["create_invoice"],
                          invoiceDate=today, invoiceDueDate=tomorrow, order_id=order_id)
            yield r, _strip_result(r)
            inv_id = _get_id(r["result"])

            if inv_id:
                r = _run_tool("search_invoices", tools["invoicing"]["search_invoices"],
                              invoiceDateFrom=month_ago, invoiceDateTo=today)
                yield r, _strip_result(r)

                r = _run_tool("register_payment", tools["invoicing"]["register_payment"],
                              invoice_id=inv_id, amount=250.0, paymentDate=today)
                yield r, _strip_result(r)

                r = _run_tool("send_invoice", tools["invoicing"]["send_invoice"],
                              invoice_id=inv_id)
                yield r, _strip_result(r)

            # Credit note flow
            order_lines2 = json.dumps([{"product_id": prod_id, "count": 1}])
            r = _run_tool("create_order (credit)", tools["invoicing"]["create_order"],
                          customer_id=cust_id, deliveryDate=today, orderLines=order_lines2)
            yield r, _strip_result(r)
            order2_id = _get_id(r["result"])
            if order2_id:
                r = _run_tool("create_invoice (credit)", tools["invoicing"]["create_invoice"],
                              invoiceDate=today, invoiceDueDate=tomorrow, order_id=order2_id)
                yield r, _strip_result(r)
                inv2_id = _get_id(r["result"])
                if inv2_id:
                    r = _run_tool("create_credit_note", tools["invoicing"]["create_credit_note"],
                                  invoice_id=inv2_id, date=today)
                    yield r, _strip_result(r)

            # Invoice reminder needs an unpaid invoice
            order_lines3 = json.dumps([{"product_id": prod_id, "count": 1}])
            r = _run_tool("create_order (reminder)", tools["invoicing"]["create_order"],
                          customer_id=cust_id, deliveryDate=today, orderLines=order_lines3)
            yield r, _strip_result(r)
            order3_id = _get_id(r["result"])
            if order3_id:
                r = _run_tool("create_invoice (reminder)", tools["invoicing"]["create_invoice"],
                              invoiceDate=today, invoiceDueDate=today, order_id=order3_id)
                yield r, _strip_result(r)
                inv3_id = _get_id(r["result"])
                if inv3_id:
                    r = _run_tool("create_invoice_reminder", tools["invoicing"]["create_invoice_reminder"],
                                  invoice_id=inv3_id)
                    yield r, _strip_result(r)

    # ── 5. ORDER EXTRAS ──
    if order_id and prod_id:
        r = _run_tool("create_order_line", tools["order"]["create_order_line"],
                      order_id=order_id, product_id=prod_id, count=1)
        yield r, _strip_result(r)
        ol_id = _get_id(r["result"])

        if ol_id:
            r = _run_tool("delete_order_line", tools["order"]["delete_order_line"],
                          order_line_id=ol_id)
            yield r, _strip_result(r)

    # ── 6. CONTACTS ──
    contact_id = 0
    if cust_id:
        r = _run_tool("create_contact", tools["contact"]["create_contact"],
                      firstName="Kontakt", lastName="Person",
                      email="kontakt@example.org", customer_id=cust_id)
        yield r, _strip_result(r)
        contact_id = _get_id(r["result"])

    r = _run_tool("search_contacts", tools["contact"]["search_contacts"],
                  firstName="Kontakt")
    yield r, _strip_result(r)

    if contact_id:
        r = _run_tool("update_contact", tools["contact"]["update_contact"],
                      contact_id=contact_id, firstName="KontaktUpdated")
        yield r, _strip_result(r)

    # ── 7. PROJECTS ──
    project_id = 0
    if cust_id:
        r = _run_tool("create_project", tools["project"]["create_project"],
                      name="Testprosjekt", customer_id=cust_id, startDate=today)
        yield r, _strip_result(r)
        project_id = _get_id(r["result"])

    r = _run_tool("search_projects", tools["project"]["search_projects"],
                  name="Testprosjekt")
    yield r, _strip_result(r)

    if project_id:
        r = _run_tool("update_project", tools["project"]["update_project"],
                      project_id=project_id, name="Testprosjekt Updated")
        yield r, _strip_result(r)

        if emp_id:
            r = _run_tool("create_project_participant", tools["project"]["create_project_participant"],
                          project_id=project_id, employee_id=emp_id)
            yield r, _strip_result(r)

    r = _run_tool("search_project_categories", tools["project"]["search_project_categories"])
    yield r, _strip_result(r)

    # ── 8. DEPARTMENTS ──
    r = _run_tool("create_department", tools["department"]["create_department"],
                  name=f"Testavdeling {ts}", departmentNumber=f"T{ts % 10000}")
    yield r, _strip_result(r)
    dept_id = _get_id(r["result"])

    r = _run_tool("search_departments", tools["department"]["search_departments"],
                  name="Testavdeling")
    yield r, _strip_result(r)

    if dept_id:
        r = _run_tool("update_department", tools["department"]["update_department"],
                      department_id=dept_id, name=f"Testavdeling Updated {ts}")
        yield r, _strip_result(r)

    # ── 9. EMPLOYMENT ──
    if emp_id:
        r = _run_tool("create_employment", tools["employment"]["create_employment"],
                      employee_id=emp_id, startDate=today)
        yield r, _strip_result(r)
        employment_id = _get_id(r["result"])

        r = _run_tool("search_employments", tools["employment"]["search_employments"],
                      employee_id=emp_id)
        yield r, _strip_result(r)

    # ── 10. TRAVEL EXPENSE ──
    travel_id = 0
    if emp_id:
        r = _run_tool("create_travel_expense", tools["travel"]["create_travel_expense"],
                      employee_id=emp_id, title="Testtur Oslo",
                      departureDate=today, returnDate=tomorrow)
        yield r, _strip_result(r)
        travel_id = _get_id(r["result"])

        r = _run_tool("search_travel_expenses", tools["travel"]["search_travel_expenses"],
                      employee_id=emp_id)
        yield r, _strip_result(r)

    # ── 10b. TRAVEL EXTRAS ──
    if travel_id:
        r = _run_tool("update_travel_expense", tools["travel_extras"]["update_travel_expense"],
                      travel_expense_id=travel_id, title="Testtur Oslo Updated")
        yield r, _strip_result(r)

        r = _run_tool("create_travel_expense_cost", tools["travel_extras"]["create_travel_expense_cost"],
                      travel_expense_id=travel_id, description="Taxi", amount=350.0,
                      date=today, category="other")
        yield r, _strip_result(r)

        r = _run_tool("search_travel_expense_costs", tools["travel_extras"]["search_travel_expense_costs"],
                      travel_expense_id=travel_id)
        yield r, _strip_result(r)

        r = _run_tool("create_mileage_allowance", tools["travel_extras"]["create_mileage_allowance"],
                      travel_expense_id=travel_id, date=today, km=120.0)
        yield r, _strip_result(r)

        r = _run_tool("create_per_diem_compensation", tools["travel_extras"]["create_per_diem_compensation"],
                      travel_expense_id=travel_id, date=today)
        yield r, _strip_result(r)

        r = _run_tool("delete_travel_expense", tools["travel"]["delete_travel_expense"],
                      travel_expense_id=travel_id)
        yield r, _strip_result(r)

    # ── 11. LEDGER ──
    r = _run_tool("get_ledger_accounts", tools["ledger"]["get_ledger_accounts"])
    yield r, _strip_result(r)

    r = _run_tool("get_ledger_accounts (1920)", tools["ledger"]["get_ledger_accounts"],
                  number="1920")
    yield r, _strip_result(r)

    r = _run_tool("get_ledger_postings", tools["ledger"]["get_ledger_postings"],
                  dateFrom=month_ago, dateTo=today)
    yield r, _strip_result(r)

    r = _run_tool("search_vouchers", tools["ledger"]["search_vouchers"],
                  dateFrom=month_ago, dateTo=today)
    yield r, _strip_result(r)

    r = _run_tool("create_voucher", tools["ledger"]["create_voucher"],
                  date=today, description="Test korrigering",
                  postings=json.dumps([
                      {"accountNumber": 1920, "amount": 100},
                      {"accountNumber": 7700, "amount": -100},
                  ]))
    yield r, _strip_result(r)
    voucher_id = _get_id(r["result"])

    if voucher_id:
        r = _run_tool("reverse_voucher", tools["ledger"]["reverse_voucher"],
                      voucher_id=voucher_id, date=today)
        yield r, _strip_result(r)

    r = _run_tool("create_ledger_account", tools["ledger"]["create_ledger_account"],
                  number=9999, name=f"Test Account {ts}")
    yield r, _strip_result(r)

    # ── 12. BANK ──
    r = _run_tool("search_bank_accounts", tools["bank"]["search_bank_accounts"])
    yield r, _strip_result(r)
    bank_accounts = _get_values(r["result"])
    bank_account_id = bank_accounts[0]["id"] if bank_accounts else 0

    r = _run_tool("search_bank_reconciliations", tools["bank"]["search_bank_reconciliations"])
    yield r, _strip_result(r)

    if bank_account_id:
        r = _run_tool("get_last_bank_reconciliation", tools["bank"]["get_last_bank_reconciliation"],
                      accountId=bank_account_id)
        yield r, _strip_result(r)

    r = _run_tool("search_bank_statements", tools["bank"]["search_bank_statements"])
    yield r, _strip_result(r)

    # ── 13. SUPPLIER ──
    r = _run_tool("create_supplier", tools["supplier"]["create_supplier"],
                  name=f"Test Leverandor {ts}", email="lev@example.org")
    yield r, _strip_result(r)
    supp_id = _get_id(r["result"])

    r = _run_tool("search_suppliers", tools["supplier"]["search_suppliers"],
                  name="Test Leverandor")
    yield r, _strip_result(r)

    if supp_id:
        r = _run_tool("update_supplier", tools["supplier"]["update_supplier"],
                      supplier_id=supp_id, name=f"Test Leverandor Updated {ts}")
        yield r, _strip_result(r)

    # ── 14. SUPPLIER INVOICE ──
    r = _run_tool("search_supplier_invoices", tools["supplier_invoice"]["search_supplier_invoices"])
    yield r, _strip_result(r)

    # ── 15. ADDRESS ──
    r = _run_tool("search_delivery_addresses", tools["address"]["search_delivery_addresses"])
    yield r, _strip_result(r)
    addresses = _get_values(r["result"])
    if addresses:
        r = _run_tool("update_delivery_address", tools["address"]["update_delivery_address"],
                      address_id=addresses[0]["id"],
                      addressLine1="Testgata 1", postalCode="0150", city="Oslo")
        yield r, _strip_result(r)

    # ── 16. BALANCE & REPORTING ──
    r = _run_tool("get_balance_sheet", tools["balance"]["get_balance_sheet"],
                  dateFrom=month_ago, dateTo=today)
    yield r, _strip_result(r)

    r = _run_tool("search_voucher_types", tools["balance"]["search_voucher_types"])
    yield r, _strip_result(r)

    r = _run_tool("search_year_ends", tools["balance"]["search_year_ends"])
    yield r, _strip_result(r)

    r = _run_tool("search_currencies", tools["balance"]["search_currencies"])
    yield r, _strip_result(r)

    r = _run_tool("get_company_info", tools["balance"]["get_company_info"])
    yield r, _strip_result(r)

    # ── 17. COMPANY ──
    r = _run_tool("get_accounting_periods", tools["company"]["get_accounting_periods"])
    yield r, _strip_result(r)

    # ── 18. ACTIVITY ──
    r = _run_tool("create_activity", tools["activity"]["create_activity"],
                  name=f"Testaktivitet {ts}")
    yield r, _strip_result(r)

    r = _run_tool("search_activities", tools["activity"]["search_activities"])
    yield r, _strip_result(r)

    # ── 19. DIVISION ──
    r = _run_tool("create_division", tools["division"]["create_division"],
                  name=f"Testdivisjon {ts}")
    yield r, _strip_result(r)
    div_id = _get_id(r["result"])

    r = _run_tool("search_divisions", tools["division"]["search_divisions"])
    yield r, _strip_result(r)

    if div_id:
        r = _run_tool("update_division", tools["division"]["update_division"],
                      division_id=div_id, name=f"Testdivisjon Updated {ts}")
        yield r, _strip_result(r)

    # ── 20. TIMESHEET ──
    if emp_id:
        r = _run_tool("create_timesheet_entry", tools["timesheet"]["create_timesheet_entry"],
                      employee_id=emp_id, date=today, hours=7.5)
        yield r, _strip_result(r)
        ts_id = _get_id(r["result"])

        r = _run_tool("search_timesheet_entries", tools["timesheet"]["search_timesheet_entries"],
                      employee_id=emp_id, dateFrom=month_ago, dateTo=today)
        yield r, _strip_result(r)

        if ts_id:
            r = _run_tool("update_timesheet_entry", tools["timesheet"]["update_timesheet_entry"],
                          entry_id=ts_id, hours=8.0)
            yield r, _strip_result(r)

            r = _run_tool("delete_timesheet_entry", tools["timesheet"]["delete_timesheet_entry"],
                          entry_id=ts_id)
            yield r, _strip_result(r)

    # ── 21. SALARY ──
    r = _run_tool("search_salary_types", tools["salary"]["search_salary_types"])
    yield r, _strip_result(r)

    r = _run_tool("search_salary_transactions", tools["salary"]["search_salary_transactions"])
    yield r, _strip_result(r)

    # ── 22. YEAR END ──
    r = _run_tool("search_year_end_annexes", tools["year_end"]["search_year_end_annexes"])
    yield r, _strip_result(r)

    r = _run_tool("get_vat_returns", tools["year_end"]["get_vat_returns"])
    yield r, _strip_result(r)

    # ── 23. INCOMING INVOICE ──
    r = _run_tool("search_incoming_invoices", tools["incoming_invoice"]["search_incoming_invoices"])
    yield r, _strip_result(r)

    # ── 24. EMPLOYEE EXTRAS ──
    r = _run_tool("search_employee_categories", tools["employee_extras"]["search_employee_categories"])
    yield r, _strip_result(r)

    r = _run_tool("search_next_of_kin", tools["employee_extras"]["search_next_of_kin"],
                  employee_id=emp_id)
    yield r, _strip_result(r)

    # ── 25. COMMON ──
    if emp_id:
        r = _run_tool("get_entity_by_id (employee)", tools["common"]["get_entity_by_id"],
                      entity_type="employee", entity_id=emp_id)
        yield r, _strip_result(r)

    if cust_id:
        r = _run_tool("get_entity_by_id (customer)", tools["common"]["get_entity_by_id"],
                      entity_type="customer", entity_id=cust_id)
        yield r, _strip_result(r)


def run_all_tool_tests(base_url: str, session_token: str) -> dict:
    """Run all tools against the sandbox. Returns structured test results."""
    client = TripletexClient(base_url, session_token)
    tools = _build_all_tool_sets(client)

    results = []
    for raw, stripped in _run_all_tests(tools, client):
        results.append(stripped)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    excp_count = sum(1 for r in results if r["status"] == "EXCEPTION")
    total_time = sum(r["elapsed"] for r in results)

    # Count unique tool names (exclude duplicates like "create_order (credit)")
    unique_tools = set()
    for ts_set in tools.values():
        unique_tools.update(ts_set.keys())

    return {
        "total": len(results),
        "unique_tools": len(unique_tools),
        "ok": ok_count,
        "fail": fail_count,
        "exception": excp_count,
        "total_time": round(total_time, 1),
        "api_calls": client._call_count,
        "api_errors": client._error_count,
        "results": results,
    }


def stream_tool_tests(base_url: str, session_token: str):
    """Generator that yields each tool test result as it completes."""
    client = TripletexClient(base_url, session_token)
    tools = _build_all_tool_sets(client)

    for raw, stripped in _run_all_tests(tools, client):
        yield stripped


# ── Tool Catalog (no credentials needed) ─────────────────────────

# Map from module name → build function
_MODULE_BUILDERS = {
    "employees": build_employee_tools,
    "customers": build_customer_tools,
    "products": build_product_tools,
    "invoicing": build_invoicing_tools,
    "travel": build_travel_tools,
    "projects": build_project_tools,
    "departments": build_department_tools,
    "ledger": build_ledger_tools,
    "contacts": build_contact_tools,
    "employment": build_employment_tools,
    "bank": build_bank_tools,
    "supplier": build_supplier_tools,
    "address": build_address_tools,
    "balance": build_balance_tools,
    "common": build_common_tools,
    "activity": build_activity_tools,
    "company": build_company_tools,
    "division": build_division_tools,
    "order": build_order_tools,
    "timesheet": build_timesheet_tools,
    "salary": build_salary_tools,
    "supplier_invoice": build_supplier_invoice_tools,
    "year_end": build_year_end_tools,
    "employee_extras": build_employee_extras_tools,
    "travel_extras": build_travel_extras_tools,
    "incoming_invoice": build_incoming_invoice_tools,
}


def get_tool_catalog() -> list[dict]:
    """Return tool metadata (name, module, docstring, params) without needing API credentials.

    Uses a dummy client — the build functions just create closures, no API calls happen.
    """

    class _DummyClient:
        """Minimal stand-in so build_*_tools() can create closures."""
        def get(self, *a, **kw): return {}
        def post(self, *a, **kw): return {}
        def put(self, *a, **kw): return {}
        def delete(self, *a, **kw): return {}

    dummy = _DummyClient()
    catalog = []

    for module_name, builder in _MODULE_BUILDERS.items():
        try:
            tool_set = builder(dummy)
        except Exception:
            continue
        for tool_name, func in tool_set.items():
            if tool_name.startswith("_"):
                continue
            # Extract docstring
            doc = inspect.getdoc(func) or ""
            summary = doc.split("\n")[0] if doc else ""
            # Extract parameters
            sig = inspect.signature(func)
            params = []
            for pname, p in sig.parameters.items():
                ptype = ""
                if p.annotation != inspect.Parameter.empty:
                    ptype = getattr(p.annotation, "__name__", str(p.annotation))
                has_default = p.default != inspect.Parameter.empty
                params.append({
                    "name": pname,
                    "type": ptype,
                    "required": not has_default,
                    "default": str(p.default) if has_default else None,
                })
            catalog.append({
                "name": tool_name,
                "module": module_name,
                "summary": summary,
                "docstring": doc,
                "params": params,
            })

    # Sort by module then name
    catalog.sort(key=lambda t: (t["module"], t["name"]))
    return catalog
