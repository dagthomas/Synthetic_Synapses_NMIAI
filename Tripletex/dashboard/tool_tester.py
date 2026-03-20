"""Run all Tripletex tools against the sandbox and return structured results."""

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


def run_all_tool_tests(base_url: str, session_token: str) -> dict:
    """Run all tools against the sandbox. Returns structured test results."""
    client = TripletexClient(base_url, session_token)

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

    ts = int(time.time())
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    results = []

    # ── 1. EMPLOYEES ──
    r = _run_tool("create_employee", employee_tools["create_employee"],
                  firstName="Test", lastName="Testesen",
                  email=f"test.{ts}@example.org")
    results.append(r)
    emp_id = _get_id(r["result"])

    results.append(_run_tool("search_employees", employee_tools["search_employees"],
                             firstName="Test"))

    if emp_id:
        results.append(_run_tool("update_employee", employee_tools["update_employee"],
                                 employee_id=emp_id, firstName="TestUpdated"))

    # ── 2. CUSTOMERS ──
    r = _run_tool("create_customer", customer_tools["create_customer"],
                  name=f"Test Kunde {ts}", email="kunde@example.org")
    results.append(r)
    cust_id = _get_id(r["result"])

    results.append(_run_tool("search_customers", customer_tools["search_customers"],
                             name="Test Kunde"))

    if cust_id:
        results.append(_run_tool("update_customer", customer_tools["update_customer"],
                                 customer_id=cust_id, name=f"Test Kunde Updated {ts}"))

    # ── 3. PRODUCTS ──
    r = _run_tool("create_product", product_tools["create_product"],
                  name=f"Testprodukt {ts}", priceExcludingVatCurrency=100.0,
                  priceIncludingVatCurrency=125.0)
    results.append(r)
    prod_id = _get_id(r["result"])

    results.append(_run_tool("search_products", product_tools["search_products"],
                             name="Testprodukt"))

    # ── 4. INVOICING ──
    if cust_id and prod_id:
        order_lines = json.dumps([{"product_id": prod_id, "count": 2}])
        r = _run_tool("create_order", invoicing_tools["create_order"],
                      customer_id=cust_id, deliveryDate=today, orderLines=order_lines)
        results.append(r)
        order_id = _get_id(r["result"])

        if order_id:
            r = _run_tool("create_invoice", invoicing_tools["create_invoice"],
                          invoiceDate=today, invoiceDueDate=tomorrow, order_id=order_id)
            results.append(r)
            inv_id = _get_id(r["result"])

            if inv_id:
                results.append(_run_tool("register_payment",
                                         invoicing_tools["register_payment"],
                                         invoice_id=inv_id, amount=250.0,
                                         paymentDate=today))

                # Credit note test
                order_lines2 = json.dumps([{"product_id": prod_id, "count": 1}])
                r2 = _run_tool("create_order (credit)", invoicing_tools["create_order"],
                               customer_id=cust_id, deliveryDate=today,
                               orderLines=order_lines2)
                results.append(r2)
                order2_id = _get_id(r2["result"])
                if order2_id:
                    r3 = _run_tool("create_invoice (credit)",
                                   invoicing_tools["create_invoice"],
                                   invoiceDate=today, invoiceDueDate=tomorrow,
                                   order_id=order2_id)
                    results.append(r3)
                    inv2_id = _get_id(r3["result"])
                    if inv2_id:
                        results.append(_run_tool("create_credit_note",
                                                 invoicing_tools["create_credit_note"],
                                                 invoice_id=inv2_id, date=today))

    # ── 5. CONTACTS ──
    if cust_id:
        results.append(_run_tool("create_contact", contact_tools["create_contact"],
                                 firstName="Kontakt", lastName="Person",
                                 email="kontakt@example.org", customer_id=cust_id))

    results.append(_run_tool("search_contacts", contact_tools["search_contacts"],
                             firstName="Kontakt"))

    # ── 6. PROJECTS ──
    if cust_id:
        results.append(_run_tool("create_project", project_tools["create_project"],
                                 name="Testprosjekt", customer_id=cust_id,
                                 startDate=today))

    # ── 7. DEPARTMENTS ──
    results.append(_run_tool("create_department", department_tools["create_department"],
                             name=f"Testavdeling {ts}",
                             departmentNumber=f"T{ts % 10000}"))

    # ── 8. EMPLOYMENT ──
    if emp_id:
        results.append(_run_tool("create_employment",
                                 employment_tools["create_employment"],
                                 employee_id=emp_id, startDate=today))
        results.append(_run_tool("search_employments",
                                 employment_tools["search_employments"],
                                 employee_id=emp_id))

    # ── 9. TRAVEL EXPENSE ──
    if emp_id:
        r = _run_tool("create_travel_expense", travel_tools["create_travel_expense"],
                      employee_id=emp_id, title="Testtur Oslo",
                      departureDate=today, returnDate=tomorrow)
        results.append(r)
        travel_id = _get_id(r["result"])

        results.append(_run_tool("search_travel_expenses",
                                 travel_tools["search_travel_expenses"],
                                 employee_id=emp_id))

        if travel_id:
            results.append(_run_tool("delete_travel_expense",
                                     travel_tools["delete_travel_expense"],
                                     travel_expense_id=travel_id))

    # ── 10. LEDGER ──
    results.append(_run_tool("get_ledger_accounts", ledger_tools["get_ledger_accounts"]))
    results.append(_run_tool("get_ledger_accounts (1920)",
                             ledger_tools["get_ledger_accounts"], number="1920"))
    results.append(_run_tool("get_ledger_postings", ledger_tools["get_ledger_postings"],
                             dateFrom=month_ago, dateTo=today))

    r = _run_tool("create_voucher", ledger_tools["create_voucher"],
                  date=today, description="Test korrigering",
                  postings=json.dumps([
                      {"accountNumber": 1920, "amount": 100},
                      {"accountNumber": 7700, "amount": -100},
                  ]))
    results.append(r)
    voucher_id = _get_id(r["result"])

    if voucher_id:
        results.append(_run_tool("delete_voucher", ledger_tools["delete_voucher"],
                                 voucher_id=voucher_id))

    # ── 11. BANK ──
    r = _run_tool("search_bank_accounts", bank_tools["search_bank_accounts"])
    results.append(r)
    bank_accounts = _get_values(r["result"])
    bank_account_id = bank_accounts[0]["id"] if bank_accounts else 0

    results.append(_run_tool("search_bank_reconciliations",
                             bank_tools["search_bank_reconciliations"]))

    if bank_account_id:
        results.append(_run_tool("get_last_bank_reconciliation",
                                 bank_tools["get_last_bank_reconciliation"],
                                 accountId=bank_account_id))

    results.append(_run_tool("search_bank_statements",
                             bank_tools["search_bank_statements"]))

    # ── 12. SUPPLIER ──
    r = _run_tool("create_supplier", supplier_tools["create_supplier"],
                  name=f"Test Leverandor {ts}", email="lev@example.org")
    results.append(r)
    supp_id = _get_id(r["result"])

    results.append(_run_tool("search_suppliers", supplier_tools["search_suppliers"],
                             name="Test Leverandor"))

    if supp_id:
        results.append(_run_tool("update_supplier", supplier_tools["update_supplier"],
                                 supplier_id=supp_id,
                                 name=f"Test Leverandor Updated {ts}"))

    # ── 13. ADDRESS ──
    r = _run_tool("search_delivery_addresses",
                  address_tools["search_delivery_addresses"])
    results.append(r)
    addresses = _get_values(r["result"])
    if addresses:
        results.append(_run_tool("update_delivery_address",
                                 address_tools["update_delivery_address"],
                                 address_id=addresses[0]["id"],
                                 addressLine1="Testgata 1", postalCode="0150",
                                 city="Oslo"))

    # ── 14. BALANCE & COMPANY ──
    results.append(_run_tool("get_balance_sheet", balance_tools["get_balance_sheet"],
                             dateFrom=month_ago, dateTo=today))
    results.append(_run_tool("search_voucher_types",
                             balance_tools["search_voucher_types"]))
    results.append(_run_tool("search_year_ends", balance_tools["search_year_ends"]))
    results.append(_run_tool("search_currencies", balance_tools["search_currencies"]))
    results.append(_run_tool("get_company_info", balance_tools["get_company_info"]))

    # ── 15. COMMON ──
    if emp_id:
        results.append(_run_tool("get_entity_by_id (employee)",
                                 common_tools["get_entity_by_id"],
                                 entity_type="employee", entity_id=emp_id))
    if cust_id:
        results.append(_run_tool("get_entity_by_id (customer)",
                                 common_tools["get_entity_by_id"],
                                 entity_type="customer", entity_id=cust_id))

    # ── Summary ──
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    excp_count = sum(1 for r in results if r["status"] == "EXCEPTION")
    total_time = sum(r["elapsed"] for r in results)

    # Strip large result data for JSON response
    for r in results:
        if r["result"] and isinstance(r["result"], dict):
            r["result_preview"] = str(r["result"])[:300]
        else:
            r["result_preview"] = ""
        del r["result"]

    return {
        "total": len(results),
        "ok": ok_count,
        "fail": fail_count,
        "exception": excp_count,
        "total_time": round(total_time, 1),
        "api_calls": client._call_count,
        "api_errors": client._error_count,
        "results": results,
    }


def _strip_result(r):
    """Strip large result data, return cleaned copy."""
    out = dict(r)
    if out["result"] and isinstance(out["result"], dict):
        out["result_preview"] = str(out["result"])[:300]
    else:
        out["result_preview"] = ""
    del out["result"]
    return out


def stream_tool_tests(base_url: str, session_token: str):
    """Generator that yields each tool test result as it completes."""
    client = TripletexClient(base_url, session_token)

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

    ts = int(time.time())
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    idx = 0

    def run_and_yield(tool_name, func, *args, **kwargs):
        nonlocal idx
        r = _run_tool(tool_name, func, *args, **kwargs)
        idx += 1
        return r, _strip_result(r)

    # ── 1. EMPLOYEES ──
    r, out = run_and_yield("create_employee", employee_tools["create_employee"],
                           firstName="Test", lastName="Testesen",
                           email=f"test.{ts}@example.org")
    yield out
    emp_id = _get_id(r["result"])

    _, out = run_and_yield("search_employees", employee_tools["search_employees"],
                           firstName="Test")
    yield out

    if emp_id:
        _, out = run_and_yield("update_employee", employee_tools["update_employee"],
                               employee_id=emp_id, firstName="TestUpdated")
        yield out

    # ── 2. CUSTOMERS ──
    r, out = run_and_yield("create_customer", customer_tools["create_customer"],
                           name=f"Test Kunde {ts}", email="kunde@example.org")
    yield out
    cust_id = _get_id(r["result"])

    _, out = run_and_yield("search_customers", customer_tools["search_customers"],
                           name="Test Kunde")
    yield out

    if cust_id:
        _, out = run_and_yield("update_customer", customer_tools["update_customer"],
                               customer_id=cust_id, name=f"Test Kunde Updated {ts}")
        yield out

    # ── 3. PRODUCTS ──
    r, out = run_and_yield("create_product", product_tools["create_product"],
                           name=f"Testprodukt {ts}", priceExcludingVatCurrency=100.0,
                           priceIncludingVatCurrency=125.0)
    yield out
    prod_id = _get_id(r["result"])

    _, out = run_and_yield("search_products", product_tools["search_products"],
                           name="Testprodukt")
    yield out

    # ── 4. INVOICING ──
    if cust_id and prod_id:
        order_lines = json.dumps([{"product_id": prod_id, "count": 2}])
        r, out = run_and_yield("create_order", invoicing_tools["create_order"],
                               customer_id=cust_id, deliveryDate=today, orderLines=order_lines)
        yield out
        order_id = _get_id(r["result"])

        if order_id:
            r, out = run_and_yield("create_invoice", invoicing_tools["create_invoice"],
                                   invoiceDate=today, invoiceDueDate=tomorrow, order_id=order_id)
            yield out
            inv_id = _get_id(r["result"])

            if inv_id:
                _, out = run_and_yield("register_payment", invoicing_tools["register_payment"],
                                       invoice_id=inv_id, amount=250.0, paymentDate=today)
                yield out

                order_lines2 = json.dumps([{"product_id": prod_id, "count": 1}])
                r2, out = run_and_yield("create_order (credit)", invoicing_tools["create_order"],
                                        customer_id=cust_id, deliveryDate=today,
                                        orderLines=order_lines2)
                yield out
                order2_id = _get_id(r2["result"])
                if order2_id:
                    r3, out = run_and_yield("create_invoice (credit)",
                                            invoicing_tools["create_invoice"],
                                            invoiceDate=today, invoiceDueDate=tomorrow,
                                            order_id=order2_id)
                    yield out
                    inv2_id = _get_id(r3["result"])
                    if inv2_id:
                        _, out = run_and_yield("create_credit_note",
                                              invoicing_tools["create_credit_note"],
                                              invoice_id=inv2_id, date=today)
                        yield out

    # ── 5. CONTACTS ──
    if cust_id:
        _, out = run_and_yield("create_contact", contact_tools["create_contact"],
                               firstName="Kontakt", lastName="Person",
                               email="kontakt@example.org", customer_id=cust_id)
        yield out

    _, out = run_and_yield("search_contacts", contact_tools["search_contacts"],
                           firstName="Kontakt")
    yield out

    # ── 6. PROJECTS ──
    if cust_id:
        _, out = run_and_yield("create_project", project_tools["create_project"],
                               name="Testprosjekt", customer_id=cust_id, startDate=today)
        yield out

    # ── 7. DEPARTMENTS ──
    _, out = run_and_yield("create_department", department_tools["create_department"],
                           name=f"Testavdeling {ts}",
                           departmentNumber=f"T{ts % 10000}")
    yield out

    # ── 8. EMPLOYMENT ──
    if emp_id:
        _, out = run_and_yield("create_employment", employment_tools["create_employment"],
                               employee_id=emp_id, startDate=today)
        yield out
        _, out = run_and_yield("search_employments", employment_tools["search_employments"],
                               employee_id=emp_id)
        yield out

    # ── 9. TRAVEL EXPENSE ──
    if emp_id:
        r, out = run_and_yield("create_travel_expense", travel_tools["create_travel_expense"],
                               employee_id=emp_id, title="Testtur Oslo",
                               departureDate=today, returnDate=tomorrow)
        yield out
        travel_id = _get_id(r["result"])

        _, out = run_and_yield("search_travel_expenses", travel_tools["search_travel_expenses"],
                               employee_id=emp_id)
        yield out

        if travel_id:
            _, out = run_and_yield("delete_travel_expense", travel_tools["delete_travel_expense"],
                                   travel_expense_id=travel_id)
            yield out

    # ── 10. LEDGER ──
    _, out = run_and_yield("get_ledger_accounts", ledger_tools["get_ledger_accounts"])
    yield out
    _, out = run_and_yield("get_ledger_accounts (1920)", ledger_tools["get_ledger_accounts"],
                           number="1920")
    yield out
    _, out = run_and_yield("get_ledger_postings", ledger_tools["get_ledger_postings"],
                           dateFrom=month_ago, dateTo=today)
    yield out

    r, out = run_and_yield("create_voucher", ledger_tools["create_voucher"],
                           date=today, description="Test korrigering",
                           postings=json.dumps([
                               {"accountNumber": 1920, "amount": 100},
                               {"accountNumber": 7700, "amount": -100},
                           ]))
    yield out
    voucher_id = _get_id(r["result"])

    if voucher_id:
        _, out = run_and_yield("delete_voucher", ledger_tools["delete_voucher"],
                               voucher_id=voucher_id)
        yield out

    # ── 11. BANK ──
    r, out = run_and_yield("search_bank_accounts", bank_tools["search_bank_accounts"])
    yield out
    bank_accounts = _get_values(r["result"])
    bank_account_id = bank_accounts[0]["id"] if bank_accounts else 0

    _, out = run_and_yield("search_bank_reconciliations",
                           bank_tools["search_bank_reconciliations"])
    yield out

    if bank_account_id:
        _, out = run_and_yield("get_last_bank_reconciliation",
                               bank_tools["get_last_bank_reconciliation"],
                               accountId=bank_account_id)
        yield out

    _, out = run_and_yield("search_bank_statements", bank_tools["search_bank_statements"])
    yield out

    # ── 12. SUPPLIER ──
    r, out = run_and_yield("create_supplier", supplier_tools["create_supplier"],
                           name=f"Test Leverandor {ts}", email="lev@example.org")
    yield out
    supp_id = _get_id(r["result"])

    _, out = run_and_yield("search_suppliers", supplier_tools["search_suppliers"],
                           name="Test Leverandor")
    yield out

    if supp_id:
        _, out = run_and_yield("update_supplier", supplier_tools["update_supplier"],
                               supplier_id=supp_id,
                               name=f"Test Leverandor Updated {ts}")
        yield out

    # ── 13. ADDRESS ──
    r, out = run_and_yield("search_delivery_addresses",
                           address_tools["search_delivery_addresses"])
    yield out
    addresses = _get_values(r["result"])
    if addresses:
        _, out = run_and_yield("update_delivery_address",
                               address_tools["update_delivery_address"],
                               address_id=addresses[0]["id"],
                               addressLine1="Testgata 1", postalCode="0150", city="Oslo")
        yield out

    # ── 14. BALANCE & COMPANY ──
    _, out = run_and_yield("get_balance_sheet", balance_tools["get_balance_sheet"],
                           dateFrom=month_ago, dateTo=today)
    yield out
    _, out = run_and_yield("search_voucher_types", balance_tools["search_voucher_types"])
    yield out
    _, out = run_and_yield("search_year_ends", balance_tools["search_year_ends"])
    yield out
    _, out = run_and_yield("search_currencies", balance_tools["search_currencies"])
    yield out
    _, out = run_and_yield("get_company_info", balance_tools["get_company_info"])
    yield out

    # ── 15. COMMON ──
    if emp_id:
        _, out = run_and_yield("get_entity_by_id (employee)", common_tools["get_entity_by_id"],
                               entity_type="employee", entity_id=emp_id)
        yield out
    if cust_id:
        _, out = run_and_yield("get_entity_by_id (customer)", common_tools["get_entity_by_id"],
                               entity_type="customer", entity_id=cust_id)
        yield out
