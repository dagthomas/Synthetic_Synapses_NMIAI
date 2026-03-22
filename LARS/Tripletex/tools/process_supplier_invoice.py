"""Compound tool: process entire supplier invoice workflow in one deterministic call.

Handles: find/create supplier → [find/create department] → create incoming invoice (ledger voucher with VAT postings).
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import logging

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def _fix_norwegian_bban(acct: str) -> str:
    """Fix mod-11 check digit on a Norwegian 11-digit bank account number.

    Takes the first 10 digits and computes the correct check digit.
    Returns valid 11-digit BBAN, or empty string if invalid.
    """
    digits = "".join(c for c in acct if c.isdigit())
    if len(digits) < 10:
        return ""
    d = [int(c) for c in digits[:10]]
    weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    s = sum(a * b for a, b in zip(d, weights))
    remainder = s % 11
    if remainder == 0:
        check = 0
    elif remainder == 1:
        d[9] = (d[9] + 1) % 10
        s = sum(a * b for a, b in zip(d, weights))
        remainder = s % 11
        check = 0 if remainder == 0 else 11 - remainder
    else:
        check = 11 - remainder
    return "".join(str(x) for x in d) + str(check)


def build_process_supplier_invoice_tools(client: TripletexClient) -> dict:
    """Build the compound supplier invoice processing tool."""

    from tools._helpers import (
        find_or_create_supplier, find_or_create_department, recover_error,
    )

    def process_supplier_invoice(
        supplier_name: str,
        amountIncludingVat: float,
        expenseAccountNumber: int,
        invoiceDate: str,
        supplierOrgNumber: str = "",
        supplierBankAccount: str = "",
        supplierAddress: str = "",
        supplierPostalCode: str = "",
        supplierCity: str = "",
        supplierEmail: str = "",
        invoiceNumber: str = "",
        vatPercentage: int = 25,
        dueDate: str = "",
        lineDescription: str = "",
        departmentName: str = "",
    ) -> dict:
        """Process a complete supplier invoice workflow in one call.

        This compound tool handles the ENTIRE supplier invoice workflow:
        1. Creates the supplier (or finds existing on duplicate)
        2. Optionally creates/finds the department
        3. Creates a ledger voucher with expense debit (+ input VAT) and payables credit

        Args:
            supplier_name: Supplier company name. REQUIRED.
            amountIncludingVat: Total invoice amount INCLUDING VAT. REQUIRED.
            expenseAccountNumber: Expense account number (e.g. 6590, 6300, 6800, 4000). REQUIRED.
            invoiceDate: Invoice date YYYY-MM-DD (fakturadato). REQUIRED.
            supplierOrgNumber: Norwegian org number (9 digits). Optional.
            supplierBankAccount: Supplier bank account number (e.g. "19048571614"). Optional.
            supplierAddress: Street address (e.g. "Storgata 1"). Optional.
            supplierPostalCode: Postal code (e.g. "0182"). Optional.
            supplierCity: City name (e.g. "Oslo"). Optional.
            supplierEmail: Supplier email. Optional.
            invoiceNumber: Vendor's invoice number (e.g. "INV-2026-001"). Optional.
            vatPercentage: VAT rate: 25 (standard), 15 (food), 12 (transport), 0 (exempt). Default 25.
            dueDate: Payment due date YYYY-MM-DD (forfallsdato). Optional.
            lineDescription: What the invoice is for (e.g. "Nettverkstjenester"). Optional.
            departmentName: Department name to post expense to (e.g. "Økonomi"). Optional.

        Returns:
            Summary with supplier_id, department_id (if applicable), voucher info, and steps log.
        """
        steps_log = []

        if not invoiceDate:
            return {"error": True, "message": "invoiceDate is required (YYYY-MM-DD)"}
        if not expenseAccountNumber:
            return {"error": True, "message": "expenseAccountNumber is required (e.g. 6590)"}
        if not amountIncludingVat:
            return {"error": True, "message": "amountIncludingVat is required"}

        # ── Step 1: Find or create supplier ──
        supplier_id = find_or_create_supplier(
            client, name=supplier_name, email=supplierEmail,
            org_number=supplierOrgNumber,
            address=supplierAddress, postal_code=supplierPostalCode,
            city=supplierCity, steps_log=steps_log,
        )
        if not supplier_id:
            return {"error": True, "message": f"Failed to create supplier '{supplier_name}'", "steps": steps_log}

        # Set bank account via PUT using bban field (best-effort)
        if supplierBankAccount:
            try:
                bban = _fix_norwegian_bban(supplierBankAccount)
                if bban:
                    ba_result = client.put(f"/supplier/{supplier_id}", json={
                        "id": supplier_id, "name": supplier_name,
                        "bankAccountPresentation": [{"bban": bban}],
                    })
                    if ba_result.get("error"):
                        recover_error(client, f"/supplier/{supplier_id}")
                    else:
                        steps_log.append(f"Set bank account on supplier ({bban})")
            except Exception:
                pass

        # ── Step 2: Optional department ──
        dept_id = 0
        if departmentName:
            dept_id = find_or_create_department(client, name=departmentName, steps_log=steps_log)

        # ── Step 3: Create incoming invoice (ledger voucher) ──
        expense_cache_key = f"acct_{expenseAccountNumber}"
        expense_id = client.get_cached(expense_cache_key)
        if not expense_id:
            accounts_to_try = [expenseAccountNumber]
            if expenseAccountNumber not in (6590, 6300, 6800, 4000):
                accounts_to_try.append(6590)
            for acct_num in accounts_to_try:
                expense_result = client.get("/ledger/account", params={
                    "number": str(acct_num), "fields": "id", "count": 1,
                })
                expense_accts = expense_result.get("values", [])
                if expense_accts:
                    expense_id = expense_accts[0]["id"]
                    client.set_cached(f"acct_{acct_num}", expense_id)
                    break
            if not expense_id:
                return {"error": True, "message": f"Expense account {expenseAccountNumber} not found",
                        "steps": steps_log}

        payables_id = client.get_cached("acct_2400")
        if not payables_id:
            payables_result = client.get("/ledger/account", params={
                "number": "2400", "fields": "id", "count": 1,
            })
            payables_accts = payables_result.get("values", [])
            if not payables_accts:
                return {"error": True, "message": "Payables account 2400 not found", "steps": steps_log}
            payables_id = payables_accts[0]["id"]
            client.set_cached("acct_2400", payables_id)

        # Resolve input VAT types
        input_vat_map = client.get_cached("input_vat_type_map") or {}
        if not input_vat_map and vatPercentage > 0:
            _IN = {1: 25, 11: 15, 13: 12}
            r = client.get("/ledger/vatType", params={"fields": "id,number"})
            for vt in (r.get("values") or []):
                n, vid = vt.get("number"), vt.get("id")
                if n is not None and vid is not None:
                    try:
                        n = int(n)
                    except (ValueError, TypeError):
                        continue
                    if n in _IN:
                        input_vat_map[_IN[n]] = vid
            if input_vat_map:
                client.set_cached("input_vat_type_map", input_vat_map)
        vat_type_id = input_vat_map.get(vatPercentage, 0) if vatPercentage > 0 else 0
        if vatPercentage > 0 and vat_type_id == 0:
            return {"error": True, "message": f"No valid input VAT type for {vatPercentage}%. Available: {list(input_vat_map.keys())}",
                    "steps": steps_log}

        amt = round(amountIncludingVat, 2)

        if lineDescription and invoiceNumber:
            description = f"{invoiceNumber} - {lineDescription}"
        elif invoiceNumber:
            description = f"{invoiceNumber} - supplier invoice"
        elif lineDescription:
            description = lineDescription
        else:
            description = "Supplier invoice"

        expense_posting = {
            "row": 1,
            "account": {"id": expense_id},
            "amountGross": amt,
            "amountGrossCurrency": amt,
            "description": description,
        }
        if vat_type_id > 0:
            expense_posting["vatType"] = {"id": vat_type_id}
        if dept_id:
            expense_posting["department"] = {"id": dept_id}

        credit_posting = {
            "row": 2,
            "account": {"id": payables_id},
            "amountGross": -amt,
            "amountGrossCurrency": -amt,
            "supplier": {"id": supplier_id},
            "description": description,
        }
        if invoiceNumber:
            credit_posting["invoiceNumber"] = invoiceNumber
        if dueDate:
            credit_posting["termOfPayment"] = dueDate

        body = {
            "date": invoiceDate,
            "description": description,
            "postings": [expense_posting, credit_posting],
        }
        if invoiceNumber:
            body["vendorInvoiceNumber"] = invoiceNumber

        voucher_result = client.post("/ledger/voucher", json=body)
        voucher_id = voucher_result.get("value", {}).get("id")
        if not voucher_id:
            return {"error": True, "message": f"Failed to create voucher: {voucher_result}",
                    "steps": steps_log}
        steps_log.append(f"Created voucher (id={voucher_id})")

        return {
            "success": True,
            "supplier_id": supplier_id,
            "department_id": dept_id if dept_id else None,
            "voucher_id": voucher_id,
            "steps": steps_log,
        }

    return {
        "process_supplier_invoice": process_supplier_invoice,
    }
