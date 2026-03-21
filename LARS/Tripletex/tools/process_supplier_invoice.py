"""Compound tool: process entire supplier invoice workflow in one deterministic call.

Handles: create supplier → [create department] → create incoming invoice (ledger voucher with VAT postings).
No LLM chaining required — all steps are hardcoded.
"""

import logging

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_supplier_invoice_tools(client: TripletexClient) -> dict:
    """Build the compound supplier invoice processing tool."""

    def _recover_error(endpoint: str):
        """Undo error count for auto-recovery on expected 422s."""
        client._error_count = max(0, client._error_count - 1)
        for entry in reversed(client._call_log):
            if not entry.get("ok") and endpoint in entry.get("url", ""):
                entry["ok"] = True
                entry["recovered"] = True
                break

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

        # ── Step 1: Create supplier ──
        sup_body = {"name": supplier_name, "isSupplier": True}
        if supplierOrgNumber:
            sup_body["organizationNumber"] = supplierOrgNumber
        if supplierEmail:
            sup_body["email"] = supplierEmail
        if supplierBankAccount:
            sup_body["bankAccountPresentation"] = [{"bankAccountNumber": supplierBankAccount}]
        if supplierAddress or supplierPostalCode or supplierCity:
            addr = {}
            if supplierAddress:
                addr["addressLine1"] = supplierAddress
            if supplierPostalCode:
                addr["postalCode"] = supplierPostalCode
            if supplierCity:
                addr["city"] = supplierCity
            sup_body["postalAddress"] = addr
            sup_body["physicalAddress"] = addr

        sup_result = client.post("/supplier", json=sup_body)
        supplier_id = sup_result.get("value", {}).get("id")

        if not supplier_id and sup_result.get("error"):
            # Auto-recover: find existing supplier
            search_params = {"fields": "id,name,organizationNumber"}
            if supplierOrgNumber:
                search_params["organizationNumber"] = supplierOrgNumber
            else:
                search_params["name"] = supplier_name
            existing = client.get("/supplier", params=search_params)
            vals = existing.get("values", [])
            if vals:
                supplier_id = vals[0]["id"]
                _recover_error("/supplier")
                steps_log.append(f"Supplier already existed (id={supplier_id})")

        if not supplier_id:
            return {"error": True, "message": f"Failed to create supplier: {sup_result}", "steps": steps_log}
        if not steps_log or "already existed" not in steps_log[-1]:
            steps_log.append(f"Created supplier '{supplier_name}' (id={supplier_id})")

        # ── Step 2: Optional department ──
        dept_id = 0
        if departmentName:
            # Search first
            dept_search = client.get("/department", params={
                "name": departmentName, "fields": "id,name", "count": 5,
            })
            dept_vals = dept_search.get("values", [])
            # Find exact or close match
            for dv in dept_vals:
                if (dv.get("name") or "").strip().lower() == departmentName.strip().lower():
                    dept_id = dv["id"]
                    steps_log.append(f"Found department '{departmentName}' (id={dept_id})")
                    break
            if not dept_id and dept_vals:
                dept_id = dept_vals[0]["id"]
                steps_log.append(f"Found department (id={dept_id})")
            if not dept_id:
                # Create department
                dept_result = client.post("/department", json={"name": departmentName})
                dept_id = dept_result.get("value", {}).get("id")
                if dept_id:
                    steps_log.append(f"Created department '{departmentName}' (id={dept_id})")

        # ── Step 3: Create incoming invoice (ledger voucher) ──
        # Look up expense account ID
        expense_cache_key = f"acct_{expenseAccountNumber}"
        expense_id = client.get_cached(expense_cache_key)
        if not expense_id:
            expense_result = client.get("/ledger/account", params={
                "number": str(expenseAccountNumber), "fields": "id", "count": 1,
            })
            expense_accts = expense_result.get("values", [])
            if not expense_accts:
                return {"error": True, "message": f"Expense account {expenseAccountNumber} not found",
                        "steps": steps_log}
            expense_id = expense_accts[0]["id"]
            client.set_cached(expense_cache_key, expense_id)

        # Look up payables account 2400
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

        # Resolve input VAT types by standard number (from prewarm cache or live lookup)
        input_vat_map = client.get_cached("input_vat_type_map") or {}
        if not input_vat_map and vatPercentage > 0:
            _IN = {1: 25, 11: 15, 13: 12}
            r = client.get("/ledger/vatType", params={"fields": "id,number"})
            for vt in (r.get("values") or []):
                n, vid = vt.get("number"), vt.get("id")
                if n is not None and vid is not None and int(n) in _IN:
                    input_vat_map[_IN[int(n)]] = vid
            if input_vat_map:
                client.set_cached("input_vat_type_map", input_vat_map)
        vat_type_id = input_vat_map.get(vatPercentage, 0) if vatPercentage > 0 else 0
        if vatPercentage > 0 and vat_type_id == 0:
            return {"error": True, "message": f"No valid input VAT type for {vatPercentage}%. Available: {list(input_vat_map.keys())}",
                    "steps": steps_log}

        amt = round(amountIncludingVat, 2)

        # Build description
        if lineDescription and invoiceNumber:
            description = f"{invoiceNumber} - {lineDescription}"
        elif invoiceNumber:
            description = f"{invoiceNumber} - supplier invoice"
        elif lineDescription:
            description = lineDescription
        else:
            description = "Supplier invoice"

        # Build expense posting (debit)
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

        # Build payables posting (credit)
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
