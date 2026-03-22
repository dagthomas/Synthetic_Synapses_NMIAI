"""Compound tool: invoice + payment workflow in one deterministic call.

Handles three flows:
  1. CREATE NEW: customer → product(s) → order → invoice → payment
  2. EXISTING: search customer → search invoice → register payment
  3. FOREIGN CURRENCY: search customer → search invoice → payment → agio voucher
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_invoice_with_payment_tools(client: TripletexClient) -> dict:
    """Build the compound invoice-with-payment tool."""

    from tools._helpers import (
        find_or_create_customer, find_or_create_product,
        ensure_bank_account, resolve_payment_type_id,
    )

    def process_invoice_with_payment(
        # Mode selection
        mode: str = "create_new",
        # Customer fields (for create_new or search)
        customer_name: str = "",
        customer_email: str = "",
        customer_org_number: str = "",
        # Product fields (for create_new only)
        products: str = "[]",
        # Invoice fields
        invoiceDate: str = "",
        invoiceDueDate: str = "",
        # Payment fields
        paymentDate: str = "",
        paymentAmount: float = 0.0,
        # Foreign currency fields
        foreignAmount: float = 0.0,
        foreignCurrency: str = "",
        oldRate: float = 0.0,
        newRate: float = 0.0,
    ) -> dict:
        """Process an invoice-with-payment workflow in one call.

        Supports three modes:
        - "create_new": Create customer + product(s) + order + invoice + payment (5-6 calls)
        - "existing": Search for existing customer and invoice, register payment (3 calls)
        - "foreign_currency": Search existing invoice + pay in foreign currency + book agio voucher (4-5 calls)

        Args:
            mode: "create_new" | "existing" | "foreign_currency"
            customer_name: Customer name (for search or create).
            customer_email: Customer email (for create_new).
            customer_org_number: Organization number (optional).
            products: JSON array of products for create_new: [{"name": "X", "price": 100, "quantity": 1, "vatPercentage": 25}].
                      price = price EXCLUDING VAT.
            invoiceDate: Invoice date YYYY-MM-DD (defaults to today).
            invoiceDueDate: Due date YYYY-MM-DD (defaults to invoiceDate).
            paymentDate: Payment date YYYY-MM-DD (defaults to today).
            paymentAmount: Payment amount override. If 0 for existing, pays full amountOutstanding.
            foreignAmount: Amount in foreign currency (e.g. 11219 EUR). Required for foreign_currency mode.
            foreignCurrency: Currency code (EUR/USD/GBP). Informational only.
            oldRate: Original exchange rate (invoice rate). Required for foreign_currency.
            newRate: New exchange rate (payment rate). Required for foreign_currency.

        Returns:
            Summary with invoice_id, payment info, or error details.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        if not invoiceDate:
            invoiceDate = today
        if not invoiceDueDate:
            invoiceDueDate = invoiceDate
        if not paymentDate:
            paymentDate = today

        # ── Detect mode from keywords if not explicitly set ──
        if mode not in ("create_new", "existing", "foreign_currency"):
            mode = "create_new"
        if foreignAmount > 0 and oldRate > 0 and newRate > 0:
            mode = "foreign_currency"

        # ══════════════════════════════════════════════════
        # MODE: CREATE NEW — customer → products → order → invoice → payment
        # ══════════════════════════════════════════════════
        if mode == "create_new":
            # Parse products
            try:
                prod_list = _json.loads(products) if isinstance(products, str) else products
            except (_json.JSONDecodeError, TypeError):
                return {"error": True, "message": f"Invalid products JSON: {products}"}
            if not prod_list:
                return {"error": True, "message": "products must be a non-empty list for create_new mode"}

            # Step 1: Find or create customer
            customer_id = find_or_create_customer(
                client, customer_name,
                email=customer_email, org_number=customer_org_number,
                steps_log=steps_log,
            )
            if not customer_id:
                return {"error": True, "message": f"Failed to create customer '{customer_name}'", "steps": steps_log}

            # Step 2: Find or create products
            product_ids = []
            order_lines = []

            for prod in prod_list:
                p_name = prod.get("name", "Product")
                p_price = prod.get("price", 0)
                p_qty = prod.get("quantity", 1)
                p_vat_pct = prod.get("vatPercentage", prod.get("vat_percentage", 25))
                p_number = prod.get("productNumber", prod.get("product_number", ""))

                prod_id = find_or_create_product(
                    client, name=p_name, price=p_price,
                    vat_percentage=p_vat_pct, product_number=p_number,
                    steps_log=steps_log,
                )
                if not prod_id:
                    return {"error": True, "message": f"Failed to create product '{p_name}'", "steps": steps_log}

                product_ids.append(prod_id)
                line = {"product": {"id": prod_id}, "count": p_qty}
                if p_price:
                    line["unitPriceExcludingVatCurrency"] = p_price
                order_lines.append(line)

            # Step 3: Create order
            order_body = {
                "customer": {"id": customer_id},
                "orderDate": invoiceDate,
                "deliveryDate": invoiceDate,
                "orderLines": order_lines,
            }
            order_result = client.post("/order", json=order_body)
            order_id = order_result.get("value", {}).get("id")
            if not order_id:
                return {"error": True, "message": f"Failed to create order: {order_result}", "steps": steps_log}
            steps_log.append(f"Created order (id={order_id})")

            # Step 4: Ensure bank account + create invoice
            ensure_bank_account(client)

            inv_body = {
                "invoiceDate": invoiceDate,
                "invoiceDueDate": invoiceDueDate,
                "orders": [{"id": order_id}],
            }
            inv_result = client.post("/invoice", json=inv_body)
            invoice_id = inv_result.get("value", {}).get("id")
            if not invoice_id:
                return {"error": True, "message": f"Failed to create invoice: {inv_result}", "steps": steps_log}

            inv_amount = inv_result.get("value", {}).get("amount", 0)
            if not inv_amount:
                inv_amount = 0
                for prod in prod_list:
                    p = prod.get("price", 0) * prod.get("quantity", 1)
                    vp = prod.get("vatPercentage", prod.get("vat_percentage", 25))
                    inv_amount += p * (1 + vp / 100)
                inv_amount = round(inv_amount, 2)

            steps_log.append(f"Created invoice (id={invoice_id}, amount={inv_amount})")

            # Step 5: Register payment
            pay_amount = paymentAmount if paymentAmount > 0 else inv_amount
            payment_type_id = resolve_payment_type_id(client)

            client.put(
                f"/invoice/{invoice_id}/:payment",
                params={
                    "paymentDate": paymentDate,
                    "paymentTypeId": payment_type_id,
                    "paidAmount": round(pay_amount, 2),
                    "paidAmountCurrency": round(pay_amount, 2),
                },
            )
            steps_log.append(f"Registered payment ({pay_amount} NOK on {paymentDate})")

            return {
                "success": True,
                "customer_id": customer_id,
                "product_ids": product_ids,
                "order_id": order_id,
                "invoice_id": invoice_id,
                "payment_amount": pay_amount,
                "steps": steps_log,
            }

        # ══════════════════════════════════════════════════
        # MODE: EXISTING — search customer → search invoice → pay
        # ══════════════════════════════════════════════════
        elif mode == "existing":
            if not customer_name:
                return {"error": True, "message": "customer_name is required for existing mode"}

            # Search customer
            cust_search = client.get("/customer", params={"name": customer_name, "fields": "id,name"})
            customers = cust_search.get("values", [])
            customer_id = None
            target = customer_name.strip().lower()
            for c in customers:
                if (c.get("name") or "").strip().lower() == target:
                    customer_id = c["id"]
                    break
            if not customer_id and customers:
                customer_id = customers[0]["id"]
            if not customer_id:
                return {"error": True, "message": f"Customer '{customer_name}' not found"}
            steps_log.append(f"Found customer '{customer_name}' (id={customer_id})")

            # Search invoices
            inv_search = client.get("/invoice", params={
                "customerId": customer_id,
                "invoiceDateFrom": "2000-01-01",
                "invoiceDateTo": "2030-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding",
            })
            invoices = inv_search.get("values", [])
            if not invoices:
                return {"error": True, "message": f"No invoices found for customer {customer_id}", "steps": steps_log}

            # Pick invoice with outstanding balance
            invoice = None
            for inv in invoices:
                if (inv.get("amountOutstanding") or 0) > 0:
                    invoice = inv
                    break
            if not invoice:
                return {"error": True, "message": "All invoices are already paid (amountOutstanding=0)", "steps": steps_log}

            invoice_id = invoice["id"]
            outstanding = invoice.get("amountOutstanding", 0)
            pay_amount = paymentAmount if paymentAmount > 0 else outstanding
            steps_log.append(f"Found invoice (id={invoice_id}, outstanding={outstanding})")

            # Register payment
            ensure_bank_account(client)
            payment_type_id = resolve_payment_type_id(client)

            client.put(
                f"/invoice/{invoice_id}/:payment",
                params={
                    "paymentDate": paymentDate,
                    "paymentTypeId": payment_type_id,
                    "paidAmount": round(pay_amount, 2),
                    "paidAmountCurrency": round(pay_amount, 2),
                },
            )
            steps_log.append(f"Registered payment ({pay_amount} NOK on {paymentDate})")

            return {
                "success": True,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "payment_amount": pay_amount,
                "steps": steps_log,
            }

        # ══════════════════════════════════════════════════
        # MODE: FOREIGN CURRENCY — search → pay → agio voucher
        # ══════════════════════════════════════════════════
        elif mode == "foreign_currency":
            if not customer_name:
                return {"error": True, "message": "customer_name is required"}
            if foreignAmount <= 0 or oldRate <= 0 or newRate <= 0:
                return {"error": True, "message": "foreignAmount, oldRate, newRate are all required for foreign_currency mode"}

            # Search customer
            cust_search = client.get("/customer", params={"name": customer_name, "fields": "id,name"})
            customers = cust_search.get("values", [])
            customer_id = None
            target = customer_name.strip().lower()
            for c in customers:
                if (c.get("name") or "").strip().lower() == target:
                    customer_id = c["id"]
                    break
            if not customer_id and customers:
                customer_id = customers[0]["id"]
            if not customer_id:
                return {"error": True, "message": f"Customer '{customer_name}' not found"}
            steps_log.append(f"Found customer '{customer_name}' (id={customer_id})")

            # Search invoices
            inv_search = client.get("/invoice", params={
                "customerId": customer_id,
                "invoiceDateFrom": "2000-01-01",
                "invoiceDateTo": "2030-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding",
            })
            invoices = inv_search.get("values", [])
            invoice = None
            for inv in invoices:
                if (inv.get("amountOutstanding") or 0) > 0:
                    invoice = inv
                    break
            if not invoice:
                return {"error": True, "message": "No unpaid invoices found", "steps": steps_log}

            invoice_id = invoice["id"]
            steps_log.append(f"Found invoice (id={invoice_id})")

            # Calculate amounts
            payment_nok = round(foreignAmount * newRate, 2)
            invoice_nok = round(foreignAmount * oldRate, 2)
            agio_diff = round(payment_nok - invoice_nok, 2)

            steps_log.append(f"Calculated: paymentNOK={payment_nok}, invoiceNOK={invoice_nok}, agio={agio_diff}")

            # Register payment
            ensure_bank_account(client)
            payment_type_id = resolve_payment_type_id(client)

            client.put(
                f"/invoice/{invoice_id}/:payment",
                params={
                    "paymentDate": paymentDate,
                    "paymentTypeId": payment_type_id,
                    "paidAmount": payment_nok,
                    "paidAmountCurrency": foreignAmount,
                },
            )
            steps_log.append(f"Registered payment ({payment_nok} NOK, {foreignAmount} {foreignCurrency})")

            # Book agio/disagio voucher
            if agio_diff != 0:
                if agio_diff > 0:
                    # Gain: debit 1500 (with customerId), credit 8060
                    postings = [
                        {"accountNumber": 1500, "amount": agio_diff, "customerId": customer_id},
                        {"accountNumber": 8060, "amount": -agio_diff},
                    ]
                    desc = f"Agiogevinst {foreignCurrency}"
                else:
                    # Loss: debit 8160, credit 1500 (with customerId)
                    postings = [
                        {"accountNumber": 8160, "amount": abs(agio_diff)},
                        {"accountNumber": 1500, "amount": -abs(agio_diff), "customerId": customer_id},
                    ]
                    desc = f"Agiotap {foreignCurrency}"

                # Resolve account IDs
                resolved_postings = []
                for p in postings:
                    acct_num = p["accountNumber"]
                    acct_search = client.get("/ledger/account", params={
                        "number": str(acct_num), "fields": "id,number"
                    })
                    acct_vals = acct_search.get("values", [])
                    acct_id = acct_vals[0]["id"] if acct_vals else 0
                    posting = {"account": {"id": acct_id}, "amount": p["amount"]}
                    if "customerId" in p:
                        posting["customer"] = {"id": p["customerId"]}
                    resolved_postings.append(posting)

                voucher_body = {
                    "date": paymentDate,
                    "description": desc,
                    "postings": resolved_postings,
                }
                client.post("/ledger/voucher", json=voucher_body)
                steps_log.append(f"Booked agio voucher: {desc} ({abs(agio_diff)} NOK)")

            return {
                "success": True,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "payment_nok": payment_nok,
                "agio": agio_diff,
                "steps": steps_log,
            }

        return {"error": True, "message": f"Unknown mode: {mode}"}

    return {
        "process_invoice_with_payment": process_invoice_with_payment,
    }
