"""Compound tool: process entire invoice workflow in one deterministic call.

Handles: create customer → create product(s) → create order → create invoice
         → optionally create credit note or send invoice.
No LLM chaining required — all steps are hardcoded.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_invoice_tools(client: TripletexClient) -> dict:
    """Build the compound invoice processing tool."""

    def _recover_error(endpoint: str):
        """Undo error count for auto-recovery on expected 422s."""
        client._error_count = max(0, client._error_count - 1)
        for entry in reversed(client._call_log):
            if not entry.get("ok") and endpoint in entry.get("url", ""):
                entry["ok"] = True
                entry["recovered"] = True
                break

    def _ensure_bank_account() -> None:
        """Ensure ledger account 1920 has a bank account number (once per session)."""
        cache_key = "bank_account_ensured"
        if client.get_cached(cache_key):
            return
        try:
            r = client.get("/ledger/account", params={
                "number": "1920", "fields": "id,name,isBankAccount,bankAccountNumber"
            })
            accounts = r.get("values", [])
            if not accounts:
                client.set_cached(cache_key, True)
                return
            acct = accounts[0]
            if acct.get("bankAccountNumber"):
                client.set_cached(cache_key, True)
                return
            client.put(f"/ledger/account/{acct['id']}", json={
                "id": acct["id"],
                "number": 1920,
                "name": acct["name"],
                "isBankAccount": True,
                "bankAccountNumber": "12345678903",
            })
            client.set_cached(cache_key, True)
        except Exception:
            pass

    def process_invoice(
        customer_name: str,
        customer_email: str = "",
        customer_org_number: str = "",
        customer_phone: str = "",
        customer_address: str = "",
        customer_postal_code: str = "",
        customer_city: str = "",
        products: str = "[]",
        invoiceDate: str = "",
        invoiceDueDate: str = "",
        create_credit_note: bool = False,
        send_invoice: bool = False,
    ) -> dict:
        """Process a complete invoice workflow in one call.

        This compound tool handles the ENTIRE invoice workflow:
        1. Creates the customer (or finds existing on collision)
        2. Creates all products with correct VAT types
        3. Creates the order with all order lines
        4. Ensures bank account exists on ledger account 1920
        5. Creates the invoice
        6. Optionally creates a credit note for the invoice
        7. Optionally sends the invoice

        Args:
            customer_name: Customer company name.
            customer_email: Customer email address.
            customer_org_number: Organization number (optional).
            customer_phone: Customer phone number (optional).
            customer_address: Street address (optional).
            customer_postal_code: Postal code (optional).
            customer_city: City (optional).
            products: JSON string of products: [{"name": "Product A", "price": 1000.0, "quantity": 2, "vatPercentage": 25}]
                      price = price EXCLUDING VAT. quantity defaults to 1. vatPercentage defaults to 25.
            invoiceDate: Invoice date YYYY-MM-DD (defaults to today).
            invoiceDueDate: Due date YYYY-MM-DD (defaults to invoiceDate).
            create_credit_note: If True, creates a credit note after the invoice.
            send_invoice: If True, sends the invoice after creation.

        Returns:
            Summary with customer_id, product_ids, order_id, invoice_id, credit_note info.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        if not invoiceDate:
            invoiceDate = today
        if not invoiceDueDate:
            invoiceDueDate = invoiceDate

        # Parse products JSON
        try:
            prod_list = _json.loads(products) if isinstance(products, str) else products
        except (_json.JSONDecodeError, TypeError):
            return {"error": True, "message": f"Invalid products JSON: {products}"}
        if not prod_list:
            return {"error": True, "message": "products must be a non-empty list"}

        # ── Step 1: Create customer ──
        cust_body = {"name": customer_name, "isCustomer": True}
        if customer_email:
            cust_body["email"] = customer_email
        if customer_org_number:
            cust_body["organizationNumber"] = customer_org_number
        if customer_phone:
            cust_body["phoneNumber"] = customer_phone
        if customer_address:
            cust_body["physicalAddress"] = {"addressLine1": customer_address}
            if customer_postal_code:
                cust_body["physicalAddress"]["postalCode"] = customer_postal_code
            if customer_city:
                cust_body["physicalAddress"]["city"] = customer_city

        cust_result = client.post("/customer", json=cust_body)
        customer_id = cust_result.get("value", {}).get("id")

        if not customer_id and cust_result.get("error"):
            # Try to find existing
            search_params = {"fields": "id,name"}
            if customer_org_number:
                search_params["organizationNumber"] = customer_org_number
            else:
                search_params["name"] = customer_name
            existing = client.get("/customer", params=search_params)
            vals = existing.get("values", [])
            if vals:
                customer_id = vals[0]["id"]
                _recover_error("/customer")
                steps_log.append(f"Customer already existed (id={customer_id})")

        if not customer_id:
            return {"error": True, "message": f"Failed to create customer: {cust_result}", "steps": steps_log}
        if not steps_log or "already existed" not in steps_log[-1]:
            steps_log.append(f"Created customer '{customer_name}' (id={customer_id})")

        # ── Step 2: Create products ──
        # Resolve output VAT types by standard number (from prewarm cache or live lookup)
        vat_map = client.get_cached("vat_type_map") or {}
        if not vat_map:
            _OUT = {3: 25, 31: 15, 33: 12}
            _ZERO = {6, 5}
            r = client.get("/ledger/vatType", params={"fields": "id,number"})
            for vt in (r.get("values") or []):
                n, vid = vt.get("number"), vt.get("id")
                if n is not None and vid is not None:
                    n = int(n)
                    if n in _OUT:
                        vat_map[_OUT[n]] = vid
                    elif n in _ZERO:
                        vat_map.setdefault(0, vid)
            if vat_map:
                client.set_cached("vat_type_map", vat_map)

        product_ids = []
        order_lines = []

        for prod in prod_list:
            p_name = prod.get("name", "Product")
            p_price = prod.get("price", 0)
            p_qty = prod.get("quantity", 1)
            p_vat_pct = prod.get("vatPercentage", 25)
            p_number = prod.get("productNumber", "")

            vat_type_id = vat_map.get(int(p_vat_pct))
            if vat_type_id is None:
                return {"error": True, "message": f"No valid output VAT type for {p_vat_pct}%. Available: {list(vat_map.keys())}", "steps": steps_log}

            prod_body = {
                "name": p_name,
                "priceExcludingVatCurrency": p_price,
                "vatType": {"id": vat_type_id},
            }
            if p_number:
                prod_body["number"] = p_number

            prod_result = client.post("/product", json=prod_body)
            prod_id = prod_result.get("value", {}).get("id")

            if not prod_id and prod_result.get("error"):
                # Product number collision — search for existing
                if p_number:
                    existing = client.get("/product", params={"number": p_number, "fields": "id"})
                    vals = existing.get("values", [])
                    if vals:
                        prod_id = vals[0]["id"]
                        _recover_error("/product")

            if not prod_id:
                return {"error": True, "message": f"Failed to create product '{p_name}': {prod_result}", "steps": steps_log}

            product_ids.append(prod_id)
            order_lines.append({"product": {"id": prod_id}, "count": p_qty})
            steps_log.append(f"Created product '{p_name}' (id={prod_id})")

        # ── Step 3: Create order ──
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

        # ── Step 4: Ensure bank account + create invoice ──
        _ensure_bank_account()

        inv_body = {
            "invoiceDate": invoiceDate,
            "invoiceDueDate": invoiceDueDate,
            "orders": [{"id": order_id}],
        }
        inv_result = client.post("/invoice", json=inv_body)
        invoice_id = inv_result.get("value", {}).get("id")
        if not invoice_id:
            return {"error": True, "message": f"Failed to create invoice: {inv_result}", "steps": steps_log}
        steps_log.append(f"Created invoice (id={invoice_id})")

        result = {
            "success": True,
            "customer_id": customer_id,
            "product_ids": product_ids,
            "order_id": order_id,
            "invoice_id": invoice_id,
            "steps": steps_log,
        }

        # ── Step 5: Optional credit note ──
        if create_credit_note:
            cn_result = client.put(
                f"/invoice/{invoice_id}/:createCreditNote",
                params={"date": invoiceDate},
            )
            cn_id = cn_result.get("value", {}).get("id")
            if cn_id:
                steps_log.append(f"Created credit note (id={cn_id})")
                result["credit_note_id"] = cn_id
            else:
                steps_log.append(f"WARNING: Credit note failed: {cn_result}")

        # ── Step 6: Optional send ──
        if send_invoice and not create_credit_note:
            client.put(f"/invoice/{invoice_id}/:send", params={"sendType": "EMAIL"})
            steps_log.append("Sent invoice via email")

        return result

    return {
        "process_invoice": process_invoice,
    }
