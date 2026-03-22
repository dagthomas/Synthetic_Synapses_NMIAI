"""Compound tool: process entire invoice workflow in one deterministic call.

Handles: create customer → create product(s) → create order → create invoice
         → optionally create credit note or send invoice.
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_invoice_tools(client: TripletexClient) -> dict:
    """Build the compound invoice processing tool."""

    from tools._helpers import (
        find_or_create_customer, find_or_create_product,
        ensure_bank_account,
    )

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

        # ── Step 1: Find or create customer ──
        customer_id = find_or_create_customer(
            client, name=customer_name, email=customer_email,
            org_number=customer_org_number, phone=customer_phone,
            address=customer_address, postal_code=customer_postal_code,
            city=customer_city, steps_log=steps_log,
        )
        if not customer_id:
            return {"error": True, "message": f"Failed to create customer '{customer_name}'", "steps": steps_log}

        # ── Step 2: Find or create products ──
        product_ids = []
        order_lines = []

        for prod in prod_list:
            p_name = prod.get("name", "Product")
            p_price = prod.get("price", 0)
            p_qty = prod.get("quantity", 1)
            p_vat_pct = prod.get("vatPercentage", 25)
            p_number = prod.get("productNumber", "")

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
