"""Compound tool: order → invoice → payment workflow in one deterministic call.

Handles: create customer → create product(s) → create order → ensure bank account
         → create invoice → register payment.
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_order_invoice_payment_tools(client: TripletexClient) -> dict:
    """Build the compound order-to-invoice-with-payment tool."""

    from tools._helpers import (
        find_or_create_customer, find_or_create_product,
        ensure_bank_account, resolve_payment_type_id,
    )

    def process_order_to_invoice_with_payment(
        customer_name: str,
        customer_email: str = "",
        customer_org_number: str = "",
        product_name: str = "",
        product_price: float = 0.0,
        product_number: str = "",
        quantity: int = 1,
        vat_percentage: int = 25,
        products: str = "[]",
        invoiceDate: str = "",
        invoiceDueDate: str = "",
        paymentDate: str = "",
    ) -> dict:
        """Process order → invoice → payment workflow in one call.

        This compound tool handles the ENTIRE order-to-invoice-with-payment workflow:
        1. Creates the customer (or finds existing on collision)
        2. Creates all products with correct VAT types
        3. Creates the order with all order lines
        4. Ensures bank account exists on ledger account 1920
        5. Creates the invoice from the order
        6. Registers full payment on the invoice

        Supports both single-product (product_name/product_price) and multi-product (products JSON).

        Args:
            customer_name: Customer company name. REQUIRED.
            customer_email: Customer email address. Optional.
            customer_org_number: Organization number. Optional.
            product_name: Single product name. Ignored if products JSON is provided.
            product_price: Single product price EXCLUDING VAT. Ignored if products JSON provided.
            product_number: Single product number/SKU. Optional.
            quantity: Quantity for single product. Default 1.
            vat_percentage: VAT rate for single product (25/15/12/0). Default 25.
            products: JSON string for MULTIPLE products: [{"name": "X", "price": 100, "quantity": 2, "vatPercentage": 25, "productNumber": ""}].
                      If non-empty array, overrides product_name/product_price.
            invoiceDate: Invoice date YYYY-MM-DD (defaults to today).
            invoiceDueDate: Due date YYYY-MM-DD (defaults to invoiceDate).
            paymentDate: Payment date YYYY-MM-DD (defaults to today).

        Returns:
            Summary with customer_id, product_ids, order_id, invoice_id, payment info.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        if not invoiceDate:
            invoiceDate = today
        if not invoiceDueDate:
            invoiceDueDate = invoiceDate
        if not paymentDate:
            paymentDate = today

        # ── Resolve product list ──
        prod_list = []
        try:
            parsed = _json.loads(products) if isinstance(products, str) else products
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                prod_list = parsed
        except (_json.JSONDecodeError, TypeError):
            pass

        if not prod_list:
            if not product_name:
                return {"error": True, "message": "Either product_name or products JSON must be provided"}
            prod_list = [{
                "name": product_name,
                "price": product_price,
                "quantity": quantity,
                "vatPercentage": vat_percentage,
                "productNumber": product_number,
            }]

        # ── Step 1: Find or create customer ──
        customer_id = find_or_create_customer(
            client, name=customer_name, email=customer_email,
            org_number=customer_org_number, steps_log=steps_log,
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

        # Get actual invoice amount from response (handles VAT correctly)
        inv_amount = inv_result.get("value", {}).get("amount", 0)
        if not inv_amount:
            # Fallback: calculate from product prices + VAT
            inv_amount = 0
            for prod in prod_list:
                p = prod.get("price", 0) * prod.get("quantity", 1)
                vp = prod.get("vatPercentage", prod.get("vat_percentage", 25))
                inv_amount += p * (1 + vp / 100)
            inv_amount = round(inv_amount, 2)

        steps_log.append(f"Created invoice (id={invoice_id}, amount={inv_amount})")

        # ── Step 5: Register payment ──
        payment_type_id = resolve_payment_type_id(client)

        pay_result = client.put(
            f"/invoice/{invoice_id}/:payment",
            params={
                "paymentDate": paymentDate,
                "paymentTypeId": payment_type_id,
                "paidAmount": round(inv_amount, 2),
                "paidAmountCurrency": round(inv_amount, 2),
            },
        )
        steps_log.append(f"Registered payment ({inv_amount} NOK on {paymentDate})")

        return {
            "success": True,
            "customer_id": customer_id,
            "product_ids": product_ids,
            "order_id": order_id,
            "invoice_id": invoice_id,
            "payment_amount": inv_amount,
            "steps": steps_log,
        }

    return {
        "process_order_to_invoice_with_payment": process_order_to_invoice_with_payment,
    }
