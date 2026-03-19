import json as _json

from tripletex_client import TripletexClient


def build_invoicing_tools(client: TripletexClient) -> dict:
    """Build invoicing-related tools."""

    def create_order(
        customer_id: int,
        deliveryDate: str,
        orderLines: str,
    ) -> dict:
        """Create a sales order for a customer. Required before creating an invoice.

        Args:
            customer_id: The ID of the customer (must exist).
            deliveryDate: Delivery date in YYYY-MM-DD format.
            orderLines: JSON string of order lines, each with 'product_id', 'count', and optionally 'unitPriceExcludingVat'. Example: '[{"product_id": 1, "count": 2}]'

        Returns:
            The created order with id, or an error message.
        """
        lines = _json.loads(orderLines) if isinstance(orderLines, str) else orderLines
        formatted_lines = []
        for line in lines:
            entry = {
                "product": {"id": line["product_id"]},
                "count": line["count"],
            }
            if "unitPriceExcludingVat" in line:
                entry["unitPriceExcludingVat"] = line["unitPriceExcludingVat"]
            formatted_lines.append(entry)

        body = {
            "customer": {"id": customer_id},
            "orderDate": deliveryDate,
            "deliveryDate": deliveryDate,
            "orderLines": formatted_lines,
        }
        return client.post("/order", json=body)

    def create_invoice(
        invoiceDate: str,
        invoiceDueDate: str,
        order_id: int,
    ) -> dict:
        """Create an invoice from an existing order.

        Args:
            invoiceDate: Invoice date in YYYY-MM-DD format.
            invoiceDueDate: Payment due date in YYYY-MM-DD format.
            order_id: The ID of the order to invoice.

        Returns:
            The created invoice with id, or an error message.
        """
        body = {
            "invoiceDate": invoiceDate,
            "invoiceDueDate": invoiceDueDate,
            "orders": [{"id": order_id}],
        }
        return client.post("/invoice", json=body)

    def register_payment(
        invoice_id: int,
        amount: float,
        paymentDate: str,
    ) -> dict:
        """Register a payment for an invoice.

        Args:
            invoice_id: The ID of the invoice being paid.
            amount: The payment amount (including VAT).
            paymentDate: Payment date in YYYY-MM-DD format.

        Returns:
            Confirmation of payment or an error message.
        """
        # Resolve paymentTypeId (e.g. "Betalt til bank")
        pt_result = client.get("/invoice/paymentType", params={"fields": "id,description", "count": 10})
        payment_types = pt_result.get("values", [])
        payment_type_id = payment_types[0]["id"] if payment_types else 0

        return client.put(
            f"/invoice/{invoice_id}/:payment",
            params={
                "paymentDate": paymentDate,
                "paymentTypeId": payment_type_id,
                "paidAmount": amount,
            },
        )

    def create_credit_note(invoice_id: int, date: str = "") -> dict:
        """Create a credit note for an existing invoice.

        Args:
            invoice_id: The ID of the invoice to credit.
            date: Credit note date in YYYY-MM-DD format. If empty, uses today's date.

        Returns:
            The created credit note or an error message.
        """
        from datetime import date as dt_date
        credit_date = date if date else dt_date.today().isoformat()
        return client.put(
            f"/invoice/{invoice_id}/:createCreditNote",
            params={"date": credit_date},
        )

    return {
        "create_order": create_order,
        "create_invoice": create_invoice,
        "register_payment": register_payment,
        "create_credit_note": create_credit_note,
    }
