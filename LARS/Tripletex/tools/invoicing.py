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
        try:
            lines = _json.loads(orderLines) if isinstance(orderLines, str) else orderLines
        except (_json.JSONDecodeError, TypeError) as e:
            return {"error": True, "message": f"Invalid JSON in orderLines: {e}. Expected: '[{{\"product_id\": 1, \"count\": 2}}]'"}
        if not lines or not isinstance(lines, list):
            return {"error": True, "message": "orderLines must be a non-empty JSON array of objects with 'product_id' and 'count'"}
        for i, line in enumerate(lines):
            if "product_id" not in line:
                return {"error": True, "message": f"Order line {i} missing 'product_id'. Each line needs: {{\"product_id\": <id>, \"count\": <qty>}}"}
            if "count" not in line:
                return {"error": True, "message": f"Order line {i} missing 'count'. Each line needs: {{\"product_id\": <id>, \"count\": <qty>}}"}
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

    def search_invoices(invoiceDateFrom: str = "", invoiceDateTo: str = "", customerId: int = 0) -> dict:
        """Search for invoices.

        Args:
            invoiceDateFrom: Filter from date YYYY-MM-DD.
            invoiceDateTo: Filter to date YYYY-MM-DD.
            customerId: Filter by customer ID (0 for all).

        Returns:
            A list of invoices.
        """
        params = {"fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,customer,amount,amountOutstanding"}
        if invoiceDateFrom:
            params["invoiceDateFrom"] = invoiceDateFrom
        if invoiceDateTo:
            params["invoiceDateTo"] = invoiceDateTo
        if customerId:
            params["customerId"] = customerId
        return client.get("/invoice", params=params)

    def search_orders(orderDateFrom: str = "", orderDateTo: str = "", customerId: int = 0) -> dict:
        """Search for orders.

        Args:
            orderDateFrom: Filter from date YYYY-MM-DD (defaults to 1 year ago).
            orderDateTo: Filter to date YYYY-MM-DD (defaults to today).
            customerId: Filter by customer ID (0 for all).

        Returns:
            A list of orders.
        """
        from datetime import date as dt_date, timedelta
        if not orderDateFrom:
            orderDateFrom = (dt_date.today() - timedelta(days=365)).isoformat()
        if not orderDateTo:
            orderDateTo = dt_date.today().isoformat()
        params = {
            "fields": "id,number,orderDate,deliveryDate,customer,isClosed",
            "orderDateFrom": orderDateFrom,
            "orderDateTo": orderDateTo,
        }
        if customerId:
            params["customerId"] = customerId
        return client.get("/order", params=params)

    def update_order(order_id: int, deliveryDate: str = "", isClosed: bool = False) -> dict:
        """Update an order.

        Args:
            order_id: ID of the order.
            deliveryDate: New delivery date YYYY-MM-DD (empty to keep).
            isClosed: Set True to close the order.

        Returns:
            Updated order or error.
        """
        _WRITABLE = {"id", "version", "customer", "orderDate", "deliveryDate", "isClosed", "orderLines", "invoiceComment"}
        current = client.get(f"/order/{order_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("customer"), dict):
            body["customer"] = {"id": body["customer"]["id"]}
        if deliveryDate:
            body["deliveryDate"] = deliveryDate
        if isClosed:
            body["isClosed"] = True
        return client.put(f"/order/{order_id}", json=body)

    def delete_order(order_id: int) -> dict:
        """Delete an order.

        Args:
            order_id: ID of the order.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/order/{order_id}")

    def send_invoice(invoice_id: int, sendType: str = "EMAIL") -> dict:
        """Send an invoice to the customer.

        Args:
            invoice_id: ID of the invoice to send.
            sendType: Send method - 'EMAIL', 'EHF', or 'EFAKTURA'.

        Returns:
            Confirmation or error.
        """
        return client.put(f"/invoice/{invoice_id}/:send", params={"sendType": sendType})

    def create_invoice_reminder(invoice_id: int, reminderType: str = "SOFT_REMINDER", date: str = "", dispatchType: str = "EMAIL") -> dict:
        """Create a reminder for an overdue invoice.

        Args:
            invoice_id: ID of the invoice.
            reminderType: Type - 'SOFT_REMINDER', 'REMINDER', or 'NOTICE_OF_DEBT_COLLECTION'.
            date: Reminder date YYYY-MM-DD (empty to auto-set after due date).
            dispatchType: How to send - 'EMAIL', 'EHF', or 'EFAKTURA'.

        Returns:
            Confirmation or error.
        """
        from datetime import date as dt_date, timedelta
        if not date:
            # Reminder date must be on or after the invoice due date
            inv = client.get(f"/invoice/{invoice_id}", params={"fields": "invoiceDueDate"})
            due = inv.get("value", {}).get("invoiceDueDate", "") if inv.get("value") else ""
            if due:
                due_dt = dt_date.fromisoformat(due)
                date = max(dt_date.today(), due_dt).isoformat()
            else:
                date = dt_date.today().isoformat()
        params = {"type": reminderType, "date": date, "dispatchType": dispatchType}
        return client.put(f"/invoice/{invoice_id}/:createReminder", params=params)

    return {
        "create_order": create_order,
        "create_invoice": create_invoice,
        "register_payment": register_payment,
        "create_credit_note": create_credit_note,
        "search_invoices": search_invoices,
        "search_orders": search_orders,
        "update_order": update_order,
        "delete_order": delete_order,
        "send_invoice": send_invoice,
        "create_invoice_reminder": create_invoice_reminder,
    }
