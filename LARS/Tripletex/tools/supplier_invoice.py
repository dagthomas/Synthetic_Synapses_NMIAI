from tripletex_client import TripletexClient


def build_supplier_invoice_tools(client: TripletexClient) -> dict:
    """Build supplier invoice (leverandoerfaktura) tools."""

    def search_supplier_invoices(
        supplierId: int = 0,
        invoiceDateFrom: str = "",
        invoiceDateTo: str = "",
    ) -> dict:
        """Search for supplier invoices.

        Args:
            supplierId: Filter by supplier ID (0 for all).
            invoiceDateFrom: Filter from date YYYY-MM-DD (defaults to 1 year ago).
            invoiceDateTo: Filter to date YYYY-MM-DD (defaults to today).

        Returns:
            A list of supplier invoices.
        """
        from datetime import date as dt_date, timedelta
        if not invoiceDateFrom:
            invoiceDateFrom = (dt_date.today() - timedelta(days=365)).isoformat()
        if not invoiceDateTo:
            invoiceDateTo = dt_date.today().isoformat()
        params = {"fields": "id,invoiceNumber,invoiceDate,amount,supplier,voucher"}
        params["invoiceDateFrom"] = invoiceDateFrom
        params["invoiceDateTo"] = invoiceDateTo
        if supplierId:
            params["supplierId"] = supplierId
        return client.get("/supplierInvoice", params=params)

    def approve_supplier_invoice(invoice_id: int) -> dict:
        """Approve a supplier invoice.

        Args:
            invoice_id: ID of the supplier invoice.

        Returns:
            Confirmation or error.
        """
        return client.put(f"/supplierInvoice/{invoice_id}/:approve")

    def reject_supplier_invoice(
        invoice_id: int,
        comment: str = "",
    ) -> dict:
        """Reject a supplier invoice.

        Args:
            invoice_id: ID of the supplier invoice.
            comment: Reason for rejection.

        Returns:
            Confirmation or error.
        """
        params = {}
        if comment:
            params["comment"] = comment
        return client.put(f"/supplierInvoice/{invoice_id}/:reject", params=params if params else None)

    def add_supplier_invoice_payment(
        invoice_id: int,
        paymentType: int = 0,
    ) -> dict:
        """Register payment for a supplier invoice.

        Args:
            invoice_id: Voucher ID of the supplier invoice.
            paymentType: Payment type ID (0 for auto-detect last used).

        Returns:
            Confirmation or error.
        """
        params = {}
        if paymentType:
            params["paymentType"] = paymentType
        return client.post(f"/supplierInvoice/{invoice_id}/:addPayment", params=params if params else None)

    return {
        "search_supplier_invoices": search_supplier_invoices,
        "approve_supplier_invoice": approve_supplier_invoice,
        "reject_supplier_invoice": reject_supplier_invoice,
        "add_supplier_invoice_payment": add_supplier_invoice_payment,
    }
