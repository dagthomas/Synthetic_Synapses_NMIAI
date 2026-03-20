from tripletex_client import TripletexClient


def build_incoming_invoice_tools(client: TripletexClient) -> dict:
    """Build incoming invoice (inngaaende faktura) tools."""

    def create_incoming_invoice(
        invoiceDate: str,
        dueDate: str,
        supplierId: int,
        invoiceNumber: str = "",
        amount: float = 0.0,
        accountId: int = 0,
    ) -> dict:
        """Create an incoming invoice (purchase invoice / leverandoerfaktura).

        Args:
            invoiceDate: Invoice date YYYY-MM-DD.
            dueDate: Due date YYYY-MM-DD.
            supplierId: Supplier ID.
            invoiceNumber: Vendor's invoice number.
            amount: Invoice amount.
            accountId: Debit account ID (0 for default).

        Returns:
            Created incoming invoice or error.
        """
        body = {
            "invoiceDate": invoiceDate,
            "dueDate": dueDate,
            "supplier": {"id": supplierId},
        }
        if invoiceNumber:
            body["invoiceNumber"] = invoiceNumber
        return client.post("/supplierInvoice", json=body)

    def search_incoming_invoices(
        supplierId: int = 0,
        invoiceDateFrom: str = "",
        invoiceDateTo: str = "",
    ) -> dict:
        """Search for incoming invoices.

        Args:
            supplierId: Filter by supplier ID (0 for all).
            invoiceDateFrom: Filter from date YYYY-MM-DD (defaults to 1 year ago).
            invoiceDateTo: Filter to date YYYY-MM-DD (defaults to today).

        Returns:
            A list of incoming invoices.
        """
        from datetime import date as dt_date, timedelta
        if not invoiceDateFrom:
            invoiceDateFrom = (dt_date.today() - timedelta(days=365)).isoformat()
        if not invoiceDateTo:
            invoiceDateTo = dt_date.today().isoformat()
        params = {"fields": "id,invoiceNumber,invoiceDate,dueDate,amount,supplier,voucher"}
        params["invoiceDateFrom"] = invoiceDateFrom
        params["invoiceDateTo"] = invoiceDateTo
        if supplierId:
            params["supplierId"] = supplierId
        return client.get("/supplierInvoice", params=params)

    return {
        "create_incoming_invoice": create_incoming_invoice,
        "search_incoming_invoices": search_incoming_invoices,
    }
