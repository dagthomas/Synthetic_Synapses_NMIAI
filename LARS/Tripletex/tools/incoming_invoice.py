from tripletex_client import TripletexClient


def build_incoming_invoice_tools(client: TripletexClient) -> dict:
    """Build incoming invoice (inngaaende faktura) tools."""

    def create_incoming_invoice(
        supplierId: int,
        invoiceNumber: str = "",
        amountIncludingVat: float = 0.0,
        expenseAccountNumber: int = 0,
        vatPercentage: int = 25,
        invoiceDate: str = "", # Make invoiceDate optional with a default empty string
    ) -> dict:
        """Create an incoming/supplier invoice (leverandoerfaktura) with VAT postings.

        Creates a ledger voucher with:
        - Debit on the expense account (with input VAT type for automatic VAT calculation)
        - Credit on account 2400 (leverandoergjeld) linked to the supplier

        Args:
            supplierId: Supplier ID (from create_supplier).
            invoiceNumber: Vendor's invoice number (e.g. INV-2026-3749).
            amountIncludingVat: Total invoice amount INCLUDING VAT.
            expenseAccountNumber: Expense account number (e.g. 6590 for office services).
            vatPercentage: VAT rate: 25 (high/hoey), 15 (medium), 12 (low), or 0 (none). Default 25.
            invoiceDate: Invoice date YYYY-MM-DD. REQUIRED.

        Returns:
            Created voucher with postings, or error.
        """
        if not invoiceDate: # Add explicit check for invoiceDate
            return {"error": True, "message": "invoiceDate is required (YYYY-MM-DD). It must be provided in the prompt."}
        if not expenseAccountNumber:
            return {"error": True, "message": "expenseAccountNumber is required (e.g. 6590 for office services)"}
        if not amountIncludingVat:
            return {"error": True, "message": "amountIncludingVat is required (total amount including VAT)"}

        # Look up expense account ID (cached per-request)
        expense_cache_key = f"acct_{expenseAccountNumber}"
        expense_id = client.get_cached(expense_cache_key)
        if not expense_id:
            expense_result = client.get("/ledger/account", params={
                "number": str(expenseAccountNumber), "fields": "id", "count": 1,
            })
            expense_accts = expense_result.get("values", [])
            if not expense_accts:
                return {"error": True, "message": f"Expense account {expenseAccountNumber} not found"}
            expense_id = expense_accts[0]["id"]
            client.set_cached(expense_cache_key, expense_id)

        # Look up payables account 2400 (cached per-request)
        payables_id = client.get_cached("acct_2400")
        if not payables_id:
            payables_result = client.get("/ledger/account", params={
                "number": "2400", "fields": "id", "count": 1,
            })
            payables_accts = payables_result.get("values", [])
            if not payables_accts:
                return {"error": True, "message": "Payables account 2400 not found"}
            payables_id = payables_accts[0]["id"]
            client.set_cached("acct_2400", payables_id)

        # Map VAT percentage to vatType ID
        # 1 = Fradrag inngaaende avgift, hoey sats (25%)
        # 11 = Fradrag inngaaende avgift, middels sats (15%)
        # 13 = Fradrag inngaaende avgift, lav sats (12%)
        # 0 = No VAT
        vat_type_map = {25: 1, 15: 11, 12: 13, 0: 0}
        vat_type_id = vat_type_map.get(vatPercentage)
        if vat_type_id is None:
            return {"error": True, "message": f"Unsupported VAT rate {vatPercentage}%. Use 25, 15, 12, or 0."}

        amt = round(amountIncludingVat, 2)
        description = f"{invoiceNumber} - supplier invoice" if invoiceNumber else "Supplier invoice"

        # Build voucher with two postings:
        # 1. Expense account (debit, gross amount, with vatType for auto-VAT split)
        # 2. Payables 2400 (credit, gross amount, linked to supplier)
        postings = [
            {
                "row": 1,
                "account": {"id": expense_id},
                "amountGross": amt,
                "amountGrossCurrency": amt,
            },
            {
                "row": 2,
                "account": {"id": payables_id},
                "amountGross": -amt,
                "amountGrossCurrency": -amt,
                "supplier": {"id": supplierId},
            },
        ]

        # Add vatType to expense posting if VAT applies
        if vat_type_id > 0:
            postings[0]["vatType"] = {"id": vat_type_id}

        body = {
            "date": invoiceDate,
            "description": description,
            "postings": postings,
        }
        return client.post("/ledger/voucher", json=body)

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
        params = {"fields": "id,invoiceNumber,invoiceDate,amount,supplier,voucher"}
        params["invoiceDateFrom"] = invoiceDateFrom
        params["invoiceDateTo"] = invoiceDateTo
        if supplierId:
            params["supplierId"] = supplierId
        return client.get("/supplierInvoice", params=params)

    return {
        "create_incoming_invoice": create_incoming_invoice,
        "search_incoming_invoices": search_incoming_invoices,
    }
