from tripletex_client import TripletexClient


def build_expense_receipt_tools(client: TripletexClient) -> dict:
    """Build expense receipt (utgiftskvittering) tools."""

    def register_expense_receipt(
        amountIncludingVat: float = 0.0,
        expenseAccountNumber: int = 0,
        vatPercentage: int = 25,
        receiptDate: str = "",
        description: str = "",
        paymentAccountNumber: int = 1920,
        departmentId: int = 0,
        projectId: int = 0,
    ) -> dict:
        """Register an expense from a receipt (kvittering/utlegg) as a ledger voucher.

        Creates a voucher with:
        - Debit on the expense account (with input VAT type for automatic VAT split)
        - Credit on the payment account (1920=bank, 1900=cash/kasse)

        This is for expenses already paid (receipt/kvittering), NOT unpaid supplier invoices.
        For unpaid supplier invoices, use create_incoming_invoice instead.

        Args:
            amountIncludingVat: Total receipt amount INCLUDING VAT. REQUIRED.
            expenseAccountNumber: Expense account number. REQUIRED. Common accounts:
                - 6300: Leie lokale (office rent)
                - 6340: Lys, varme (utilities)
                - 6500: Verktøy/inventar (tools/equipment <15000)
                - 6590: Annen kontorkostnad (other office costs)
                - 6800: Kontorrekvisita (office supplies)
                - 6900: Telefon/internett (phone/internet)
                - 7100: Bilkostnader (car costs)
                - 7140: Reisekostnad (travel costs)
                - 7350: Representasjon (entertainment/client meals)
                - 7770: Rekvisita/forbruksmateriell (consumables)
            vatPercentage: VAT rate: 25 (standard), 15 (food), 12 (transport), 0 (exempt). Default 25.
            receiptDate: Receipt/purchase date YYYY-MM-DD. REQUIRED.
            description: What the expense is for (e.g. "Kontorrekvisita fra Staples"). REQUIRED.
            paymentAccountNumber: Account that was debited for payment. Default 1920 (bank).
                - 1920: Bankkonto (bank account) — most common
                - 1900: Kontanter/kasse (cash)
            departmentId: Department ID to link the expense to. Optional.
            projectId: Project ID to link the expense to. Optional.

        Returns:
            Created voucher with postings, or error.
        """
        if not receiptDate:
            return {"error": True, "message": "receiptDate is required (YYYY-MM-DD)"}
        if not expenseAccountNumber:
            return {"error": True, "message": "expenseAccountNumber is required (e.g. 6800 for office supplies)"}
        if not amountIncludingVat:
            return {"error": True, "message": "amountIncludingVat is required (total amount including VAT)"}
        if not description:
            return {"error": True, "message": "description is required (what the expense is for)"}

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

        # Look up payment account ID (cached per-request)
        payment_cache_key = f"acct_{paymentAccountNumber}"
        payment_id = client.get_cached(payment_cache_key)
        if not payment_id:
            payment_result = client.get("/ledger/account", params={
                "number": str(paymentAccountNumber), "fields": "id", "count": 1,
            })
            payment_accts = payment_result.get("values", [])
            if not payment_accts:
                return {"error": True, "message": f"Payment account {paymentAccountNumber} not found"}
            payment_id = payment_accts[0]["id"]
            client.set_cached(payment_cache_key, payment_id)

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
        if vat_type_id is None:
            return {"error": True, "message": f"Unsupported VAT rate {vatPercentage}%. Use 25, 15, 12, or 0."}

        amt = round(amountIncludingVat, 2)

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
        if departmentId:
            expense_posting["department"] = {"id": departmentId}
        if projectId:
            expense_posting["project"] = {"id": projectId}

        # Build payment posting (credit)
        payment_posting = {
            "row": 2,
            "account": {"id": payment_id},
            "amountGross": -amt,
            "amountGrossCurrency": -amt,
            "description": description,
        }

        body = {
            "date": receiptDate,
            "description": description,
            "postings": [expense_posting, payment_posting],
        }

        return client.post("/ledger/voucher", json=body)

    return {
        "register_expense_receipt": register_expense_receipt,
    }
