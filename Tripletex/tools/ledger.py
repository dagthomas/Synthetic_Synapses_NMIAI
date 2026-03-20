import json as _json

from tripletex_client import TripletexClient


def build_ledger_tools(client: TripletexClient) -> dict:
    """Build ledger tools for Tier 3 tasks."""

    def get_ledger_accounts(number: str = "", name: str = "") -> dict:
        """Search the chart of accounts.

        Args:
            number: Filter by account number (partial match).
            name: Filter by account name (partial match).

        Returns:
            A list of matching ledger accounts.
        """
        params = {"fields": "id,number,name,description"}
        if number:
            params["number"] = number
        if name:
            params["name"] = name
        return client.get("/ledger/account", params=params)

    def get_ledger_postings(dateFrom: str, dateTo: str, accountNumber: str = "") -> dict:
        """Query ledger postings within a date range.

        Args:
            dateFrom: Start date in YYYY-MM-DD format.
            dateTo: End date in YYYY-MM-DD format.
            accountNumber: Optional filter by account number.

        Returns:
            A list of ledger postings.
        """
        params = {
            "dateFrom": dateFrom,
            "dateTo": dateTo,
            "fields": "id,date,description,amount,account",
        }
        if accountNumber:
            params["accountNumber"] = accountNumber
        return client.get("/ledger/posting", params=params)

    def create_voucher(date: str, description: str, postings: str) -> dict:
        """Create a ledger voucher with postings. Used for corrections and manual entries.

        Args:
            date: Voucher date in YYYY-MM-DD format.
            description: Description of the voucher.
            postings: JSON string of postings, each with 'accountId' (ledger account ID) and 'amount' (positive=debit, negative=credit). Example: '[{"accountId": 123, "amount": 1000}, {"accountId": 456, "amount": -1000}]'. You can also use 'accountNumber' and the tool will look up the account ID.

        Returns:
            The created voucher with id, or an error message.
        """
        posting_list = _json.loads(postings) if isinstance(postings, str) else postings
        formatted = []
        for i, p in enumerate(posting_list):
            entry = {"row": i + 1}
            if "accountId" in p:
                entry["account"] = {"id": p["accountId"]}
            elif "accountNumber" in p:
                # Look up account by number
                acct_result = client.get("/ledger/account", params={"number": str(p["accountNumber"]), "fields": "id", "count": 1})
                accts = acct_result.get("values", [])
                if accts:
                    entry["account"] = {"id": accts[0]["id"]}
                else:
                    return {"error": True, "message": f"Account {p['accountNumber']} not found"}
            # Tripletex uses 'amount' (positive=debit, negative=credit)
            amount = p.get("amount", 0)
            if not amount:
                # Support legacy debit/credit format
                debit = p.get("debitAmount", 0)
                credit = p.get("creditAmount", 0)
                amount = debit - credit
            entry["amount"] = amount
            entry["amountCurrency"] = amount
            formatted.append(entry)
        body = {
            "date": date,
            "description": description,
            "postings": formatted,
        }
        return client.post("/ledger/voucher", json=body)

    def delete_voucher(voucher_id: int) -> dict:
        """Delete a ledger voucher.

        Args:
            voucher_id: The ID of the voucher to delete.

        Returns:
            Confirmation or error message.
        """
        return client.delete(f"/ledger/voucher/{voucher_id}")

    def create_ledger_account(number: int, name: str, description: str = "") -> dict:
        """Create a new ledger account in the chart of accounts.

        Args:
            number: Account number (e.g. 1920, 7700).
            name: Account name.
            description: Optional description.

        Returns:
            Created account or error.
        """
        body = {"number": number, "name": name}
        if description:
            body["description"] = description
        return client.post("/ledger/account", json=body)

    def reverse_voucher(voucher_id: int, date: str = "") -> dict:
        """Reverse a voucher, creating a new counter-voucher.

        Args:
            voucher_id: ID of the voucher to reverse.
            date: Date for the reversal YYYY-MM-DD (empty for today).

        Returns:
            The reversed voucher or error.
        """
        params = {}
        if date:
            params["date"] = date
        return client.put(f"/ledger/voucher/{voucher_id}/:reverse", params=params if params else None)

    def create_opening_balance(voucherDate: str, balancePostings: str) -> dict:
        """Create an opening balance. Zeroes out all movements before this date.

        Args:
            voucherDate: Date for opening balance YYYY-MM-DD (must be first day of month).
            balancePostings: JSON string of postings, each with 'accountId' (or 'accountNumber') and 'amount'. Example: '[{"accountNumber": 1920, "amount": 50000}]'

        Returns:
            Created opening balance or error.
        """
        posting_list = _json.loads(balancePostings) if isinstance(balancePostings, str) else balancePostings
        formatted = []
        for p in posting_list:
            entry = {}
            if "accountId" in p:
                entry["account"] = {"id": p["accountId"]}
            elif "accountNumber" in p:
                acct = client.get("/ledger/account", params={"number": str(p["accountNumber"]), "fields": "id", "count": 1})
                accts = acct.get("values", [])
                if accts:
                    entry["account"] = {"id": accts[0]["id"]}
                else:
                    return {"error": True, "message": f"Account {p['accountNumber']} not found"}
            entry["amount"] = p.get("amount", 0)
            formatted.append(entry)
        body = {"voucherDate": voucherDate, "balancePostings": formatted}
        return client.post("/ledger/voucher/openingBalance", json=body)

    def search_vouchers(dateFrom: str = "", dateTo: str = "") -> dict:
        """Search for vouchers within a date range.

        Args:
            dateFrom: Start date YYYY-MM-DD.
            dateTo: End date YYYY-MM-DD.

        Returns:
            A list of vouchers.
        """
        params = {"fields": "id,date,description,voucherType,number"}
        if dateFrom:
            params["dateFrom"] = dateFrom
        if dateTo:
            params["dateTo"] = dateTo
        return client.get("/ledger/voucher", params=params)

    def update_voucher(voucher_id: int, date: str = "", description: str = "") -> dict:
        """Update a voucher's date or description.

        Args:
            voucher_id: ID of the voucher.
            date: New date YYYY-MM-DD (empty to keep).
            description: New description (empty to keep).

        Returns:
            Updated voucher or error.
        """
        current = client.get(f"/ledger/voucher/{voucher_id}", params={"fields": "*"})
        full = current.get("value", {})
        _WRITABLE = {"id", "version", "date", "description", "voucherType", "postings", "externalVoucherNumber"}
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("voucherType"), dict):
            body["voucherType"] = {"id": body["voucherType"]["id"]}
        if date:
            body["date"] = date
        if description:
            body["description"] = description
        return client.put(f"/ledger/voucher/{voucher_id}", json=body)

    return {
        "get_ledger_accounts": get_ledger_accounts,
        "get_ledger_postings": get_ledger_postings,
        "create_voucher": create_voucher,
        "delete_voucher": delete_voucher,
        "create_ledger_account": create_ledger_account,
        "reverse_voucher": reverse_voucher,
        "create_opening_balance": create_opening_balance,
        "search_vouchers": search_vouchers,
        "update_voucher": update_voucher,
    }
