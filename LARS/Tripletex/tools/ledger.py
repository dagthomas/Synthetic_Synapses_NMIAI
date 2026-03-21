import json as _json
from datetime import date as _date

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
            "fields": "id,date,description,amount,account,voucher",
        }
        if accountNumber:
            params["accountNumber"] = accountNumber
        return client.get("/ledger/posting", params=params)

    def create_voucher(date: str, description: str, postings: str) -> dict:
        """Create a ledger voucher with postings. Used for corrections and manual entries.

        Args:
            date: Voucher date in YYYY-MM-DD format.
            description: Description of the voucher.
            postings: JSON string of postings, each with 'accountId' (ledger account ID) and 'amount' (positive=debit, negative=credit). Example: '[{"accountId": 123, "amount": 1000}, {"accountId": 456, "amount": -1000}]'. Alternatively, within each posting object, you can use 'accountNumber' instead of 'accountId', and the tool will look up the account ID. Optionally include 'customerId' to link a posting to a customer (REQUIRED for accounts 1500-1599 kundefordringer), or 'supplierId' for supplier link (REQUIRED for accounts 2400-2499 leverandørgjeld). Optionally include 'projectId' or 'departmentId' to link a posting to a project or department dimension. For free accounting dimensions, include 'dimensionValueId' (the ID from create_dimension_value) and 'dimensionIndex' (1, 2, or 3) — or just 'dimensionValueId' if already known.

        Returns:
            The created voucher with id, or an error message.
        """
        try:
            posting_list = _json.loads(postings) if isinstance(postings, str) else postings
        except (_json.JSONDecodeError, TypeError) as e:
            return {"error": True, "message": f"Invalid JSON in postings: {e}. Expected: '[{{\"accountNumber\": 1920, \"amount\": 1000}}, ...]'"}
        if not posting_list or not isinstance(posting_list, list):
            return {"error": True, "message": "postings must be a non-empty JSON array of objects with 'accountNumber'/'accountId' and 'amount'"}

        # Pre-validate: postings must balance (sum to 0)
        total = 0
        for p in posting_list:
            amt = p.get("amount", 0)
            if not amt:
                amt = p.get("debitAmount", 0) - p.get("creditAmount", 0)
            total += amt
        if abs(total) > 0.01:
            return {"error": True, "message": f"Postings do not balance: sum={total}. Debit (positive) and credit (negative) amounts must sum to 0. Adjust your amounts."}

        # Cache account lookups within this call
        acct_cache = {}
        formatted = []
        for i, p in enumerate(posting_list):
            entry = {"row": i + 1}
            if "accountId" in p:
                entry["account"] = {"id": p["accountId"]}
            elif "accountNumber" in p:
                acct_num = str(p["accountNumber"])
                if acct_num in acct_cache:
                    entry["account"] = {"id": acct_cache[acct_num]}
                else:
                    acct_result = client.get("/ledger/account", params={"number": acct_num, "fields": "id", "count": 1})
                    if acct_result.get("error"):
                        return acct_result  # Propagate API error (e.g. 403)
                    accts = acct_result.get("values", [])
                    if accts:
                        acct_cache[acct_num] = accts[0]["id"]
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
            entry["amountGross"] = amount
            entry["amountGrossCurrency"] = amount
            # Optional customer/supplier references (required for receivables/payables accounts)
            if p.get("customerId"):
                entry["customer"] = {"id": p["customerId"]}
            if p.get("supplierId"):
                entry["supplier"] = {"id": p["supplierId"]}
            # Optional dimension references
            if p.get("projectId"):
                entry["project"] = {"id": p["projectId"]}
            if p.get("departmentId"):
                entry["department"] = {"id": p["departmentId"]}
            # Free accounting dimension reference
            dim_val_id = p.get("dimensionValueId")
            if dim_val_id:
                dim_idx = p.get("dimensionIndex", 1)
                entry[f"freeAccountingDimension{dim_idx}"] = {"id": dim_val_id}
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
        If the account number already exists, returns the existing account.

        Args:
            number: Account number (e.g. 1920, 7700).
            name: Account name.
            description: Optional description.

        Returns:
            Created or existing account, or error.
        """
        existing = client.get("/ledger/account", params={"number": str(number), "fields": "id,number,name"})
        existing_list = existing.get("values", [])
        if existing_list:
            return {"value": existing_list[0]}
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
        if not date:
            date = _date.today().isoformat()
        return client.put(f"/ledger/voucher/{voucher_id}/:reverse", params={"date": date})

    def create_opening_balance(voucherDate: str, balancePostings: str) -> dict:
        """Create an opening balance. Zeroes out all movements before this date.

        Args:
            voucherDate: Date for opening balance YYYY-MM-DD (must be first day of month).
            balancePostings: JSON string of postings, each with 'accountId' (or 'accountNumber') and 'amount'. Example: '[{"accountNumber": 1920, "amount": 50000}]'

        Returns:
            Created opening balance or error.
        """
        try:
            posting_list = _json.loads(balancePostings) if isinstance(balancePostings, str) else balancePostings
        except (_json.JSONDecodeError, TypeError) as e:
            return {"error": True, "message": f"Invalid JSON in balancePostings: {e}. Expected: '[{{\"accountNumber\": 1920, \"amount\": 50000}}]'"}
        if not posting_list or not isinstance(posting_list, list):
            return {"error": True, "message": "balancePostings must be a non-empty JSON array"}

        acct_cache = {}
        formatted = []
        for p in posting_list:
            entry = {}
            if "accountId" in p:
                entry["account"] = {"id": p["accountId"]}
            elif "accountNumber" in p:
                acct_num = str(p["accountNumber"])
                if acct_num in acct_cache:
                    entry["account"] = {"id": acct_cache[acct_num]}
                else:
                    acct = client.get("/ledger/account", params={"number": acct_num, "fields": "id", "count": 1})
                    if acct.get("error"):
                        return acct  # Propagate API error (e.g. 403)
                    accts = acct.get("values", [])
                    if accts:
                        acct_cache[acct_num] = accts[0]["id"]
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
            dateFrom: Start date YYYY-MM-DD (defaults to Jan 1 of current year).
            dateTo: End date YYYY-MM-DD (defaults to today).

        Returns:
            A list of vouchers.
        """
        today = _date.today()
        if not dateFrom:
            dateFrom = f"{today.year}-01-01"
        if not dateTo:
            dateTo = today.isoformat()
        params = {"fields": "id,date,description,voucherType,number",
                  "dateFrom": dateFrom, "dateTo": dateTo}
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

    # ── Free accounting dimension tools ──────────────────────────────

    def create_accounting_dimension(name: str, description: str = "") -> dict:
        """Create a free (user-defined) accounting dimension.

        Tripletex supports up to 3 free dimensions (index 1, 2, 3).
        The next available index is assigned automatically.

        Args:
            name: The dimension name (e.g. "Marked", "Region").
            description: Optional description.

        Returns:
            The created dimension with id and dimensionIndex, or error.
        """
        body: dict = {"dimensionName": name}
        if description:
            body["description"] = description
        return client.post("/ledger/accountingDimensionName", json=body)

    def create_dimension_value(dimensionIndex: int, name: str, number: str = "") -> dict:
        """Create a value for a free accounting dimension.

        Args:
            dimensionIndex: The dimension index (1, 2, or 3) from create_accounting_dimension.
            name: Display name (e.g. "Offentlig", "Privat").
            number: Optional value number/code.

        Returns:
            The created dimension value with id, or error.
        """
        body: dict = {
            "displayName": name,
            "dimensionIndex": dimensionIndex,
            "active": True,
            "showInVoucherRegistration": True,
        }
        if number:
            body["number"] = number
        return client.post("/ledger/accountingDimensionValue", json=body)

    def search_accounting_dimensions() -> dict:
        """List all free accounting dimensions.

        Returns:
            A list of accounting dimension names with id, dimensionName, dimensionIndex.
        """
        return client.get("/ledger/accountingDimensionName",
                          params={"fields": "id,dimensionName,description,dimensionIndex,active"})

    def search_dimension_values(dimensionIndex: int = 0) -> dict:
        """Search for dimension values, optionally filtered by dimension index.

        Args:
            dimensionIndex: Filter by dimension index (1, 2, or 3). 0 for all.

        Returns:
            A list of dimension values.
        """
        params: dict = {"fields": "id,displayName,dimensionIndex,number,active"}
        if dimensionIndex:
            params["dimensionIndex"] = dimensionIndex
        return client.get("/ledger/accountingDimensionValue/search", params=params)

    def analyze_ledger_changes(periodA_from: str, periodA_to: str,
                               periodB_from: str, periodB_to: str,
                               top_n: int = 10) -> dict:
        """Compare ledger posting totals between two periods, grouped by account.

        Returns accounts sorted by largest increase first.
        Useful for identifying which expense accounts grew the most.

        Args:
            periodA_from: Start date of first period (YYYY-MM-DD), e.g. "2026-01-01".
            periodA_to: End date of first period (YYYY-MM-DD), e.g. "2026-01-31".
            periodB_from: Start date of second period (YYYY-MM-DD), e.g. "2026-02-01".
            periodB_to: End date of second period (YYYY-MM-DD), e.g. "2026-02-28".
            top_n: Number of top accounts to return (default 10).

        Returns:
            A sorted list of accounts with periodA total, periodB total, and change amount.
        """
        from collections import defaultdict
        a_resp = client.get("/ledger/posting", params={
            "dateFrom": periodA_from, "dateTo": periodA_to,
            "fields": "amount,account",
        })
        b_resp = client.get("/ledger/posting", params={
            "dateFrom": periodB_from, "dateTo": periodB_to,
            "fields": "amount,account",
        })
        totals_a: dict[str, float] = defaultdict(float)
        totals_b: dict[str, float] = defaultdict(float)
        names: dict[str, str] = {}
        for p in a_resp.get("values", []):
            acct = p.get("account") or {}
            num = str(acct.get("number", ""))
            totals_a[num] += p.get("amount", 0)
            if num and num not in names:
                names[num] = acct.get("name", "")
        for p in b_resp.get("values", []):
            acct = p.get("account") or {}
            num = str(acct.get("number", ""))
            totals_b[num] += p.get("amount", 0)
            if num and num not in names:
                names[num] = acct.get("name", "")
        rows = []
        for num in set(totals_a) | set(totals_b):
            a_tot = round(totals_a.get(num, 0), 2)
            b_tot = round(totals_b.get(num, 0), 2)
            rows.append({
                "accountNumber": num,
                "accountName": names.get(num, ""),
                "periodA_total": a_tot,
                "periodB_total": b_tot,
                "change": round(b_tot - a_tot, 2),
            })
        rows.sort(key=lambda r: r["change"], reverse=True)
        top_n = max(1, min(top_n, len(rows)))
        return {"accounts": rows[:top_n], "total_accounts": len(rows)}

    return {
        "get_ledger_accounts": get_ledger_accounts,
        "get_ledger_postings": get_ledger_postings,
        "analyze_ledger_changes": analyze_ledger_changes,
        "create_voucher": create_voucher,
        "delete_voucher": delete_voucher,
        "create_ledger_account": create_ledger_account,
        "reverse_voucher": reverse_voucher,
        "create_opening_balance": create_opening_balance,
        "search_vouchers": search_vouchers,
        "update_voucher": update_voucher,
        "create_accounting_dimension": create_accounting_dimension,
        "create_dimension_value": create_dimension_value,
        "search_accounting_dimensions": search_accounting_dimensions,
        "search_dimension_values": search_dimension_values,
    }
