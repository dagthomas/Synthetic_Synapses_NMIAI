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
            postings: JSON string of postings, each with 'accountNumber', 'debitAmount', 'creditAmount'. Example: '[{"accountNumber": 1920, "debitAmount": 1000, "creditAmount": 0}]'

        Returns:
            The created voucher with id, or an error message.
        """
        posting_list = _json.loads(postings) if isinstance(postings, str) else postings
        formatted = []
        for p in posting_list:
            formatted.append({
                "account": {"number": p["accountNumber"]},
                "debitAmount": p.get("debitAmount", 0),
                "creditAmount": p.get("creditAmount", 0),
            })
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

    return {
        "get_ledger_accounts": get_ledger_accounts,
        "get_ledger_postings": get_ledger_postings,
        "create_voucher": create_voucher,
        "delete_voucher": delete_voucher,
    }
