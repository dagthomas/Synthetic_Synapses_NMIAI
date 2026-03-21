from tripletex_client import TripletexClient


def build_year_end_tools(client: TripletexClient) -> dict:
    """Build year-end (aarsoppgjoer) tools."""

    def search_year_end_annexes(year_end_id: int = 0) -> dict:
        """Search for year-end annexes.

        Args:
            year_end_id: ID of the year-end to get annexes for (0 to auto-detect most recent).

        Returns:
            A list of year-end annexes.
        """
        if not year_end_id:
            ye = client.get("/yearEnd", params={"fields": "id", "count": 1})
            ye_list = ye.get("values", [])
            year_end_id = ye_list[0]["id"] if ye_list else 0
        if not year_end_id:
            return {"values": [], "fullResultSize": 0}
        return client.get("/yearEnd/annex", params={"yearEndId": year_end_id, "fields": "*"})

    def create_year_end_note(
        year_end_id: int = 0, # Make year_end_id optional with default 0
        note: str = "",
    ) -> dict:
        """Create a year-end note.

        Args:
            year_end_id: ID of the year-end (0 to auto-detect most recent).
            note: Note content.

        Returns:
            Created note or error.
        """
        if not year_end_id:
            ye = client.get("/yearEnd", params={"fields": "id", "count": 1})
            ye_list = ye.get("values", [])
            year_end_id = ye_list[0]["id"] if ye_list else 0
        if not year_end_id:
            return {"error": "No year-end found to attach note to."}
        body = {"yearEnd": {"id": year_end_id}}
        if note:
            body["note"] = note
        return client.post("/yearEnd/note", json=body)

    def get_vat_returns() -> dict:
        """Get VAT types (MVA-typer).

        Returns:
            A list of VAT types.
        """
        return client.get("/ledger/vatType", params={"fields": "id,number,name,percentage"})

    def get_result_before_tax(dateFrom: str, dateTo: str) -> dict:
        """Calculate the profit/loss (resultat) before tax for a period.

        Queries the balance sheet for all P&L accounts (3000-8699) and sums them.
        In Norwegian accounting, income accounts (3xxx) have negative balances, so
        the result is negated: a positive return value means profit.

        Use this AFTER booking depreciation and expense reversals, so the result
        includes those entries. Then multiply by the tax rate to get the tax amount.

        Args:
            dateFrom: Period start date YYYY-MM-DD (e.g. "2025-01-01").
            dateTo: Period end date YYYY-MM-DD (e.g. "2025-12-31").

        Returns:
            Dict with 'result_before_tax' (positive=profit, negative=loss),
            'total_income', 'total_expenses', and 'account_count'.
        """
        resp = client.get("/balanceSheet", params={
            "dateFrom": dateFrom,
            "dateTo": dateTo,
            "accountNumberFrom": 3000,
            "accountNumberTo": 8699,
            "fields": "*",
        })
        if resp.get("error"):
            return resp

        entries = resp.get("values", [])
        total = 0.0
        total_income = 0.0
        total_expenses = 0.0
        details = []
        for e in entries:
            bal = e.get("balanceIn", 0) or 0
            bal += e.get("balanceChange", 0) or 0
            acct = e.get("account", {})
            acct_num = acct.get("number", 0) if isinstance(acct, dict) else 0
            acct_name = acct.get("name", "") if isinstance(acct, dict) else ""
            if bal != 0:
                details.append({"accountNumber": acct_num, "name": acct_name, "balance": bal})
                total += bal
                if 3000 <= acct_num <= 3999:
                    total_income += bal
                else:
                    total_expenses += bal

        result = round(-total, 2)
        return {
            "result_before_tax": result,
            "total_income": round(-total_income, 2),
            "total_expenses": round(total_expenses, 2),
            "account_count": len(details),
        }

    return {
        "search_year_end_annexes": search_year_end_annexes,
        "create_year_end_note": create_year_end_note,
        "get_vat_returns": get_vat_returns,
        "get_result_before_tax": get_result_before_tax,
    }
