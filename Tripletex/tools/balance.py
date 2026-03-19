from tripletex_client import TripletexClient


def build_balance_tools(client: TripletexClient) -> dict:
    """Build balance sheet and year-end tools."""

    def get_balance_sheet(dateFrom: str, dateTo: str, accountNumberFrom: int = 0, accountNumberTo: int = 0) -> dict:
        """Get the balance sheet for a date range.

        Args:
            dateFrom: Start date in YYYY-MM-DD format.
            dateTo: End date in YYYY-MM-DD format.
            accountNumberFrom: Filter from this account number (0 for all).
            accountNumberTo: Filter to this account number (0 for all).

        Returns:
            Balance sheet data with account balances.
        """
        params = {
            "dateFrom": dateFrom,
            "dateTo": dateTo,
            "fields": "*",
        }
        if accountNumberFrom:
            params["accountNumberFrom"] = accountNumberFrom
        if accountNumberTo:
            params["accountNumberTo"] = accountNumberTo
        return client.get("/balanceSheet", params=params)

    def search_periods() -> dict:
        """Search for accounting periods.

        Returns:
            A list of accounting periods with id, start, end dates.
        """
        return client.get("/period", params={"fields": "*"})

    def search_voucher_types() -> dict:
        """Search for available voucher types.

        Returns:
            A list of voucher types with id and name.
        """
        return client.get("/voucherType", params={"fields": "*"})

    def get_year_end(yearEndId: int) -> dict:
        """Get a year-end entry by ID.

        Args:
            yearEndId: The ID of the year-end entry.

        Returns:
            The year-end entry data.
        """
        return client.get(f"/yearEnd/{yearEndId}", params={"fields": "*"})

    def search_year_ends() -> dict:
        """Search for year-end entries.

        Returns:
            A list of year-end entries.
        """
        return client.get("/yearEnd", params={"fields": "*"})

    def search_currencies(code: str = "") -> dict:
        """Search for currencies.

        Args:
            code: Filter by currency code, e.g. 'NOK', 'EUR', 'USD'.

        Returns:
            A list of currencies with id, code, and description.
        """
        params = {"fields": "id,code,description"}
        if code:
            params["code"] = code
        return client.get("/currency", params=params)

    def get_company_info() -> dict:
        """Get the current company information.

        Returns:
            Company details including name, org number, modules.
        """
        return client.get("/company", params={"fields": "*"})

    return {
        "get_balance_sheet": get_balance_sheet,
        "search_periods": search_periods,
        "search_voucher_types": search_voucher_types,
        "get_year_end": get_year_end,
        "search_year_ends": search_year_ends,
        "search_currencies": search_currencies,
        "get_company_info": get_company_info,
    }
