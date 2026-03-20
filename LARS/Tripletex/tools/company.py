from tripletex_client import TripletexClient


def build_company_tools(client: TripletexClient) -> dict:
    """Build company management tools."""

    def update_company(
        name: str = "",
        email: str = "",
        phoneNumber: str = "",
        organizationNumber: str = "",
    ) -> dict:
        """Update the company information.

        Args:
            name: New company name (empty to keep).
            email: New email (empty to keep).
            phoneNumber: New phone number (empty to keep).
            organizationNumber: New org number (empty to keep).

        Returns:
            Updated company or error.
        """
        who = client.get("/token/session/>whoAmI", params={"fields": "companyId"})
        cid = who.get("value", {}).get("companyId", 0) if isinstance(who.get("value"), dict) else 0
        if not cid:
            return {"error": True, "message": "Could not determine company ID"}
        current = client.get(f"/company/{cid}", params={"fields": "*"})
        full = current.get("value", {})
        _WRITABLE = {
            "id", "version", "name", "startDate", "endDate",
            "organizationNumber", "email", "phoneNumber",
            "phoneNumberMobile", "faxNumber", "address", "type", "currency",
        }
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("address"), dict):
            addr = body["address"]
            body["address"] = {
                k: v for k, v in addr.items()
                if k in ("id", "addressLine1", "addressLine2", "postalCode", "city", "country")
                and v is not None
            }
            if isinstance(body["address"].get("country"), dict):
                body["address"]["country"] = {"id": body["address"]["country"]["id"]}
        if isinstance(body.get("currency"), dict):
            body["currency"] = {"id": body["currency"]["id"]}
        if name:
            body["name"] = name
        if email:
            body["email"] = email
        if phoneNumber:
            body["phoneNumber"] = phoneNumber
        if organizationNumber:
            body["organizationNumber"] = organizationNumber
        return client.put("/company", json=body)

    def get_accounting_periods(dateFrom: str = "", dateTo: str = "") -> dict:
        """Get accounting periods.

        Args:
            dateFrom: Filter from date YYYY-MM-DD.
            dateTo: Filter to date YYYY-MM-DD.

        Returns:
            A list of accounting periods.
        """
        params = {"fields": "id,start,end,isClosed"}
        if dateFrom:
            params["dateFrom"] = dateFrom
        if dateTo:
            params["dateTo"] = dateTo
        return client.get("/ledger/accountingPeriod", params=params)

    return {
        "update_company": update_company,
        "get_accounting_periods": get_accounting_periods,
    }
