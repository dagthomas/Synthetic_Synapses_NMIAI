from tripletex_client import TripletexClient


def build_travel_extras_tools(client: TripletexClient) -> dict:
    """Build extended travel expense tools (costs, mileage, per diem)."""

    def create_travel_expense_cost(
        travelExpenseId: int,
        amount: float,
        paymentType_id: int = 0,
        vatType_id: int = 0,
        currency_id: int = 0,
    ) -> dict:
        """Add a cost/expense line to a travel expense report.

        Args:
            travelExpenseId: ID of the travel expense report.
            amount: Cost amount including VAT.
            paymentType_id: Payment type ID (0 to auto-detect first available).
            vatType_id: VAT type ID (0 for default).
            currency_id: Currency ID (0 for NOK default).

        Returns:
            Created cost or error.
        """
        if not paymentType_id:
            pt = client.get("/travelExpense/paymentType", params={"fields": "id", "count": 1})
            pts = pt.get("values", [])
            paymentType_id = pts[0]["id"] if pts else 0
        body = {
            "travelExpense": {"id": travelExpenseId},
            "amountCurrencyIncVat": amount,
            "paymentType": {"id": paymentType_id},
        }
        if vatType_id:
            body["vatType"] = {"id": vatType_id}
        if currency_id:
            body["currency"] = {"id": currency_id}
        return client.post("/travelExpense/cost", json=body)

    def search_travel_expense_costs(travelExpenseId: int) -> dict:
        """Search for costs on a travel expense.

        Args:
            travelExpenseId: ID of the travel expense.

        Returns:
            A list of costs.
        """
        return client.get("/travelExpense/cost", params={"travelExpenseId": travelExpenseId, "fields": "*"})

    def create_mileage_allowance(
        travelExpenseId: int,
        date: str,
        km: float,
        rateType_id: int = 0,
        departureLocation: str = "",
        destination: str = "",
    ) -> dict:
        """Create a mileage allowance on a travel expense.

        Args:
            travelExpenseId: ID of the travel expense.
            date: Date YYYY-MM-DD.
            km: Number of kilometers.
            rateType_id: Rate type ID (0 for default).
            departureLocation: Starting location.
            destination: Destination.

        Returns:
            Created mileage allowance or error.
        """
        body = {
            "travelExpense": {"id": travelExpenseId},
            "date": date,
            "km": km,
        }
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        if departureLocation:
            body["departureLocation"] = departureLocation
        if destination:
            body["destination"] = destination
        return client.post("/travelExpense/mileageAllowance", json=body)

    def create_per_diem_compensation(
        travelExpenseId: int,
        location: str,
        rateType_id: int = 0,
        accommodation: str = "",
    ) -> dict:
        """Create a per diem compensation on a travel expense.

        Args:
            travelExpenseId: ID of the travel expense.
            location: Travel location (required).
            rateType_id: Rate type ID (0 for default).
            accommodation: Accommodation type.

        Returns:
            Created per diem or error.
        """
        body = {
            "travelExpense": {"id": travelExpenseId},
            "location": location,
        }
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        if accommodation:
            body["accommodation"] = accommodation
        return client.post("/travelExpense/perDiemCompensation", json=body)

    def update_travel_expense(
        travelExpenseId: int,
        title: str = "",
        description: str = "",
    ) -> dict:
        """Update a travel expense report.

        Args:
            travelExpenseId: ID of the travel expense.
            title: New title (empty to keep).
            description: New description (empty to keep).

        Returns:
            Updated travel expense or error.
        """
        _WRITABLE = {"id", "version", "employee", "title", "travelDetails"}
        current = client.get(f"/travelExpense/{travelExpenseId}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("employee"), dict):
            body["employee"] = {"id": body["employee"]["id"]}
        if title:
            body["title"] = title
        if description and isinstance(body.get("travelDetails"), dict):
            body["travelDetails"]["detailedJourneyDescription"] = description
        return client.put(f"/travelExpense/{travelExpenseId}", json=body)

    return {
        "create_travel_expense_cost": create_travel_expense_cost,
        "search_travel_expense_costs": search_travel_expense_costs,
        "create_mileage_allowance": create_mileage_allowance,
        "create_per_diem_compensation": create_per_diem_compensation,
        "update_travel_expense": update_travel_expense,
    }
