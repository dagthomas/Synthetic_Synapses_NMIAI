from tripletex_client import TripletexClient


def build_travel_extras_tools(client: TripletexClient) -> dict:
    """Build extended travel expense tools (costs, mileage, per diem)."""

    def create_travel_expense_cost(
        travel_expense_id: int,
        amount: float,
        description: str = "",
        date: str = "",
        category: str = "",
        paymentType_id: int = 0,
        vatType_id: int = 0,
        currency_id: int = 0,
    ) -> dict:
        """Add a cost/expense line to a travel expense report.

        Args:
            travel_expense_id: ID of the travel expense report.
            amount: Cost amount including VAT.
            description: Description of the cost.
            date: Date of the cost YYYY-MM-DD.
            category: Cost category (e.g. 'other', 'food', 'transport').
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
            "travelExpense": {"id": travel_expense_id},
            "amountCurrencyIncVat": amount,
            "paymentType": {"id": paymentType_id},
        }
        if date:
            body["date"] = date
        if category:
            body["category"] = category
        if vatType_id:
            body["vatType"] = {"id": vatType_id}
        if currency_id:
            body["currency"] = {"id": currency_id}
        return client.post("/travelExpense/cost", json=body)

    def search_travel_expense_costs(travel_expense_id: int) -> dict:
        """Search for costs on a travel expense.

        Args:
            travel_expense_id: ID of the travel expense.

        Returns:
            A list of costs.
        """
        return client.get("/travelExpense/cost", params={"travelExpenseId": travel_expense_id, "fields": "*"})

    def create_mileage_allowance(
        travel_expense_id: int,
        date: str,
        km: float,
        rateType_id: int = 0,
        departureLocation: str = "",
        destination: str = "",
    ) -> dict:
        """Create a mileage allowance on a travel expense.

        Args:
            travel_expense_id: ID of the travel expense.
            date: Date YYYY-MM-DD.
            km: Number of kilometers.
            rateType_id: Rate type ID (0 for default).
            departureLocation: Starting location.
            destination: Destination.

        Returns:
            Created mileage allowance or error.
        """
        if not departureLocation:
            departureLocation = "N/A"
        if not destination:
            destination = "N/A"
        body = {
            "travelExpense": {"id": travel_expense_id},
            "date": date,
            "km": km,
            "departureLocation": departureLocation,
            "destination": destination,
        }
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        return client.post("/travelExpense/mileageAllowance", json=body)

    def create_per_diem_compensation(
        travel_expense_id: int,
        date: str = "",
        location: str = "",
        rateType_id: int = 0,
        accommodation: str = "",
    ) -> dict:
        """Create a per diem compensation on a travel expense.

        Args:
            travel_expense_id: ID of the travel expense.
            date: Date YYYY-MM-DD (used as location fallback if location empty).
            location: Travel location.
            rateType_id: Rate type ID (0 for default).
            accommodation: Accommodation type.

        Returns:
            Created per diem or error.
        """
        if not location:
            location = "Norge"
        body = {
            "travelExpense": {"id": travel_expense_id},
            "location": location,
        }
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        if accommodation:
            body["accommodation"] = accommodation
        return client.post("/travelExpense/perDiemCompensation", json=body)

    def update_travel_expense(
        travel_expense_id: int,
        title: str = "",
        description: str = "",
    ) -> dict:
        """Update a travel expense report.

        Args:
            travel_expense_id: ID of the travel expense.
            title: New title (empty to keep).
            description: New description (empty to keep).

        Returns:
            Updated travel expense or error.
        """
        _WRITABLE = {"id", "version", "employee", "title", "travelDetails"}
        current = client.get(f"/travelExpense/{travel_expense_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("employee"), dict):
            body["employee"] = {"id": body["employee"]["id"]}
        if title:
            body["title"] = title
        if description and isinstance(body.get("travelDetails"), dict):
            body["travelDetails"]["detailedJourneyDescription"] = description
        return client.put(f"/travelExpense/{travel_expense_id}", json=body)

    return {
        "create_travel_expense_cost": create_travel_expense_cost,
        "search_travel_expense_costs": search_travel_expense_costs,
        "create_mileage_allowance": create_mileage_allowance,
        "create_per_diem_compensation": create_per_diem_compensation,
        "update_travel_expense": update_travel_expense,
    }
