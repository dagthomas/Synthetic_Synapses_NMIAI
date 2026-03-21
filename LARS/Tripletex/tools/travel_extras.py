from tripletex_client import TripletexClient


def build_travel_extras_tools(client: TripletexClient) -> dict:
    """Build extended travel expense tools (costs, mileage, per diem)."""

    def create_travel_expense_cost(
        travel_expense_id: int,
        amount: float,
        category: str = "",
        comments: str = "",
        date: str = "",
        paymentType_id: int = 0,
        costCategory_id: int = 0,
        vatType_id: int = 0,
        currency_id: int = 0,
    ) -> dict:
        """Add a cost/expense line to a travel expense report.

        Use this for actual receipt-based expenses (hotel bills, taxi, flights, etc.).
        For fixed-rate overnight allowance (nattillegg), use create_accommodation_allowance instead.

        Args:
            travel_expense_id: ID of the travel expense report.
            amount: Cost amount including VAT (beloep inkl. mva).
            category: Cost category string (e.g. 'other', 'food', 'transport', 'accommodation').
            comments: Description of the cost (e.g. 'Overnatting Thon Hotel Oslo 2 netter').
            date: Date of the expense YYYY-MM-DD.
            paymentType_id: Payment type ID (0 to auto-detect first available).
            costCategory_id: Travel cost category ID from /travelExpense/costCategory (0 to skip).
            vatType_id: VAT type ID (0 for default). Hotel in Norway = 12% (vatType 13).
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
        if category:
            body["category"] = category
        if comments:
            body["comments"] = comments
        if date:
            body["date"] = date
        if costCategory_id:
            body["costCategory"] = {"id": costCategory_id}
        if vatType_id:
            body["vatType"] = {"id": vatType_id}
        if currency_id:
            body["currency"] = {"id": currency_id}
        return client.post("/travelExpense/cost", json=body)

    def create_accommodation_allowance(
        travel_expense_id: int,
        count: int = 1,
        location: str = "",
        address: str = "",
        rateType_id: int = 0,
        rateCategory_id: int = 0,
    ) -> dict:
        """Create an accommodation allowance (nattillegg) on a travel expense.

        This is for fixed-rate overnight allowance (statssats), NOT actual hotel receipts.
        For actual hotel bills with receipts, use create_travel_expense_cost instead.

        Args:
            travel_expense_id: ID of the travel expense.
            count: Number of nights (antall netter).
            location: Location of the stay (e.g. 'Oslo').
            address: Address of the accommodation.
            rateType_id: Rate type ID (0 for default rate).
            rateCategory_id: Rate category ID (0 for default).

        Returns:
            Created accommodation allowance or error.
        """
        body = {
            "travelExpense": {"id": travel_expense_id},
            "count": count,
        }
        if location:
            body["location"] = location
        if address:
            body["address"] = address
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        if rateCategory_id:
            body["rateCategory"] = {"id": rateCategory_id}
        return client.post("/travelExpense/accommodationAllowance", json=body)

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
        location: str = "",
        rate: float = 0,
        amount: float = 0,
        count: int = 0,
        overnight_accommodation: str = "",
        is_deduction_for_breakfast: bool = False,
        is_deduction_for_lunch: bool = False,
        is_deduction_for_dinner: bool = False,
        rateType_id: int = 0,
        rateCategory_id: int = 0,
    ) -> dict:
        """Create a per diem compensation (diett/dagpenger) on a travel expense.

        Per diem covers the full travel period (departureDate to returnDate).
        One call creates per diem for the entire trip — do NOT call per day.

        Args:
            travel_expense_id: ID of the travel expense.
            location: Travel location (default 'Norge').
            rate: Daily per diem rate in NOK (e.g. 800 for 800 NOK/day).
            amount: Total per diem amount in NOK (e.g. 1600 for 2 days × 800).
                    If rate is set, amount is usually auto-calculated.
            count: Number of days (0 = auto from departure/return dates).
            overnight_accommodation: Accommodation type (e.g. 'HOTEL', 'NONE').
            is_deduction_for_breakfast: Deduct breakfast from per diem.
            is_deduction_for_lunch: Deduct lunch from per diem.
            is_deduction_for_dinner: Deduct dinner from per diem.
            rateType_id: Rate type ID (0 for default).
            rateCategory_id: Rate category ID (0 for default).

        Returns:
            Created per diem or error.
        """
        if not location:
            location = "Norge"
        body = {
            "travelExpense": {"id": travel_expense_id},
            "location": location,
        }
        if rate:
            body["rate"] = rate
        if amount:
            body["amount"] = amount
        if count:
            body["count"] = count
        if overnight_accommodation:
            body["overnightAccommodation"] = overnight_accommodation
        if is_deduction_for_breakfast:
            body["isDeductionForBreakfast"] = True
        if is_deduction_for_lunch:
            body["isDeductionForLunch"] = True
        if is_deduction_for_dinner:
            body["isDeductionForDinner"] = True
        if rateType_id:
            body["rateType"] = {"id": rateType_id}
        if rateCategory_id:
            body["rateCategory"] = {"id": rateCategory_id}
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
        "create_accommodation_allowance": create_accommodation_allowance,
        "search_travel_expense_costs": search_travel_expense_costs,
        "create_mileage_allowance": create_mileage_allowance,
        "create_per_diem_compensation": create_per_diem_compensation,
        "update_travel_expense": update_travel_expense,
    }
