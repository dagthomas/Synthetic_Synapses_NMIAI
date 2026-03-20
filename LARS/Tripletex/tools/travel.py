from tripletex_client import TripletexClient


def build_travel_tools(client: TripletexClient) -> dict:
    """Build travel expense tools."""

    def create_travel_expense(
        employee_id: int,
        title: str,
        departureDate: str,
        returnDate: str,
        description: str = "",
    ) -> dict:
        """Create a travel expense report.

        Args:
            employee_id: The ID of the employee filing the expense.
            title: Title of the travel expense report.
            departureDate: Departure date in YYYY-MM-DD format.
            returnDate: Return date in YYYY-MM-DD format.
            description: Optional description of the travel.

        Returns:
            The created travel expense with id, or an error message.
        """
        body = {
            "employee": {"id": employee_id},
            "title": title,
            "travelDetails": {
                "departureDate": departureDate,
                "returnDate": returnDate,
            },
        }
        if description:
            body["travelDetails"]["detailedJourneyDescription"] = description
        return client.post("/travelExpense", json=body)

    def delete_travel_expense(travel_expense_id: int) -> dict:
        """Delete a travel expense report.

        Args:
            travel_expense_id: The ID of the travel expense to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/travelExpense/{travel_expense_id}")

    def create_travel_expense_cost(
        travel_expense_id: int,
        amount: float,
        description: str,
        date: str = "",
        vatPercentage: float = 25.0,
        accountNumber: str = "7140",
    ) -> dict:
        """Add a travel expense cost item to an existing travel expense report.

        Args:
            travel_expense_id: The ID of the travel expense report.
            amount: The amount of the expense.
            description: A description of the expense (e.g., "Flight ticket", "Taxi").
            date: The date of the expense in YYYY-MM-DD format (optional).
            vatPercentage: The VAT percentage for the expense (default 25.0).
            accountNumber: The account number for the expense (default 7140 for travel expenses).

        Returns:
            The created travel expense cost item, or an error message.
        """
        body = {
            "amount": amount,
            "description": description,
            "vatType": {"id": client.get_vat_type_id(vatPercentage)},
            "account": {"id": client.get_account_id(accountNumber)},
        }
        if date:
            body["date"] = date
        return client.post(f"/travelExpense/{travel_expense_id}/travelExpenseCost", json=body)

    def create_per_diem_compensation(
        travel_expense_id: int,
        date: str,
        amount: float,
        currency: str = "NOK",
        perDiemType: str = "DOMESTIC_OVERNIGHT",
        location: str = "Norway",
    ) -> dict:
        """Add a per diem compensation item to an existing travel expense report.

        Args:
            travel_expense_id: The ID of the travel expense report.
            date: The date for which the per diem applies in YYYY-MM-DD format.
            amount: The daily per diem amount.
            currency: The currency of the per diem (e.g., "NOK", "EUR").
            perDiemType: The type of per diem (e.g., "DOMESTIC_OVERNIGHT", "FOREIGN_OVERNIGHT").
            location: The location/country for the per diem.

        Returns:
            The created per diem compensation item, or an error message.
        """
        body = {
            "date": date,
            "amount": amount,
            "currency": {"id": client.get_currency_id(currency)},
            "perDiemType": {"id": client.get_per_diem_type_id(perDiemType)},
            "location": location,
        }
        return client.post(f"/travelExpense/{travel_expense_id}/perDiemCompensation", json=body)

    def search_travel_expenses(employee_id: int = 0) -> dict:
        """Search for travel expense reports.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of travel expenses.
        """
        params = {"fields": "id,title,employee,travelDetails,date"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/travelExpense", params=params)

    return {
        "create_travel_expense": create_travel_expense,
        "create_travel_expense_cost": create_travel_expense_cost,
        "create_per_diem_compensation": create_per_diem_compensation,
        "delete_travel_expense": delete_travel_expense,
        "search_travel_expenses": search_travel_expenses,
    }
