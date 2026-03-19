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
            "departureDate": departureDate,
            "returnDate": returnDate,
        }
        if description:
            body["description"] = description
        return client.post("/travelExpense", json=body)

    def delete_travel_expense(travel_expense_id: int) -> dict:
        """Delete a travel expense report.

        Args:
            travel_expense_id: The ID of the travel expense to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/travelExpense/{travel_expense_id}")

    def search_travel_expenses(employee_id: int = 0) -> dict:
        """Search for travel expense reports.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of travel expenses.
        """
        params = {"fields": "id,title,employee,departureDate,returnDate"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/travelExpense", params=params)

    return {
        "create_travel_expense": create_travel_expense,
        "delete_travel_expense": delete_travel_expense,
        "search_travel_expenses": search_travel_expenses,
    }
