from tripletex_client import TripletexClient


def build_employment_tools(client: TripletexClient) -> dict:
    """Build employment tools."""

    def create_employment(
        employee_id: int,
        startDate: str,
    ) -> dict:
        """Create an employment record for an employee.

        This sets up the formal employment relationship (ansettelsesforhold)
        including start date. The employee must already exist.

        NOTE: The employee must have a dateOfBirth set. If not, this tool
        will automatically set it to 1990-01-01 before creating the employment.

        Args:
            employee_id: The ID of the employee.
            startDate: Employment start date in YYYY-MM-DD format.

        Returns:
            The created employment with id, or an error message.
        """
        # Tripletex requires dateOfBirth on the employee before creating employment
        emp = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
        emp_val = emp.get("value", emp)
        if not emp_val.get("dateOfBirth"):
            client.put(f"/employee/{employee_id}", json={"dateOfBirth": "1990-01-01"})

        body = {
            "employee": {"id": employee_id},
            "startDate": startDate,
            "employmentDetails": [
                {
                    "date": startDate,
                    "employmentType": "ORDINARY",
                    "workingHoursScheme": "NOT_SHIFT",
                }
            ],
        }
        # Auto-detect division if any exist (required by Tripletex when divisions are present)
        divs = client.get("/division", params={"fields": "id", "count": 1})
        div_list = divs.get("values", [])
        if div_list:
            body["division"] = {"id": div_list[0]["id"]}
        return client.post("/employee/employment", json=body)

    def search_employments(employee_id: int = 0) -> dict:
        """Search for employment records.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of employment records.
        """
        params = {"fields": "id,employee,startDate,endDate"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/employee/employment", params=params)

    return {
        "create_employment": create_employment,
        "search_employments": search_employments,
    }
