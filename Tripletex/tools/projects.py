from datetime import date

from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def _ensure_employee_ready(employee_id: int):
        """Ensure employee has dateOfBirth and employment (required for projectManager)."""
        emp = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
        emp_val = emp.get("value", emp)

        # Set dateOfBirth if missing (required for employment)
        if not emp_val.get("dateOfBirth"):
            client.put(f"/employee/{employee_id}", json={"dateOfBirth": "1990-01-01"})

        # Check if employment exists
        emp_result = client.get("/employee/employment", params={"employeeId": employee_id, "fields": "id", "count": 1})
        if not emp_result.get("values"):
            today = date.today().isoformat()
            client.post("/employee/employment", json={
                "employee": {"id": employee_id},
                "startDate": today,
                "employmentDetails": [{"date": today, "employmentType": "ORDINARY", "workingHoursScheme": "NOT_SHIFT"}],
            })

    def create_project(
        name: str,
        customer_id: int = 0,
        projectManagerId: int = 0,
        startDate: str = "",
        description: str = "",
    ) -> dict:
        """Create a project in Tripletex.

        Args:
            name: Project name.
            customer_id: ID of the customer linked to this project (0 if none).
            projectManagerId: ID of the employee managing the project (0 to auto-assign).
            startDate: Project start date in YYYY-MM-DD format (defaults to today).
            description: Optional project description.

        Returns:
            The created project with id, or an error message.
        """
        if not startDate:
            startDate = date.today().isoformat()

        body = {"name": name, "startDate": startDate}
        if customer_id:
            body["customer"] = {"id": customer_id}

        # Tripletex requires a projectManager with valid employment
        if projectManagerId:
            _ensure_employee_ready(projectManagerId)
            body["projectManager"] = {"id": projectManagerId}
        else:
            emp_result = client.get("/employee", params={"fields": "id", "count": 1})
            emps = emp_result.get("values", [])
            if emps:
                body["projectManager"] = {"id": emps[0]["id"]}

        if description:
            body["description"] = description
        return client.post("/project", json=body)

    return {"create_project": create_project}
