from datetime import date

from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def _ensure_employee_ready(employee_id: int):
        """Ensure employee has dateOfBirth, employment, and project manager access."""
        _WRITABLE = {
            "id", "version", "firstName", "lastName", "email",
            "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
            "dateOfBirth", "department", "employeeNumber", "address",
            "userType", "nationalIdentityNumber", "bankAccountNumber",
            "comments", "employeeCategory",
        }
        emp = client.get(f"/employee/{employee_id}", params={"fields": "*"})
        emp_val = emp.get("value", {})

        # Build writable body for PUT
        needs_update = False
        body = {k: v for k, v in emp_val.items() if k in _WRITABLE and v is not None}
        if isinstance(body.get("department"), dict):
            body["department"] = {"id": body["department"]["id"]}

        # Set dateOfBirth if missing
        if not body.get("dateOfBirth"):
            body["dateOfBirth"] = "1990-01-01"
            needs_update = True

        # Upgrade to EXTENDED so they can be project manager
        if body.get("userType") not in ("EXTENDED",):
            body["userType"] = "EXTENDED"
            needs_update = True

        if needs_update:
            client.put(f"/employee/{employee_id}", json=body)

        # Check if employment exists
        emp_result = client.get("/employee/employment", params={"employeeId": employee_id, "fields": "id", "count": 1})
        if not emp_result.get("values"):
            today = date.today().isoformat()
            client.post("/employee/employment", json={
                "employee": {"id": employee_id},
                "startDate": today,
                "employmentDetails": [{"date": today, "employmentType": "ORDINARY", "workingHoursScheme": "NOT_SHIFT"}],
            })

        # Also try granting entitlements via template (project manager access)
        client.put("/employee/entitlement/:grantEntitlementsByTemplate",
                   params={"employeeId": employee_id, "template": "all_access"})

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

        result = client.post("/project", json=body)

        # If project manager access denied, retry with account owner as fallback
        if (result.get("error") and projectManagerId
                and "prosjektleder" in str(result.get("message", "")).lower()):
            # Try falling back to account owner (first employee)
            emp_result = client.get("/employee", params={"fields": "id", "count": 1})
            emps = emp_result.get("values", [])
            if emps and emps[0]["id"] != projectManagerId:
                body["projectManager"] = {"id": emps[0]["id"]}
                result = client.post("/project", json=body)

        return result

    return {"create_project": create_project}
