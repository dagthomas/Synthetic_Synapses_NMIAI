from datetime import date

from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def _ensure_employee_ready(employee_id: int):
        """Ensure employee has dateOfBirth, employment, and project manager access.

        Optimized: checks current state first and only makes calls that are needed.
        """
        import logging
        _log = logging.getLogger("projects")

        _WRITABLE = {
            "id", "version", "firstName", "lastName", "email",
            "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
            "dateOfBirth", "department", "employeeNumber", "address",
            "userType", "nationalIdentityNumber", "bankAccountNumber",
            "comments", "employeeCategory",
        }
        emp = client.get(f"/employee/{employee_id}", params={"fields": "*"})
        emp_val = emp.get("value", {})

        # Check if employee needs updating (dateOfBirth, userType)
        needs_update = False
        body = {k: v for k, v in emp_val.items() if k in _WRITABLE and v is not None}
        if isinstance(body.get("department"), dict):
            body["department"] = {"id": body["department"]["id"]}

        if not body.get("dateOfBirth"):
            body["dateOfBirth"] = "1990-01-01"
            needs_update = True

        if body.get("userType") not in ("EXTENDED",):
            body["userType"] = "EXTENDED"
            needs_update = True

        if needs_update:
            client.put(f"/employee/{employee_id}", json=body)

        # Create employment (always needed for fresh employees)
        today = date.today().isoformat()
        emp_result = client.get("/employee/employment", params={"employeeId": employee_id, "fields": "id", "count": 1})
        if not emp_result.get("values"):
            client.post("/employee/employment", json={
                "employee": {"id": employee_id},
                "startDate": today,
                "employmentDetails": [{"date": today, "employmentType": "ORDINARY", "workingHoursScheme": "NOT_SHIFT"}],
            })

        # Grant entitlements — try both approaches for robustness
        r1 = client.put("/employee/entitlement/:grantClientEntitlements",
                        params={"employeeId": employee_id})
        if r1.get("error"):
            _log.warning(f"grantClientEntitlements failed for {employee_id}: {r1.get('message', '')}")

        r2 = client.put("/employee/entitlement/:grantEntitlementsByTemplate",
                        params={"employeeId": employee_id, "template": "all_access"})
        if r2.get("error"):
            _log.warning(f"grantEntitlementsByTemplate failed for {employee_id}: {r2.get('message', '')}")

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

        # If project manager access denied, retry entitlement granting
        if (result.get("error") and projectManagerId
                and "prosjektleder" in str(result.get("message", "")).lower()):
            import logging
            _log = logging.getLogger("projects")
            _log.warning(f"PM access denied for {projectManagerId}, retrying entitlements...")

            # Try additional entitlement approaches
            client.put("/employee/entitlement/:grantEntitlementsByTemplate",
                        params={"employeeId": projectManagerId, "template": "project_leader"})
            client.put("/employee/entitlement/:grantEntitlementsByTemplate",
                        params={"employeeId": projectManagerId, "template": "internal_accountant"})

            # Retry project creation with the CORRECT PM (never substitute wrong PM)
            result = client.post("/project", json=body)

        return result

    def search_projects(name: str = "", isClosed: bool = False) -> dict:
        """Search for projects.

        Args:
            name: Filter by project name (partial match).
            isClosed: Filter by closed status.

        Returns:
            A list of projects.
        """
        params = {"fields": "id,name,number,description,startDate,endDate,projectManager,customer,isClosed"}
        if name:
            params["name"] = name
        if isClosed:
            params["isClosed"] = True
        return client.get("/project", params=params)

    def update_project(
        project_id: int,
        name: str = "",
        description: str = "",
        isClosed: bool = False,
        endDate: str = "",
    ) -> dict:
        """Update a project.

        Args:
            project_id: ID of the project.
            name: New name (empty to keep).
            description: New description (empty to keep).
            isClosed: Set to True to close the project.
            endDate: End date YYYY-MM-DD (empty to keep).

        Returns:
            Updated project or error.
        """
        _WRITABLE = {
            "id", "version", "name", "number", "description", "projectManager",
            "department", "startDate", "endDate", "customer", "isClosed",
            "isInternal", "isOffer", "projectCategory",
        }
        current = client.get(f"/project/{project_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        for ref in ("projectManager", "department", "customer", "projectCategory"):
            if isinstance(body.get(ref), dict) and "id" in body[ref]:
                body[ref] = {"id": body[ref]["id"]}
        if name:
            body["name"] = name
        if description:
            body["description"] = description
        if isClosed:
            body["isClosed"] = True
        if endDate:
            body["endDate"] = endDate
        return client.put(f"/project/{project_id}", json=body)

    def delete_project(project_id: int) -> dict:
        """Delete a project.

        Args:
            project_id: ID of the project.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/project/{project_id}")

    def create_project_category(name: str, number: str = "", description: str = "") -> dict:
        """Create a project category.

        Args:
            name: Category name.
            number: Category number.
            description: Category description.

        Returns:
            Created category or error.
        """
        body = {"name": name}
        if number:
            body["number"] = number
        if description:
            body["description"] = description
        return client.post("/project/category", json=body)

    def search_project_categories() -> dict:
        """Search for project categories.

        Returns:
            A list of project categories.
        """
        return client.get("/project/category", params={"fields": "id,name,number,description"})

    def create_project_participant(project_id: int, employee_id: int, adminAccess: bool = False) -> dict:
        """Add a participant to a project.

        Args:
            project_id: ID of the project.
            employee_id: ID of the employee to add.
            adminAccess: Whether the participant has admin access.

        Returns:
            Created participant or error.
        """
        body = {
            "project": {"id": project_id},
            "employee": {"id": employee_id},
            "adminAccess": adminAccess,
        }
        return client.post("/project/participant", json=body)

    return {
        "create_project": create_project,
        "search_projects": search_projects,
        "update_project": update_project,
        "delete_project": delete_project,
        "create_project_category": create_project_category,
        "search_project_categories": search_project_categories,
        "create_project_participant": create_project_participant,
    }
