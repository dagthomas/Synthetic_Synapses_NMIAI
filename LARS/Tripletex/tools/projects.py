from datetime import date

from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def _ensure_employee_ready(employee_id: int, fresh_create: bool = False) -> bool:
        """Ensure employee has dateOfBirth, employment, and project manager access.

        Args:
            employee_id: Employee ID to set up as PM.
            fresh_create: If True, skip the GET check — employee was just created
                         with dateOfBirth + userType=EXTENDED (saves 1-2 API calls).

        Returns True if PM entitlements were granted, False if not possible.
        """
        import logging
        _log = logging.getLogger("projects")

        # Fast path: already ensured in this request
        cache_key = f"pm_ready_{employee_id}"
        if client.get_cached(cache_key):
            return True

        if fresh_create:
            # Employee was just created with dateOfBirth + EXTENDED — skip GET+PUT
            has_dob = True
            has_extended = True
            has_employment = False
        else:
            _WRITABLE = {
                "id", "version", "firstName", "lastName", "email",
                "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
                "dateOfBirth", "department", "employeeNumber", "address",
                "userType", "nationalIdentityNumber", "bankAccountNumber",
                "comments", "employeeCategory",
            }
            emp = client.get(f"/employee/{employee_id}", params={"fields": "*"})
            emp_val = emp.get("value", {})

            has_dob = bool(emp_val.get("dateOfBirth"))
            has_extended = emp_val.get("userType") == "EXTENDED"
            has_employment = bool(emp_val.get("employments"))

            # Already fully set up → skip all setup, cache result
            if has_dob and has_extended and has_employment:
                client.set_cached(cache_key, True)
                return True

            # Build update body if needed
            if not has_dob or not has_extended:
                body = {k: v for k, v in emp_val.items() if k in _WRITABLE and v is not None}
                if isinstance(body.get("department"), dict):
                    body["department"] = {"id": body["department"]["id"]}
                if not has_dob:
                    body["dateOfBirth"] = "1990-01-01"
                if not has_extended:
                    body["userType"] = "EXTENDED"
                put_result = client.put(f"/employee/{employee_id}", json=body)
                if put_result.get("error"):
                    _log.warning(f"PUT employee {employee_id} failed: {put_result.get('message', '')}")

        # Create employment if needed
        if not has_employment:
            today = date.today().isoformat()
            division_id = client.get_cached("default_division")
            if division_id is None:
                div_result = client.get("/company/divisions", params={"fields": "id", "count": 1})
                divs = div_result.get("values", [])
                division_id = divs[0]["id"] if divs else 0
                client.set_cached("default_division", division_id)
            division_id = division_id or None

            emp_body = {
                "employee": {"id": employee_id},
                "startDate": today,
                "employmentDetails": [{"date": today, "employmentType": "ORDINARY", "workingHoursScheme": "NOT_SHIFT"}],
            }
            if division_id:
                emp_body["division"] = {"id": division_id}
            client.post("/employee/employment", json=emp_body)

        # Grant PM entitlements via POST /employee/entitlement
        # Must be granted in dependency order: 45 → 10 → 8
        company_id = client.get_cached("company_id")
        if not company_id:
            who = client.get("/token/session/>whoAmI", params={"fields": "companyId"})
            company_id = who.get("value", {}).get("companyId")
            if company_id:
                client.set_cached("company_id", company_id)
        if not company_id:
            _log.warning(f"Could not get companyId from whoAmI for entitlements")
            return False

        _PM_ENTITLEMENTS = [45, 10, 8]
        for eid in _PM_ENTITLEMENTS:
            r = client.post("/employee/entitlement", json={
                "employee": {"id": employee_id},
                "entitlementId": eid,
                "customer": {"id": company_id},
            })
            if r.get("error"):
                msg = str(r.get("message", ""))
                if "utvidet tilgang" in msg.lower():
                    _log.warning(f"Cannot grant PM entitlements to {employee_id} — needs EXTENDED userType (not settable via PUT)")
                    return False
                _log.warning(f"Entitlement {eid} failed for employee {employee_id}: {msg}")

        client.set_cached(cache_key, True)
        return True

    def create_project(
        name: str,
        customer_id: int = 0,
        projectManagerId: int = 0,
        startDate: str = "",
        description: str = "",
        fixedPriceAmount: float = 0.0,
        isInternal: bool = False,
        _pm_fresh_create: bool = False,
    ) -> dict:
        """Create a project in Tripletex.

        Args:
            name: Project name.
            customer_id: ID of the customer linked to this project (0 if none).
            projectManagerId: ID of the employee managing the project (0 to auto-assign).
            startDate: Project start date in YYYY-MM-DD format (defaults to today).
            description: Optional project description.
            fixedPriceAmount: The fixed price for the project (0.0 if not a fixed price project).

        Returns:
            The created project with id, or an error message.
        """
        if not startDate:
            startDate = date.today().isoformat()

        body = {"name": name, "startDate": startDate}
        if customer_id:
            body["customer"] = {"id": customer_id}

        # Resolve project manager (cached to avoid repeated GET)
        pm_id = projectManagerId
        if not pm_id:
            pm_id = client.get_cached("admin_employee_id")
        if not pm_id:
            emp_result = client.get("/employee", params={"fields": "id", "count": 1})
            emps = emp_result.get("values", [])
            if emps:
                pm_id = emps[0]["id"]
                client.set_cached("admin_employee_id", pm_id)

        if pm_id:
            body["projectManager"] = {"id": pm_id}

        if description:
            body["description"] = description
        if fixedPriceAmount > 0:
            body["isFixedPrice"] = True
            body["fixedprice"] = fixedPriceAmount
        if isInternal: # Added logic to set isInternal
            body["isInternal"] = True

        # Pre-ensure PM has entitlements BEFORE attempting project creation
        # This avoids a wasted POST + 422 error + retry cycle
        if pm_id:
            _ensure_employee_ready(pm_id, fresh_create=_pm_fresh_create)
            # Always attempt project creation with intended PM regardless of entitlement result.
            # Tripletex sandbox often accepts the PM even without full entitlements.
            # Falling back to admin causes verifier failure (wrong PM name).

        result = client.post("/project", json=body)
        # If project creation failed due to PM entitlements, retry without PM
        if result.get("error") and pm_id:
            err_msg = str(result.get("message", "")).lower()
            if "entitlement" in err_msg or "tilgang" in err_msg or "rettighet" in err_msg:
                import logging
                logging.getLogger("projects").warning(
                    f"Project creation failed with PM {pm_id}, retrying without PM")
                body.pop("projectManager", None)
                client._error_count = max(0, client._error_count - 1)
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

    def create_voucher(date: str, description: str, postings: list[dict]) -> dict:
        """Create a general ledger voucher.

        Args:
            date: The date of the voucher in YYYY-MM-DD format.
            description: A description for the voucher.
            postings: A list of dictionaries, each with 'accountNumber' (int) and 'amount' (float).
                      Amounts must balance (sum to 0). Positive = debit, negative = credit.

        Returns:
            The created voucher with id and fields, or an error message.
        """
        voucher_lines = []
        for p in postings:
            # Tripletex API expects account ID, not number directly in voucherLines
            account_id_result = client.get("/account", params={"number": p["accountNumber"], "fields": "id", "count": 1})
            account_id = account_id_result.get("values", [{}])[0].get("id")
            if not account_id:
                return {"error": f"Account with number {p['accountNumber']} not found."}
            voucher_lines.append({"account": {"id": account_id}, "amount": p["amount"]})

        body = {
            "date": date,
            "description": description,
            "voucherLines": voucher_lines
        }
        return client.post("/voucher", json=body)

    def create_activity(name: str, project_id: int) -> dict:
        """Create an activity for a project.

        Args:
            name: Name of the activity.
            project_id: ID of the project this activity belongs to.

        Returns:
            The created activity or an error message.
        """
        body = {
            "name": name,
            "project": {"id": project_id},
            "isInactive": False, # Activities are typically active by default
        }
        return client.post("/project/activity", json=body)

    return {
        "create_project": create_project,
        "search_projects": search_projects,
        "update_project": update_project,
        "delete_project": delete_project,
        "create_project_category": create_project_category,
        "search_project_categories": search_project_categories,
        "create_project_participant": create_project_participant,
        "create_activity": create_activity, # Add the new tool
    }
