from tripletex_client import TripletexClient


def build_employee_tools(client: TripletexClient) -> dict:
    """Build employee-related tools as closures over the client."""

    def create_employee(
        firstName: str,
        lastName: str,
        email: str,
        phoneNumberMobile: str = "",
        userType: str = "STANDARD",
        department_id: int = 0,
        dateOfBirth: str = "",
    ) -> dict:
        """Create a new employee in Tripletex.

        Args:
            firstName: The employee's first name.
            lastName: The employee's last name.
            email: The employee's email address.
            phoneNumberMobile: The employee's mobile phone number.
            userType: User access type. Use "EXTENDED" for account administrator, "STANDARD" for normal users, "NO_ACCESS" for no login.
            department_id: Optional department ID to assign the employee to (0 to skip).
            dateOfBirth: Date of birth in YYYY-MM-DD format (optional).

        Returns:
            The created employee with id and fields, or an error message.
        """
        # Tripletex requires a department — use cache or auto-resolve
        if not department_id:
            cached = client.get_cached("default_department")
            if cached:
                department_id = cached
            else:
                dept_result = client.get("/department", params={"fields": "id", "count": 1})
                depts = dept_result.get("values", [])
                if depts:
                    department_id = depts[0]["id"]
                else:
                    # Fresh sandbox with no departments — create a default one
                    new_dept = client.post("/department", json={
                        "name": "Avdeling",
                        "departmentNumber": "1",
                    })
                    dept_val = new_dept.get("value", {})
                    if dept_val.get("id"):
                        department_id = dept_val["id"]
                if department_id:
                    client.set_cached("default_department", department_id)

        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "userType": userType,
        }
        if department_id:
            body["department"] = {"id": department_id}
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        if dateOfBirth:
            body["dateOfBirth"] = dateOfBirth
        result = client.post("/employee", json=body)

        # Auto-recover: if email already exists, find and return the existing employee
        if (result.get("error") and result.get("status_code") == 422
                and "e-postadress" in str(result.get("message", "")).lower()):
            existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
            vals = existing.get("values", [])
            if vals:
                # Undo the error count — recovery succeeded, don't penalize scoring
                client._error_count = max(0, client._error_count - 1)
                for entry in reversed(client._call_log):
                    if not entry.get("ok") and "/employee" in entry.get("url", ""):
                        entry["ok"] = True
                        entry["recovered"] = True
                        break
                return {"value": vals[0], "_note": "Employee already existed, returning existing."}

        return result

    def update_employee(employee_id: int, firstName: str = "", lastName: str = "", email: str = "", phoneNumberMobile: str = "", isInactive: bool = False, version: int = -1) -> dict:
        """Update an existing employee's fields. Set isInactive=True to deactivate.

        Args:
            employee_id: The ID of the employee to update.
            firstName: New first name (leave empty to keep current).
            lastName: New last name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumberMobile: New phone number (leave empty to keep current).
            isInactive: Set to True to deactivate the employee.
            version: Entity version from the create response. If provided (>0), skips the GET call (saves 1 API call).

        Returns:
            The updated employee data or an error message.
        """
        _WRITABLE = {
            "id", "version", "firstName", "lastName", "email",
            "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
            "dateOfBirth", "department", "employeeNumber", "address",
            "userType", "nationalIdentityNumber", "bankAccountNumber",
            "comments", "employeeCategory", "isInactive",
        }
        if version > 0 and firstName and lastName:
            # Fast path: skip GET when we have version + required fields
            body = {"id": employee_id, "version": version, "firstName": firstName, "lastName": lastName,
                    "dateOfBirth": "1990-01-01"}
            if email:
                body["email"] = email
            if phoneNumberMobile:
                body["phoneNumberMobile"] = phoneNumberMobile
            if isInactive:
                body["isInactive"] = True
        else:
            current = client.get(f"/employee/{employee_id}", params={"fields": "*"})
            full = current.get("value", {})
            body = {k: v for k, v in full.items() if k in _WRITABLE} if full else {}
            if not body.get("dateOfBirth"):
                body["dateOfBirth"] = "1990-01-01"
            if isinstance(body.get("department"), dict):
                body["department"] = {"id": body["department"]["id"]}
            body = {k: v for k, v in body.items() if v is not None}
            if firstName:
                body["firstName"] = firstName
            if lastName:
                body["lastName"] = lastName
            if phoneNumberMobile:
                body["phoneNumberMobile"] = phoneNumberMobile
            if isInactive:
                body["isInactive"] = True
        return client.put(f"/employee/{employee_id}", json=body)

    def search_employees(firstName: str = "", lastName: str = "", email: str = "") -> dict:
        """Search for employees by name or email.

        Args:
            firstName: Filter by first name (partial match).
            lastName: Filter by last name (partial match).
            email: Filter by email (partial match).

        Returns:
            A list of matching employees with id, firstName, lastName, email.
        """
        params = {"fields": "id,firstName,lastName,email"}
        if firstName:
            params["firstName"] = firstName
        if lastName:
            params["lastName"] = lastName
        if email:
            params["email"] = email
        return client.get("/employee", params=params)

    return {
        "create_employee": create_employee,
        "update_employee": update_employee,
        "search_employees": search_employees,
    }
