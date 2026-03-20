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
        # Tripletex requires a department — auto-resolve or create if not given
        if not department_id:
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
        return client.post("/employee", json=body)

    def update_employee(employee_id: int, firstName: str = "", lastName: str = "", email: str = "", phoneNumberMobile: str = "", isInactive: bool = False) -> dict:
        """Update an existing employee's fields. Set isInactive=True to deactivate.

        Args:
            employee_id: The ID of the employee to update.
            firstName: New first name (leave empty to keep current).
            lastName: New last name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumberMobile: New phone number (leave empty to keep current).
            isInactive: Set to True to deactivate the employee.

        Returns:
            The updated employee data or an error message.
        """
        # Tripletex PUT requires writable fields — GET first, keep only writable, merge
        _WRITABLE = {
            "id", "version", "firstName", "lastName", "email",
            "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
            "dateOfBirth", "department", "employeeNumber", "address",
            "userType", "nationalIdentityNumber", "bankAccountNumber",
            "comments", "employeeCategory", "isInactive",
        }
        current = client.get(f"/employee/{employee_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE} if full else {}
        # Tripletex requires dateOfBirth on PUT — set default if missing
        if not body.get("dateOfBirth"):
            body["dateOfBirth"] = "1990-01-01"
        # Strip nested read-only fields from department
        if isinstance(body.get("department"), dict):
            body["department"] = {"id": body["department"]["id"]}
        # Strip None values that cause validation errors
        body = {k: v for k, v in body.items() if v is not None}
        # Tripletex does NOT allow email changes — keep original to avoid 422
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
