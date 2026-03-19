from tripletex_client import TripletexClient


def build_employee_tools(client: TripletexClient) -> dict:
    """Build employee-related tools as closures over the client."""

    def create_employee(
        firstName: str,
        lastName: str,
        email: str,
        phoneNumberMobile: str = "",
        userType: str = "STANDARD",
    ) -> dict:
        """Create a new employee in Tripletex.

        Args:
            firstName: The employee's first name.
            lastName: The employee's last name.
            email: The employee's email address.
            phoneNumberMobile: The employee's mobile phone number.
            userType: User access type. Use "STANDARD" for normal users, "NO_ACCESS" for no login.

        Returns:
            The created employee with id and fields, or an error message.
        """
        # Tripletex requires a department — auto-resolve one
        dept_result = client.get("/department", params={"fields": "id", "count": 1})
        depts = dept_result.get("values", [])
        if depts:
            dept_id = depts[0]["id"]
        else:
            dept_create = client.post("/department", json={"name": "Generell", "departmentNumber": "1"})
            dept_id = dept_create.get("value", {}).get("id", 0)

        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "userType": userType,
        }
        if dept_id:
            body["department"] = {"id": dept_id}
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        return client.post("/employee", json=body)

    def update_employee(employee_id: int, firstName: str = "", lastName: str = "", email: str = "", phoneNumberMobile: str = "") -> dict:
        """Update an existing employee's fields.

        Args:
            employee_id: The ID of the employee to update.
            firstName: New first name (leave empty to keep current).
            lastName: New last name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumberMobile: New phone number (leave empty to keep current).

        Returns:
            The updated employee data or an error message.
        """
        body = {}
        if firstName:
            body["firstName"] = firstName
        if lastName:
            body["lastName"] = lastName
        if email:
            body["email"] = email
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
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
