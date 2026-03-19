from tripletex_client import TripletexClient


def build_department_tools(client: TripletexClient) -> dict:
    """Build department tools."""

    def create_department(
        name: str,
        departmentNumber: str = "",
    ) -> dict:
        """Create a department in Tripletex. May require enabling the department module first.

        Args:
            name: Department name.
            departmentNumber: Optional department number/code.

        Returns:
            The created department with id, or an error message.
        """
        body = {"name": name}
        if departmentNumber:
            body["number"] = departmentNumber
        return client.post("/department", json=body)

    return {"create_department": create_department}
