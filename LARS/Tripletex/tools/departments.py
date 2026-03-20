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
            body["departmentNumber"] = departmentNumber
        return client.post("/department", json=body)

    def search_departments(name: str = "") -> dict:
        """Search for departments.

        Args:
            name: Filter by name (partial match).

        Returns:
            A list of departments.
        """
        params = {"fields": "id,name,departmentNumber,departmentManager"}
        if name:
            params["name"] = name
        return client.get("/department", params=params)

    def update_department(department_id: int, name: str = "", departmentNumber: str = "", version: int = -1) -> dict:
        """Update a department.

        Args:
            department_id: The ID of the department to update.
            name: New name (empty to keep current).
            departmentNumber: New number (empty to keep current).
            version: Entity version from the create response. If provided, skips the GET call (saves 1 API call).

        Returns:
            The updated department or error.
        """
        if version >= 0:
            body = {"id": department_id, "version": version}
            if name:
                body["name"] = name
            if departmentNumber:
                body["departmentNumber"] = departmentNumber
            return client.put(f"/department/{department_id}", json=body)

        _WRITABLE = {"id", "version", "name", "departmentNumber", "departmentManager", "isInactive"}
        current = client.get(f"/department/{department_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("departmentManager"), dict):
            body["departmentManager"] = {"id": body["departmentManager"]["id"]}
        if name:
            body["name"] = name
        if departmentNumber:
            body["departmentNumber"] = departmentNumber
        return client.put(f"/department/{department_id}", json=body)

    def delete_department(department_id: int) -> dict:
        """Delete a department.

        Args:
            department_id: ID of the department to delete.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/department/{department_id}")

    return {
        "create_department": create_department,
        "search_departments": search_departments,
        "update_department": update_department,
        "delete_department": delete_department,
    }
