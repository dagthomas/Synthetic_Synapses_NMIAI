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
        result = client.post("/department", json=body)

        # Auto-recover: if duplicate (422), search by name and return existing
        if result.get("error") and result.get("status_code") == 422:
            params = {"fields": "id,name,departmentNumber,departmentManager"}
            params["name"] = name
            existing = client.get("/department", params=params)
            vals = existing.get("values", [])
            # Filter for exact match, as API's 'name' parameter might be a partial match or return results in arbitrary order.
            exact_match = next((d for d in vals if d.get("name") == name), None)
            if exact_match:
                return {"value": exact_match, "_note": "Department already existed, returning existing."}

        return result

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
            version: Entity version from the create response. If provided (>0), skips the GET call (saves 1 API call).

        Returns:
            The updated department or error.
        """
        _WRITABLE = {"id", "version", "name", "departmentNumber", "departmentManager", "isInactive"}
        if version > 0 and name:
            # Fast path: skip GET when we have version + required field (name)
            body = {"id": department_id, "version": version, "name": name}
            if departmentNumber:
                body["departmentNumber"] = departmentNumber
        else:
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
