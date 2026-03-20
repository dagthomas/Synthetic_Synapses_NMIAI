from tripletex_client import TripletexClient


def build_timesheet_tools(client: TripletexClient) -> dict:
    """Build timesheet tools."""

    def create_timesheet_entry(
        employee_id: int,
        project_id: int,
        activity_id: int,
        date: str,
        hours: float,
        comment: str = "",
    ) -> dict:
        """Create a timesheet entry.

        Args:
            employee_id: Employee ID.
            project_id: Project ID.
            activity_id: Activity ID.
            date: Date YYYY-MM-DD.
            hours: Number of hours.
            comment: Optional comment.

        Returns:
            Created timesheet entry or error.
        """
        body = {
            "employee": {"id": employee_id},
            "project": {"id": project_id},
            "activity": {"id": activity_id},
            "date": date,
            "hours": hours,
        }
        if comment:
            body["comment"] = comment
        return client.post("/timesheet/entry", json=body)

    def search_timesheet_entries(
        employeeId: int = 0,
        dateFrom: str = "",
        dateTo: str = "",
        projectId: int = 0,
    ) -> dict:
        """Search for timesheet entries.

        Args:
            employeeId: Filter by employee ID (0 for all).
            dateFrom: Filter from date YYYY-MM-DD.
            dateTo: Filter to date YYYY-MM-DD.
            projectId: Filter by project ID (0 for all).

        Returns:
            A list of timesheet entries.
        """
        params = {"fields": "id,employee,project,activity,date,hours,comment"}
        if employeeId:
            params["employeeId"] = employeeId
        if dateFrom:
            params["dateFrom"] = dateFrom
        if dateTo:
            params["dateTo"] = dateTo
        if projectId:
            params["projectId"] = projectId
        return client.get("/timesheet/entry", params=params)

    def update_timesheet_entry(
        entry_id: int,
        hours: float = 0,
        comment: str = "",
    ) -> dict:
        """Update a timesheet entry.

        Args:
            entry_id: ID of the entry.
            hours: New hours (0 to keep).
            comment: New comment (empty to keep).

        Returns:
            Updated entry or error.
        """
        _WRITABLE = {"id", "version", "project", "activity", "date", "hours", "employee", "comment"}
        current = client.get(f"/timesheet/entry/{entry_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        for ref in ("project", "activity", "employee"):
            if isinstance(body.get(ref), dict) and "id" in body[ref]:
                body[ref] = {"id": body[ref]["id"]}
        if hours:
            body["hours"] = hours
        if comment:
            body["comment"] = comment
        return client.put(f"/timesheet/entry/{entry_id}", json=body)

    def delete_timesheet_entry(entry_id: int) -> dict:
        """Delete a timesheet entry.

        Args:
            entry_id: ID of the entry.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/timesheet/entry/{entry_id}")

    return {
        "create_timesheet_entry": create_timesheet_entry,
        "search_timesheet_entries": search_timesheet_entries,
        "update_timesheet_entry": update_timesheet_entry,
        "delete_timesheet_entry": delete_timesheet_entry,
    }
