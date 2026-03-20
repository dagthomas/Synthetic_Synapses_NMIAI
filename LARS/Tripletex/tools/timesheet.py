from tripletex_client import TripletexClient


def build_timesheet_tools(client: TripletexClient) -> dict:
    """Build timesheet tools."""

    def create_timesheet_entry(
        employee_id: int,
        date: str,
        hours: float,
        project_id: int = 0,
        activity_id: int = 0,
        comment: str = "",
    ) -> dict:
        """Create a timesheet entry.

        Args:
            employee_id: Employee ID.
            date: Date YYYY-MM-DD.
            hours: Number of hours.
            project_id: Project ID (0 to auto-detect first available).
            activity_id: Activity ID (0 to auto-detect first available).
            comment: Optional comment.

        Returns:
            Created timesheet entry or error.
        """
        if not project_id:
            projs = client.get("/project", params={"fields": "id", "count": 1})
            proj_list = projs.get("values", [])
            project_id = proj_list[0]["id"] if proj_list else 0
        if not activity_id:
            # Find a project-compatible activity (isProjectActivity=True)
            acts = client.get("/activity", params={"fields": "id", "isProjectActivity": True, "count": 1})
            act_list = acts.get("values", [])
            if not act_list:
                acts = client.get("/activity", params={"fields": "id", "count": 1})
                act_list = acts.get("values", [])
            activity_id = act_list[0]["id"] if act_list else 0
        if not project_id or not activity_id:
            return {"error": True, "message": "No project or activity found for timesheet entry"}
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
        employee_id: int = 0,
        dateFrom: str = "",
        dateTo: str = "",
        projectId: int = 0,
    ) -> dict:
        """Search for timesheet entries.

        Args:
            employee_id: Filter by employee ID (0 for all).
            dateFrom: Filter from date YYYY-MM-DD.
            dateTo: Filter to date YYYY-MM-DD.
            projectId: Filter by project ID (0 for all).

        Returns:
            A list of timesheet entries.
        """
        params = {"fields": "id,employee,project,activity,date,hours,comment"}
        if employee_id:
            params["employeeId"] = employee_id
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
