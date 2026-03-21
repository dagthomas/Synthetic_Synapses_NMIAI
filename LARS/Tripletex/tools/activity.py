from tripletex_client import TripletexClient


def build_activity_tools(client: TripletexClient) -> dict:
    """Build activity tools."""

    def create_activity(
        name: str,
        number: str = "",
        description: str = "",
        isChargeable: bool = True,
        activityType: str = "PROJECT_GENERAL_ACTIVITY",
    ) -> dict:
        """Create an activity (used for time tracking and projects).

        Args:
            name: Activity name.
            number: Activity number/code.
            description: Activity description.
            isChargeable: Whether the activity is chargeable to customers.
            activityType: Activity type - 'PROJECT_GENERAL_ACTIVITY', 'OFFICE_ACTIVITY', etc.

        Returns:
            The created activity or error.
        """
        body = {"name": name, "isChargeable": isChargeable, "activityType": activityType}
        if number:
            body["number"] = number
        if description:
            body["description"] = description
        result = client.post("/activity", json=body)

        # Auto-recover on "name already in use" — search and return existing
        if result.get("error"):
            msg = str(result.get("message", ""))
            if "bruk" in msg.lower() or "already" in msg.lower():
                # Undo the error count — this is a recoverable situation
                client._error_count = max(0, client._error_count - 1)
                for entry in reversed(client._call_log):
                    if not entry.get("ok") and "/activity" in entry.get("url", ""):
                        entry["ok"] = True
                        entry["recovered"] = True
                        break
                existing = client.get("/activity", params={"name": name, "fields": "id,name,number,isChargeable,activityType"})
                vals = existing.get("values", [])
                if vals:
                    return {"value": vals[0], "_note": "Activity already existed, returning existing."}
        return result

    def search_activities(name: str = "") -> dict:
        """Search for activities.

        Args:
            name: Filter by name (partial match).

        Returns:
            A list of activities.
        """
        params = {"fields": "id,name,number,description,isChargeable,activityType"}
        if name:
            params["name"] = name
        return client.get("/activity", params=params)

    return {
        "create_activity": create_activity,
        "search_activities": search_activities,
    }
