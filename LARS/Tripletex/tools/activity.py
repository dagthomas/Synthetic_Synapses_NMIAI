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
        return client.post("/activity", json=body)

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
