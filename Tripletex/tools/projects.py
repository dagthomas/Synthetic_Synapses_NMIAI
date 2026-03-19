from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def create_project(
        name: str,
        customer_id: int = 0,
        projectManagerId: int = 0,
        startDate: str = "",
        description: str = "",
    ) -> dict:
        """Create a project in Tripletex.

        Args:
            name: Project name.
            customer_id: ID of the customer linked to this project (0 if none).
            projectManagerId: ID of the employee managing the project (0 if none).
            startDate: Project start date in YYYY-MM-DD format.
            description: Optional project description.

        Returns:
            The created project with id, or an error message.
        """
        body = {"name": name}
        if customer_id:
            body["customer"] = {"id": customer_id}
        if projectManagerId:
            body["projectManager"] = {"id": projectManagerId}
        if startDate:
            body["startDate"] = startDate
        if description:
            body["description"] = description
        return client.post("/project", json=body)

    return {"create_project": create_project}
