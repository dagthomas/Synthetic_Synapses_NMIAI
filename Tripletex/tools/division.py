from tripletex_client import TripletexClient


def build_division_tools(client: TripletexClient) -> dict:
    """Build division tools."""

    def create_division(
        name: str,
        startDate: str = "",
        organizationNumber: str = "",
        municipalityDate: str = "",
        municipality_id: int = 0,
        endDate: str = "",
    ) -> dict:
        """Create a division (underenhet).

        Args:
            name: Division name.
            startDate: Start date YYYY-MM-DD (defaults to today).
            organizationNumber: Org number for the division (auto-generated if empty).
            municipalityDate: Municipality date YYYY-MM-DD (defaults to startDate).
            municipality_id: Municipality ID (0 to auto-detect).
            endDate: End date YYYY-MM-DD (empty for no end).

        Returns:
            Created division or error.
        """
        from datetime import date as dt_date
        import random
        if not startDate:
            startDate = dt_date.today().isoformat()
        if not organizationNumber:
            organizationNumber = str(900000000 + random.randint(0, 99999999))
        body = {"name": name, "startDate": startDate}
        body["organizationNumber"] = organizationNumber
        if not municipalityDate:
            municipalityDate = startDate
        body["municipalityDate"] = municipalityDate
        if municipality_id:
            body["municipality"] = {"id": municipality_id}
        else:
            # Try to find a municipality
            muni = client.get("/municipality", params={"fields": "id", "count": 1})
            munis = muni.get("values", [])
            if munis:
                body["municipality"] = {"id": munis[0]["id"]}
        if endDate:
            body["endDate"] = endDate
        return client.post("/division", json=body)

    def search_divisions() -> dict:
        """Search for divisions.

        Returns:
            A list of divisions.
        """
        return client.get("/division", params={"fields": "id,name,organizationNumber,startDate,endDate"})

    def update_division(
        division_id: int,
        name: str = "",
        organizationNumber: str = "",
    ) -> dict:
        """Update a division.

        Args:
            division_id: ID of the division.
            name: New name (empty to keep).
            organizationNumber: New org number (empty to keep).

        Returns:
            Updated division or error.
        """
        _WRITABLE = {
            "id", "version", "name", "startDate", "endDate",
            "organizationNumber", "municipalityDate", "municipality",
        }
        current = client.get(f"/division/{division_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("municipality"), dict):
            body["municipality"] = {"id": body["municipality"]["id"]}
        if name:
            body["name"] = name
        if organizationNumber:
            body["organizationNumber"] = organizationNumber
        return client.put(f"/division/{division_id}", json=body)

    return {
        "create_division": create_division,
        "search_divisions": search_divisions,
        "update_division": update_division,
    }
