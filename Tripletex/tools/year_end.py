from tripletex_client import TripletexClient


def build_year_end_tools(client: TripletexClient) -> dict:
    """Build year-end (aarsoppgjoer) tools."""

    def search_year_end_annexes(year_end_id: int = 0) -> dict:
        """Search for year-end annexes.

        Args:
            year_end_id: ID of the year-end to get annexes for (0 to auto-detect most recent).

        Returns:
            A list of year-end annexes.
        """
        if not year_end_id:
            ye = client.get("/yearEnd", params={"fields": "id", "count": 1})
            ye_list = ye.get("values", [])
            year_end_id = ye_list[0]["id"] if ye_list else 0
        if not year_end_id:
            return {"values": [], "fullResultSize": 0}
        return client.get("/yearEnd/annex", params={"yearEndId": year_end_id, "fields": "*"})

    def create_year_end_note(
        year_end_id: int,
        note: str = "",
    ) -> dict:
        """Create a year-end note.

        Args:
            year_end_id: ID of the year-end.
            note: Note content.

        Returns:
            Created note or error.
        """
        body = {"yearEnd": {"id": year_end_id}}
        if note:
            body["note"] = note
        return client.post("/yearEnd/note", json=body)

    def get_vat_returns() -> dict:
        """Get VAT types (MVA-typer).

        Returns:
            A list of VAT types.
        """
        return client.get("/ledger/vatType", params={"fields": "id,number,name,percentage"})

    return {
        "search_year_end_annexes": search_year_end_annexes,
        "create_year_end_note": create_year_end_note,
        "get_vat_returns": get_vat_returns,
    }
