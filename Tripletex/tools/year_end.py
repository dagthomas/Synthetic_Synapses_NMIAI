from tripletex_client import TripletexClient


def build_year_end_tools(client: TripletexClient) -> dict:
    """Build year-end (aarsoppgjoer) tools."""

    def search_year_end_annexes(yearEndId: int) -> dict:
        """Search for year-end annexes.

        Args:
            yearEndId: ID of the year-end to get annexes for.

        Returns:
            A list of year-end annexes.
        """
        return client.get("/yearEnd/annex", params={"yearEndId": yearEndId, "fields": "*"})

    def create_year_end_note(
        yearEndId: int,
        note: str = "",
    ) -> dict:
        """Create a year-end note.

        Args:
            yearEndId: ID of the year-end.
            note: Note content.

        Returns:
            Created note or error.
        """
        body = {"yearEnd": {"id": yearEndId}}
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
