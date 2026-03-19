from tripletex_client import TripletexClient


def build_address_tools(client: TripletexClient) -> dict:
    """Build delivery address tools."""

    def search_delivery_addresses() -> dict:
        """Search for delivery addresses.

        Returns:
            A list of delivery addresses.
        """
        return client.get("/deliveryAddress", params={"fields": "*"})

    def update_delivery_address(
        address_id: int,
        addressLine1: str = "",
        postalCode: str = "",
        city: str = "",
        country_id: int = 0,
    ) -> dict:
        """Update an existing delivery address.

        Args:
            address_id: The ID of the delivery address to update.
            addressLine1: Street address line 1.
            postalCode: Postal code.
            city: City name.
            country_id: Country ID (0 to keep unchanged).

        Returns:
            The updated address or an error message.
        """
        body = {}
        if addressLine1:
            body["addressLine1"] = addressLine1
        if postalCode:
            body["postalCode"] = postalCode
        if city:
            body["city"] = city
        if country_id:
            body["country"] = {"id": country_id}
        return client.put(f"/deliveryAddress/{address_id}", json=body)

    return {
        "search_delivery_addresses": search_delivery_addresses,
        "update_delivery_address": update_delivery_address,
    }
