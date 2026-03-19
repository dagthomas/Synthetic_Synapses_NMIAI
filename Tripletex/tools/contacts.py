from tripletex_client import TripletexClient


def build_contact_tools(client: TripletexClient) -> dict:
    """Build contact person tools."""

    def create_contact(
        firstName: str,
        lastName: str,
        email: str,
        customer_id: int,
        phoneNumberMobile: str = "",
    ) -> dict:
        """Create a contact person linked to an existing customer.

        Args:
            firstName: Contact person's first name.
            lastName: Contact person's last name.
            email: Contact person's email address.
            customer_id: The ID of the customer this contact belongs to.
            phoneNumberMobile: Optional mobile phone number.

        Returns:
            The created contact with id, or an error message.
        """
        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "customer": {"id": customer_id},
        }
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        return client.post("/contact", json=body)

    def search_contacts(firstName: str = "", lastName: str = "", customer_id: int = 0) -> dict:
        """Search for contact persons.

        Args:
            firstName: Filter by first name (partial match).
            lastName: Filter by last name (partial match).
            customer_id: Filter by customer ID (0 for all).

        Returns:
            A list of matching contacts.
        """
        params = {"fields": "id,firstName,lastName,email,customer"}
        if firstName:
            params["firstName"] = firstName
        if lastName:
            params["lastName"] = lastName
        if customer_id:
            params["customerId"] = customer_id
        return client.get("/contact", params=params)

    return {
        "create_contact": create_contact,
        "search_contacts": search_contacts,
    }
