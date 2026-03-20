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

    def update_contact(
        contact_id: int,
        firstName: str = "",
        lastName: str = "",
        email: str = "",
        phoneNumberMobile: str = "",
    ) -> dict:
        """Update a contact person.

        Args:
            contact_id: ID of the contact to update.
            firstName: New first name (empty to keep).
            lastName: New last name (empty to keep).
            email: New email (empty to keep).
            phoneNumberMobile: New phone (empty to keep).

        Returns:
            Updated contact or error.
        """
        _WRITABLE = {
            "id", "version", "firstName", "lastName", "email",
            "phoneNumberMobile", "phoneNumberWork", "customer",
            "department", "isInactive",
        }
        current = client.get(f"/contact/{contact_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if isinstance(body.get("customer"), dict):
            body["customer"] = {"id": body["customer"]["id"]}
        if isinstance(body.get("department"), dict):
            body["department"] = {"id": body["department"]["id"]}
        if firstName:
            body["firstName"] = firstName
        if lastName:
            body["lastName"] = lastName
        if email:
            body["email"] = email
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        return client.put(f"/contact/{contact_id}", json=body)

    def delete_contact(contact_id: int) -> dict:
        """Delete a contact person.

        Args:
            contact_id: ID of the contact to delete.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/contact/{contact_id}")

    return {
        "create_contact": create_contact,
        "search_contacts": search_contacts,
        "update_contact": update_contact,
        "delete_contact": delete_contact,
    }
