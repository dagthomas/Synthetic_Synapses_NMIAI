from tripletex_client import TripletexClient


def build_customer_tools(client: TripletexClient) -> dict:
    """Build customer-related tools."""

    def create_customer(
        name: str,
        email: str = "",
        isCustomer: bool = True,
        isSupplier: bool = False,
        phoneNumber: str = "",
        organizationNumber: str = "",
        addressLine1: str = "",
        postalCode: str = "",
        city: str = "",
    ) -> dict:
        """Create a new customer (or supplier) in Tripletex.

        Args:
            name: The company or person name.
            email: Contact email address.
            isCustomer: Whether this is a customer.
            isSupplier: Whether this is a supplier.
            phoneNumber: Contact phone number.
            organizationNumber: Norwegian org number.
            addressLine1: Street address (e.g. "Storgata 1").
            postalCode: Postal/zip code (e.g. "0182").
            city: City name (e.g. "Oslo").

        Returns:
            The created customer with id and fields, or an error message.
        """
        body = {"name": name, "isCustomer": isCustomer}
        if email:
            body["email"] = email
        if isSupplier:
            body["isSupplier"] = True
        if phoneNumber:
            body["phoneNumber"] = phoneNumber
        if organizationNumber:
            body["organizationNumber"] = organizationNumber
        if addressLine1 or postalCode or city:
            addr = {}
            if addressLine1:
                addr["addressLine1"] = addressLine1
            if postalCode:
                addr["postalCode"] = postalCode
            if city:
                addr["city"] = city
            body["postalAddress"] = addr
            body["physicalAddress"] = addr
        result = client.post("/customer", json=body)

        # Auto-recover: if duplicate (422), search by name and return existing
        if result.get("error") and result.get("status_code") == 422:
            params = {"fields": "id,name,email,organizationNumber,isCustomer,isSupplier"}
            if organizationNumber:
                params["organizationNumber"] = organizationNumber
            else:
                params["name"] = name
            existing = client.get("/customer", params=params)
            vals = existing.get("values", [])
            if vals:
                return {"value": vals[0], "_note": "Customer already existed, returning existing."}

        return result

    def update_customer(customer_id: int, name: str = "", email: str = "", phoneNumber: str = "", version: int = -1) -> dict:
        """Update an existing customer's fields.

        Args:
            customer_id: The ID of the customer to update.
            name: New name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumber: New phone number (leave empty to keep current).
            version: Entity version from the create response. If provided (>0), skips the GET call (saves 1 API call).

        Returns:
            The updated customer data or an error message.
        """
        _WRITABLE = {
            "id", "version", "name", "email", "phoneNumber", "phoneNumberMobile",
            "organizationNumber", "isCustomer", "isSupplier", "accountManager",
            "description", "language", "invoiceEmail", "category1", "category2",
            "category3", "bankAccounts", "invoiceSendMethod",
        }
        if version > 0 and name:
            # Fast path: skip GET when we have version + required field (name)
            body = {"id": customer_id, "version": version, "name": name, "isCustomer": True}
            if email:
                body["email"] = email
            if phoneNumber:
                body["phoneNumber"] = phoneNumber
        else:
            current = client.get(f"/customer/{customer_id}", params={"fields": "*"})
            full = current.get("value", {})
            body = {k: v for k, v in full.items() if k in _WRITABLE} if full else {}
            body = {k: v for k, v in body.items() if v is not None}
            if name:
                body["name"] = name
            if email:
                body["email"] = email
            if phoneNumber:
                body["phoneNumber"] = phoneNumber
        return client.put(f"/customer/{customer_id}", json=body)

    def search_customers(name: str = "", email: str = "") -> dict:
        """Search for customers by name or email.

        Args:
            name: Filter by customer name (partial match).
            email: Filter by email (partial match).

        Returns:
            A list of matching customers with id, name, email.
        """
        params = {"fields": "id,name,email,isCustomer,isSupplier"}
        if name:
            params["name"] = name
        if email:
            params["email"] = email
        return client.get("/customer", params=params)

    def delete_customer(customer_id: int) -> dict:
        """Delete a customer.

        Args:
            customer_id: ID of the customer.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/customer/{customer_id}")

    return {
        "create_customer": create_customer,
        "update_customer": update_customer,
        "search_customers": search_customers,
        "delete_customer": delete_customer,
    }
