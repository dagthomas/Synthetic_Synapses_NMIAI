from tripletex_client import TripletexClient


def build_supplier_tools(client: TripletexClient) -> dict:
    """Build supplier management tools."""

    def create_supplier(
        name: str,
        email: str = "",
        phoneNumber: str = "",
        organizationNumber: str = "",
        addressLine1: str = "",
        postalCode: str = "",
        city: str = "",
    ) -> dict:
        """Create a new supplier.

        Args:
            name: The supplier company name.
            email: Supplier email address.
            phoneNumber: Supplier phone number.
            organizationNumber: Norwegian org number (9 digits).
            addressLine1: Street address (e.g. "Storgata 1").
            postalCode: Postal/zip code (e.g. "0182").
            city: City name (e.g. "Oslo").

        Returns:
            The created supplier with id, or an error message.
        """
        body = {"name": name, "isSupplier": True}
        if email:
            body["email"] = email
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
        result = client.post("/supplier", json=body)

        # Auto-recover: if duplicate (422), search by name and return existing
        if result.get("error") and result.get("status_code") == 422:
            params = {"fields": "id,name,email,organizationNumber,phoneNumber"}
            if organizationNumber:
                params["organizationNumber"] = organizationNumber
            else:
                params["name"] = name
            existing = client.get("/supplier", params=params)
            vals = existing.get("values", [])
            if vals:
                return {"value": vals[0], "_note": "Supplier already existed, returning existing."}

        return result

    def search_suppliers(name: str = "", organizationNumber: str = "") -> dict:
        """Search for suppliers by name or org number.

        Args:
            name: Filter by supplier name (partial match).
            organizationNumber: Filter by organization number.

        Returns:
            A list of matching suppliers.
        """
        params = {"fields": "id,name,email,organizationNumber,phoneNumber"}
        if name:
            params["name"] = name
        if organizationNumber:
            params["organizationNumber"] = organizationNumber
        return client.get("/supplier", params=params)

    def update_supplier(supplier_id: int, name: str = "", email: str = "", phoneNumber: str = "", version: int = -1) -> dict:
        """Update an existing supplier.

        Args:
            supplier_id: The ID of the supplier to update.
            name: New supplier name (empty to keep unchanged).
            email: New email (empty to keep unchanged).
            phoneNumber: New phone number (empty to keep unchanged).
            version: Entity version from the create response. If provided (>0), skips the GET call (saves 1 API call).

        Returns:
            The updated supplier or an error message.
        """
        _WRITABLE = {
            "id", "version", "name", "email", "phoneNumber", "phoneNumberMobile",
            "organizationNumber", "isSupplier", "isCustomer", "accountManager",
            "description", "bankAccounts",
        }
        if version > 0 and name:
            # Fast path: skip GET when we have version + required field (name)
            body = {"id": supplier_id, "version": version, "name": name, "isSupplier": True}
            if email:
                body["email"] = email
            if phoneNumber:
                body["phoneNumber"] = phoneNumber
        else:
            current = client.get(f"/supplier/{supplier_id}", params={"fields": "*"})
            full = current.get("value", {})
            body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
            if name:
                body["name"] = name
            if email:
                body["email"] = email
            if phoneNumber:
                body["phoneNumber"] = phoneNumber
        return client.put(f"/supplier/{supplier_id}", json=body)

    def delete_supplier(supplier_id: int) -> dict:
        """Delete a supplier.

        Args:
            supplier_id: ID of the supplier.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/supplier/{supplier_id}")

    return {
        "create_supplier": create_supplier,
        "search_suppliers": search_suppliers,
        "update_supplier": update_supplier,
        "delete_supplier": delete_supplier,
    }
