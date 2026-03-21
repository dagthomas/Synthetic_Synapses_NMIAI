from tripletex_client import TripletexClient


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

    # Hardcoded fallback for output VAT (utgående MVA) type IDs
    _OUTPUT_VAT_FALLBACK = {25: 3, 15: 31, 12: 33, 0: 6}
    _DYNAMIC_VAT_MAP = {}
    _VAT_MAP_INITIALIZED = False

    def _initialize_vat_map(client: TripletexClient):
        nonlocal _VAT_MAP_INITIALIZED
        if _VAT_MAP_INITIALIZED:
            return

        # 1. Check if client pre-warmed the VAT map (free — no API call counted)
        cached = client.get_cached("vat_type_map")
        if cached:
            _DYNAMIC_VAT_MAP.update(cached)
            _VAT_MAP_INITIALIZED = True
            return

        # 2. Fetch from correct endpoint: /ledger/vatType (not /vatType)
        #    Must filter for OUTPUT VAT types — input types have same percentages but different IDs
        result = client.get("/ledger/vatType", params={"fields": "id,number,name,percentage"})
        # Temporary storage for all VAT types found
        all_vat_types_by_percentage = {}

        if result and result.get("values"):
            for vat_type in result["values"]:
                percentage = vat_type.get("percentage")
                vat_id = vat_type.get("id")
                name = (vat_type.get("name") or "").lower()
                if percentage is not None and vat_id is not None:
                    percentage_int = int(percentage)
                    if percentage_int not in all_vat_types_by_percentage:
                        all_vat_types_by_percentage[percentage_int] = []
                    all_vat_types_by_percentage[percentage_int].append({"id": vat_id, "name": name})

        # Populate _DYNAMIC_VAT_MAP
        for percentage, vat_types_list in all_vat_types_by_percentage.items():
            # Prioritize "utgående" types for all percentages
            found_id = None
            for vt in vat_types_list:
                if "utgående" in vt["name"] or "utg." in vt["name"]:
                    found_id = vt["id"]
                    break
            
            # If no "utgående" type found, take the first one for 25% (standard)
            # This is a specific heuristic for the common 25% standard rate,
            # as its name might not always contain "utgående".
            if found_id is None and percentage == 25 and vat_types_list:
                found_id = vat_types_list[0]["id"]

            if found_id is not None:
                _DYNAMIC_VAT_MAP[percentage] = found_id

        # Fallback to hardcoded output VAT IDs for any percentages not dynamically mapped
        for percentage, vat_id in _OUTPUT_VAT_FALLBACK.items():
            if percentage not in _DYNAMIC_VAT_MAP:
                _DYNAMIC_VAT_MAP[percentage] = vat_id

        _VAT_MAP_INITIALIZED = True

    def create_product(
        name: str,
        priceExcludingVatCurrency: float = 0.0,
        priceIncludingVatCurrency: float = 0.0,
        productNumber: str = "",
        vatPercentage: int = -1,
    ) -> dict:
        """Create a new product in Tripletex.

        Args:
            name: The product name.
            priceExcludingVatCurrency: Price excluding VAT in NOK. Preferred — Tripletex auto-calculates incl VAT.
            priceIncludingVatCurrency: Price including VAT in NOK. Only send this if you do NOT send priceExcludingVatCurrency.
            productNumber: Optional product number/SKU.
            vatPercentage: VAT rate for this product: 25 (standard), 15 (food/mat), 12 (transport), 0 (exempt). Default -1 = Tripletex default (25%).

        Returns:
            The created product with id and fields, or an error message.
        """
        _initialize_vat_map(client) # Ensure VAT map is initialized

        # Check if product number already exists before POSTing (avoids 422 error)
        if productNumber:
            existing = client.get("/product", params={
                "number": productNumber,
                "fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency",
            })
            vals = existing.get("values", [])
            if vals:
                return {"value": vals[0], "_note": "Product number already existed, returning existing."}

        body = {"name": name}
        # Only send ONE price — sending both causes validation errors when they don't match the product's VAT type
        if priceExcludingVatCurrency:
            body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
        elif priceIncludingVatCurrency:
            body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
        if productNumber:
            body["number"] = productNumber
        if vatPercentage >= 0:
            vat_id = _DYNAMIC_VAT_MAP.get(vatPercentage)
            if vat_id is None:
                return {"error": True, "message": f"Unsupported VAT rate {vatPercentage}%. Could not find matching VAT type in Tripletex."}
            body["vatType"] = {"id": vat_id}

        result = client.post("/product", json=body)

        # If product number already exists, fetch and return the existing product
        if result.get("error") and result.get("status_code") == 422 and productNumber:
            existing = client.get("/product", params={
                "number": productNumber,
                "fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency",
            })
            vals = existing.get("values", [])
            if vals:
                return {"value": vals[0], "_note": "Product number already existed, returning existing."}

        return result

    def search_products(name: str = "", number: str = "") -> dict:
        """Search for products by name or product number.

        Args:
            name: Filter by product name (partial match).
            number: Filter by product number/SKU (exact match).

        Returns:
            A list of matching products with id, name, number, price fields.
        """
        params = {"fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency"}
        if name:
            params["name"] = name
        if number:
            params["number"] = number
        return client.get("/product", params=params)

    def update_product(
        product_id: int,
        name: str = "",
        priceExcludingVatCurrency: float = 0.0,
        priceIncludingVatCurrency: float = 0.0,
        description: str = "",
        version: int = -1,
    ) -> dict:
        """Update an existing product.

        Args:
            product_id: The ID of the product to update.
            name: New name (empty to keep current).
            priceExcludingVatCurrency: New price excl VAT (0 to keep current).
            priceIncludingVatCurrency: New price incl VAT (0 to keep current).
            description: New description (empty to keep current).
            version: Entity version from the create response. If provided (>0), skips the GET call (saves 1 API call).

        Returns:
            The updated product or an error message.
        """
        _WRITABLE = {
            "id", "version", "name", "number", "description",
            "priceExcludingVatCurrency", "priceIncludingVatCurrency",
            "costExcludingVatCurrency", "isInactive", "isStockItem",
        }
        if version > 0 and name:
            # Fast path: skip GET when we have version + required field (name)
            body = {"id": product_id, "version": version, "name": name}
            if priceExcludingVatCurrency:
                body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
            elif priceIncludingVatCurrency:
                body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
            if description:
                body["description"] = description
        else:
            current = client.get(f"/product/{product_id}", params={"fields": "*"})
            full = current.get("value", {})
            body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
            if name:
                body["name"] = name
            if priceExcludingVatCurrency:
                body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
                body.pop("priceIncludingVatCurrency", None)
            if priceIncludingVatCurrency:
                body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
                body.pop("priceExcludingVatCurrency", None)
            if description:
                body["description"] = description
        return client.put(f"/product/{product_id}", json=body)

    def delete_product(product_id: int) -> dict:
        """Delete a product by ID.

        Args:
            product_id: The ID of the product to delete.

        Returns:
            Confirmation or error message.
        """
        return client.delete(f"/product/{product_id}")

    return {
        "create_product": create_product,
        "search_products": search_products,
        "update_product": update_product,
        "delete_product": delete_product,
    }
