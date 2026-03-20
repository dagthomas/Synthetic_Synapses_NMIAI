from tripletex_client import TripletexClient


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

    # Output VAT type mapping: percentage → Tripletex vatType ID
    # 3 = Utgående mva høy sats (25%), 31 = middels (15%), 33 = lav (12%), 6 = utenfor mva-loven (0%)
    _OUTPUT_VAT_MAP = {25: 3, 15: 31, 12: 33, 0: 6}

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
        body = {"name": name}
        # Only send ONE price — sending both causes validation errors when they don't match the product's VAT type
        if priceExcludingVatCurrency:
            body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
        elif priceIncludingVatCurrency:
            body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
        if productNumber:
            body["number"] = productNumber
        if vatPercentage >= 0:
            vat_id = _OUTPUT_VAT_MAP.get(vatPercentage)
            if vat_id is None:
                return {"error": True, "message": f"Unsupported VAT rate {vatPercentage}%. Use 25, 15, 12, or 0."}
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
