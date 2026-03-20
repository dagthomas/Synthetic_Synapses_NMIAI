from tripletex_client import TripletexClient


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

    def create_product(
        name: str,
        priceExcludingVatCurrency: float = 0.0,
        priceIncludingVatCurrency: float = 0.0,
        productNumber: str = "",
    ) -> dict:
        """Create a new product in Tripletex.

        Args:
            name: The product name.
            priceExcludingVatCurrency: Price excluding VAT in NOK.
            priceIncludingVatCurrency: Price including VAT in NOK.
            productNumber: Optional product number/SKU.

        Returns:
            The created product with id and fields, or an error message.
        """
        body = {"name": name}
        if priceExcludingVatCurrency:
            body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
        if priceIncludingVatCurrency:
            body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
        if productNumber:
            body["number"] = productNumber
        return client.post("/product", json=body)

    def search_products(name: str = "") -> dict:
        """Search for products by name.

        Args:
            name: Filter by product name (partial match).

        Returns:
            A list of matching products with id, name, number, price fields.
        """
        params = {"fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency"}
        if name:
            params["name"] = name
        return client.get("/product", params=params)

    def update_product(
        product_id: int,
        name: str = "",
        priceExcludingVatCurrency: float = 0.0,
        priceIncludingVatCurrency: float = 0.0,
        description: str = "",
    ) -> dict:
        """Update an existing product.

        Args:
            product_id: The ID of the product to update.
            name: New name (empty to keep current).
            priceExcludingVatCurrency: New price excl VAT (0 to keep current).
            priceIncludingVatCurrency: New price incl VAT (0 to keep current).
            description: New description (empty to keep current).

        Returns:
            The updated product or an error message.
        """
        _WRITABLE = {
            "id", "version", "name", "number", "description",
            "priceExcludingVatCurrency", "priceIncludingVatCurrency",
            "costExcludingVatCurrency", "isInactive", "isStockItem",
        }
        current = client.get(f"/product/{product_id}", params={"fields": "*"})
        full = current.get("value", {})
        body = {k: v for k, v in full.items() if k in _WRITABLE and v is not None} if full else {}
        if name:
            body["name"] = name
        if priceExcludingVatCurrency:
            body["priceExcludingVatCurrency"] = priceExcludingVatCurrency
        if priceIncludingVatCurrency:
            body["priceIncludingVatCurrency"] = priceIncludingVatCurrency
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
