from tripletex_client import TripletexClient


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

    def create_product(
        name: str,
        priceExcludingVat: float = 0.0,
        priceIncludingVat: float = 0.0,
        productNumber: str = "",
    ) -> dict:
        """Create a new product in Tripletex.

        Args:
            name: The product name.
            priceExcludingVat: Price excluding VAT.
            priceIncludingVat: Price including VAT.
            productNumber: Optional product number/SKU.

        Returns:
            The created product with id and fields, or an error message.
        """
        body = {"name": name}
        if priceExcludingVat:
            body["priceExcludingVat"] = priceExcludingVat
        if priceIncludingVat:
            body["priceIncludingVat"] = priceIncludingVat
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
        params = {"fields": "id,name,number,priceExcludingVat,priceIncludingVat"}
        if name:
            params["name"] = name
        return client.get("/product", params=params)

    return {
        "create_product": create_product,
        "search_products": search_products,
    }
