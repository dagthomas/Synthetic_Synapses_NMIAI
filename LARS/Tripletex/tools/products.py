from tripletex_client import TripletexClient
from tools._helpers import recover_error, resolve_vat_map, find_or_create_product


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

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
        price = priceExcludingVatCurrency or priceIncludingVatCurrency
        vat_pct = vatPercentage if vatPercentage >= 0 else 25

        # Build product body
        body = {"name": name}
        if price:
            body["priceExcludingVatCurrency"] = price
        if productNumber:
            body["number"] = productNumber
        # Only resolve VAT types for non-standard rates (15, 12, 0)
        # Tripletex uses 25% by default, so skip the GET for standard rate
        if vat_pct != 25 and not client.get_cached("skip_vat_type"):
            vat_map = resolve_vat_map(client)
            vat_type_id = vat_map.get(int(vat_pct))
            if vat_type_id is not None:
                body["vatType"] = {"id": vat_type_id}

        result = client.post("/product", json=body)
        prod_id = result.get("value", {}).get("id")

        # Handle collisions: search existing and UPDATE with correct values
        if not prod_id and result.get("error"):
            err_msg = str(result.get("message", "")).lower()
            # VAT type rejected — retry without
            if "vatType" in body and ("mva-kode" in err_msg or "vattype" in err_msg):
                body.pop("vatType", None)
                client.set_cached("skip_vat_type", True)
                recover_error(client, "/product")
                result = client.post("/product", json=body)
                prod_id = result.get("value", {}).get("id")

            if not prod_id:
                recover_error(client, "/product")
                # Search by name and update existing with correct values
                existing = client.get("/product", params={"name": name, "fields": "id,name,version"})
                for v in (existing.get("values") or []):
                    if (v.get("name") or "").strip().lower() == name.strip().lower():
                        prod_id = v["id"]
                        upd = {"id": prod_id, "version": v.get("version", 0), "name": name}
                        if price:
                            upd["priceExcludingVatCurrency"] = price
                        if productNumber:
                            upd["number"] = productNumber
                        client.put(f"/product/{prod_id}", json=upd)
                        break

        if prod_id:
            # Return from POST/search response when possible (saves 1 GET)
            val = result.get("value") if isinstance(result, dict) else None
            if val and isinstance(val, dict) and val.get("id") == prod_id:
                return {"value": val}
            return client.get(f"/product/{prod_id}", params={
                "fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency,version",
            })
        return {"error": True, "message": f"Failed to create product '{name}'"}

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
            params["productNumber"] = number
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
