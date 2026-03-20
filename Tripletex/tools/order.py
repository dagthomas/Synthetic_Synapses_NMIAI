from tripletex_client import TripletexClient


def build_order_tools(client: TripletexClient) -> dict:
    """Build order line and order group tools."""

    def create_order_line(
        order_id: int,
        product_id: int = 0,
        description: str = "",
        count: float = 1,
        unitPriceExcludingVatCurrency: float = 0.0,
    ) -> dict:
        """Create an order line on an existing order.

        Args:
            order_id: The order to add the line to.
            product_id: Product ID (0 if using description only).
            description: Line description.
            count: Quantity.
            unitPriceExcludingVatCurrency: Unit price excl VAT.

        Returns:
            Created order line or error.
        """
        body = {"order": {"id": order_id}, "count": count}
        if product_id:
            body["product"] = {"id": product_id}
        if description:
            body["description"] = description
        if unitPriceExcludingVatCurrency:
            body["unitPriceExcludingVatCurrency"] = unitPriceExcludingVatCurrency
        return client.post("/order/orderline", json=body)

    def delete_order_line(order_line_id: int) -> dict:
        """Delete an order line.

        Args:
            order_line_id: ID of the order line.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/order/orderline/{order_line_id}")

    def create_order_group(
        order_id: int,
        title: str,
        comment: str = "",
    ) -> dict:
        """Create an order group on an existing order.

        Args:
            order_id: The order to add the group to.
            title: Group title.
            comment: Group comment.

        Returns:
            Created order group or error.
        """
        body = {"order": {"id": order_id}, "title": title}
        if comment:
            body["comment"] = comment
        return client.post("/order/orderGroup", json=body)

    return {
        "create_order_line": create_order_line,
        "delete_order_line": delete_order_line,
        "create_order_group": create_order_group,
    }
