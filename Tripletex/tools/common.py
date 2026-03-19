from tripletex_client import TripletexClient


def build_common_tools(client: TripletexClient) -> dict:
    """Build common utility tools."""

    def get_entity_by_id(entity_type: str, entity_id: int) -> dict:
        """Retrieve any Tripletex entity by type and ID.

        Args:
            entity_type: The API entity type, e.g. 'employee', 'customer', 'invoice'.
            entity_id: The numeric ID of the entity.

        Returns:
            The entity data or an error message.
        """
        return client.get(f"/{entity_type}/{entity_id}", params={"fields": "*"})

    def delete_entity(entity_type: str, entity_id: int) -> dict:
        """Delete a Tripletex entity by type and ID.

        Args:
            entity_type: The API entity type, e.g. 'travelExpense', 'ledger/voucher'.
            entity_id: The numeric ID of the entity to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/{entity_type}/{entity_id}")

    return {
        "get_entity_by_id": get_entity_by_id,
        "delete_entity": delete_entity,
    }
