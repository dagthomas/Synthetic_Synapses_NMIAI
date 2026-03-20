import json as _json

from tripletex_client import TripletexClient


def build_bank_tools(client: TripletexClient) -> dict:
    """Build bank and bank reconciliation tools."""

    def search_bank_accounts() -> dict:
        """Search for bank accounts registered in the company.

        Returns:
            A list of bank accounts with id, register number, and reconciliation support.
        """
        return client.get("/bank", params={"fields": "id,registerNumbers,name,displayName"})

    def search_bank_reconciliations(accountId: int = 0) -> dict:
        """Search for bank reconciliations, optionally filtered by account.

        Args:
            accountId: Filter by bank account ID (0 for all).

        Returns:
            A list of bank reconciliations.
        """
        params = {"fields": "*"}
        if accountId:
            params["accountId"] = accountId
        return client.get("/bank/reconciliation", params=params)

    def get_last_bank_reconciliation(accountId: int) -> dict:
        """Get the most recently created bank reconciliation for an account.

        Args:
            accountId: The ledger account ID for the bank account.

        Returns:
            The last bank reconciliation or an error message.
        """
        return client.get("/bank/reconciliation/>last", params={"accountId": accountId, "fields": "*"})

    def get_last_closed_bank_reconciliation(accountId: int) -> dict:
        """Get the last closed bank reconciliation for an account.

        Args:
            accountId: The ledger account ID for the bank account.

        Returns:
            The last closed bank reconciliation or an error message.
        """
        return client.get("/bank/reconciliation/>lastClosed", params={"accountId": accountId, "fields": "*"})

    def create_bank_reconciliation(accountId: int, accountingPeriodId: int) -> dict:
        """Create a new bank reconciliation for an account and period.

        Args:
            accountId: The ledger account ID for the bank account.
            accountingPeriodId: The accounting period ID for the reconciliation.

        Returns:
            The created bank reconciliation with id, or an error message.
        """
        body = {
            "account": {"id": accountId},
            "accountingPeriod": {"id": accountingPeriodId},
        }
        return client.post("/bank/reconciliation", json=body)

    def adjust_bank_reconciliation(reconciliationId: int, adjustments: str) -> dict:
        """Apply adjustments to a bank reconciliation.

        Args:
            reconciliationId: The ID of the bank reconciliation.
            adjustments: JSON string array of adjustments, each with 'amount', 'date', 'description', and optionally 'accountId'. Example: '[{"amount": 100.0, "date": "2026-01-15", "description": "Missing deposit"}]'

        Returns:
            The created adjustments or an error message.
        """
        items = _json.loads(adjustments) if isinstance(adjustments, str) else adjustments
        return client.put(f"/bank/reconciliation/{reconciliationId}/:adjustment", json=items)

    def close_bank_reconciliation(reconciliationId: int) -> dict:
        """Close/finalize a bank reconciliation by updating it.

        Args:
            reconciliationId: The ID of the bank reconciliation to close.

        Returns:
            The updated bank reconciliation or an error message.
        """
        return client.put(f"/bank/reconciliation/{reconciliationId}", json={"isClosed": True})

    def delete_bank_reconciliation(reconciliationId: int) -> dict:
        """Delete a bank reconciliation.

        Args:
            reconciliationId: The ID of the bank reconciliation to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/bank/reconciliation/{reconciliationId}")

    def get_bank_reconciliation_match_count(bankReconciliationId: int) -> dict:
        """Get the number of matched transactions for a bank reconciliation.

        Args:
            bankReconciliationId: The bank reconciliation ID.

        Returns:
            The count of matches.
        """
        return client.get("/bank/reconciliation/match/count", params={"bankReconciliationId": bankReconciliationId})

    def search_bank_statements(bankAccountId: int = 0) -> dict:
        """Search for bank statements.

        Args:
            bankAccountId: Filter by bank account ID (0 for all).

        Returns:
            A list of bank statements.
        """
        params = {"fields": "*"}
        if bankAccountId:
            params["bankAccountId"] = bankAccountId
        return client.get("/bank/statement", params=params)

    return {
        "search_bank_accounts": search_bank_accounts,
        "search_bank_reconciliations": search_bank_reconciliations,
        "get_last_bank_reconciliation": get_last_bank_reconciliation,
        "get_last_closed_bank_reconciliation": get_last_closed_bank_reconciliation,
        "create_bank_reconciliation": create_bank_reconciliation,
        "adjust_bank_reconciliation": adjust_bank_reconciliation,
        "close_bank_reconciliation": close_bank_reconciliation,
        "delete_bank_reconciliation": delete_bank_reconciliation,
        "get_bank_reconciliation_match_count": get_bank_reconciliation_match_count,
        "search_bank_statements": search_bank_statements,
    }
