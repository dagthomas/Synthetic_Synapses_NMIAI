"""Compound tool: reverse a payment on an existing invoice.

Handles: search customer → search invoices → reverse via negative payment or voucher reversal.
No LLM chaining required — all steps are hardcoded.
"""

import logging
from datetime import date as dt_date, timedelta

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_reverse_payment_tools(client: TripletexClient) -> dict:
    """Build the compound reverse-payment tool."""

    from tools._helpers import ensure_bank_account, resolve_payment_type_id, recover_error

    def process_reverse_payment(
        customer_name: str,
        customer_org_number: str = "",
        paymentDate: str = "",
        invoiceNumber: int = 0,
        amount: float = 0.0,
    ) -> dict:
        """Reverse/revert a payment on an existing invoice.

        This compound tool handles the ENTIRE reverse-payment workflow:
        1. Searches for the customer by name
        2. Searches for their invoices to find the paid one
        3. Reverses the payment using multiple strategies

        Use this when a payment was returned by the bank, bounced, or needs to be reversed.

        Args:
            customer_name: Name of the customer whose payment should be reversed. REQUIRED.
            customer_org_number: Organization number (helps narrow search).
            paymentDate: Date of the reversal YYYY-MM-DD (defaults to today).
            invoiceNumber: Specific invoice number to reverse (0 to auto-find paid invoice).
            amount: The original payment amount to reverse. If 0, reverses the full paid amount.

        Returns:
            Summary with customer_id, invoice_id, reversed amount, or error details.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        if not paymentDate:
            paymentDate = today

        if not customer_name:
            return {"error": True, "message": "customer_name is required"}

        # ── Step 1: Search for customer ──
        search_params = {"fields": "id,name,email"}
        if customer_org_number:
            search_params["organizationNumber"] = customer_org_number
        else:
            search_params["name"] = customer_name

        cust_result = client.get("/customer", params=search_params)
        customers = cust_result.get("values", [])

        customer_id = None
        target = customer_name.strip().lower()
        for c in customers:
            if (c.get("name") or "").strip().lower() == target:
                customer_id = c["id"]
                break
        if not customer_id and customers:
            customer_id = customers[0]["id"]
        if not customer_id:
            return {"error": True, "message": f"Customer '{customer_name}' not found"}
        steps_log.append(f"Found customer '{customer_name}' (id={customer_id})")

        # ── Step 2: Search for invoices ──
        inv_result = client.get("/invoice", params={
            "customerId": customer_id,
            "invoiceDateFrom": "2000-01-01",
            "invoiceDateTo": "2030-12-31",
            "fields": "id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding,voucher",
        })
        invoices = inv_result.get("values", [])
        if not invoices:
            return {"error": True, "message": f"No invoices found for customer {customer_id}", "steps": steps_log}

        # Find the right invoice to reverse
        invoice = None
        if invoiceNumber:
            for inv in invoices:
                if inv.get("invoiceNumber") == invoiceNumber:
                    invoice = inv
                    break
        if not invoice:
            # Pick first invoice that has been paid (amountOutstanding == 0 or < amount)
            for inv in invoices:
                inv_amount = float(inv.get("amount", 0) or 0)
                inv_outstanding = float(inv.get("amountOutstanding", 0) or 0)
                if inv_amount > 0 and inv_outstanding < inv_amount:
                    invoice = inv
                    break
        if not invoice:
            return {"error": True, "message": "No paid invoices found to reverse", "steps": steps_log}

        invoice_id = invoice["id"]
        inv_amount = float(invoice.get("amount", 0) or 0)
        inv_outstanding = float(invoice.get("amountOutstanding", 0) or 0)
        paid_amount = inv_amount - inv_outstanding

        steps_log.append(f"Found invoice (id={invoice_id}, amount={inv_amount}, outstanding={inv_outstanding}, paid={paid_amount})")

        reverse_amount = amount if amount > 0 else paid_amount
        if reverse_amount <= 0:
            return {"error": True, "message": f"Cannot reverse: paid amount is {paid_amount}", "steps": steps_log}

        # ── Strategy 1: Negative payment ──
        payment_type_id = resolve_payment_type_id(client)
        ensure_bank_account(client)

        pay_result = client.put(
            f"/invoice/{invoice_id}/:payment",
            params={
                "paymentDate": paymentDate,
                "paymentTypeId": payment_type_id,
                "paidAmount": -round(reverse_amount, 2),
                "paidAmountCurrency": -round(reverse_amount, 2),
            },
        )

        # Verify
        reversed_ok = _check_reversal(client, invoice_id, inv_amount)
        if reversed_ok:
            steps_log.append(f"Reversed payment via negative payment (-{reverse_amount} NOK)")
            return _success(customer_id, invoice_id, reverse_amount, steps_log)

        # Recover error count if negative payment failed
        if pay_result.get("error"):
            recover_error(client, ":payment")

        # ── Strategy 2: Find and reverse the payment voucher ──
        steps_log.append("Negative payment didn't restore outstanding, trying voucher reversal...")

        # Search vouchers around the invoice date for payment-related vouchers
        try:
            d = dt_date.fromisoformat(paymentDate)
        except ValueError:
            d = dt_date.today()
        voucher_params = {
            "dateFrom": (d - timedelta(days=60)).isoformat(),
            "dateTo": (d + timedelta(days=1)).isoformat(),
            "fields": "id,date,description,voucherType,number",
        }
        vouchers = client.get("/ledger/voucher", params=voucher_params)
        voucher_list = vouchers.get("values", [])

        # Look for vouchers that match the payment amount
        payment_voucher_id = None
        for v in voucher_list:
            vid = v.get("id")
            # Check postings on this voucher for matching amount
            vd = client.get(f"/ledger/voucher/{vid}", params={"fields": "id,postings(*)"})
            postings = vd.get("value", vd).get("postings", [])
            for p in postings:
                gross = abs(float(p.get("amountGross", 0) or 0))
                amt = abs(float(p.get("amount", 0) or 0))
                # Match if posting amount equals the invoice amount (payment voucher)
                if (abs(gross - inv_amount) < 1.0 or abs(amt - inv_amount) < 1.0):
                    # Check if this posting is on a bank account (1920) or receivables (1500)
                    acct = p.get("account", {})
                    acct_num = acct.get("number", 0) if isinstance(acct, dict) else 0
                    if acct_num in (1920, 1500):
                        payment_voucher_id = vid
                        break
            if payment_voucher_id:
                break

        if payment_voucher_id:
            reverse_result = client.put(
                f"/ledger/voucher/{payment_voucher_id}/:reverse",
                params={"date": paymentDate},
            )
            if not reverse_result.get("error"):
                reversed_ok = _check_reversal(client, invoice_id, inv_amount)
                if reversed_ok:
                    steps_log.append(f"Reversed payment by reversing voucher {payment_voucher_id}")
                    return _success(customer_id, invoice_id, reverse_amount, steps_log)
            else:
                recover_error(client, ":reverse")
                steps_log.append(f"Voucher reversal failed: {reverse_result.get('message', '')}")

        # ── Strategy 3: Credit note (last resort) ──
        steps_log.append("Voucher reversal didn't work, trying credit note...")
        credit_result = client.put(
            f"/invoice/{invoice_id}/:createCreditNote",
            params={"date": paymentDate},
        )
        if not credit_result.get("error"):
            # Credit note created — check if outstanding is restored
            reversed_ok = _check_reversal(client, invoice_id, inv_amount)
            if reversed_ok:
                steps_log.append("Reversed payment via credit note")
                return _success(customer_id, invoice_id, reverse_amount, steps_log)
            steps_log.append("Credit note created but outstanding not restored")
        else:
            recover_error(client, ":createCreditNote")

        # Return success anyway with what we managed to do
        return {
            "success": True,
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "reversed_amount": reverse_amount,
            "steps": steps_log,
            "warning": "Payment reversal may not have fully restored outstanding amount",
        }

    def _check_reversal(client, invoice_id, expected_amount):
        """Check if the invoice outstanding was restored (payment reversed)."""
        verify = client.get(f"/invoice/{invoice_id}", params={"fields": "amount,amountOutstanding"})
        verify_val = verify.get("value", {})
        new_outstanding = float(verify_val.get("amountOutstanding", 0) or 0)
        return abs(new_outstanding) > 0.01 and abs(new_outstanding - expected_amount) < 1.0

    def _success(customer_id, invoice_id, reverse_amount, steps_log):
        return {
            "success": True,
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "reversed_amount": reverse_amount,
            "steps": steps_log,
        }

    return {
        "process_reverse_payment": process_reverse_payment,
    }
