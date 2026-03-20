import json as _json

from tripletex_client import TripletexClient


def build_salary_tools(client: TripletexClient) -> dict:
    """Build salary tools."""

    def search_salary_types() -> dict:
        """Search for salary types (loennsarter).

        Returns:
            A list of salary types.
        """
        return client.get("/salary/type", params={"fields": "id,number,name,description"})

    def create_salary_transaction(
        date: str,
        year: int,
        month: int,
        employee_ids: str = "",
        payslip_lines: str = "",
    ) -> dict:
        """Create a salary transaction (loennskjoering).

        Args:
            date: Transaction date YYYY-MM-DD.
            year: The year.
            month: The month (1-12).
            employee_ids: JSON string of employee IDs, e.g. '[1, 2, 3]'. If empty, tries to include all employees.
            payslip_lines: JSON string of salary specifications per employee.
                Format: '[{"employee_id": 123, "lines": [{"salary_type_number": 10, "rate": 58350, "count": 1}]}]'
                Each line needs: salary_type_number (from search_salary_types), rate (amount in NOK), count (usually 1).

        Returns:
            Created salary transaction or error.
        """
        payslips = []
        if payslip_lines:
            specs = _json.loads(payslip_lines) if isinstance(payslip_lines, str) else payslip_lines
            for spec in specs:
                eid = spec["employee_id"]
                lines = spec.get("lines", [])
                payslip = {"employee": {"id": eid}}
                if lines:
                    payslip["specifications"] = [
                        {
                            "salaryType": {"number": line["salary_type_number"]},
                            "rate": line["rate"],
                            "count": line.get("count", 1),
                        }
                        for line in lines
                    ]
                payslips.append(payslip)
        elif employee_ids:
            ids = _json.loads(employee_ids) if isinstance(employee_ids, str) else employee_ids
            payslips = [{"employee": {"id": eid}} for eid in ids]
        else:
            emps = client.get("/employee", params={"fields": "id"})
            ids = [e["id"] for e in emps.get("values", [])]
            payslips = [{"employee": {"id": eid}} for eid in ids]
        body = {
            "date": date,
            "year": year,
            "month": month,
            "payslips": payslips,
        }
        return client.post("/salary/transaction", json=body)

    def delete_salary_transaction(transaction_id: int) -> dict:
        """Delete a salary transaction.

        Args:
            transaction_id: ID of the transaction.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/salary/transaction/{transaction_id}")

    def search_salary_transactions(
        yearFrom: int = 0,
        yearTo: int = 0,
    ) -> dict:
        """Search for salary transactions.

        Args:
            yearFrom: Filter from year.
            yearTo: Filter to year.

        Returns:
            A list of salary transactions.
        """
        params = {"fields": "id,date,year,month"}
        if yearFrom:
            params["yearFrom"] = yearFrom
        if yearTo:
            params["yearTo"] = yearTo
        return client.get("/salary/transaction", params=params)

    def update_salary_settings(payrollTaxCalcMethod: str = "") -> dict:
        """Update salary settings.

        Args:
            payrollTaxCalcMethod: Payroll tax calculation method.

        Returns:
            Updated settings or error.
        """
        body = {}
        if payrollTaxCalcMethod:
            body["payrollTaxCalcMethod"] = payrollTaxCalcMethod
        return client.put("/salary/settings", json=body)

    return {
        "search_salary_types": search_salary_types,
        "create_salary_transaction": create_salary_transaction,
        "delete_salary_transaction": delete_salary_transaction,
        "search_salary_transactions": search_salary_transactions,
        "update_salary_settings": update_salary_settings,
    }
