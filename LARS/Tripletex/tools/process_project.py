"""Compound tool: process entire project_lifecycle workflow in one deterministic call.

Handles: find/create customer → create PM → create project → employee hours →
         supplier cost → customer invoice.
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_project_tools(client: TripletexClient) -> dict:
    """Build the compound project lifecycle tool."""

    from tools._helpers import (
        recover_error, find_or_create_customer, find_or_create_supplier,
        find_or_create_employment, find_or_create_product, ensure_bank_account,
    )
    from tools.projects import build_project_tools as _build_project_tools

    def _find_or_create_employee(firstName, lastName, email, userType="STANDARD", steps_log=None):
        """Create an employee or find existing by email on collision.

        Returns (emp_id, just_created) tuple.
        """
        if steps_log is None:
            steps_log = []

        # Search by email first to avoid 422
        if email:
            existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName"})
            vals = existing.get("values", [])
            if vals:
                steps_log.append(f"Found existing employee {firstName} {lastName} (id={vals[0]['id']})")
                return vals[0]["id"], False

        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "userType": userType,
            "dateOfBirth": "1990-01-01",
        }
        result = client.post("/employee", json=body)
        emp_id = result.get("value", {}).get("id")

        if not emp_id and result.get("error") and result.get("status_code") == 422:
            msg = str(result.get("message", "")).lower()
            if "e-postadress" in msg or "email" in msg:
                recover_error(client, "/employee")
                existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName"})
                vals = existing.get("values", [])
                if vals:
                    emp_id = vals[0]["id"]
                    steps_log.append(f"Employee {firstName} {lastName} already existed (id={emp_id})")
                    return emp_id, False

        if emp_id:
            steps_log.append(f"Created employee {firstName} {lastName} (id={emp_id})")
            return emp_id, True

        return None, False

    def execute_project_lifecycle(
        project_name: str,
        customer_name: str,
        customer_org_number: str = "",
        pm_firstName: str = "",
        pm_lastName: str = "",
        pm_email: str = "",
        budget: float = 0.0,
        employees: str = "[]",
        supplier_name: str = "",
        supplier_org_number: str = "",
        supplier_cost: float = 0.0,
        supplier_expense_account: int = 4000,
        supplier_vat_percentage: int = 25,
        create_customer_invoice: bool = False,
        invoice_vat_percentage: int = 25,
    ) -> dict:
        """Execute a complete project lifecycle in one call.

        This compound tool handles the ENTIRE project lifecycle workflow:
        1. Creates the customer
        2. Creates the PM employee (with EXTENDED access)
        3. Creates the project with budget and PM
        4. For each employee: creates employee, employment, project participant, and timesheet entry
        5. Optionally registers supplier cost as incoming invoice
        6. Optionally creates customer invoice for the project budget

        Args:
            project_name: Name of the project.
            customer_name: Name of the customer company.
            customer_org_number: Customer organization number (optional).
            pm_firstName: Project manager first name.
            pm_lastName: Project manager last name.
            pm_email: Project manager email address.
            budget: Project budget / fixed price amount (including VAT).
            employees: JSON string of employee list: [{"firstName":"...","lastName":"...","email":"...","hours":28}]
            supplier_name: Supplier company name (empty to skip supplier cost).
            supplier_org_number: Supplier organization number (optional).
            supplier_cost: Supplier cost amount including VAT (0 to skip).
            supplier_expense_account: Expense account number for supplier invoice (default 4000).
            supplier_vat_percentage: VAT percentage for supplier invoice (default 25).
            create_customer_invoice: Whether to create a customer invoice for the project budget.
            invoice_vat_percentage: VAT percentage for customer invoice (default 25).

        Returns:
            Summary with project_id, customer_id, pm_id, employee_ids, supplier_id, invoice_id.
        """
        today = dt_date.today().isoformat()
        steps_log = []
        employee_ids = []
        supplier_id = None
        invoice_id = None

        # Parse employees JSON
        try:
            emp_list = _json.loads(employees) if isinstance(employees, str) else employees
        except (_json.JSONDecodeError, TypeError):
            emp_list = []

        # ── Step 1: Find or create customer ──
        customer_id = find_or_create_customer(
            client, name=customer_name, org_number=customer_org_number,
            steps_log=steps_log,
        )
        if not customer_id:
            return {"error": True, "message": f"Failed to create customer '{customer_name}'", "steps": steps_log}

        # ── Step 2: Create PM employee (EXTENDED) ──
        pm_id = 0
        if pm_email:
            pm_id, _pm_created = _find_or_create_employee(pm_firstName, pm_lastName, pm_email,
                                              userType="EXTENDED", steps_log=steps_log)
            if not pm_id:
                return {"error": True, "message": "Failed to create PM employee", "steps": steps_log}

        # ── Step 3: Create project ──
        proj_tools = _build_project_tools(client)
        proj_result = proj_tools["create_project"](
            name=project_name,
            customer_id=customer_id,
            projectManagerId=pm_id,
            startDate=today,
            fixedPriceAmount=budget,
        )
        project_id = proj_result.get("value", {}).get("id")
        if not project_id:
            return {"error": True, "message": f"Failed to create project: {proj_result}", "steps": steps_log}
        steps_log.append(f"Created project '{project_name}' (id={project_id})")

        # ── Step 4: Employee hours ──
        for emp_spec in emp_list:
            emp_email = emp_spec.get("email", "")
            emp_firstName = emp_spec.get("firstName", "")
            emp_lastName = emp_spec.get("lastName", "")
            emp_hours = emp_spec.get("hours", 0)

            # Reuse PM if same email
            if emp_email and emp_email == pm_email and pm_id:
                emp_id = pm_id
                find_or_create_employment(client, employee_id=emp_id, start_date=today,
                                          steps_log=steps_log, skip_dob_check=True)
            else:
                emp_id, emp_created = _find_or_create_employee(emp_firstName, emp_lastName, emp_email,
                                                   steps_log=steps_log)
                if not emp_id:
                    steps_log.append(f"WARNING: Skipping employee {emp_firstName} {emp_lastName} — creation failed")
                    continue

                find_or_create_employment(client, employee_id=emp_id, start_date=today,
                                          steps_log=steps_log, skip_dob_check=emp_created)

                try:
                    client.post("/project/participant", json={
                        "project": {"id": project_id},
                        "employee": {"id": emp_id},
                    })
                    steps_log.append(f"Added employee {emp_id} as project participant")
                except Exception:
                    pass

            employee_ids.append(emp_id)

            # Register timesheet hours
            if emp_hours > 0:
                activity_id = 0
                act_result = client.get("/activity", params={"projectId": project_id, "fields": "id", "count": 1})
                act_vals = act_result.get("values", [])
                if act_vals:
                    activity_id = act_vals[0]["id"]
                else:
                    act_result2 = client.get("/activity", params={"fields": "id", "count": 1})
                    act_vals2 = act_result2.get("values", [])
                    if act_vals2:
                        activity_id = act_vals2[0]["id"]

                ts_body = {
                    "employee": {"id": emp_id},
                    "project": {"id": project_id},
                    "date": today,
                    "hours": emp_hours,
                }
                if activity_id:
                    ts_body["activity"] = {"id": activity_id}
                ts_result = client.post("/timesheet/entry", json=ts_body)
                ts_id = ts_result.get("value", {}).get("id")
                if ts_id:
                    steps_log.append(f"Registered {emp_hours}h for employee {emp_id} (timesheet={ts_id})")
                else:
                    steps_log.append(f"WARNING: Timesheet entry failed for employee {emp_id}: {ts_result}")

        # ── Step 5: Supplier cost ──
        if supplier_name and supplier_cost > 0:
            supplier_id = find_or_create_supplier(
                client, name=supplier_name, org_number=supplier_org_number,
                steps_log=steps_log,
            )
            if supplier_id:
                from tools.incoming_invoice import build_incoming_invoice_tools
                inv_tools = build_incoming_invoice_tools(client)
                inv_result = inv_tools["create_incoming_invoice"](
                    supplierId=supplier_id,
                    invoiceNumber=f"PROJECT_COST_{today}",
                    amountIncludingVat=supplier_cost,
                    expenseAccountNumber=supplier_expense_account,
                    vatPercentage=supplier_vat_percentage,
                    invoiceDate=today,
                    projectId=project_id,
                )
                inv_id = inv_result.get("value", {}).get("id") if isinstance(inv_result, dict) else None
                if inv_id:
                    steps_log.append(f"Created incoming invoice for supplier cost (id={inv_id})")
                else:
                    steps_log.append(f"WARNING: Incoming invoice failed: {inv_result}")
            else:
                steps_log.append(f"WARNING: Failed to create supplier '{supplier_name}'")

        # ── Step 6: Customer invoice ──
        if create_customer_invoice and budget > 0:
            prod_id = find_or_create_product(
                client, name=f"Project Services - {project_name}",
                price=budget, vat_percentage=invoice_vat_percentage,
                steps_log=steps_log,
            )
            if prod_id:
                order_body = {
                    "customer": {"id": customer_id},
                    "orderDate": today,
                    "deliveryDate": today,
                    "orderLines": [{"product": {"id": prod_id}, "count": 1}],
                    "project": {"id": project_id},
                }
                order_result = client.post("/order", json=order_body)
                order_id = order_result.get("value", {}).get("id")

                if not order_id and order_result.get("error"):
                    order_body.pop("project", None)
                    recover_error(client, "/order")
                    order_result = client.post("/order", json=order_body)
                    order_id = order_result.get("value", {}).get("id")

                if order_id:
                    steps_log.append(f"Created order (id={order_id})")

                    ensure_bank_account(client)
                    inv_result = client.post("/invoice", json={
                        "invoiceDate": today,
                        "invoiceDueDate": today,
                        "orders": [{"id": order_id}],
                    })
                    invoice_id = inv_result.get("value", {}).get("id")
                    if invoice_id:
                        steps_log.append(f"Created invoice (id={invoice_id})")
                    else:
                        steps_log.append(f"WARNING: Invoice creation failed: {inv_result}")
                else:
                    steps_log.append(f"WARNING: Order creation failed: {order_result}")
            else:
                steps_log.append("WARNING: Product creation failed")

        return {
            "success": True,
            "project_id": project_id,
            "customer_id": customer_id,
            "pm_id": pm_id,
            "employee_ids": employee_ids,
            "supplier_id": supplier_id,
            "invoice_id": invoice_id,
            "steps": steps_log,
        }

    def create_project_with_billing(
        project_name: str,
        customer_name: str,
        customer_org_number: str = "",
        pm_firstName: str = "",
        pm_lastName: str = "",
        pm_email: str = "",
        budget: float = 0.0,
        employees: str = "[]",
        supplier_name: str = "",
        supplier_org_number: str = "",
        supplier_cost: float = 0.0,
        supplier_expense_account: int = 4000,
        supplier_vat_percentage: int = 25,
        create_customer_invoice: bool = False,
        invoice_vat_percentage: int = 25,
    ) -> dict:
        """Create a project with full billing cycle in one call.

        Handles: customer → PM employee → project with budget → employee hours →
        supplier cost (incoming invoice) → customer invoice.

        Args:
            project_name: Name of the project.
            customer_name: Name of the customer company.
            customer_org_number: Customer organization number (optional).
            pm_firstName: Project manager first name.
            pm_lastName: Project manager last name.
            pm_email: Project manager email address.
            budget: Project budget / fixed price amount (including VAT).
            employees: JSON string of employee list: [{"firstName":"...","lastName":"...","email":"...","hours":28}]
            supplier_name: Supplier company name (empty to skip supplier cost).
            supplier_org_number: Supplier organization number (optional).
            supplier_cost: Supplier cost amount including VAT (0 to skip).
            supplier_expense_account: Expense account number for supplier invoice (default 4000).
            supplier_vat_percentage: VAT percentage for supplier invoice (default 25).
            create_customer_invoice: Whether to create a customer invoice for the project budget.
            invoice_vat_percentage: VAT percentage for customer invoice (default 25).

        Returns:
            Summary with project_id, customer_id, pm_id, employee_ids, supplier_id, invoice_id.
        """
        return execute_project_lifecycle(
            project_name=project_name,
            customer_name=customer_name,
            customer_org_number=customer_org_number,
            pm_firstName=pm_firstName,
            pm_lastName=pm_lastName,
            pm_email=pm_email,
            budget=budget,
            employees=employees,
            supplier_name=supplier_name,
            supplier_org_number=supplier_org_number,
            supplier_cost=supplier_cost,
            supplier_expense_account=supplier_expense_account,
            supplier_vat_percentage=supplier_vat_percentage,
            create_customer_invoice=create_customer_invoice,
            invoice_vat_percentage=invoice_vat_percentage,
        )

    return {
        "execute_project_lifecycle": execute_project_lifecycle,
        "create_project_with_billing": create_project_with_billing,
    }
