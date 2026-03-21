"""Compound tool: process entire project_lifecycle workflow in one deterministic call.

Handles: create customer → create PM → create project → employee hours →
         supplier cost → customer invoice.
No LLM chaining required — all steps are hardcoded.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_project_tools(client: TripletexClient) -> dict:
    """Build the compound project lifecycle tool."""

    # ── Import helper from projects module ──
    from tools.projects import build_project_tools as _build_project_tools

    def _recover_error(endpoint: str):
        """Undo error count for auto-recovery on expected 422s."""
        client._error_count = max(0, client._error_count - 1)
        for entry in reversed(client._call_log):
            if not entry.get("ok") and endpoint in entry.get("url", ""):
                entry["ok"] = True
                entry["recovered"] = True
                break

    def _find_or_create_employee(firstName, lastName, email, userType="STANDARD", steps_log=None):
        """Create an employee or find existing by email on collision.

        Returns (emp_id, just_created) tuple.
        """
        if steps_log is None:
            steps_log = []

        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "userType": userType,
            "dateOfBirth": "1990-01-01",
        }
        result = client.post("/employee", json=body)
        emp_id = None
        just_created = False

        if result.get("error") and result.get("status_code") == 422:
            msg = str(result.get("message", "")).lower()
            if "e-postadress" in msg or "email" in msg:
                existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName"})
                vals = existing.get("values", [])
                if vals:
                    emp_id = vals[0]["id"]
                    _recover_error("/employee")
                    steps_log.append(f"Employee {firstName} {lastName} already existed (id={emp_id})")
        if not emp_id:
            emp_val = result.get("value", {})
            emp_id = emp_val.get("id")
            if emp_id:
                just_created = True
                steps_log.append(f"Created employee {firstName} {lastName} (id={emp_id})")

        return emp_id, just_created

    def _find_or_create_employment(employee_id, startDate, steps_log=None, skip_dob_check=False):
        """Create employment or find existing on collision."""
        if steps_log is None:
            steps_log = []

        # Ensure dateOfBirth is set (required for employment)
        # Skip if employee was just created with dateOfBirth
        if not skip_dob_check:
            emp_check = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
            emp_data = emp_check.get("value", emp_check)
            if not emp_data.get("dateOfBirth"):
                client.put(f"/employee/{employee_id}", json={"dateOfBirth": "1990-01-01"})

        # Get or create division
        div_id = client.get_cached("default_division")
        if div_id is None:
            divs = client.get("/division", params={"fields": "id", "count": 1})
            div_list = divs.get("values", [])
            if div_list:
                div_id = div_list[0]["id"]
            client.set_cached("default_division", div_id or 0)

        emp_body = {
            "employee": {"id": employee_id},
            "startDate": startDate,
            "employmentDetails": [{"date": startDate, "employmentType": "ORDINARY",
                                    "workingHoursScheme": "NOT_SHIFT"}],
        }
        if div_id:
            emp_body["division"] = {"id": div_id}

        result = client.post("/employee/employment", json=emp_body)
        employment_id = result.get("value", {}).get("id")

        if not employment_id:
            # May already exist — find it
            existing = client.get("/employee/employment", params={
                "employeeId": employee_id, "fields": "id"
            })
            vals = existing.get("values", [])
            if vals:
                employment_id = vals[0]["id"]
                _recover_error("/employee/employment")
                steps_log.append(f"Employment already existed for employee {employee_id} (id={employment_id})")
            else:
                steps_log.append(f"WARNING: Failed to create employment for employee {employee_id}: {result}")
        else:
            steps_log.append(f"Created employment for employee {employee_id} (id={employment_id})")

        return employment_id

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

        # ── Step 1: Create customer ──
        cust_body = {"name": customer_name, "isCustomer": True}
        if customer_org_number:
            cust_body["organizationNumber"] = customer_org_number
        cust_result = client.post("/customer", json=cust_body)
        customer_id = cust_result.get("value", {}).get("id")

        # Handle duplicate customer (422 collision)
        if not customer_id and cust_result.get("error"):
            search_params = {"fields": "id,name"}
            if customer_org_number:
                search_params["organizationNumber"] = customer_org_number
            else:
                search_params["name"] = customer_name
            existing = client.get("/customer", params=search_params)
            vals = existing.get("values", [])
            if vals:
                customer_id = vals[0]["id"]
                _recover_error("/customer")
                steps_log.append(f"Customer '{customer_name}' already existed (id={customer_id})")

        if not customer_id:
            return {"error": True, "message": f"Failed to create customer: {cust_result}", "steps": steps_log}
        if not steps_log or "already existed" not in steps_log[-1]:
            steps_log.append(f"Created customer '{customer_name}' (id={customer_id})")

        # ── Step 2: Create PM employee (EXTENDED) ──
        pm_id = 0
        if pm_email:
            pm_id, _pm_created = _find_or_create_employee(pm_firstName, pm_lastName, pm_email,
                                              userType="EXTENDED", steps_log=steps_log)
            if not pm_id:
                return {"error": True, "message": "Failed to create PM employee", "steps": steps_log}

        # ── Step 3: Create project ──
        # Use the project tool which handles PM entitlement errors automatically
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
                # Ensure PM has employment (create_project may not have created it)
                # PM was created with dateOfBirth, skip DOB check
                _find_or_create_employment(emp_id, today, steps_log=steps_log, skip_dob_check=True)
            else:
                emp_id, emp_created = _find_or_create_employee(emp_firstName, emp_lastName, emp_email,
                                                   steps_log=steps_log)
                if not emp_id:
                    steps_log.append(f"WARNING: Skipping employee {emp_firstName} {emp_lastName} — creation failed")
                    continue

                # Non-PM employees need employment record
                _find_or_create_employment(emp_id, today, steps_log=steps_log, skip_dob_check=emp_created)

                # Add as project participant (best-effort)
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
                # Resolve activity for timesheet
                activity_id = 0
                act_result = client.get("/activity", params={"projectId": project_id, "fields": "id", "count": 1})
                act_vals = act_result.get("values", [])
                if act_vals:
                    activity_id = act_vals[0]["id"]
                else:
                    # Get any activity
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
            sup_body = {"name": supplier_name, "isSupplier": True}
            if supplier_org_number:
                sup_body["organizationNumber"] = supplier_org_number
            sup_result = client.post("/supplier", json=sup_body)
            supplier_id = sup_result.get("value", {}).get("id")
            if supplier_id:
                steps_log.append(f"Created supplier '{supplier_name}' (id={supplier_id})")

                # Create incoming invoice linked to project (tool handles VAT/account resolution)
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
                steps_log.append(f"WARNING: Failed to create supplier: {sup_result}")

        # ── Step 6: Customer invoice ──
        if create_customer_invoice and budget > 0:
            # Create product
            from tools.products import build_product_tools
            prod_tools = build_product_tools(client)
            prod_result = prod_tools["create_product"](
                name=f"Project Services - {project_name}",
                priceExcludingVatCurrency=budget,
                vatPercentage=invoice_vat_percentage,
            )
            prod_id = prod_result.get("value", {}).get("id")
            if prod_id:
                steps_log.append(f"Created product for invoice (id={prod_id})")

                # Create order
                from tools.invoicing import build_invoicing_tools
                inv_tools = build_invoicing_tools(client)
                order_lines = _json.dumps([{"product_id": prod_id, "count": 1}])
                order_result = inv_tools["create_order"](
                    customer_id=customer_id,
                    deliveryDate=today,
                    orderLines=order_lines,
                    project_id=project_id,
                )
                order_id = order_result.get("value", {}).get("id")
                if order_id:
                    steps_log.append(f"Created order (id={order_id})")

                    # Create invoice
                    inv_result = inv_tools["create_invoice"](
                        invoiceDate=today,
                        invoiceDueDate=today,
                        order_id=order_id,
                    )
                    invoice_id = inv_result.get("value", {}).get("id")
                    if invoice_id:
                        steps_log.append(f"Created invoice (id={invoice_id})")
                    else:
                        steps_log.append(f"WARNING: Invoice creation failed: {inv_result}")
                else:
                    steps_log.append(f"WARNING: Order creation failed: {order_result}")
            else:
                steps_log.append(f"WARNING: Product creation failed: {prod_result}")

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
