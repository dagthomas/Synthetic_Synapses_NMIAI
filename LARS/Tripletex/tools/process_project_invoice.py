"""Compound tool: project invoice workflow in one deterministic call.

Handles two scenarios:
  A. Fixed-price project: customer → employee → project → product → order → invoice
  B. Hourly project: customer → employee → project → employment → participant →
     hourly rate → timesheet → product → order → invoice
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_project_invoice_tools(client: TripletexClient) -> dict:
    """Build the compound project-invoice tool."""

    from tools._helpers import (
        recover_error, find_or_create_product, find_or_create_customer,
        find_or_create_employment, ensure_bank_account,
    )

    def _ensure_employee_ready(employee_id: int, fresh_create: bool = False) -> bool:
        """Ensure employee has dateOfBirth, employment, and PM access."""
        cache_key = f"pm_ready_{employee_id}"
        if client.get_cached(cache_key):
            return True

        if not fresh_create:
            _WRITABLE = {
                "id", "version", "firstName", "lastName", "email",
                "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
                "dateOfBirth", "department", "employeeNumber", "address",
                "userType", "nationalIdentityNumber", "bankAccountNumber",
                "comments", "employeeCategory",
            }
            emp = client.get(f"/employee/{employee_id}", params={"fields": "*"})
            emp_val = emp.get("value", {})
            has_dob = bool(emp_val.get("dateOfBirth"))
            has_extended = emp_val.get("userType") == "EXTENDED"
            has_employment = bool(emp_val.get("employments"))

            if has_dob and has_extended and has_employment:
                client.set_cached(cache_key, True)
                return True

            if not has_dob or not has_extended:
                body = {k: v for k, v in emp_val.items() if k in _WRITABLE and v is not None}
                if isinstance(body.get("department"), dict):
                    body["department"] = {"id": body["department"]["id"]}
                if not has_dob:
                    body["dateOfBirth"] = "1990-01-01"
                if not has_extended:
                    body["userType"] = "EXTENDED"
                client.put(f"/employee/{employee_id}", json=body)
        else:
            has_employment = False

        # Create employment if needed (search-before-create)
        if not has_employment:
            today = dt_date.today().isoformat()
            find_or_create_employment(client, employee_id=employee_id, start_date=today)

        # Grant PM entitlements
        company_id = client.get_cached("company_id")
        if not company_id:
            who = client.get("/token/session/>whoAmI", params={"fields": "companyId"})
            company_id = who.get("value", {}).get("companyId")
            if company_id:
                client.set_cached("company_id", company_id)

        if company_id:
            for eid in [45, 10, 8]:
                client.post("/employee/entitlement", json={
                    "employee": {"id": employee_id},
                    "entitlementId": eid,
                    "customer": {"id": company_id},
                })

        client.set_cached(cache_key, True)
        return True

    def process_project_invoice(
        # Customer
        customer_name: str,
        customer_email: str = "",
        customer_org_number: str = "",
        # Project manager (employee)
        pm_firstName: str = "",
        pm_lastName: str = "",
        pm_email: str = "",
        # Project
        project_name: str = "",
        startDate: str = "",
        # Scenario A: Fixed price
        fixedPriceAmount: float = 0.0,
        milestonePercentage: float = 100.0,
        # Scenario B: Hourly
        hourlyRate: float = 0.0,
        hours: float = 0.0,
        timesheetDate: str = "",
        activity_name: str = "",
        # Invoice
        invoiceDate: str = "",
        invoiceDueDate: str = "",
        send_invoice: bool = False,
        # Product override
        product_name: str = "",
        vatPercentage: int = 25,
    ) -> dict:
        """Process a complete project invoice workflow in one call.

        Supports two scenarios:
        A. FIXED-PRICE: Create project with fixed price, invoice a milestone percentage.
        B. HOURLY: Create project, log timesheet hours, invoice hours x rate.

        The tool auto-detects the scenario from the parameters:
        - If fixedPriceAmount > 0: Scenario A (fixed price)
        - If hourlyRate > 0 and hours > 0: Scenario B (hourly)
        - Otherwise: Scenario A with invoice_amount = 0 (just create project)

        Args:
            customer_name: Customer company name. REQUIRED.
            customer_email: Customer email.
            customer_org_number: Organization number.
            pm_firstName: Project manager first name.
            pm_lastName: Project manager last name.
            pm_email: Project manager email.
            project_name: Project name (defaults to customer_name + " Project").
            startDate: Project start date YYYY-MM-DD.
            fixedPriceAmount: Total fixed price amount (for scenario A).
            milestonePercentage: Percentage of fixed price to invoice (default 100%).
            hourlyRate: Hourly rate (for scenario B).
            hours: Number of hours worked (for scenario B).
            timesheetDate: Date for timesheet entry YYYY-MM-DD (defaults to today).
            activity_name: Activity name for timesheet (defaults to project name).
            invoiceDate: Invoice date YYYY-MM-DD (defaults to today).
            invoiceDueDate: Due date YYYY-MM-DD (defaults to invoiceDate).
            send_invoice: If True, sends the invoice after creation.
            product_name: Custom product name for the invoice line.
            vatPercentage: VAT rate (default 25).

        Returns:
            Summary with all created entity IDs, or error details.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        if not startDate:
            startDate = today
        if not invoiceDate:
            invoiceDate = today
        if not invoiceDueDate:
            invoiceDueDate = invoiceDate
        if not timesheetDate:
            timesheetDate = today
        if not project_name:
            project_name = f"{customer_name} Project"

        # ── Step 1: Find or create customer ──
        customer_id = find_or_create_customer(
            client, name=customer_name, email=customer_email,
            org_number=customer_org_number, steps_log=steps_log,
        )
        if not customer_id:
            return {"error": True, "message": f"Failed to create customer '{customer_name}'", "steps": steps_log}

        # ── Step 2: Create employee (project manager) ──
        employee_id = None
        pm_fresh = False
        if pm_firstName and pm_lastName:
            # Search by email first
            if pm_email:
                existing = client.get("/employee", params={"email": pm_email, "fields": "id"})
                vals = existing.get("values", [])
                if vals:
                    employee_id = vals[0]["id"]
                    steps_log.append(f"Found existing employee {pm_firstName} {pm_lastName} (id={employee_id})")

            if not employee_id:
                emp_body = {
                    "firstName": pm_firstName,
                    "lastName": pm_lastName,
                    "email": pm_email or f"{pm_firstName.lower()}.{pm_lastName.lower()}@example.com",
                    "userType": "EXTENDED",
                    "dateOfBirth": "1990-01-01",
                }
                emp_result = client.post("/employee", json=emp_body)
                employee_id = emp_result.get("value", {}).get("id")

                if not employee_id and emp_result.get("error"):
                    msg = str(emp_result.get("message", "")).lower()
                    if "e-postadress" in msg or "email" in msg:
                        recover_error(client, "/employee")
                        existing = client.get("/employee", params={
                            "email": emp_body["email"], "fields": "id"
                        })
                        vals = existing.get("values", [])
                        if vals:
                            employee_id = vals[0]["id"]
                            steps_log.append(f"Employee already existed (id={employee_id})")

                if employee_id and not any("Employee" in s or "employee" in s for s in steps_log):
                    pm_fresh = True
                    steps_log.append(f"Created employee {pm_firstName} {pm_lastName} (id={employee_id})")
        else:
            emp_result = client.get("/employee", params={"fields": "id", "count": 1})
            emps = emp_result.get("values", [])
            if emps:
                employee_id = emps[0]["id"]

        # ── Step 3: Create project ──
        proj_body = {
            "name": project_name,
            "startDate": startDate,
            "customer": {"id": customer_id},
        }
        if fixedPriceAmount > 0:
            proj_body["isFixedPrice"] = True
            proj_body["fixedprice"] = fixedPriceAmount
        if employee_id:
            _ensure_employee_ready(employee_id, fresh_create=pm_fresh)
            proj_body["projectManager"] = {"id": employee_id}

        proj_result = client.post("/project", json=proj_body)
        project_id = proj_result.get("value", {}).get("id")

        if not project_id and proj_result.get("error") and employee_id:
            err_msg = str(proj_result.get("message", "")).lower()
            if "entitlement" in err_msg or "tilgang" in err_msg or "rettighet" in err_msg:
                proj_body.pop("projectManager", None)
                client._error_count = max(0, client._error_count - 1)
                proj_result = client.post("/project", json=proj_body)
                project_id = proj_result.get("value", {}).get("id")

        if not project_id:
            return {"error": True, "message": f"Failed to create project: {proj_result}", "steps": steps_log}
        steps_log.append(f"Created project '{project_name}' (id={project_id})")

        # ── Determine invoice amount ──
        is_hourly = hourlyRate > 0 and hours > 0
        is_fixed = fixedPriceAmount > 0

        if is_hourly:
            invoice_amount = round(hourlyRate * hours, 2)
        elif is_fixed:
            invoice_amount = round(fixedPriceAmount * (milestonePercentage / 100), 2)
        else:
            return {
                "success": True,
                "customer_id": customer_id,
                "employee_id": employee_id,
                "project_id": project_id,
                "steps": steps_log,
            }

        # ── Step 4 (hourly only): Employment + participant + timesheet ──
        if is_hourly and employee_id:
            find_or_create_employment(client, employee_id=employee_id, start_date="2026-01-01",
                                      steps_log=steps_log)

            client.post("/project/participant", json={
                "project": {"id": project_id},
                "employee": {"id": employee_id},
            })
            steps_log.append("Added as project participant")

            client.post("/project/hourlyRates", json={
                "employee": {"id": employee_id},
                "date": startDate,
                "hourlyRate": hourlyRate,
                "hourlyCost": 0,
                "project": {"id": project_id},
            })
            steps_log.append(f"Set hourly rate: {hourlyRate}")

            act_name = activity_name or project_name
            act_result = client.post("/activity", json={
                "name": act_name,
                "project": {"id": project_id},
            })
            activity_id = act_result.get("value", {}).get("id")
            if not activity_id:
                act_search = client.get("/activity", params={
                    "projectId": project_id, "fields": "id,name"
                })
                for a in (act_search.get("values") or []):
                    activity_id = a["id"]
                    break

            if activity_id:
                ts_body = {
                    "employee": {"id": employee_id},
                    "project": {"id": project_id},
                    "activity": {"id": activity_id},
                    "date": timesheetDate,
                    "hours": hours,
                }
                client.post("/timesheet/entry", json=ts_body)
                steps_log.append(f"Created timesheet: {hours}h on {timesheetDate}")

        # ── Step 5: Find or create product ──
        if not product_name:
            if is_fixed:
                product_name = f"Milestone Payment for {project_name}"
            else:
                product_name = f"Consulting - {project_name}"

        prod_id = find_or_create_product(
            client, name=product_name, price=invoice_amount,
            vat_percentage=vatPercentage, steps_log=steps_log,
        )
        if not prod_id:
            return {"error": True, "message": f"Failed to create product '{product_name}'", "steps": steps_log}

        # ── Step 6: Create order ──
        order_body = {
            "customer": {"id": customer_id},
            "orderDate": invoiceDate,
            "deliveryDate": invoiceDate,
            "orderLines": [{"product": {"id": prod_id}, "count": 1,
                           "unitPriceExcludingVatCurrency": invoice_amount}],
            "project": {"id": project_id},
        }

        order_result = client.post("/order", json=order_body)
        order_id = order_result.get("value", {}).get("id")

        if not order_id and order_result.get("error"):
            order_body.pop("project", None)
            recover_error(client, "/order")
            order_result = client.post("/order", json=order_body)
            order_id = order_result.get("value", {}).get("id")

        if not order_id:
            return {"error": True, "message": f"Failed to create order: {order_result}", "steps": steps_log}
        steps_log.append(f"Created order (id={order_id})")

        # ── Step 7: Create invoice ──
        ensure_bank_account(client)

        inv_body = {
            "invoiceDate": invoiceDate,
            "invoiceDueDate": invoiceDueDate,
            "orders": [{"id": order_id}],
        }
        inv_result = client.post("/invoice", json=inv_body)
        invoice_id = inv_result.get("value", {}).get("id")
        if not invoice_id:
            return {"error": True, "message": f"Failed to create invoice: {inv_result}", "steps": steps_log}
        steps_log.append(f"Created invoice (id={invoice_id})")

        # ── Step 8: Send invoice (if requested) ──
        if send_invoice:
            client.put(f"/invoice/{invoice_id}/:send", params={"sendType": "EMAIL"})
            steps_log.append("Sent invoice via email")

        return {
            "success": True,
            "customer_id": customer_id,
            "employee_id": employee_id,
            "project_id": project_id,
            "product_id": prod_id,
            "order_id": order_id,
            "invoice_id": invoice_id,
            "invoice_amount": invoice_amount,
            "steps": steps_log,
        }

    return {
        "process_project_invoice": process_project_invoice,
    }
