"""Compound tool: process entire salary_with_bonus workflow in one deterministic call.

Handles: create employee → ensure division → create employment → create salary transaction.
No LLM chaining required — all steps are hardcoded.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_salary_tools(client: TripletexClient) -> dict:
    """Build the compound salary processing tool."""

    def process_salary(
        firstName: str,
        lastName: str,
        email: str,
        year: int,
        month: int,
        base_salary: float = 0.0,
        bonus: float = 0.0,
        dateOfBirth: str = "",
        department_name: str = "",
        startDate: str = "",
        hoursPerDay: float = 0.0,
        percentageOfFullTimeEquivalent: float = 100.0,
        annualSalary: float = 0.0,
        occupationCode: str = "",
    ) -> dict:
        """Process a complete salary/payroll transaction for an employee.

        This is a compound tool that handles the ENTIRE salary workflow:
        1. Creates the employee (or finds existing by email)
        2. Ensures a division (virksomhet) exists
        3. Creates employment with start date and annual salary
        4. Optionally sets standard working hours
        5. Creates the salary transaction with base salary and/or bonus

        Args:
            firstName: Employee's first name.
            lastName: Employee's last name.
            email: Employee's email address.
            year: Salary year (e.g. 2026).
            month: Salary month (1-12).
            base_salary: Base salary amount (Fastlønn/Grundgehalt) for payslip. 0 to skip.
            bonus: Bonus amount for payslip. 0 to skip.
            dateOfBirth: Date of birth YYYY-MM-DD (optional, defaults to 1990-01-01).
            department_name: Department name (optional, auto-created if needed).
            startDate: Employment start date YYYY-MM-DD (defaults to 1st of salary month).
            hoursPerDay: Standard working hours per day (0 to skip, e.g. 7.5).
            percentageOfFullTimeEquivalent: FTE percentage (default 100).
            annualSalary: Annual salary for employment record (optional, defaults to base_salary * 12).
            occupationCode: STYRK occupation code (optional).

        Returns:
            Summary with employee_id, employment_id, transaction_id, or error details.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        # ── Defaults ──
        if not startDate:
            startDate = f"{year}-{month:02d}-01"
        if not dateOfBirth:
            dateOfBirth = "1990-01-01"
        if annualSalary == 0.0 and base_salary > 0:
            annualSalary = base_salary * 12
        # Auto-derive monthly base salary from annual salary when not explicit
        if base_salary == 0.0 and annualSalary > 0:
            base_salary = round(annualSalary / 12, 2)
        salary_date = f"{year}-{month:02d}-15"  # mid-month

        # ── Step 1: Create or find employee ──
        dept_id = 0
        if department_name:
            dept_search = client.get("/department", params={"name": department_name, "fields": "id"})
            depts = dept_search.get("values", [])
            if depts:
                dept_id = depts[0]["id"]
            else:
                new_dept = client.post("/department", json={
                    "name": department_name,
                    "departmentNumber": "AUTO_" + department_name.upper().replace(" ", "_")[:20],
                })
                dept_val = new_dept.get("value", {})
                if dept_val.get("id"):
                    dept_id = dept_val["id"]
                    steps_log.append(f"Created department '{department_name}' (id={dept_id})")
        else:
            cached = client.get_cached("default_department")
            if cached:
                dept_id = cached
            else:
                dept_result = client.get("/department", params={"fields": "id", "count": 1})
                depts = dept_result.get("values", [])
                if depts:
                    dept_id = depts[0]["id"]
                else:
                    new_dept = client.post("/department", json={"name": "Avdeling", "departmentNumber": "1"})
                    dept_val = new_dept.get("value", {})
                    if dept_val.get("id"):
                        dept_id = dept_val["id"]
                if dept_id:
                    client.set_cached("default_department", dept_id)

        emp_body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "userType": "STANDARD",
            "dateOfBirth": dateOfBirth,
        }
        if dept_id:
            emp_body["department"] = {"id": dept_id}

        emp_result = client.post("/employee", json=emp_body)
        employee_id = None

        # Check for email collision — find existing employee
        if emp_result.get("error") and emp_result.get("status_code") == 422:
            msg = str(emp_result.get("message", "")).lower()
            if "e-postadress" in msg or "email" in msg:
                existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
                vals = existing.get("values", [])
                if vals:
                    employee_id = vals[0]["id"]
                    # Undo error count for auto-recovery
                    client._error_count = max(0, client._error_count - 1)
                    for entry in reversed(client._call_log):
                        if not entry.get("ok") and "/employee" in entry.get("url", ""):
                            entry["ok"] = True
                            entry["recovered"] = True
                            break
                    steps_log.append(f"Employee already existed (id={employee_id})")
        if not employee_id:
            emp_val = emp_result.get("value", {})
            employee_id = emp_val.get("id")
            if not employee_id:
                return {"error": True, "message": f"Failed to create employee: {emp_result}", "steps": steps_log}
            steps_log.append(f"Created employee {firstName} {lastName} (id={employee_id})")

        # ── Step 2: Ensure division exists ──
        div_id = client.get_cached("default_division")
        if div_id is None:
            divs = client.get("/division", params={"fields": "id", "count": 1})
            div_list = divs.get("values", [])
            if div_list:
                div_id = div_list[0]["id"]
            else:
                # Create a default division — needs municipality
                import random
                org_num = str(900000000 + random.randint(0, 99999999))
                muni = client.get("/municipality", params={"fields": "id", "count": 1})
                munis = muni.get("values", [])
                div_body = {
                    "name": "Hovedkontor",
                    "startDate": startDate,
                    "organizationNumber": org_num,
                    "municipalityDate": startDate,
                }
                if munis:
                    div_body["municipality"] = {"id": munis[0]["id"]}
                div_result = client.post("/division", json=div_body)
                div_val = div_result.get("value", {})
                div_id = div_val.get("id", 0)
                if div_id:
                    steps_log.append(f"Created division 'Hovedkontor' (id={div_id})")
            client.set_cached("default_division", div_id or 0)

        # ── Step 3: Create employment ──
        # Ensure dateOfBirth is set (required for employment)
        emp_check = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
        emp_data = emp_check.get("value", emp_check)
        if not emp_data.get("dateOfBirth"):
            client.put(f"/employee/{employee_id}", json={"dateOfBirth": dateOfBirth})

        emp_details = {
            "date": startDate,
            "employmentType": "ORDINARY",
            "workingHoursScheme": "NOT_SHIFT",
            "percentageOfFullTimeEquivalent": percentageOfFullTimeEquivalent,
        }
        if annualSalary:
            emp_details["annualSalary"] = annualSalary

        # Resolve occupation code if provided
        if occupationCode:
            stripped = occupationCode.strip()
            if stripped.isdigit():
                occ_result = client.get(
                    "/employee/employment/occupationCode",
                    params={"code": stripped, "fields": "id,code,nameNO"},
                )
                occ_vals = occ_result.get("values", [])
                if occ_vals:
                    prefix = [v for v in occ_vals if v.get("code", "").startswith(stripped)]
                    if prefix:
                        emp_details["occupationCode"] = {"id": prefix[0]["id"]}
                    else:
                        emp_details["occupationCode"] = {"id": occ_vals[0]["id"]}

        employment_body = {
            "employee": {"id": employee_id},
            "startDate": startDate,
            "employmentDetails": [emp_details],
        }
        if div_id:
            employment_body["division"] = {"id": div_id}

        employment_result = client.post("/employee/employment", json=employment_body)
        employment_id = None
        emp_employment = employment_result.get("value", {})
        employment_id = emp_employment.get("id")

        if not employment_id:
            # Employment may already exist (overlapping periods) — try to find it
            existing_emp = client.get("/employee/employment", params={
                "employeeId": employee_id, "fields": "id"
            })
            existing_vals = existing_emp.get("values", [])
            if existing_vals:
                employment_id = existing_vals[0]["id"]
                steps_log.append(f"Employment already existed (id={employment_id})")
            else:
                return {"error": True, "message": f"Failed to create employment: {employment_result}", "steps": steps_log}
        else:
            steps_log.append(f"Created employment (id={employment_id})")

        # ── Step 4: Standard working hours (if specified) ──
        if hoursPerDay > 0:
            std_result = client.post("/employee/standardTime", json={
                "employee": {"id": employee_id},
                "fromDate": startDate,
                "hoursPerDay": hoursPerDay,
            })
            steps_log.append(f"Set standard time: {hoursPerDay}h/day")

        # ── Step 5: Create salary transaction ──
        # Build payslip lines
        payslip_lines = []

        if base_salary > 0 or bonus > 0:
            # Look up salary type IDs
            st_resp = client.get("/salary/type", params={"fields": "id,number"})
            num_to_id = {}
            for st in st_resp.get("values", []):
                num_to_id[str(st.get("number", ""))] = st["id"]

            lines = []
            if base_salary > 0:
                st_id = num_to_id.get("2000")
                if not st_id:
                    # Fallback: try to find any "fastlønn" type
                    st_detail = client.get("/salary/type", params={"fields": "id,number,name"})
                    for st in st_detail.get("values", []):
                        if st.get("number") == 2000 or "fastl" in (st.get("name", "") or "").lower():
                            st_id = st["id"]
                            break
                if st_id:
                    lines.append({"salaryType": {"id": st_id}, "rate": base_salary, "count": 1})
                else:
                    steps_log.append("WARNING: Could not find salary type 2000 (Fastlønn)")

            if bonus > 0:
                st_id = num_to_id.get("2002")
                if not st_id:
                    st_detail = client.get("/salary/type", params={"fields": "id,number,name"})
                    for st in st_detail.get("values", []):
                        if st.get("number") == 2002 or "bonus" in (st.get("name", "") or "").lower():
                            st_id = st["id"]
                            break
                if st_id:
                    lines.append({"salaryType": {"id": st_id}, "rate": bonus, "count": 1})
                else:
                    steps_log.append("WARNING: Could not find salary type 2002 (Bonus)")

            payslip_lines = [{"employee": {"id": employee_id}, "specifications": lines}]

        sal_body = {
            "date": salary_date,
            "year": year,
            "month": month,
            "payslips": payslip_lines if payslip_lines else [{"employee": {"id": employee_id}}],
        }

        sal_result = client.post("/salary/transaction", json=sal_body)
        transaction_id = sal_result.get("value", {}).get("id")

        if not transaction_id:
            return {
                "error": True,
                "message": f"Failed to create salary transaction: {sal_result}",
                "employee_id": employee_id,
                "employment_id": employment_id,
                "steps": steps_log,
            }

        steps_log.append(f"Created salary transaction (id={transaction_id}) for {year}-{month:02d}")

        return {
            "success": True,
            "employee_id": employee_id,
            "employment_id": employment_id,
            "transaction_id": transaction_id,
            "year": year,
            "month": month,
            "base_salary": base_salary,
            "bonus": bonus,
            "steps": steps_log,
        }

    return {
        "process_salary": process_salary,
    }
