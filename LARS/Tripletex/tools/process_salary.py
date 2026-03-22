"""Compound tool: process entire salary_with_bonus workflow in one deterministic call.

Handles: find/create employee → ensure division → find/create employment → create salary transaction.
No LLM chaining required — all steps are hardcoded.

Uses search-before-create pattern to avoid 422 errors and efficiency penalties.
"""

import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_salary_tools(client: TripletexClient) -> dict:
    """Build the compound salary processing tool."""

    from tools._helpers import (
        find_or_create_department, find_or_create_employment, recover_error,
    )

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
        today = dt_date.today()
        today_str = today.isoformat()
        steps_log = []

        # ── Sanity check: auto-correct year if it's clearly wrong ──
        if year < today.year - 1:
            log.warning("process_salary: year=%d looks wrong (today=%s), correcting to %d", year, today_str, today.year)
            year = today.year

        # ── Defaults ──
        if not startDate:
            startDate = f"{year}-{month:02d}-01"
        if not dateOfBirth:
            dateOfBirth = "1990-01-01"
        if annualSalary == 0.0 and base_salary > 0:
            annualSalary = base_salary * 12
        if base_salary == 0.0 and annualSalary > 0:
            base_salary = round(annualSalary / 12, 2)
        salary_date = f"{year}-{month:02d}-15"

        # ── Validate: salary period must not be before employment start ──
        salary_ym = f"{year}-{month:02d}"
        start_ym = startDate[:7]
        if salary_ym < start_ym:
            log.warning("process_salary: salary period %s is before employment start %s, adjusting startDate", salary_ym, startDate)
            startDate = f"{year}-{month:02d}-01"

        # ── Step 1: Find or create department ──
        dept_id = 0
        if department_name:
            dept_id = find_or_create_department(client, name=department_name, steps_log=steps_log)
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
                    dept_id = find_or_create_department(client, name="Avdeling", steps_log=steps_log)
                if dept_id:
                    client.set_cached("default_department", dept_id)

        # ── Step 2: Create or find employee (search by email first) ──
        employee_id = None
        emp_just_created = False

        existing = client.get("/employee", params={"email": email, "fields": "id,firstName,lastName,email"})
        vals = existing.get("values", [])
        if vals:
            employee_id = vals[0]["id"]
            steps_log.append(f"Found existing employee '{firstName} {lastName}' (id={employee_id})")
        else:
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
            employee_id = emp_result.get("value", {}).get("id")

            if not employee_id and emp_result.get("error"):
                msg = str(emp_result.get("message", "")).lower()
                if "e-postadress" in msg or "email" in msg:
                    recover_error(client, "/employee")
                    existing2 = client.get("/employee", params={"email": email, "fields": "id"})
                    vals2 = existing2.get("values", [])
                    if vals2:
                        employee_id = vals2[0]["id"]
                        steps_log.append(f"Employee already existed (id={employee_id})")

            if not employee_id:
                return {"error": True, "message": "Failed to create employee", "steps": steps_log}
            if not any("Employee" in s or "employee" in s for s in steps_log):
                emp_just_created = True
                steps_log.append(f"Created employee {firstName} {lastName} (id={employee_id})")

        # ── Step 3: Find or create employment ──
        employment_id = find_or_create_employment(
            client, employee_id=employee_id, start_date=startDate,
            annual_salary=annualSalary,
            percentage=percentageOfFullTimeEquivalent,
            skip_dob_check=emp_just_created,
            steps_log=steps_log,
        )
        if not employment_id:
            return {"error": True, "message": f"Failed to create employment for employee {employee_id}", "steps": steps_log}

        # ── Step 4: Standard working hours (if specified) ──
        if hoursPerDay > 0:
            client.post("/employee/standardTime", json={
                "employee": {"id": employee_id},
                "fromDate": startDate,
                "hoursPerDay": hoursPerDay,
            })
            steps_log.append(f"Set standard time: {hoursPerDay}h/day")

        # ── Step 5: Ensure salary module is active ──
        # Some sandboxes don't have the salary module enabled by default
        try:
            _test = client.get("/salary/type", params={"fields": "id", "count": 1})
            if _test.get("status") == 403 or "permission" in str(_test.get("message", "")).lower():
                # Try to activate salary module
                client.post("/company/salesmodules", json={"name": "MAMUT_SALARY"})
                # Also try the internal modules endpoint (sandbox-specific)
                client.put("/company/modules", json={"moduleSalary": True, "moduleHRM": True})
                steps_log.append("Attempted to enable salary module")
        except Exception:
            pass

        # ── Step 6: Create salary transaction ──
        payslip_lines = []

        if base_salary > 0 or bonus > 0:
            st_resp = client.get("/salary/type", params={"fields": "id,number,name"})
            num_to_id = {}
            name_to_id = {}
            for st in st_resp.get("values", []):
                num_to_id[str(st.get("number", ""))] = st["id"]
                st_name = (st.get("name", "") or "").lower()
                if st_name:
                    name_to_id[st_name] = st["id"]

            lines = []
            if base_salary > 0:
                st_id = num_to_id.get("2000")
                if not st_id:
                    # Try common names for base salary type
                    for keyword in ["fastlønn", "fastlonn", "fast lønn", "grundgehalt", "base salary"]:
                        for name, sid in name_to_id.items():
                            if keyword in name:
                                st_id = sid
                                break
                        if st_id:
                            break
                if not st_id:
                    # Fallback: use the first salary type with number 2000-2099
                    for num_str, sid in num_to_id.items():
                        try:
                            n = int(num_str)
                            if 2000 <= n < 2100:
                                st_id = sid
                                break
                        except ValueError:
                            pass
                if st_id:
                    lines.append({"salaryType": {"id": st_id}, "rate": base_salary, "count": 1})
                else:
                    steps_log.append("WARNING: Could not find salary type for Fastlønn, using first available")
                    # Last resort: use first salary type from list
                    all_types = st_resp.get("values", [])
                    if all_types:
                        lines.append({"salaryType": {"id": all_types[0]["id"]}, "rate": base_salary, "count": 1})

            if bonus > 0:
                st_id = num_to_id.get("2002")
                if not st_id:
                    for keyword in ["bonus", "tillegg", "zulage", "prime"]:
                        for name, sid in name_to_id.items():
                            if keyword in name:
                                st_id = sid
                                break
                        if st_id:
                            break
                if st_id:
                    lines.append({"salaryType": {"id": st_id}, "rate": bonus, "count": 1})
                else:
                    steps_log.append("WARNING: Could not find salary type 2002 (Bonus)")

            if lines:
                payslip_lines = [{"employee": {"id": employee_id}, "specifications": lines}]

        # Always include the employee in the payslips — even without lines
        if not payslip_lines:
            payslip_lines = [{"employee": {"id": employee_id}}]

        sal_body = {
            "date": salary_date,
            "year": year,
            "month": month,
            "payslips": payslip_lines,
        }

        sal_result = client.post("/salary/transaction", json=sal_body)
        transaction_id = sal_result.get("value", {}).get("id")

        if not transaction_id:
            # Check if the error is recoverable — sometimes the transaction is created
            # but the response doesn't include the ID
            err_msg = str(sal_result.get("message", "")).lower()
            log.warning("Salary transaction POST failed: %s", sal_result)

            # Try to find the transaction we just created
            verify = client.get("/salary/transaction", params={
                "employeeId": employee_id,
                "yearFrom": year, "yearTo": year,
                "fields": "id,year,month",
                "count": 5,
            })
            for txn in verify.get("values", []):
                if txn.get("year") == year and txn.get("month") == month:
                    transaction_id = txn["id"]
                    steps_log.append(f"Found salary transaction after POST error (id={transaction_id})")
                    break

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
