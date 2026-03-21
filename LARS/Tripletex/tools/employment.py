from tripletex_client import TripletexClient


def build_employment_tools(client: TripletexClient) -> dict:
    """Build employment tools."""

    def create_employment(
        employee_id: int,
        startDate: str,
        annualSalary: float = 0.0,
        percentageOfFullTimeEquivalent: float = 100.0,
        occupationCode: str = "",
    ) -> dict:
        """Create an employment record for an employee.

        This sets up the formal employment relationship (ansettelsesforhold)
        including start date. The employee must already exist.

        NOTE: This tool automatically creates initial employment details for startDate.
        Pass annualSalary and percentageOfFullTimeEquivalent here to set them on the
        initial details — do NOT call create_employment_details separately for startDate.

        NOTE: The employee must have a dateOfBirth set. If not, this tool
        will automatically set it to 1990-01-01 before creating the employment.

        Args:
            employee_id: The ID of the employee.
            startDate: Employment start date in YYYY-MM-DD format.
            annualSalary: Annual salary (optional, default 0 = not set).
            percentageOfFullTimeEquivalent: FTE percentage (default 100).
            occupationCode: Occupation code string (e.g. '3112', 'yrkeskode'). The tool will look up the code and include it in employment details.

        Returns:
            The created employment with id, or an error message.
        """
        # Tripletex requires dateOfBirth on the employee before creating employment
        emp = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
        emp_val = emp.get("value", emp)
        if not emp_val.get("dateOfBirth"):
            client.put(f"/employee/{employee_id}", json={"dateOfBirth": "1990-01-01"})

        details = {
            "date": startDate,
            "employmentType": "ORDINARY",
            "workingHoursScheme": "NOT_SHIFT",
            "percentageOfFullTimeEquivalent": percentageOfFullTimeEquivalent,
        }
        if annualSalary:
            details["annualSalary"] = annualSalary

        # Resolve occupation code by searching the Tripletex occupation code registry
        if occupationCode:
            stripped = occupationCode.strip()
            occ_vals = []
            is_numeric = stripped.isdigit()

            if is_numeric:
                # STYRK codes are numeric — search by code field first
                occ_result = client.get(
                    "/employee/employment/occupationCode",
                    params={"code": stripped, "fields": "id,code,nameNO"},
                )
                occ_vals = occ_result.get("values", [])
                # Filter for prefix match (4-digit STYRK → 7-digit Tripletex codes)
                if occ_vals:
                    prefix = [v for v in occ_vals if v.get("code", "").startswith(stripped)]
                    if prefix:
                        occ_vals = prefix
                # Fallback: broad search filtered by code prefix
                if not occ_vals and len(stripped) <= 4:
                    occ_result = client.get(
                        "/employee/employment/occupationCode",
                        params={"fields": "id,code,nameNO", "count": 8000},
                    )
                    all_codes = occ_result.get("values", [])
                    occ_vals = [v for v in all_codes if v.get("code", "").startswith(stripped)]
            else:
                # Text description — search by name
                occ_result = client.get(
                    "/employee/employment/occupationCode",
                    params={"nameNOContains": stripped, "fields": "id,code,nameNO"},
                )
                occ_vals = occ_result.get("values", [])

            if occ_vals:
                details["occupationCode"] = {"id": occ_vals[0]["id"]}

        body = {
            "employee": {"id": employee_id},
            "startDate": startDate,
            "employmentDetails": [details],
        }
        # Auto-detect division — use cache or fetch once
        div_id = client.get_cached("default_division")
        if div_id is None:
            divs = client.get("/division", params={"fields": "id", "count": 1})
            div_list = divs.get("values", [])
            if div_list:
                div_id = div_list[0]["id"]
                client.set_cached("default_division", div_id)
            else:
                div_id = 0
                client.set_cached("default_division", 0)
        if div_id:
            body["division"] = {"id": div_id}
        return client.post("/employee/employment", json=body)

    def search_employments(employee_id: int = 0) -> dict:
        """Search for employment records.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of employment records.
        """
        params = {"fields": "id,employee,startDate,endDate"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/employee/employment", params=params)

    return {
        "create_employment": create_employment,
        "search_employments": search_employments,
    }
