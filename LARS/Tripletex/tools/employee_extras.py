from tripletex_client import TripletexClient


def build_employee_extras_tools(client: TripletexClient) -> dict:
    """Build extended employee tools (categories, next of kin, leave, hourly rates, standard time)."""

    def create_employee_category(
        name: str,
        number: str = "",
        description: str = "",
    ) -> dict:
        """Create an employee category.

        Args:
            name: Category name.
            number: Category number.
            description: Category description.

        Returns:
            Created category or error.
        """
        body = {"name": name}
        if number:
            body["number"] = number
        if description:
            body["description"] = description
        return client.post("/employee/category", json=body)

    def search_employee_categories() -> dict:
        """Search for employee categories.

        Returns:
            A list of employee categories.
        """
        return client.get("/employee/category", params={"fields": "id,name,number,description"})

    def delete_employee_category(category_id: int) -> dict:
        """Delete an employee category.

        Args:
            category_id: ID of the category.

        Returns:
            Confirmation or error.
        """
        return client.delete(f"/employee/category/{category_id}")

    def create_next_of_kin(
        employee_id: int,
        name: str,
        phoneNumber: str = "",
        typeOfRelationship: str = "",
    ) -> dict:
        """Create a next-of-kin record for an employee.

        Args:
            employee_id: ID of the employee.
            name: Name of the next of kin.
            phoneNumber: Phone number.
            typeOfRelationship: Relationship type (e.g. 'SPOUSE', 'PARENT', 'CHILD', 'SIBLING', 'OTHER').

        Returns:
            Created record or error.
        """
        body = {"employee": {"id": employee_id}, "name": name}
        if phoneNumber:
            body["phoneNumber"] = phoneNumber
        if typeOfRelationship:
            body["typeOfRelationship"] = typeOfRelationship
        return client.post("/employee/nextOfKin", json=body)

    def search_next_of_kin(employee_id: int = 0) -> dict:
        """Search for next-of-kin records.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of next-of-kin records.
        """
        params = {"fields": "id,employee,name,phoneNumber,typeOfRelationship"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/employee/nextOfKin", params=params)

    def create_leave_of_absence(
        employment_id: int,
        startDate: str,
        endDate: str = "",
        percentage: float = 100.0,
        leaveType: str = "MILITARY_SERVICE",
        isWageDeduction: bool = False,
    ) -> dict:
        """Create a leave of absence for an employment.

        Args:
            employment_id: ID of the employment record.
            startDate: Start date YYYY-MM-DD.
            endDate: End date YYYY-MM-DD (empty for ongoing).
            percentage: Leave percentage (default 100%).
            leaveType: Leave type string. Valid values: MILITARY_SERVICE, PARENTAL_LEAVE, EDUCATION, COMPASSIONATE, LEAVE_WITH_PAY, VOLUNTARY, FURLOUGH, OTHER.
            isWageDeduction: Whether there is wage deduction.

        Returns:
            Created leave record or error.
        """
        body = {
            "employment": {"id": employment_id},
            "startDate": startDate,
            "percentage": percentage,
            "type": leaveType,
            "isWageDeduction": isWageDeduction,
        }
        if endDate:
            body["endDate"] = endDate
        return client.post("/employee/employment/leaveOfAbsence", json=body)

    def create_hourly_cost_and_rate(
        employee_id: int,
        date: str,
        rate: float = 0.0,
        hourCostRate: float = 0.0,
    ) -> dict:
        """Create hourly cost and rate for an employee.

        Args:
            employee_id: ID of the employee.
            date: Effective date YYYY-MM-DD.
            rate: Hourly rate.
            hourCostRate: Hourly cost rate.

        Returns:
            Created record or error.
        """
        body = {"employee": {"id": employee_id}, "date": date}
        if rate:
            body["rate"] = rate
        if hourCostRate:
            body["hourCostRate"] = hourCostRate
        return client.post("/employee/hourlyCostAndRate", json=body)

    def create_standard_time(
        employee_id: int,
        fromDate: str,
        hoursPerDay: float = 7.5,
    ) -> dict:
        """Set standard working hours for an employee.

        Args:
            employee_id: ID of the employee.
            fromDate: Effective from date YYYY-MM-DD.
            hoursPerDay: Hours per day (default 7.5).

        Returns:
            Created record or error.
        """
        body = {"employee": {"id": employee_id}, "fromDate": fromDate, "hoursPerDay": hoursPerDay}
        return client.post("/employee/standardTime", json=body)

    def create_employment_details(
        employment_id: int,
        date: str,
        employmentType: str = "ORDINARY",
        workingHoursScheme: str = "NOT_SHIFT",
        annualSalary: float = 0.0,
        percentageOfFullTimeEquivalent: float = 100.0,
    ) -> dict:
        """Create employment details (salary, work arrangement).

        Args:
            employment_id: ID of the employment.
            date: Effective date YYYY-MM-DD.
            employmentType: Type - 'ORDINARY', 'MARITIME', 'FREELANCE'.
            workingHoursScheme: Scheme - 'NOT_SHIFT', 'ROUND_THE_CLOCK', 'SHIFT_365', 'OFFSHORE_336', 'CONTINUOUS', 'OTHER_SHIFT'.
            annualSalary: Annual salary amount.
            percentageOfFullTimeEquivalent: FTE percentage.

        Returns:
            Created details or error.
        """
        body = {
            "employment": {"id": employment_id},
            "date": date,
            "employmentType": employmentType,
            "workingHoursScheme": workingHoursScheme,
            "percentageOfFullTimeEquivalent": percentageOfFullTimeEquivalent,
        }
        if annualSalary:
            body["annualSalary"] = annualSalary
        return client.post("/employee/employment/details", json=body)

    def grant_entitlements(
        employee_id: int,
        template: str = "all_access",
    ) -> dict:
        """Grant entitlements to an employee using a template.

        Args:
            employee_id: ID of the employee.
            template: Template name - 'all_access', 'standard', 'extended'.

        Returns:
            Confirmation or error.
        """
        return client.put(
            "/employee/entitlement/client/:grantClientEntitlementsByTemplate",
            params={"employeeId": employee_id, "template": template},
        )

    return {
        "create_employee_category": create_employee_category,
        "search_employee_categories": search_employee_categories,
        "delete_employee_category": delete_employee_category,
        "create_next_of_kin": create_next_of_kin,
        "search_next_of_kin": search_next_of_kin,
        "create_leave_of_absence": create_leave_of_absence,
        "create_hourly_cost_and_rate": create_hourly_cost_and_rate,
        "create_standard_time": create_standard_time,
        "create_employment_details": create_employment_details,
        "grant_entitlements": grant_entitlements,
    }
