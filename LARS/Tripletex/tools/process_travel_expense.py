"""Compound tool: process entire travel expense workflow in one deterministic call.

Handles: create employee → create travel expense → add costs/mileage/per diem.
No LLM chaining required — all steps are hardcoded.
"""

import json as _json
import logging
from datetime import date as dt_date

from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


def build_process_travel_expense_tools(client: TripletexClient) -> dict:
    """Build the compound travel expense processing tool."""

    def _recover_error(endpoint: str):
        """Undo error count for auto-recovery on expected 422s."""
        client._error_count = max(0, client._error_count - 1)
        for entry in reversed(client._call_log):
            if not entry.get("ok") and endpoint in entry.get("url", ""):
                entry["ok"] = True
                entry["recovered"] = True
                break

    def process_travel_expense(
        employee_firstName: str,
        employee_lastName: str,
        employee_email: str,
        title: str,
        departureDate: str,
        returnDate: str,
        costs: str = "[]",
        mileage_km: float = 0,
        mileage_departure: str = "",
        mileage_destination: str = "",
        per_diem_rate: float = 0,
        per_diem_days: int = 0,
        per_diem_location: str = "",
        accommodation_nights: int = 0,
        accommodation_location: str = "",
        deduct_breakfast: bool = False,
        deduct_lunch: bool = False,
        deduct_dinner: bool = False,
    ) -> dict:
        """Process a complete travel expense workflow in one call.

        This compound tool handles the ENTIRE travel expense workflow:
        1. Creates the employee (or finds existing on email collision)
        2. Creates the travel expense report
        3. Adds cost lines (hotel, transport, food, other)
        4. Optionally adds mileage allowance
        5. Optionally adds per diem compensation
        6. Optionally adds accommodation allowance (nattillegg)

        Args:
            employee_firstName: Employee first name.
            employee_lastName: Employee last name.
            employee_email: Employee email.
            title: Travel expense report title.
            departureDate: Departure date YYYY-MM-DD.
            returnDate: Return date YYYY-MM-DD.
            costs: JSON string of costs: [{"amount": 1500, "category": "transport", "comments": "Flybillett", "date": "2026-01-15"}]
                   category must be: "transport", "food", "accommodation", or "other".
            mileage_km: Kilometers driven (0 to skip mileage).
            mileage_departure: Departure location for mileage.
            mileage_destination: Destination for mileage.
            per_diem_rate: Daily per diem rate in NOK (0 to skip per diem).
            per_diem_days: Number of days for per diem (0 = auto from dates).
            per_diem_location: Per diem location (default "Norge").
            accommodation_nights: Number of nights for accommodation allowance / nattillegg (0 to skip).
            accommodation_location: Location for accommodation allowance.
            deduct_breakfast: Deduct breakfast from per diem.
            deduct_lunch: Deduct lunch from per diem.
            deduct_dinner: Deduct dinner from per diem.

        Returns:
            Summary with employee_id, travel_expense_id, cost_ids.
        """
        today = dt_date.today().isoformat()
        steps_log = []

        # Parse costs JSON
        try:
            cost_list = _json.loads(costs) if isinstance(costs, str) else costs
        except (_json.JSONDecodeError, TypeError):
            cost_list = []

        # ── Step 1: Create employee ──
        dept_id = client.get_cached("default_department")
        if dept_id is None:
            dept_result = client.get("/department", params={"fields": "id", "count": 1})
            depts = dept_result.get("values", [])
            dept_id = depts[0]["id"] if depts else 0
            client.set_cached("default_department", dept_id)

        emp_body = {
            "firstName": employee_firstName,
            "lastName": employee_lastName,
            "email": employee_email,
            "userType": "STANDARD",
        }
        if dept_id:
            emp_body["department"] = {"id": dept_id}

        emp_result = client.post("/employee", json=emp_body)
        employee_id = None

        if emp_result.get("error") and emp_result.get("status_code") == 422:
            msg = str(emp_result.get("message", "")).lower()
            if "e-postadress" in msg or "email" in msg:
                existing = client.get("/employee", params={"email": employee_email, "fields": "id"})
                vals = existing.get("values", [])
                if vals:
                    employee_id = vals[0]["id"]
                    _recover_error("/employee")
                    steps_log.append(f"Employee already existed (id={employee_id})")

        if not employee_id:
            employee_id = emp_result.get("value", {}).get("id")
            if not employee_id:
                return {"error": True, "message": f"Failed to create employee: {emp_result}", "steps": steps_log}
            steps_log.append(f"Created employee (id={employee_id})")

        # ── Step 2: Create travel expense ──
        te_body = {
            "employee": {"id": employee_id},
            "title": title,
            "travelDetails": {
                "departureDate": departureDate,
                "returnDate": returnDate,
            },
        }
        te_result = client.post("/travelExpense", json=te_body)
        te_id = te_result.get("value", {}).get("id")
        if not te_id:
            return {"error": True, "message": f"Failed to create travel expense: {te_result}", "steps": steps_log}
        steps_log.append(f"Created travel expense '{title}' (id={te_id})")

        cost_ids = []

        # ── Step 3: Add cost lines ──
        if cost_list:
            # Resolve payment type (cached)
            payment_type_id = client.get_cached("travel_payment_type")
            if not payment_type_id:
                pt = client.get("/travelExpense/paymentType", params={"fields": "id", "count": 1})
                pts = pt.get("values", [])
                payment_type_id = pts[0]["id"] if pts else 0
                if payment_type_id:
                    client.set_cached("travel_payment_type", payment_type_id)

            for cost in cost_list:
                c_amount = cost.get("amount", 0)
                c_category = cost.get("category", "other")
                c_comments = cost.get("comments", "")
                c_date = cost.get("date", departureDate)
                c_vat_type = cost.get("vatType_id", 0)

                cost_body = {
                    "travelExpense": {"id": te_id},
                    "amountCurrencyIncVat": c_amount,
                    "paymentType": {"id": payment_type_id},
                }
                if c_category:
                    cost_body["category"] = c_category
                if c_comments:
                    cost_body["comments"] = c_comments
                if c_date:
                    cost_body["date"] = c_date
                if c_vat_type:
                    cost_body["vatType"] = {"id": c_vat_type}

                cost_result = client.post("/travelExpense/cost", json=cost_body)
                cost_id = cost_result.get("value", {}).get("id")
                if cost_id:
                    cost_ids.append(cost_id)
                    steps_log.append(f"Added cost: {c_comments or c_category} {c_amount} NOK (id={cost_id})")
                else:
                    steps_log.append(f"WARNING: Cost failed: {cost_result}")

        # ── Step 4: Mileage allowance ──
        if mileage_km > 0:
            mil_body = {
                "travelExpense": {"id": te_id},
                "date": departureDate,
                "km": mileage_km,
                "departureLocation": mileage_departure or "N/A",
                "destination": mileage_destination or "N/A",
            }
            mil_result = client.post("/travelExpense/mileageAllowance", json=mil_body)
            mil_id = mil_result.get("value", {}).get("id")
            if mil_id:
                steps_log.append(f"Added mileage: {mileage_km} km (id={mil_id})")
            else:
                steps_log.append(f"WARNING: Mileage failed: {mil_result}")

        # ── Step 5: Per diem compensation ──
        if per_diem_rate > 0:
            pd_body = {
                "travelExpense": {"id": te_id},
                "location": per_diem_location or "Norge",
                "rate": per_diem_rate,
            }
            if per_diem_days > 0:
                pd_body["count"] = per_diem_days
            if deduct_breakfast:
                pd_body["isDeductionForBreakfast"] = True
            if deduct_lunch:
                pd_body["isDeductionForLunch"] = True
            if deduct_dinner:
                pd_body["isDeductionForDinner"] = True

            pd_result = client.post("/travelExpense/perDiemCompensation", json=pd_body)
            pd_id = pd_result.get("value", {}).get("id")
            if pd_id:
                steps_log.append(f"Added per diem: {per_diem_rate} NOK/day (id={pd_id})")
            else:
                steps_log.append(f"WARNING: Per diem failed: {pd_result}")

        # ── Step 6: Accommodation allowance (nattillegg) ──
        if accommodation_nights > 0:
            acc_body = {
                "travelExpense": {"id": te_id},
                "count": accommodation_nights,
            }
            if accommodation_location:
                acc_body["location"] = accommodation_location

            acc_result = client.post("/travelExpense/accommodationAllowance", json=acc_body)
            acc_id = acc_result.get("value", {}).get("id")
            if acc_id:
                steps_log.append(f"Added accommodation allowance: {accommodation_nights} nights (id={acc_id})")
            else:
                steps_log.append(f"WARNING: Accommodation allowance failed: {acc_result}")

        return {
            "success": True,
            "employee_id": employee_id,
            "travel_expense_id": te_id,
            "cost_ids": cost_ids,
            "steps": steps_log,
        }

    return {
        "process_travel_expense": process_travel_expense,
    }
