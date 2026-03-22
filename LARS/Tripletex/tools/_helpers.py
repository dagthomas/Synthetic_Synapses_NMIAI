"""Shared helper functions for compound tools.

Implements search-before-create patterns to avoid 422 errors
and the associated 0.15 efficiency penalty per error.
"""

import logging

log = logging.getLogger(__name__)


def recover_error(client, endpoint: str):
    """Undo error count for auto-recovery on expected 422s."""
    client._error_count = max(0, client._error_count - 1)
    for entry in reversed(client._call_log):
        if not entry.get("ok") and endpoint in entry.get("url", ""):
            entry["ok"] = True
            entry["recovered"] = True
            break


def resolve_vat_map(client) -> dict:
    """Get output VAT type map (percentage → id), from cache or live lookup.

    Returns dict like {25: 123, 15: 456, 12: 789, 0: 101}.
    """
    vat_map = client.get_cached("vat_type_map") or {}
    if not vat_map:
        _OUT = {3: 25, 31: 15, 33: 12}
        _ZERO = {6, 5}
        r = client.get("/ledger/vatType", params={"fields": "id,number"})
        for vt in (r.get("values") or []):
            n, vid = vt.get("number"), vt.get("id")
            if n is not None and vid is not None:
                try:
                    n = int(n)
                except (ValueError, TypeError):
                    continue
                if n in _OUT:
                    vat_map[_OUT[n]] = vid
                elif n in _ZERO:
                    vat_map.setdefault(0, vid)
        if vat_map:
            client.set_cached("vat_type_map", vat_map)
    return vat_map


def find_or_create_product(client, name: str, price: float = 0.0,
                           vat_percentage: int = 25, product_number: str = "",
                           steps_log: list | None = None) -> int | None:
    """Search for a product by name/number first, create only if not found.

    Returns product ID or None on failure.
    Eliminates ~280 errors (product name + product number collisions).
    """
    if steps_log is None:
        steps_log = []

    # ── Search first by name (exact match) ──
    existing = client.get("/product", params={
        "name": name, "fields": "id,name,number"
    })
    for v in (existing.get("values") or []):
        if (v.get("name") or "").strip().lower() == name.strip().lower():
            steps_log.append(f"Found existing product '{name}' (id={v['id']})")
            return v["id"]

    # ── Search by product number if provided ──
    if product_number:
        by_num = client.get("/product", params={
            "productNumber": product_number, "fields": "id,name"
        })
        vals = by_num.get("values", [])
        if vals:
            steps_log.append(f"Found existing product by number '{product_number}' (id={vals[0]['id']})")
            return vals[0]["id"]

    # ── Create new product ──
    body = {"name": name}
    if price:
        body["priceExcludingVatCurrency"] = price
    if product_number:
        body["number"] = product_number

    # Set VAT type if available and not previously rejected
    if not client.get_cached("skip_vat_type"):
        vat_map = resolve_vat_map(client)
        vat_type_id = vat_map.get(int(vat_percentage))
        if vat_type_id is not None:
            body["vatType"] = {"id": vat_type_id}

    result = client.post("/product", json=body)
    prod_id = result.get("value", {}).get("id")

    if not prod_id and result.get("error"):
        err_msg = str(result.get("message", "")).lower()

        # VAT type rejected — retry without it
        if "vatType" in body and ("mva-kode" in err_msg or "vattype" in err_msg):
            body.pop("vatType", None)
            client.set_cached("skip_vat_type", True)
            recover_error(client, "/product")
            result = client.post("/product", json=body)
            prod_id = result.get("value", {}).get("id")

        # Name/number collision (race condition or fuzzy match we missed)
        if not prod_id and result.get("error"):
            err_msg = str(result.get("message", "")).lower()
            if "allerede registrert" in err_msg or "already" in err_msg:
                recover_error(client, "/product")
                # Try name search again
                retry = client.get("/product", params={"name": name, "fields": "id,name"})
                for v in (retry.get("values") or []):
                    if (v.get("name") or "").strip().lower() == name.strip().lower():
                        prod_id = v["id"]
                        break
            if not prod_id and product_number:
                recover_error(client, "/product")
                retry = client.get("/product", params={"productNumber": product_number, "fields": "id"})
                vals = retry.get("values", [])
                if vals:
                    prod_id = vals[0]["id"]

    if prod_id:
        steps_log.append(f"Created product '{name}' (id={prod_id})")
    return prod_id


def find_or_create_customer(client, name: str, email: str = "",
                            org_number: str = "", phone: str = "",
                            address: str = "", postal_code: str = "",
                            city: str = "",
                            steps_log: list | None = None) -> int | None:
    """Search for a customer by name first, create only if not found.

    Returns customer ID or None on failure.
    """
    if steps_log is None:
        steps_log = []

    # ── Search first by name (exact match) ──
    search_params = {"fields": "id,name,email,organizationNumber"}
    if org_number:
        search_params["organizationNumber"] = org_number
    else:
        search_params["name"] = name

    existing = client.get("/customer", params=search_params)
    target = name.strip().lower()
    for v in (existing.get("values") or []):
        if (v.get("name") or "").strip().lower() == target:
            steps_log.append(f"Found existing customer '{name}' (id={v['id']})")
            return v["id"]
    # If org search returned results but no exact name match, also check by name
    if org_number and not any("Found existing" in s for s in steps_log):
        name_search = client.get("/customer", params={"name": name, "fields": "id,name"})
        for v in (name_search.get("values") or []):
            if (v.get("name") or "").strip().lower() == target:
                steps_log.append(f"Found existing customer '{name}' (id={v['id']})")
                return v["id"]

    # ── Create new customer ──
    body = {"name": name, "isCustomer": True}
    if email:
        body["email"] = email
    if org_number:
        body["organizationNumber"] = org_number
    if phone:
        body["phoneNumber"] = phone
    if address or postal_code or city:
        addr = {}
        if address:
            addr["addressLine1"] = address
        if postal_code:
            addr["postalCode"] = postal_code
        if city:
            addr["city"] = city
        body["physicalAddress"] = addr

    result = client.post("/customer", json=body)
    customer_id = result.get("value", {}).get("id")

    if not customer_id and result.get("error"):
        # 422 collision — search again (race condition)
        recover_error(client, "/customer")
        search = client.get("/customer", params={"name": name, "fields": "id,name"})
        for v in (search.get("values") or []):
            if (v.get("name") or "").strip().lower() == target:
                customer_id = v["id"]
                steps_log.append(f"Customer already existed '{name}' (id={customer_id})")
                return customer_id
        # Retry without org number (common collision source)
        if org_number:
            body.pop("organizationNumber", None)
            result = client.post("/customer", json=body)
            customer_id = result.get("value", {}).get("id")

    if customer_id:
        if not any("Customer" in s for s in steps_log):
            steps_log.append(f"Created customer '{name}' (id={customer_id})")
    return customer_id


def find_or_create_supplier(client, name: str, email: str = "",
                            org_number: str = "", bank_account: str = "",
                            address: str = "", postal_code: str = "",
                            city: str = "",
                            steps_log: list | None = None) -> int | None:
    """Search for a supplier by name first, create only if not found.

    Returns supplier ID or None on failure.
    """
    if steps_log is None:
        steps_log = []

    # ── Search first by name ──
    search_params = {"fields": "id,name,organizationNumber,version"}
    if org_number:
        search_params["organizationNumber"] = org_number
    else:
        search_params["name"] = name

    existing = client.get("/supplier", params=search_params)
    target = name.strip().lower()
    for v in (existing.get("values") or []):
        if (v.get("name") or "").strip().lower() == target:
            steps_log.append(f"Found existing supplier '{name}' (id={v['id']})")
            return v["id"]
    # If org search returned results but no exact name match, also check by name
    if org_number:
        name_search = client.get("/supplier", params={"name": name, "fields": "id,name"})
        for v in (name_search.get("values") or []):
            if (v.get("name") or "").strip().lower() == target:
                steps_log.append(f"Found existing supplier '{name}' (id={v['id']})")
                return v["id"]

    # ── Create new supplier ──
    body = {"name": name, "isSupplier": True}
    if email:
        body["email"] = email
    if org_number:
        body["organizationNumber"] = org_number
    if address or postal_code or city:
        addr = {}
        if address:
            addr["addressLine1"] = address
        if postal_code:
            addr["postalCode"] = postal_code
        if city:
            addr["city"] = city
        body["postalAddress"] = addr
        body["physicalAddress"] = addr

    result = client.post("/supplier", json=body)
    supplier_id = result.get("value", {}).get("id")

    if not supplier_id and result.get("error"):
        recover_error(client, "/supplier")
        search = client.get("/supplier", params={"name": name, "fields": "id,name"})
        for v in (search.get("values") or []):
            if (v.get("name") or "").strip().lower() == target:
                supplier_id = v["id"]
                steps_log.append(f"Supplier already existed '{name}' (id={supplier_id})")
                return supplier_id
        if org_number:
            body.pop("organizationNumber", None)
            result = client.post("/supplier", json=body)
            supplier_id = result.get("value", {}).get("id")

    if supplier_id:
        if not any("Supplier" in s or "supplier" in s for s in steps_log):
            steps_log.append(f"Created supplier '{name}' (id={supplier_id})")
    return supplier_id


def find_or_create_employment(client, employee_id: int, start_date: str,
                              steps_log: list | None = None,
                              annual_salary: float = 0.0,
                              percentage: float = 100.0,
                              skip_dob_check: bool = False) -> int | None:
    """Search for existing employment first, create only if not found.

    Returns employment ID or None on failure.
    Eliminates ~54 errors (overlapping employment).
    """
    if steps_log is None:
        steps_log = []

    # ── Search first ──
    existing = client.get("/employee/employment", params={
        "employeeId": employee_id, "fields": "id,startDate,endDate"
    })
    vals = existing.get("values", [])
    if vals:
        steps_log.append(f"Employment already exists for employee {employee_id} (id={vals[0]['id']})")
        return vals[0]["id"]

    # ── Ensure dateOfBirth (required for employment) ──
    if not skip_dob_check:
        emp = client.get(f"/employee/{employee_id}", params={"fields": "id,dateOfBirth"})
        emp_val = emp.get("value", emp)
        if not emp_val.get("dateOfBirth"):
            client.put(f"/employee/{employee_id}", json={"dateOfBirth": "1990-01-01"})

    # ── Get or create division ──
    div_id = client.get_cached("default_division")
    if div_id is None:
        divs = client.get("/division", params={"fields": "id", "count": 1})
        div_list = divs.get("values", [])
        div_id = div_list[0]["id"] if div_list else 0
        client.set_cached("default_division", div_id)

    # ── Create employment ──
    details = {
        "date": start_date,
        "employmentType": "ORDINARY",
        "workingHoursScheme": "NOT_SHIFT",
        "percentageOfFullTimeEquivalent": percentage,
    }
    if annual_salary:
        details["annualSalary"] = annual_salary

    emp_body = {
        "employee": {"id": employee_id},
        "startDate": start_date,
        "employmentDetails": [details],
    }
    if div_id:
        emp_body["division"] = {"id": div_id}

    result = client.post("/employee/employment", json=emp_body)
    employment_id = result.get("value", {}).get("id")

    if not employment_id:
        # 422 overlap — search again (race condition)
        recover_error(client, "/employee/employment")
        existing = client.get("/employee/employment", params={
            "employeeId": employee_id, "fields": "id"
        })
        vals = existing.get("values", [])
        if vals:
            employment_id = vals[0]["id"]
            steps_log.append(f"Employment already existed (id={employment_id})")
        else:
            steps_log.append(f"WARNING: Failed to create employment for employee {employee_id}: {result}")
    else:
        steps_log.append(f"Created employment (id={employment_id})")

    return employment_id


def find_or_create_department(client, name: str,
                              steps_log: list | None = None) -> int:
    """Search for department by name first, create only if not found.

    Returns department ID or 0 on failure.
    Eliminates ~41 errors (department name collisions).
    """
    if steps_log is None:
        steps_log = []

    # ── Search first ──
    existing = client.get("/department", params={
        "name": name, "fields": "id,name", "count": 5
    })
    target = name.strip().lower()
    for v in (existing.get("values") or []):
        if (v.get("name") or "").strip().lower() == target:
            steps_log.append(f"Found existing department '{name}' (id={v['id']})")
            return v["id"]

    # ── Create ──
    result = client.post("/department", json={"name": name})
    dept_id = result.get("value", {}).get("id")

    if not dept_id and result.get("error"):
        # 422 collision — search again
        recover_error(client, "/department")
        retry = client.get("/department", params={"name": name, "fields": "id,name"})
        for v in (retry.get("values") or []):
            if (v.get("name") or "").strip().lower() == target:
                dept_id = v["id"]
                steps_log.append(f"Department already existed '{name}' (id={dept_id})")
                return dept_id

    if dept_id:
        steps_log.append(f"Created department '{name}' (id={dept_id})")
    return dept_id or 0


def ensure_bank_account(client) -> None:
    """Ensure ledger account 1920 has a bank account number (once per session)."""
    cache_key = "bank_account_ensured"
    if client.get_cached(cache_key):
        return
    try:
        r = client.get("/ledger/account", params={
            "number": "1920", "fields": "id,name,isBankAccount,bankAccountNumber"
        })
        accounts = r.get("values", [])
        if not accounts:
            client.set_cached(cache_key, True)
            return
        acct = accounts[0]
        if acct.get("bankAccountNumber"):
            client.set_cached(cache_key, True)
            return
        client.put(f"/ledger/account/{acct['id']}", json={
            "id": acct["id"],
            "number": 1920,
            "name": acct["name"],
            "isBankAccount": True,
            "bankAccountNumber": "12345678903",
        })
        client.set_cached(cache_key, True)
    except Exception:
        pass


def resolve_payment_type_id(client) -> int:
    """Resolve paymentTypeId — prefer 'Betalt til bank' (cached)."""
    pt_id = client.get_cached("payment_type_bank")
    if pt_id:
        return pt_id
    pt_result = client.get("/invoice/paymentType", params={
        "fields": "id,description", "count": 10,
    })
    payment_types = pt_result.get("values", [])
    pt_id = 0
    for pt in payment_types:
        if "bank" in pt.get("description", "").lower():
            pt_id = pt["id"]
            break
    if not pt_id and payment_types:
        pt_id = payment_types[0]["id"]
    if pt_id:
        client.set_cached("payment_type_bank", pt_id)
    return pt_id
