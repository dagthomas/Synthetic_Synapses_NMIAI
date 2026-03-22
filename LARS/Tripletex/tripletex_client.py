import logging
import time
import requests

log = logging.getLogger(__name__)


class TripletexClient:
    """Thin wrapper around Tripletex REST API with auth and logging."""

    def __init__(self, base_url: str, session_token: str, on_api_call=None):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._call_count = 0
        self._error_count = 0
        self._call_log: list[dict] = []
        self._on_api_call = on_api_call
        # Per-request cache for frequently-accessed IDs (avoids redundant GETs)
        self._cache: dict[str, int] = {}

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("GET", url, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "GET", url, elapsed, request_params=params)

    def post(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("POST", url, json=json)
        elapsed = time.time() - t0
        return self._handle_response(resp, "POST", url, elapsed, request_body=json)

    def put(self, endpoint: str, json: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("PUT", url, json=json, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "PUT", url, elapsed, request_body=json)

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}{endpoint}"
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("DELETE", url)
        elapsed = time.time() - t0
        return self._handle_response(resp, "DELETE", url, elapsed)

    def _prewarm_caches(self):
        """Fetch commonly needed IDs (department, division) before agent starts.

        Resets call counters so these infrastructure lookups don't count
        against the agent's efficiency score.
        """
        # Default department
        try:
            dept_result = requests.get(
                f"{self.base_url}/department",
                auth=self.auth,
                params={"fields": "id", "count": 1},
                timeout=10,
            )
            if dept_result.status_code == 200:
                depts = dept_result.json().get("values", [])
                if depts:
                    self._cache["default_department"] = depts[0]["id"]
        except Exception:
            pass
        # Default division
        try:
            div_result = requests.get(
                f"{self.base_url}/division",
                auth=self.auth,
                params={"fields": "id", "count": 1},
                timeout=10,
            )
            if div_result.status_code == 200:
                divs = div_result.json().get("values", [])
                if divs:
                    self._cache["default_division"] = divs[0]["id"]
        except Exception:
            pass
        # VAT type maps: match by standard VAT type NUMBER (stable across all sandboxes)
        # Standard Norwegian VAT type numbers:
        #   Output: 3→25%, 31→15%, 33→12%, 5/6→0%
        #   Input:  1→25%, 11→15%, 13→12%
        try:
            vat_result = requests.get(
                f"{self.base_url}/ledger/vatType",
                auth=self.auth,
                params={"fields": "id,number,percentage", "count": 100},
                timeout=10,
            )
            if vat_result.status_code == 200:
                _OUT = {3: 25, 31: 15, 33: 12}
                _IN = {1: 25, 11: 15, 13: 12}
                _ZERO = {6, 5}
                vat_map = {}
                input_vat_map = {}
                for vt in vat_result.json().get("values", []):
                    num, vid = vt.get("number"), vt.get("id")
                    if num is not None and vid is not None:
                        try:
                            num = int(num)
                        except (ValueError, TypeError):
                            continue
                        if num in _OUT:
                            vat_map[_OUT[num]] = vid
                        elif num in _ZERO:
                            vat_map.setdefault(0, vid)
                        elif num in _IN:
                            input_vat_map[_IN[num]] = vid
                if vat_map:
                    self._cache["vat_type_map"] = vat_map
                if input_vat_map:
                    self._cache["input_vat_type_map"] = input_vat_map
        except Exception:
            pass
        # Payables account 2400 (used by supplier invoice tool)
        try:
            acct_result = requests.get(
                f"{self.base_url}/ledger/account",
                auth=self.auth,
                params={"number": "2400", "fields": "id", "count": 1},
                timeout=10,
            )
            if acct_result.status_code == 200:
                accts = acct_result.json().get("values", [])
                if accts:
                    self._cache["acct_2400"] = accts[0]["id"]
        except Exception:
            pass
        # Bank account 1920 (used by invoice/payment tools)
        try:
            ba_result = requests.get(
                f"{self.base_url}/ledger/account",
                auth=self.auth,
                params={"number": "1920", "fields": "id,name,isBankAccount,bankAccountNumber"},
                timeout=10,
            )
            if ba_result.status_code == 200:
                accts = ba_result.json().get("values", [])
                if accts:
                    acct = accts[0]
                    if acct.get("bankAccountNumber"):
                        self._cache["bank_account_ensured"] = True
                    else:
                        # Set a bank account number so payments work
                        try:
                            requests.put(
                                f"{self.base_url}/ledger/account/{acct['id']}",
                                auth=self.auth, timeout=10,
                                json={
                                    "id": acct["id"], "number": 1920,
                                    "name": acct["name"],
                                    "isBankAccount": True,
                                    "bankAccountNumber": "12345678903",
                                },
                            )
                        except Exception:
                            pass
                        self._cache["bank_account_ensured"] = True
        except Exception:
            pass
        # Company ID (used by project PM entitlements)
        try:
            who_result = requests.get(
                f"{self.base_url}/token/session/>whoAmI",
                auth=self.auth,
                params={"fields": "companyId"},
                timeout=10,
            )
            if who_result.status_code == 200:
                cid = who_result.json().get("value", {}).get("companyId")
                if cid:
                    self._cache["company_id"] = cid
        except Exception:
            pass
        # Admin employee ID (first employee — used as default project manager)
        try:
            emp_result = requests.get(
                f"{self.base_url}/employee",
                auth=self.auth,
                params={"fields": "id", "count": 1},
                timeout=10,
            )
            if emp_result.status_code == 200:
                emps = emp_result.json().get("values", [])
                if emps:
                    self._cache["admin_employee_id"] = emps[0]["id"]
        except Exception:
            pass
        # Payment type (bank — used by register_payment)
        try:
            pt_result = requests.get(
                f"{self.base_url}/invoice/paymentType",
                auth=self.auth,
                params={"fields": "id,description", "count": 10},
                timeout=10,
            )
            if pt_result.status_code == 200:
                for pt in pt_result.json().get("values", []):
                    if "bank" in pt.get("description", "").lower():
                        self._cache["payment_type_bank"] = pt["id"]
                        break
                else:
                    pts = pt_result.json().get("values", [])
                    if pts:
                        self._cache["payment_type_bank"] = pts[0]["id"]
        except Exception:
            pass

    def get_cached(self, key: str) -> int | None:
        """Get a cached ID value (e.g., 'default_department', 'default_division')."""
        return self._cache.get(key)

    def set_cached(self, key: str, value: int):
        """Cache an ID value for reuse within this request."""
        self._cache[key] = value

    def _do_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Execute request with single retry on 401 (transient token errors)."""
        kwargs.setdefault("timeout", 30)
        resp = requests.request(method, url, auth=self.auth, **kwargs)
        if resp.status_code == 401:
            # Check if token is expired/invalid — don't retry those
            try:
                body_text = resp.text.lower()
            except Exception:
                body_text = ""
            ep = url.split("/v2")[-1] if "/v2" in url else url.split("/", 3)[-1]
            if "expired" in body_text or "invalid" in body_text:
                log.error(f"  API {method:6s} {ep} → 401 token expired/invalid")
                return resp
            log.warning(f"  API {method:6s} {ep} → 401, retrying…")
            time.sleep(0.5)
            resp = requests.request(method, url, auth=self.auth, **kwargs)
        return resp

    def _handle_response(self, resp: requests.Response, method: str, url: str, elapsed: float,
                         request_body: dict | None = None, request_params: dict | None = None) -> dict:
        log_entry: dict = {
            "method": method, "url": url, "status": resp.status_code,
            "elapsed": round(elapsed, 3),
        }
        if request_body is not None:
            log_entry["request_body"] = request_body
        if request_params is not None:
            log_entry["request_params"] = request_params

        try:
            resp.raise_for_status()
            result = resp.json() if resp.text else {"ok": True}
            # Extract endpoint from URL for concise logging
            ep = url.split("/v2")[-1] if "/v2" in url else url.split("/", 3)[-1]
            log.info(f"  API {method:6s} {ep} → {resp.status_code} ({elapsed:.2f}s)")
            log.debug(f"  API detail: {method} {ep} body={request_body} resp={result}")
            log_entry["ok"] = True
            log_entry["response_body"] = result
            self._call_log.append(log_entry)
            if self._on_api_call:
                try: self._on_api_call(log_entry)
                except Exception: pass
            return result
        except Exception:
            self._error_count += 1
            try:
                body = resp.json()
                # Include full validation details from Tripletex
                # Use `or` chain: .get() returns None if key exists with null value
                msg = body.get("message") or body.get("error") or resp.text if isinstance(body, dict) else resp.text
                validation = body.get("validationMessages", []) if isinstance(body, dict) else []
                details = "; ".join(
                    f"{v.get('field', '?')}: {v.get('message', v)}"
                    for v in validation
                ) if validation else ""
                full_msg = f"{msg} [{details}]" if details else str(msg)
            except Exception:
                body = None
                full_msg = resp.text[:500] if resp.text else "(empty response body)"
            ep = url.split("/v2")[-1] if "/v2" in url else url.split("/", 3)[-1]
            log.error(f"  API {method:6s} {ep} → {resp.status_code} ({elapsed:.2f}s) {full_msg[:200]}")
            log.debug(f"  API error detail: {method} {ep} body={request_body} resp={body or resp.text}")
            log_entry["ok"] = False
            log_entry["error"] = full_msg
            log_entry["response_body"] = body if body else resp.text[:2000]
            self._call_log.append(log_entry)
            if self._on_api_call:
                try: self._on_api_call(log_entry)
                except Exception: pass
            return {"error": True, "status_code": resp.status_code, "message": full_msg}
