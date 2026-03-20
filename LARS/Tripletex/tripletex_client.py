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
        log.info(f"[API #{self._call_count+1}] GET {url} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("GET", url, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "GET", url, elapsed, request_params=params)

    def post(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] POST {url} body={json}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("POST", url, json=json)
        elapsed = time.time() - t0
        return self._handle_response(resp, "POST", url, elapsed, request_body=json)

    def put(self, endpoint: str, json: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] PUT {url} body={json} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("PUT", url, json=json, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "PUT", url, elapsed, request_body=json)

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] DELETE {url}")
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
            if "expired" in body_text or "invalid" in body_text:
                log.error(f"[API] {method} {url} -> 401 (token expired/invalid, not retrying)")
                return resp
            log.warning(f"[API] {method} {url} -> 401, retrying once after 0.5s...")
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
            # Log success with response summary
            summary = str(result)[:300]
            log.info(f"[API] {method} {url} -> {resp.status_code} ({elapsed:.2f}s) {summary}")
            log.debug(f"[API DETAIL] {method} {url} request_body={request_body} response_body={result}")
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
            log.error(f"[API ERROR] {method} {url} -> {resp.status_code} ({elapsed:.2f}s) {full_msg}")
            log.debug(f"[API ERROR DETAIL] {method} {url} request_body={request_body} response_body={body or resp.text}")
            log_entry["ok"] = False
            log_entry["error"] = full_msg
            log_entry["response_body"] = body if body else resp.text[:2000]
            self._call_log.append(log_entry)
            if self._on_api_call:
                try: self._on_api_call(log_entry)
                except Exception: pass
            return {"error": True, "status_code": resp.status_code, "message": full_msg}
