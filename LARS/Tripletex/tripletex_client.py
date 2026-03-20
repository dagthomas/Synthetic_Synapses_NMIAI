import logging
import time
import requests

log = logging.getLogger(__name__)


class TripletexClient:
    """Thin wrapper around Tripletex REST API with auth and logging."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._call_count = 0
        self._error_count = 0
        self._call_log: list[dict] = []
        # Per-request cache for frequently-accessed IDs (avoids redundant GETs)
        self._cache: dict[str, int] = {}

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] GET {url} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("GET", url, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "GET", url, elapsed)

    def post(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] POST {url} body={json}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("POST", url, json=json)
        elapsed = time.time() - t0
        return self._handle_response(resp, "POST", url, elapsed)

    def put(self, endpoint: str, json: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] PUT {url} body={json} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = self._do_request("PUT", url, json=json, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "PUT", url, elapsed)

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
            )
            if div_result.status_code == 200:
                divs = div_result.json().get("values", [])
                if divs:
                    self._cache["default_division"] = divs[0]["id"]
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

    def _handle_response(self, resp: requests.Response, method: str, url: str, elapsed: float) -> dict:
        try:
            resp.raise_for_status()
            result = resp.json() if resp.text else {"ok": True}
            # Log success with response summary
            summary = str(result)[:300]
            log.info(f"[API] {method} {url} -> {resp.status_code} ({elapsed:.2f}s) {summary}")
            self._call_log.append({
                "method": method, "url": url, "status": resp.status_code,
                "elapsed": round(elapsed, 3), "ok": True,
            })
            return result
        except Exception:
            self._error_count += 1
            try:
                body = resp.json()
                # Include full validation details from Tripletex
                msg = body.get("message", body.get("error", resp.text))
                validation = body.get("validationMessages", [])
                details = "; ".join(
                    f"{v.get('field', '?')}: {v.get('message', v)}"
                    for v in validation
                ) if validation else ""
                full_msg = f"{msg} [{details}]" if details else str(msg)
            except Exception:
                full_msg = resp.text[:500]
            log.error(f"[API ERROR] {method} {url} -> {resp.status_code} ({elapsed:.2f}s) {full_msg}")
            self._call_log.append({
                "method": method, "url": url, "status": resp.status_code,
                "elapsed": round(elapsed, 3), "ok": False, "error": full_msg,
            })
            return {"error": True, "status_code": resp.status_code, "message": full_msg}
