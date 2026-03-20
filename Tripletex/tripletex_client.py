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

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] GET {url} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = requests.get(url, auth=self.auth, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "GET", url, elapsed)

    def post(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] POST {url} body={json}")
        self._call_count += 1
        t0 = time.time()
        resp = requests.post(url, auth=self.auth, json=json)
        elapsed = time.time() - t0
        return self._handle_response(resp, "POST", url, elapsed)

    def put(self, endpoint: str, json: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] PUT {url} body={json} params={params}")
        self._call_count += 1
        t0 = time.time()
        resp = requests.put(url, auth=self.auth, json=json, params=params)
        elapsed = time.time() - t0
        return self._handle_response(resp, "PUT", url, elapsed)

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"[API #{self._call_count+1}] DELETE {url}")
        self._call_count += 1
        t0 = time.time()
        resp = requests.delete(url, auth=self.auth)
        elapsed = time.time() - t0
        return self._handle_response(resp, "DELETE", url, elapsed)

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
