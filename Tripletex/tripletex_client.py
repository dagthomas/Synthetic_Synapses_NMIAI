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
            return result
        except Exception:
            self._error_count += 1
            try:
                body = resp.json()
                msg = body.get("message", body.get("error", resp.text))
            except Exception:
                msg = resp.text
            log.error(f"[API ERROR] {method} {url} -> {resp.status_code} ({elapsed:.2f}s) {msg}")
            return {"error": True, "status_code": resp.status_code, "message": str(msg)}
