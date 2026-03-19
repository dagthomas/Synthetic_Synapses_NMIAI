import logging
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
        log.info(f"GET {url} params={params}")
        self._call_count += 1
        resp = requests.get(url, auth=self.auth, params=params)
        return self._handle_response(resp)

    def post(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"POST {url} body={json}")
        self._call_count += 1
        resp = requests.post(url, auth=self.auth, json=json)
        return self._handle_response(resp)

    def put(self, endpoint: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"PUT {url} body={json}")
        self._call_count += 1
        resp = requests.put(url, auth=self.auth, json=json)
        return self._handle_response(resp)

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}{endpoint}"
        log.info(f"DELETE {url}")
        self._call_count += 1
        resp = requests.delete(url, auth=self.auth)
        return self._handle_response(resp)

    def _handle_response(self, resp: requests.Response) -> dict:
        try:
            resp.raise_for_status()
            return resp.json() if resp.text else {"ok": True}
        except Exception:
            self._error_count += 1
            try:
                body = resp.json()
                msg = body.get("message", body.get("error", resp.text))
            except Exception:
                msg = resp.text
            log.warning(f"API error {resp.status_code}: {msg}")
            return {"error": True, "status_code": resp.status_code, "message": str(msg)}
