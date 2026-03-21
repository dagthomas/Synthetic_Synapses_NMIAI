import time
from collections import deque

import requests

from config import ASTAR_TOKEN, BASE_URL


class RateLimiter:
    """Deque-based sliding window rate limiter."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def wait(self):
        now = time.monotonic()
        # Remove expired timestamps
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.calls.append(time.monotonic())


class AstarIslandClient:
    """API client for Astar Island with rate limiting and budget tracking."""

    def __init__(self, token: str = None):
        self.token = token or ASTAR_TOKEN
        if not self.token:
            raise ValueError("ASTAR_TOKEN not set. Set env var or pass token directly.")

        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

        self._simulate_limiter = RateLimiter(max_calls=5, period=1.0)
        self._submit_limiter = RateLimiter(max_calls=2, period=1.0)

        self._queries_used = 0
        self._queries_max = 50

        # Verify auth with generous retry
        for attempt in range(5):
            try:
                self.get_rounds()
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Auth verification failed: {e}")

    def _get(self, path: str, retries: int = 5, **kwargs) -> dict | list:
        url = f"{BASE_URL}/astar-island{path}"
        for attempt in range(retries):
            resp = self.session.get(url, **kwargs)
            if resp.status_code == 429 and attempt < retries - 1:
                wait = 2 ** attempt + 1
                print(f"  Rate limited on GET, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, json_data: dict, limiter: RateLimiter = None, retries: int = 3) -> dict:
        url = f"{BASE_URL}/astar-island{path}"
        for attempt in range(retries):
            if limiter:
                limiter.wait()
            resp = self.session.post(url, json=json_data)
            if resp.status_code == 429 and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()

    # --- Free endpoints (no query cost) ---

    def get_rounds(self) -> list:
        """List all rounds."""
        return self._get("/rounds")

    def get_active_round(self) -> dict | None:
        """Find the currently active round, or None."""
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> dict:
        """Get round details including initial states for all seeds."""
        return self._get(f"/rounds/{round_id}")

    def get_budget(self) -> dict:
        """Get remaining query budget for active round."""
        data = self._get("/budget")
        self._queries_used = data.get("queries_used", 0)
        self._queries_max = data.get("queries_max", 50)
        return data

    def get_my_rounds(self) -> list:
        """Get rounds with your scores, rank, budget."""
        return self._get("/my-rounds")

    def get_my_predictions(self, round_id: str) -> list:
        """Get your predictions with argmax/confidence for a round."""
        return self._get(f"/my-predictions/{round_id}")

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        """Get post-round ground truth comparison."""
        return self._get(f"/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self) -> list:
        """Get the leaderboard."""
        return self._get("/leaderboard")

    # --- Budget-consuming ---

    def simulate(self, round_id: str, seed_index: int, x: int = 0, y: int = 0,
                 w: int = 15, h: int = 15) -> dict:
        """Run one simulation query. Costs 1 query from budget.

        Args:
            round_id: UUID of active round
            seed_index: 0-4
            x, y: Top-left corner of viewport
            w, h: Viewport dimensions (5-15)
        """
        remaining = self._queries_max - self._queries_used
        if remaining <= 0:
            raise RuntimeError("Query budget exhausted (0 remaining)")
        if remaining <= 10:
            print(f"  WARNING: Only {remaining} queries remaining!")

        data = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        result = self._post("/simulate", data, limiter=self._simulate_limiter)
        self._queries_used = result.get("queries_used", self._queries_used + 1)
        self._queries_max = result.get("queries_max", self._queries_max)
        return result

    # --- Submission ---

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        """Submit prediction for one seed.

        Args:
            round_id: UUID of active round
            seed_index: 0-4
            prediction: H×W×6 probability tensor as nested list
        """
        data = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        return self._post("/submit", data, limiter=self._submit_limiter)

    # --- Budget tracking ---

    @property
    def queries_remaining(self) -> int:
        return self._queries_max - self._queries_used

    @property
    def queries_used(self) -> int:
        return self._queries_used

    def refresh_budget(self) -> int:
        """Refresh budget from API and return remaining queries."""
        self.get_budget()
        return self.queries_remaining
