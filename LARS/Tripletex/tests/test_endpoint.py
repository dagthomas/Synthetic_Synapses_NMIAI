"""
End-to-end test: send a task prompt to /solve and verify it works.
Requires: server running on localhost:8000, valid .env with GOOGLE_API_KEY.

For real sandbox testing, set TRIPLETEX_BASE_URL and TRIPLETEX_TOKEN env vars.
"""
import os
import requests

ENDPOINT = "http://localhost:8000/solve"
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")

# Use real sandbox if available, otherwise mock
BASE_URL = os.environ.get("TRIPLETEX_BASE_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
TOKEN = os.environ.get("TRIPLETEX_TOKEN", "fake-token-for-smoke-test")


def test_create_employee():
    """Test: create a simple employee."""
    headers = {}
    if AGENT_API_KEY:
        headers["Authorization"] = f"Bearer {AGENT_API_KEY}"

    payload = {
        "prompt": "Opprett en ansatt med navn Ola Nordmann og epostadresse ola@example.no.",
        "files": [],
        "tripletex_credentials": {
            "base_url": BASE_URL,
            "session_token": TOKEN,
        },
    }

    resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


if __name__ == "__main__":
    test_create_employee()
    print("Test passed!")
