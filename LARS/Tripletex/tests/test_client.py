from unittest.mock import patch, MagicMock
from tripletex_client import TripletexClient


def test_get_sends_auth_and_params():
    client = TripletexClient("https://example.com/v2", "test-token")
    with patch("tripletex_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"fullResultSize": 1, "values": [{"id": 1}]},
            text='{"fullResultSize": 1, "values": [{"id": 1}]}',
            raise_for_status=lambda: None,
        )
        result = client.get("/employee", params={"firstName": "Ola"})
        mock_get.assert_called_once_with(
            "https://example.com/v2/employee",
            auth=("0", "test-token"),
            params={"firstName": "Ola"},
        )
        assert result == {"fullResultSize": 1, "values": [{"id": 1}]}


def test_post_sends_json_body():
    client = TripletexClient("https://example.com/v2", "test-token")
    with patch("tripletex_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"value": {"id": 42, "name": "Acme"}},
            text='{"value": {"id": 42, "name": "Acme"}}',
            raise_for_status=lambda: None,
        )
        result = client.post("/customer", json={"name": "Acme"})
        mock_post.assert_called_once_with(
            "https://example.com/v2/customer",
            auth=("0", "test-token"),
            json={"name": "Acme"},
        )
        assert result == {"value": {"id": 42, "name": "Acme"}}


def test_error_returns_message():
    client = TripletexClient("https://example.com/v2", "test-token")
    with patch("tripletex_client.requests.post") as mock_post:
        resp = MagicMock(
            status_code=422,
            text='{"message": "Field email is required"}',
        )
        resp.json.return_value = {"message": "Field email is required"}
        resp.raise_for_status.side_effect = Exception("422")
        mock_post.return_value = resp
        result = client.post("/employee", json={"firstName": "Ola"})
        assert result["error"] is True
        assert "email is required" in result["message"]
