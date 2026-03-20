from unittest.mock import MagicMock
from tools.employees import build_employee_tools


def test_create_employee_calls_post():
    client = MagicMock()
    client.post.return_value = {"value": {"id": 1, "firstName": "Ola", "lastName": "Nordmann"}}
    tools = build_employee_tools(client)
    create_fn = tools["create_employee"]
    result = create_fn(firstName="Ola", lastName="Nordmann", email="ola@test.no")
    client.post.assert_called_once()
    args = client.post.call_args
    assert args[0][0] == "/employee"
    assert args[1]["json"]["firstName"] == "Ola"
    assert "id" in str(result)


def test_search_employees_calls_get():
    client = MagicMock()
    client.get.return_value = {"fullResultSize": 1, "values": [{"id": 1, "firstName": "Ola"}]}
    tools = build_employee_tools(client)
    search_fn = tools["search_employees"]
    result = search_fn(firstName="Ola")
    client.get.assert_called_once()
    assert len(result["values"]) == 1
