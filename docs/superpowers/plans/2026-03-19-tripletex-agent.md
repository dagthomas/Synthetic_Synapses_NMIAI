# Tripletex AI Accounting Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Google ADK agent that completes accounting tasks in Tripletex via a ReAct tool-calling loop, served as a FastAPI endpoint.

**Architecture:** Single ADK Agent (Gemini 2.5 Pro) with ~25 entity-specific tools wrapping the Tripletex REST API. FastAPI receives `/solve` POST requests, creates per-request agent instances with credentials injected via closures, and returns `{"status": "completed"}`.

**Tech Stack:** Python 3.11+, Google ADK (`google-adk`), Gemini 2.5 Pro, FastAPI, uvicorn, pdfplumber, requests, python-dotenv

**Spec:** `docs/superpowers/specs/2026-03-19-tripletex-agent-design.md`

---

## File Structure

```
Tripletex/
├── main.py                  # FastAPI app, /solve endpoint, agent orchestration
├── agent.py                 # ADK Agent factory, system instruction
├── tripletex_client.py      # HTTP wrapper: auth, logging, error parsing
├── tools/
│   ├── __init__.py          # Re-exports all tool builder functions
│   ├── employees.py         # create_employee, update_employee, search_employees
│   ├── customers.py         # create_customer, update_customer, search_customers
│   ├── products.py          # create_product, search_products
│   ├── invoicing.py         # create_order, create_invoice, register_payment, create_credit_note
│   ├── travel.py            # create_travel_expense, delete_travel_expense, search_travel_expenses
│   ├── projects.py          # create_project
│   ├── departments.py       # create_department
│   ├── ledger.py            # get_ledger_accounts, get_ledger_postings, create_voucher, delete_voucher
│   ├── files.py             # extract_file_content (PDF + image)
│   └── common.py            # get_entity_by_id, delete_entity, enable_module
├── config.py                # Load env vars
├── requirements.txt
├── .env.example             # Template for env vars
└── tests/
    ├── test_client.py       # TripletexClient unit tests
    ├── test_tools.py        # Tool function unit tests (mocked client)
    └── test_endpoint.py     # /solve integration test
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `Tripletex/requirements.txt`
- Create: `Tripletex/.env.example`
- Create: `Tripletex/config.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi
uvicorn[standard]
google-adk
google-genai
pdfplumber
Pillow
requests
python-dotenv
```

- [ ] **Step 2: Create .env.example**

```
GOOGLE_API_KEY=your-google-api-key
AGENT_API_KEY=optional-bearer-token-to-protect-solve-endpoint
```

- [ ] **Step 3: Create config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"
MAX_AGENT_TURNS = 25
```

- [ ] **Step 4: Create .env with real key**

Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY`.

- [ ] **Step 5: Create tests/__init__.py and conftest.py**

```python
# tests/__init__.py (empty)
```

```python
# tests/conftest.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

- [ ] **Step 6: Install dependencies**

Run: `cd Tripletex && pip install -r requirements.txt`

- [ ] **Step 7: Verify ADK installation**

Run: `python -c "from google.adk.agents import LlmAgent; print('ADK OK')"`
Expected: `ADK OK`

- [ ] **Step 8: Commit**

```bash
git add Tripletex/requirements.txt Tripletex/.env.example Tripletex/config.py Tripletex/tests/
git commit -m "feat(tripletex): project setup with dependencies and config"
```

---

### Task 2: Tripletex HTTP Client

**Files:**
- Create: `Tripletex/tripletex_client.py`
- Create: `Tripletex/tests/test_client.py`

- [ ] **Step 1: Write failing tests for TripletexClient**

```python
# tests/test_client.py
import json
from unittest.mock import patch, MagicMock
from tripletex_client import TripletexClient


def test_get_sends_auth_and_params():
    client = TripletexClient("https://example.com/v2", "test-token")
    with patch("tripletex_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"fullResultSize": 1, "values": [{"id": 1}]},
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Tripletex && python -m pytest tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripletex_client'`

- [ ] **Step 3: Implement TripletexClient**

```python
# tripletex_client.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Tripletex && python -m pytest tests/test_client.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add Tripletex/tripletex_client.py Tripletex/tests/test_client.py
git commit -m "feat(tripletex): HTTP client with auth, logging, error handling"
```

---

### Task 3: Core Tools — Employees, Customers, Products

**Files:**
- Create: `Tripletex/tools/__init__.py`
- Create: `Tripletex/tools/employees.py`
- Create: `Tripletex/tools/customers.py`
- Create: `Tripletex/tools/products.py`
- Create: `Tripletex/tools/common.py`
- Create: `Tripletex/tests/test_tools.py`

- [ ] **Step 1: Write failing tests for employee tools**

```python
# tests/test_tools.py
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
    assert "id" in result


def test_search_employees_calls_get():
    client = MagicMock()
    client.get.return_value = {"fullResultSize": 1, "values": [{"id": 1, "firstName": "Ola"}]}
    tools = build_employee_tools(client)
    search_fn = tools["search_employees"]
    result = search_fn(firstName="Ola")
    client.get.assert_called_once()
    assert len(result["values"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Tripletex && python -m pytest tests/test_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tools/common.py**

```python
# tools/common.py
from tripletex_client import TripletexClient


def build_common_tools(client: TripletexClient) -> dict:
    """Build common utility tools."""

    def get_entity_by_id(entity_type: str, entity_id: int) -> dict:
        """Retrieve any Tripletex entity by type and ID.

        Args:
            entity_type: The API entity type, e.g. 'employee', 'customer', 'invoice'.
            entity_id: The numeric ID of the entity.

        Returns:
            The entity data or an error message.
        """
        return client.get(f"/{entity_type}/{entity_id}", params={"fields": "*"})

    def delete_entity(entity_type: str, entity_id: int) -> dict:
        """Delete a Tripletex entity by type and ID.

        Args:
            entity_type: The API entity type, e.g. 'travelExpense', 'ledger/voucher'.
            entity_id: The numeric ID of the entity to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/{entity_type}/{entity_id}")

    def enable_module(module_name: str) -> dict:
        """Enable a Tripletex module. Call this if a task fails because a module is not activated.

        Args:
            module_name: The module to enable, e.g. 'moduleDepartment', 'moduleProjectEconomy'.

        Returns:
            Confirmation or error message.
        """
        return client.put("/company/modules", json={module_name: True})

    return {
        "get_entity_by_id": get_entity_by_id,
        "delete_entity": delete_entity,
        "enable_module": enable_module,
    }
```

- [ ] **Step 4: Implement tools/employees.py**

```python
# tools/employees.py
from tripletex_client import TripletexClient


def build_employee_tools(client: TripletexClient) -> dict:
    """Build employee-related tools as closures over the client."""

    def create_employee(
        firstName: str,
        lastName: str,
        email: str,
        isAdministrator: bool = False,
        phoneNumberMobile: str = "",
    ) -> dict:
        """Create a new employee in Tripletex.

        Args:
            firstName: The employee's first name.
            lastName: The employee's last name.
            email: The employee's email address.
            isAdministrator: Whether the employee should be an account administrator.
            phoneNumberMobile: The employee's mobile phone number.

        Returns:
            The created employee with id and fields, or an error message.
        """
        body = {
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
        }
        if isAdministrator:
            body["isAdministrator"] = True
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        return client.post("/employee", json=body)

    def update_employee(employee_id: int, firstName: str = "", lastName: str = "", email: str = "", phoneNumberMobile: str = "") -> dict:
        """Update an existing employee's fields.

        Args:
            employee_id: The ID of the employee to update.
            firstName: New first name (leave empty to keep current).
            lastName: New last name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumberMobile: New phone number (leave empty to keep current).

        Returns:
            The updated employee data or an error message.
        """
        body = {}
        if firstName:
            body["firstName"] = firstName
        if lastName:
            body["lastName"] = lastName
        if email:
            body["email"] = email
        if phoneNumberMobile:
            body["phoneNumberMobile"] = phoneNumberMobile
        return client.put(f"/employee/{employee_id}", json=body)

    def search_employees(firstName: str = "", lastName: str = "", email: str = "") -> dict:
        """Search for employees by name or email.

        Args:
            firstName: Filter by first name (partial match).
            lastName: Filter by last name (partial match).
            email: Filter by email (partial match).

        Returns:
            A list of matching employees with id, firstName, lastName, email.
        """
        params = {"fields": "id,firstName,lastName,email"}
        if firstName:
            params["firstName"] = firstName
        if lastName:
            params["lastName"] = lastName
        if email:
            params["email"] = email
        return client.get("/employee", params=params)

    return {
        "create_employee": create_employee,
        "update_employee": update_employee,
        "search_employees": search_employees,
    }
```

- [ ] **Step 5: Implement tools/customers.py**

```python
# tools/customers.py
from tripletex_client import TripletexClient


def build_customer_tools(client: TripletexClient) -> dict:
    """Build customer-related tools."""

    def create_customer(
        name: str,
        email: str = "",
        isCustomer: bool = True,
        isSupplier: bool = False,
        phoneNumber: str = "",
        organizationNumber: str = "",
    ) -> dict:
        """Create a new customer (or supplier) in Tripletex.

        Args:
            name: The company or person name.
            email: Contact email address.
            isCustomer: Whether this is a customer.
            isSupplier: Whether this is a supplier.
            phoneNumber: Contact phone number.
            organizationNumber: Norwegian org number.

        Returns:
            The created customer with id and fields, or an error message.
        """
        body = {"name": name, "isCustomer": isCustomer}
        if email:
            body["email"] = email
        if isSupplier:
            body["isSupplier"] = True
        if phoneNumber:
            body["phoneNumber"] = phoneNumber
        if organizationNumber:
            body["organizationNumber"] = organizationNumber
        return client.post("/customer", json=body)

    def update_customer(customer_id: int, name: str = "", email: str = "", phoneNumber: str = "") -> dict:
        """Update an existing customer's fields.

        Args:
            customer_id: The ID of the customer to update.
            name: New name (leave empty to keep current).
            email: New email (leave empty to keep current).
            phoneNumber: New phone number (leave empty to keep current).

        Returns:
            The updated customer data or an error message.
        """
        body = {}
        if name:
            body["name"] = name
        if email:
            body["email"] = email
        if phoneNumber:
            body["phoneNumber"] = phoneNumber
        return client.put(f"/customer/{customer_id}", json=body)

    def search_customers(name: str = "", email: str = "") -> dict:
        """Search for customers by name or email.

        Args:
            name: Filter by customer name (partial match).
            email: Filter by email (partial match).

        Returns:
            A list of matching customers with id, name, email.
        """
        params = {"fields": "id,name,email,isCustomer,isSupplier"}
        if name:
            params["name"] = name
        if email:
            params["email"] = email
        return client.get("/customer", params=params)

    return {
        "create_customer": create_customer,
        "update_customer": update_customer,
        "search_customers": search_customers,
    }
```

- [ ] **Step 6: Implement tools/products.py**

```python
# tools/products.py
from tripletex_client import TripletexClient


def build_product_tools(client: TripletexClient) -> dict:
    """Build product-related tools."""

    def create_product(
        name: str,
        priceExcludingVat: float = 0.0,
        priceIncludingVat: float = 0.0,
        productNumber: str = "",
    ) -> dict:
        """Create a new product in Tripletex.

        Args:
            name: The product name.
            priceExcludingVat: Price excluding VAT.
            priceIncludingVat: Price including VAT.
            productNumber: Optional product number/SKU.

        Returns:
            The created product with id and fields, or an error message.
        """
        body = {"name": name}
        if priceExcludingVat:
            body["priceExcludingVat"] = priceExcludingVat
        if priceIncludingVat:
            body["priceIncludingVat"] = priceIncludingVat
        if productNumber:
            body["number"] = productNumber
        return client.post("/product", json=body)

    def search_products(name: str = "") -> dict:
        """Search for products by name.

        Args:
            name: Filter by product name (partial match).

        Returns:
            A list of matching products with id, name, number, price fields.
        """
        params = {"fields": "id,name,number,priceExcludingVat,priceIncludingVat"}
        if name:
            params["name"] = name
        return client.get("/product", params=params)

    return {
        "create_product": create_product,
        "search_products": search_products,
    }
```

- [ ] **Step 7: Create tools/__init__.py**

```python
# tools/__init__.py
from tripletex_client import TripletexClient
from tools.employees import build_employee_tools
from tools.customers import build_customer_tools
from tools.products import build_product_tools
from tools.common import build_common_tools


def build_all_tools(client: TripletexClient) -> list:
    """Build all tool functions as closures over the given client.

    Returns a flat list of callable tool functions for ADK Agent.
    """
    all_tools = {}
    all_tools.update(build_employee_tools(client))
    all_tools.update(build_customer_tools(client))
    all_tools.update(build_product_tools(client))
    all_tools.update(build_common_tools(client))
    return list(all_tools.values())
```

- [ ] **Step 8: Run tests**

Run: `cd Tripletex && python -m pytest tests/test_tools.py -v`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add Tripletex/tools/
git commit -m "feat(tripletex): employee, customer, product, common tools"
```

---

### Task 4: Invoicing Tools

**Files:**
- Create: `Tripletex/tools/invoicing.py`
- Modify: `Tripletex/tools/__init__.py` — add invoicing import

- [ ] **Step 1: Implement tools/invoicing.py**

```python
# tools/invoicing.py
from tripletex_client import TripletexClient


def build_invoicing_tools(client: TripletexClient) -> dict:
    """Build invoicing-related tools."""

    def create_order(
        customer_id: int,
        deliveryDate: str,
        orderLines: str,
    ) -> dict:
        """Create a sales order for a customer. Required before creating an invoice.

        Args:
            customer_id: The ID of the customer (must exist).
            deliveryDate: Delivery date in YYYY-MM-DD format.
            orderLines: JSON string of order lines, each with 'product_id', 'count', and optionally 'unitPriceExcludingVat'. Example: '[{"product_id": 1, "count": 2}]'

        Returns:
            The created order with id, or an error message.
        """
        import json
        lines = json.loads(orderLines) if isinstance(orderLines, str) else orderLines
        formatted_lines = []
        for line in lines:
            entry = {
                "product": {"id": line["product_id"]},
                "count": line["count"],
            }
            if "unitPriceExcludingVat" in line:
                entry["unitPriceExcludingVat"] = line["unitPriceExcludingVat"]
            formatted_lines.append(entry)

        body = {
            "customer": {"id": customer_id},
            "deliveryDate": deliveryDate,
            "orderLines": formatted_lines,
        }
        return client.post("/order", json=body)

    def create_invoice(
        invoiceDate: str,
        invoiceDueDate: str,
        order_id: int,
    ) -> dict:
        """Create an invoice from an existing order.

        Args:
            invoiceDate: Invoice date in YYYY-MM-DD format.
            invoiceDueDate: Payment due date in YYYY-MM-DD format.
            order_id: The ID of the order to invoice.

        Returns:
            The created invoice with id, or an error message.
        """
        body = {
            "invoiceDate": invoiceDate,
            "invoiceDueDate": invoiceDueDate,
            "orders": [{"id": order_id}],
        }
        return client.post("/invoice", json=body)

    def register_payment(
        invoice_id: int,
        amount: float,
        paymentDate: str,
    ) -> dict:
        """Register a payment for an invoice.

        Args:
            invoice_id: The ID of the invoice being paid.
            amount: The payment amount.
            paymentDate: Payment date in YYYY-MM-DD format.

        Returns:
            Confirmation of payment or an error message.
        """
        body = {
            "invoice": {"id": invoice_id},
            "amount": amount,
            "date": paymentDate,
        }
        return client.post("/payment", json=body)

    def create_credit_note(invoice_id: int) -> dict:
        """Create a credit note for an existing invoice.

        Args:
            invoice_id: The ID of the invoice to credit.

        Returns:
            The created credit note or an error message.
        """
        return client.post(f"/invoice/{invoice_id}/:createCreditNote", json={})

    return {
        "create_order": create_order,
        "create_invoice": create_invoice,
        "register_payment": register_payment,
        "create_credit_note": create_credit_note,
    }
```

- [ ] **Step 2: Update tools/__init__.py to include invoicing**

Add `from tools.invoicing import build_invoicing_tools` and `all_tools.update(build_invoicing_tools(client))`.

- [ ] **Step 3: Commit**

```bash
git add Tripletex/tools/invoicing.py Tripletex/tools/__init__.py
git commit -m "feat(tripletex): invoicing tools (order, invoice, payment, credit note)"
```

---

### Task 5: Travel, Projects, Departments, Ledger Tools

**Files:**
- Create: `Tripletex/tools/travel.py`
- Create: `Tripletex/tools/projects.py`
- Create: `Tripletex/tools/departments.py`
- Create: `Tripletex/tools/ledger.py`
- Modify: `Tripletex/tools/__init__.py` — add all imports

- [ ] **Step 1: Implement tools/travel.py**

```python
# tools/travel.py
from tripletex_client import TripletexClient


def build_travel_tools(client: TripletexClient) -> dict:
    """Build travel expense tools."""

    def create_travel_expense(
        employee_id: int,
        title: str,
        departureDate: str,
        returnDate: str,
        description: str = "",
    ) -> dict:
        """Create a travel expense report.

        Args:
            employee_id: The ID of the employee filing the expense.
            title: Title of the travel expense report.
            departureDate: Departure date in YYYY-MM-DD format.
            returnDate: Return date in YYYY-MM-DD format.
            description: Optional description of the travel.

        Returns:
            The created travel expense with id, or an error message.
        """
        body = {
            "employee": {"id": employee_id},
            "title": title,
            "departureDate": departureDate,
            "returnDate": returnDate,
        }
        if description:
            body["description"] = description
        return client.post("/travelExpense", json=body)

    def delete_travel_expense(travel_expense_id: int) -> dict:
        """Delete a travel expense report.

        Args:
            travel_expense_id: The ID of the travel expense to delete.

        Returns:
            Confirmation of deletion or an error message.
        """
        return client.delete(f"/travelExpense/{travel_expense_id}")

    def search_travel_expenses(employee_id: int = 0) -> dict:
        """Search for travel expense reports.

        Args:
            employee_id: Filter by employee ID (0 for all).

        Returns:
            A list of travel expenses.
        """
        params = {"fields": "id,title,employee,departureDate,returnDate"}
        if employee_id:
            params["employeeId"] = employee_id
        return client.get("/travelExpense", params=params)

    return {
        "create_travel_expense": create_travel_expense,
        "delete_travel_expense": delete_travel_expense,
        "search_travel_expenses": search_travel_expenses,
    }
```

- [ ] **Step 2: Implement tools/projects.py**

```python
# tools/projects.py
from tripletex_client import TripletexClient


def build_project_tools(client: TripletexClient) -> dict:
    """Build project tools."""

    def create_project(
        name: str,
        customer_id: int = 0,
        projectManagerId: int = 0,
        startDate: str = "",
        description: str = "",
    ) -> dict:
        """Create a project in Tripletex.

        Args:
            name: Project name.
            customer_id: ID of the customer linked to this project (0 if none).
            projectManagerId: ID of the employee managing the project (0 if none).
            startDate: Project start date in YYYY-MM-DD format.
            description: Optional project description.

        Returns:
            The created project with id, or an error message.
        """
        body = {"name": name}
        if customer_id:
            body["customer"] = {"id": customer_id}
        if projectManagerId:
            body["projectManager"] = {"id": projectManagerId}
        if startDate:
            body["startDate"] = startDate
        if description:
            body["description"] = description
        return client.post("/project", json=body)

    return {"create_project": create_project}
```

- [ ] **Step 3: Implement tools/departments.py**

```python
# tools/departments.py
from tripletex_client import TripletexClient


def build_department_tools(client: TripletexClient) -> dict:
    """Build department tools."""

    def create_department(
        name: str,
        departmentNumber: str = "",
    ) -> dict:
        """Create a department in Tripletex. May require enabling the department module first.

        Args:
            name: Department name.
            departmentNumber: Optional department number/code.

        Returns:
            The created department with id, or an error message.
        """
        body = {"name": name}
        if departmentNumber:
            body["number"] = departmentNumber
        return client.post("/department", json=body)

    return {"create_department": create_department}
```

- [ ] **Step 4: Implement tools/ledger.py**

```python
# tools/ledger.py
from tripletex_client import TripletexClient


def build_ledger_tools(client: TripletexClient) -> dict:
    """Build ledger tools for Tier 3 tasks."""

    def get_ledger_accounts(number: str = "", name: str = "") -> dict:
        """Search the chart of accounts.

        Args:
            number: Filter by account number (partial match).
            name: Filter by account name (partial match).

        Returns:
            A list of matching ledger accounts.
        """
        params = {"fields": "id,number,name,description"}
        if number:
            params["number"] = number
        if name:
            params["name"] = name
        return client.get("/ledger/account", params=params)

    def get_ledger_postings(dateFrom: str, dateTo: str, accountNumber: str = "") -> dict:
        """Query ledger postings within a date range.

        Args:
            dateFrom: Start date in YYYY-MM-DD format.
            dateTo: End date in YYYY-MM-DD format.
            accountNumber: Optional filter by account number.

        Returns:
            A list of ledger postings.
        """
        params = {
            "dateFrom": dateFrom,
            "dateTo": dateTo,
            "fields": "id,date,description,amount,account",
        }
        if accountNumber:
            params["accountNumber"] = accountNumber
        return client.get("/ledger/posting", params=params)

    def create_voucher(date: str, description: str, postings: str) -> dict:
        """Create a ledger voucher with postings. Used for corrections and manual entries.

        Args:
            date: Voucher date in YYYY-MM-DD format.
            description: Description of the voucher.
            postings: JSON string of postings, each with 'accountNumber', 'debitAmount', 'creditAmount'. Example: '[{"accountNumber": 1920, "debitAmount": 1000, "creditAmount": 0}]'

        Returns:
            The created voucher with id, or an error message.
        """
        import json
        posting_list = json.loads(postings) if isinstance(postings, str) else postings
        formatted = []
        for p in posting_list:
            formatted.append({
                "account": {"number": p["accountNumber"]},
                "debitAmount": p.get("debitAmount", 0),
                "creditAmount": p.get("creditAmount", 0),
            })
        body = {
            "date": date,
            "description": description,
            "postings": formatted,
        }
        return client.post("/ledger/voucher", json=body)

    def delete_voucher(voucher_id: int) -> dict:
        """Delete a ledger voucher.

        Args:
            voucher_id: The ID of the voucher to delete.

        Returns:
            Confirmation or error message.
        """
        return client.delete(f"/ledger/voucher/{voucher_id}")

    return {
        "get_ledger_accounts": get_ledger_accounts,
        "get_ledger_postings": get_ledger_postings,
        "create_voucher": create_voucher,
        "delete_voucher": delete_voucher,
    }
```

- [ ] **Step 5: Update tools/__init__.py with all imports**

```python
# tools/__init__.py
from tripletex_client import TripletexClient
from tools.employees import build_employee_tools
from tools.customers import build_customer_tools
from tools.products import build_product_tools
from tools.invoicing import build_invoicing_tools
from tools.travel import build_travel_tools
from tools.projects import build_project_tools
from tools.departments import build_department_tools
from tools.ledger import build_ledger_tools
from tools.common import build_common_tools


def build_all_tools(client: TripletexClient) -> list:
    """Build all tool functions as closures over the given client."""
    all_tools = {}
    all_tools.update(build_employee_tools(client))
    all_tools.update(build_customer_tools(client))
    all_tools.update(build_product_tools(client))
    all_tools.update(build_invoicing_tools(client))
    all_tools.update(build_travel_tools(client))
    all_tools.update(build_project_tools(client))
    all_tools.update(build_department_tools(client))
    all_tools.update(build_ledger_tools(client))
    all_tools.update(build_common_tools(client))
    return list(all_tools.values())
```

- [ ] **Step 6: Commit**

```bash
git add Tripletex/tools/
git commit -m "feat(tripletex): travel, project, department, ledger tools"
```

---

### Task 6: File Extraction Tool

**Files:**
- Create: `Tripletex/tools/files.py`
- Modify: `Tripletex/tools/__init__.py` — add files import

- [ ] **Step 1: Implement tools/files.py**

```python
# tools/files.py
import os
import logging
import pdfplumber
from PIL import Image
import google.genai as genai

from config import GEMINI_MODEL

log = logging.getLogger(__name__)


def build_file_tools(files_dir: str) -> dict:
    """Build file extraction tools. files_dir is the per-request temp directory."""

    def extract_file_content(filename: str) -> dict:
        """Extract text content from a PDF or image file attachment.

        Args:
            filename: The name of the attached file to extract text from. Supports PDF and image files (PNG, JPG, JPEG).

        Returns:
            The extracted text content, or an error message.
        """
        filepath = os.path.join(files_dir, filename)
        if not os.path.exists(filepath):
            return {"error": True, "message": f"File not found: {filename}"}

        lower = filename.lower()
        if lower.endswith(".pdf"):
            return _extract_pdf(filepath)
        elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return _extract_image(filepath)
        else:
            return {"error": True, "message": f"Unsupported file type: {filename}. Supported: .pdf, .png, .jpg"}

    return {"extract_file_content": extract_file_content}


def _extract_pdf(filepath: str) -> dict:
    """Extract text from PDF. Falls back to Gemini vision for scanned PDFs."""
    try:
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            return {"text": full_text}

        # Fallback: scanned PDF — render pages to images and use Gemini vision
        log.info("PDF has no text layer, falling back to Gemini vision")
        return _extract_pdf_with_vision(filepath)
    except Exception as e:
        return {"error": True, "message": f"Failed to extract PDF: {str(e)}"}


def _extract_pdf_with_vision(filepath: str) -> dict:
    """Render PDF pages to images and extract text with Gemini vision."""
    try:
        images = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=200).original
                images.append(img)

        client = genai.Client()
        texts = []
        for img in images:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=["Extract ALL text from this image exactly as written. Preserve formatting, numbers, and special characters:", img],
            )
            texts.append(response.text)
        return {"text": "\n\n".join(texts)}
    except Exception as e:
        return {"error": True, "message": f"Vision extraction failed: {str(e)}"}


def _extract_image(filepath: str) -> dict:
    """Extract text from image using Gemini vision."""
    try:
        img = Image.open(filepath)
        client = genai.Client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=["Extract ALL text from this image exactly as written. Preserve formatting, numbers, and special characters:", img],
        )
        return {"text": response.text}
    except Exception as e:
        return {"error": True, "message": f"Image extraction failed: {str(e)}"}
```

- [ ] **Step 2: Update tools/__init__.py**

Add `from tools.files import build_file_tools` and modify `build_all_tools` to accept `files_dir` parameter:

```python
def build_all_tools(client: TripletexClient, files_dir: str = "") -> list:
    """Build all tool functions as closures over the given client."""
    all_tools = {}
    all_tools.update(build_employee_tools(client))
    all_tools.update(build_customer_tools(client))
    all_tools.update(build_product_tools(client))
    all_tools.update(build_invoicing_tools(client))
    all_tools.update(build_travel_tools(client))
    all_tools.update(build_project_tools(client))
    all_tools.update(build_department_tools(client))
    all_tools.update(build_ledger_tools(client))
    all_tools.update(build_common_tools(client))
    if files_dir:
        all_tools.update(build_file_tools(files_dir))
    return list(all_tools.values())
```

- [ ] **Step 3: Commit**

```bash
git add Tripletex/tools/files.py Tripletex/tools/__init__.py
git commit -m "feat(tripletex): PDF file extraction tool"
```

---

### Task 7: ADK Agent Factory

**Files:**
- Create: `Tripletex/agent.py`

- [ ] **Step 1: Implement agent.py with system instruction**

```python
# agent.py
from google.adk.agents import LlmAgent
from config import GEMINI_MODEL, MAX_AGENT_TURNS

SYSTEM_INSTRUCTION = """You are an expert accounting assistant that executes tasks in Tripletex, Norway's cloud accounting platform.

You receive a task prompt in one of 7 languages: Norwegian (bokmal), English, Spanish, Portuguese, Nynorsk, German, or French. Regardless of language, you must:

1. Read the task carefully and identify exactly what needs to be done
2. Create prerequisites first if needed (e.g. create customer before invoice, create product before order)
3. Execute the task using your available tools
4. The account starts COMPLETELY EMPTY — everything must be created from scratch

CRITICAL RULES:
- Use EXACTLY the names, emails, amounts, dates etc. from the task prompt. Never guess or invent values.
- Do not specify fields that are not mentioned in the prompt.
- Norwegian characters (ae, oe, aa) must be preserved exactly as given.
- If a tool returns an error, read the error message carefully and correct your input in ONE retry. Do not retry more than once.
- EFFICIENCY MATTERS: Do not make unnecessary API calls. Do not call search/get to verify something you just created — the create response already confirms it.

COMMON TASK PATTERNS:
- Create employee: create_employee directly
- Create customer: create_customer directly
- Create invoice: create_customer -> create_product -> create_order -> create_invoice
- Register payment: (find or create invoice) -> register_payment
- Credit note: (find invoice) -> create_credit_note
- Travel expense: (find or create employee) -> create_travel_expense
- Create project: (find or create customer) -> create_project
- Create department: enable_module('moduleDepartment') if needed -> create_department
- Ledger corrections: get_ledger_accounts -> get_ledger_postings -> create_voucher or delete_voucher
- Delete/reverse: search for entity -> delete_entity or delete-specific tool

If a tool fails with a module-not-enabled error, call enable_module with the relevant module name and retry.

When you have completed the task, stop calling tools. Do not make verification calls."""


def create_agent(tools: list) -> LlmAgent:
    """Create an ADK agent with the given tools."""
    return LlmAgent(
        name="tripletex_accountant",
        model=GEMINI_MODEL,
        description="AI accounting assistant for Tripletex tasks",
        instruction=SYSTEM_INSTRUCTION,
        tools=tools,
    )
```

- [ ] **Step 2: Verify import works**

Run: `cd Tripletex && python -c "from agent import create_agent; print('Agent module OK')"`
Expected: `Agent module OK`

- [ ] **Step 3: Commit**

```bash
git add Tripletex/agent.py
git commit -m "feat(tripletex): ADK agent factory with system instruction"
```

---

### Task 8: FastAPI Endpoint

**Files:**
- Create: `Tripletex/main.py`

- [ ] **Step 1: Implement main.py**

```python
# main.py
import base64
import logging
import os
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from agent import create_agent
from config import AGENT_API_KEY, GOOGLE_API_KEY, MAX_AGENT_TURNS
from tripletex_client import TripletexClient
from tools import build_all_tools

# Set Google API key for ADK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")


@app.post("/solve")
async def solve(request: Request):
    # Optional Bearer token auth
    if AGENT_API_KEY:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {AGENT_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    # Per-request isolation
    request_id = str(uuid.uuid4())
    files_dir = os.path.join("/tmp", f"tripletex_{request_id}")

    # Decode attachments
    file_names = []
    if files:
        os.makedirs(files_dir, exist_ok=True)
        for f in files:
            data = base64.b64decode(f["content_base64"])
            filepath = os.path.join(files_dir, f["filename"])
            with open(filepath, "wb") as fh:
                fh.write(data)
            file_names.append(f["filename"])
            log.info(f"Saved attachment: {f['filename']} ({len(data)} bytes)")

    # Build per-request client and tools
    client = TripletexClient(creds["base_url"], creds["session_token"])
    tools = build_all_tools(client, files_dir=files_dir if files else "")

    # Build agent and runner
    agent = create_agent(tools)
    runner = InMemoryRunner(agent=agent)

    # Build user message
    user_text = prompt
    if file_names:
        user_text += f"\n\nAttached files: {', '.join(file_names)}"
        user_text += "\nUse extract_file_content to read file contents."

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_text)],
    )

    # Pre-activate common modules (1 API call, prevents module-not-enabled errors)
    client.put("/company/modules", json={
        "moduleDepartment": True,
        "moduleProjectEconomy": True,
    })

    # Run agent with turn limit
    log.info(f"Running agent for request {request_id}")
    log.info(f"Prompt: {prompt[:200]}...")

    final_text = ""
    turn_count = 0
    try:
        async for event in runner.run_async(
            user_id="tripletex",
            session_id=request_id,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text = part.text
                    if hasattr(part, "function_call") and part.function_call:
                        turn_count += 1
                        log.info(f"Tool call #{turn_count}: {part.function_call.name}({part.function_call.args})")
                        if turn_count >= MAX_AGENT_TURNS:
                            log.warning(f"Turn limit reached ({MAX_AGENT_TURNS}), stopping agent")
                            break
            if turn_count >= MAX_AGENT_TURNS:
                break
    except Exception as e:
        log.error(f"Agent error: {e}")

    log.info(f"Agent done. API calls: {client._call_count}, errors: {client._error_count}")
    log.info(f"Final response: {final_text[:200]}")

    # Cleanup temp files
    if files_dir and os.path.exists(files_dir):
        import shutil
        shutil.rmtree(files_dir, ignore_errors=True)

    return JSONResponse({"status": "completed"})
```

- [ ] **Step 2: Verify ADK runner API (async vs sync)**

Run: `cd Tripletex && python -c "from google.adk.runners import InMemoryRunner; import inspect; print('run_async' if hasattr(InMemoryRunner, 'run_async') else 'run')"`

If output is `run` (not `run_async`), change `runner.run_async(...)` to `runner.run(...)` and `async for` to `for` in main.py.

- [ ] **Step 3: Test server starts**

Run: `cd Tripletex && python -m uvicorn main:app --host 0.0.0.0 --port 8000 &`
Then: `curl -s http://localhost:8000/docs | head -5`
Expected: FastAPI docs HTML

- [ ] **Step 3: Commit**

```bash
git add Tripletex/main.py
git commit -m "feat(tripletex): FastAPI /solve endpoint with ADK agent orchestration"
```

---

### Task 9: End-to-End Test with Sandbox

**Files:**
- Create: `Tripletex/tests/test_endpoint.py`

- [ ] **Step 1: Create integration test script**

```python
# tests/test_endpoint.py
"""
End-to-end test: send a task prompt to /solve and verify it works.
Requires: server running on localhost:8000, valid .env with GOOGLE_API_KEY.
Does NOT require Tripletex sandbox (uses mock credentials for smoke test).

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
```

- [ ] **Step 2: Run smoke test (with server running)**

Run: `cd Tripletex && python tests/test_endpoint.py`
Expected: `Test passed!` (with real sandbox token, actual employee created in Tripletex)

- [ ] **Step 3: Commit**

```bash
git add Tripletex/tests/test_endpoint.py
git commit -m "feat(tripletex): end-to-end integration test"
```

---

### Task 10: Cloudflared Tunnel & Final Verification

- [ ] **Step 1: Start the server**

Run: `cd Tripletex && python -m uvicorn main:app --host 0.0.0.0 --port 8000`

- [ ] **Step 2: Start cloudflared tunnel**

Run (in separate terminal): `npx cloudflared tunnel --url http://localhost:8000`
Note the HTTPS URL printed (e.g. `https://random-name.trycloudflare.com`)

- [ ] **Step 3: Test via tunnel**

Run: `curl -X POST https://<tunnel-url>/solve -H "Content-Type: application/json" -d '{"prompt": "test", "files": [], "tripletex_credentials": {"base_url": "https://example.com/v2", "session_token": "test"}}'`
Expected: `{"status": "completed"}`

- [ ] **Step 4: Submit URL at app.ainm.no**

Go to `https://app.ainm.no/submit/tripletex` and submit the tunnel HTTPS URL.

- [ ] **Step 5: Final commit**

```bash
git add -A Tripletex/
git commit -m "feat(tripletex): complete agent ready for competition"
```
