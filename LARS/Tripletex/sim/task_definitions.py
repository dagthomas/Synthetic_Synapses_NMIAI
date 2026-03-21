"""Task type definitions for the Tripletex submission simulator.

Each task defines:
- How to generate a prompt (gen_instruction for the LLM)
- How to verify the result (entity_type, search_fields, field_checks)
- Scoring weights per field
- Baseline API call count for efficiency scoring

Only confirmed competition task types are included here (18 confirmed from payloads).
"""

from dataclasses import dataclass, field
from typing import Optional


LANGUAGES = {
    "no": "Norwegian (bokmål)",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "nn": "Nynorsk",
    "de": "German",
    "fr": "French",
}


@dataclass
class FieldCheck:
    """A field to verify with its point value."""
    field: str
    points: int


@dataclass
class TaskDef:
    """Definition of a task type for simulation."""
    name: str
    tier: int
    description: str
    gen_instruction: str
    entity_type: str  # API entity for verification lookup
    search_fields: list[str]  # which expected fields to use for searching
    field_checks: list[FieldCheck]
    baseline_calls: int
    extra_verifications: list[dict] = field(default_factory=list)
    pre_create: Optional[dict] = None
    sandbox_broken: bool = False  # True = cannot be verified in dev sandbox


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Basic entity creation (×1 multiplier)
# ═══════════════════════════════════════════════════════════════════════

CREATE_CUSTOMER = TaskDef(
    name="create_customer",
    tier=1,
    description="Create a new customer",
    gen_instruction="""\
Generate a task to create a customer in Tripletex.
Always include: name (company name ending in AS, ANS, or DA), email.
30% chance: include organizationNumber (9-digit Norwegian org number).
30% chance: include phoneNumber (+47 format).
50% chance: include a postal address with addressLine1 (street), postalCode (4-digit), city.

Expected fields:
- name (string)
- email (string)
- organizationNumber (string, only if included)
- phoneNumber (string, only if included)
- addressLine1 (string, only if address included)
- postalCode (string, only if address included)
- city (string, only if address included)""",
    entity_type="customer",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 1),
        FieldCheck("email", 1),
        FieldCheck("organizationNumber", 1),
        FieldCheck("phoneNumber", 1),
        FieldCheck("addressLine1", 1),
        FieldCheck("postalCode", 1),
        FieldCheck("city", 1),
    ],
    baseline_calls=1,
)

CREATE_PRODUCT = TaskDef(
    name="create_product",
    tier=1,
    description="Create a new product",
    gen_instruction="""\
Generate a task to create a product in Tripletex.
Always include: name (realistic product/service name), priceExcludingVatCurrency (50-10000).
50% chance: include a productNumber (alphanumeric like "PRD-001").

Expected fields:
- name (string)
- priceExcludingVatCurrency (number)
- number (string, only if productNumber included)""",
    entity_type="product",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("priceExcludingVatCurrency", 3),
        FieldCheck("number", 2),
    ],
    baseline_calls=1,
)

CREATE_DEPARTMENT = TaskDef(
    name="create_department",
    tier=1,
    description="Create a department",
    gen_instruction="""\
Generate a task to create a department in Tripletex.
Always include: name (realistic: "Salg", "IT", "HR", "Regnskap", "Marked", "Logistikk").
50% chance: include departmentNumber (1-3 digit string).

Expected fields:
- name (string)
- departmentNumber (string, only if departmentNumber included)""",
    entity_type="department",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 3),
        FieldCheck("name", 4),
        FieldCheck("departmentNumber", 3),
    ],
    baseline_calls=1,
)

CREATE_SUPPLIER = TaskDef(
    name="create_supplier",
    tier=1,
    description="Create a supplier",
    gen_instruction="""\
Generate a task to create a supplier in Tripletex.
Always include: name (company name ending in AS), email.
50% chance: include organizationNumber (9-digit).
30% chance: include phoneNumber (+47 format).
40% chance: include a postal address with addressLine1 (street), postalCode (4-digit), city.

The prompt should make clear this is a SUPPLIER (leverandør), not a customer.

Expected fields:
- name (string)
- email (string)
- organizationNumber (string, only if included)
- phoneNumber (string, only if included)
- addressLine1 (string, only if address included)
- postalCode (string, only if address included)
- city (string, only if address included)""",
    entity_type="supplier",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("email", 2),
        FieldCheck("organizationNumber", 1),
        FieldCheck("phoneNumber", 1),
        FieldCheck("addressLine1", 1),
        FieldCheck("postalCode", 1),
        FieldCheck("city", 1),
    ],
    baseline_calls=1,
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Multi-step tasks (×2 multiplier)
# ═══════════════════════════════════════════════════════════════════════

CREATE_INVOICE = TaskDef(
    name="create_invoice",
    tier=2,
    description="Create an invoice for a customer",
    gen_instruction="""\
Generate a task to create an invoice in Tripletex.
Include ALL of these:
- customer_name: company name ending in AS
- customer_email: email for the customer
- product_name: a product or service name
- product_price: price excluding VAT (100-10000)
- quantity: number of items (1-20)
- invoice_date: a date in March 2026 (YYYY-MM-DD)
- due_date: 14-30 days after invoice_date (YYYY-MM-DD)

The prompt should instruct to create customer, product, and then invoice.

Expected fields:
- customer_name (string)
- customer_email (string)
- product_name (string)
- product_price (number)
- quantity (integer)
- invoice_date (string, YYYY-MM-DD)
- due_date (string, YYYY-MM-DD)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_invoice_found", 2),
        FieldCheck("customer_name", 1),
        FieldCheck("product_name", 1),
        FieldCheck("quantity", 1),
        FieldCheck("invoice_date", 2),
        FieldCheck("due_date", 2),
    ],
    baseline_calls=4,
)

CREATE_MULTI_LINE_INVOICE = TaskDef(
    name="create_multi_line_invoice",
    tier=2,
    description="Create an invoice with multiple products",
    gen_instruction="""\
Generate a task to create an invoice with 2-3 different products/services.
Include ALL:
- customer_name: company name ending in AS
- customer_email: email
- products: list of 2-3 products, each with name, price, and quantity
- invoice_date: March 2026 (YYYY-MM-DD)
- due_date: 14-30 days after invoice_date (YYYY-MM-DD)

Expected fields:
- customer_name (string)
- customer_email (string)
- product_count (integer, number of distinct products: 2 or 3)
- invoice_date (string, YYYY-MM-DD)
- due_date (string, YYYY-MM-DD)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_invoice_found", 2),
        FieldCheck("customer_name", 1),
        FieldCheck("product_count", 2),
        FieldCheck("invoice_date", 2),
        FieldCheck("due_date", 2),
    ],
    baseline_calls=6,  # customer + 2-3 products + order + invoice
)

CREATE_PROJECT = TaskDef(
    name="create_project",
    tier=2,
    description="Create a project linked to a customer",
    gen_instruction="""\
Generate a task to create a project in Tripletex linked to a customer.
Include ALL:
- project_name: realistic project name
- customer_name: company name ending in AS
- start_date: a date in 2026 (YYYY-MM-DD)
50% chance: include description.

Expected fields:
- project_name (string)
- customer_name (string)
- start_date (string, YYYY-MM-DD)
- description (string, only if included)""",
    entity_type="project",
    search_fields=["project_name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("project_name", 2),
        FieldCheck("customer_name", 2),
        FieldCheck("start_date", 2),
        FieldCheck("description", 1),
    ],
    baseline_calls=2,
)

INVOICE_WITH_PAYMENT = TaskDef(
    name="invoice_with_payment",
    tier=2,
    description="Create an invoice and register payment",
    gen_instruction="""\
Generate a task to create an invoice AND register full payment in Tripletex.
Include ALL:
- customer_name: company name ending in AS
- customer_email: email
- product_name: product or service name
- product_price: price excluding VAT (100-10000)
- quantity: number of items (1-10)
- invoice_date: March 2026 (YYYY-MM-DD)
- due_date: 14-30 days after invoice_date (YYYY-MM-DD)
- payment_date: same as or a few days after invoice_date (YYYY-MM-DD)
- payment_amount: total = product_price * quantity (NO VAT — sandbox products have no VAT by default)

Expected fields:
- customer_name, customer_email, product_name, product_price (number)
- quantity (integer), invoice_date, due_date, payment_date (strings, YYYY-MM-DD)
- payment_amount (number)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_invoice_found", 1),
        FieldCheck("_payment_found", 2),
        FieldCheck("customer_name", 1),
        FieldCheck("invoice_date", 1),
        FieldCheck("due_date", 1),
        FieldCheck("payment_amount", 2),
    ],
    baseline_calls=5,
)

CREATE_CREDIT_NOTE = TaskDef(
    name="create_credit_note",
    tier=2,
    description="Create an invoice and then credit it",
    gen_instruction="""\
Generate a task to create an invoice and then create a credit note for it.
Include ALL:
- customer_name: company name ending in AS
- customer_email: email
- product_name: product or service name
- product_price: price excluding VAT (100-10000)
- quantity: number of items (1-5)
- invoice_date: March 2026 (YYYY-MM-DD)
- due_date: 14-30 days after invoice_date

The prompt should say: create the invoice, then immediately create a credit note for it.

Expected fields:
- customer_name (string)
- customer_email (string)
- product_name (string)
- invoice_date (string, YYYY-MM-DD)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_invoice_found", 1),
        FieldCheck("_credit_note_found", 4),
        FieldCheck("customer_name", 1),
        FieldCheck("invoice_date", 1),
    ],
    baseline_calls=5,  # customer + product + order + invoice + creditNote
)

CREATE_EMPLOYEE_WITH_EMPLOYMENT = TaskDef(
    name="create_employee_with_employment",
    tier=2,
    description="Create employee with employment details",
    gen_instruction="""\
Generate a task to create an employee with employment details in Tripletex.
Include ALL:
- firstName, lastName — use varied, uncommon Scandinavian names (avoid Lars, Erik, Henrik, Silje, Johansen, Olsen, Hansen).
  Examples: Bjarte, Halvard, Torgeir, Solveig, Ragnhild, Ingvild, Kjærstad, Thorsnes, Grønnli, Bergsland.
- email: use the pattern firstname.lastname@firma.no or test.no
- start_date: employment start date (YYYY-MM-DD, in 2026)
- employment_type: "maritim" or "ordinær"

The prompt should say to create the employee and set up their employment.

Expected fields:
- firstName (string)
- lastName (string)
- email (string)
- start_date (string, YYYY-MM-DD)""",
    entity_type="employee",
    search_fields=["firstName", "lastName", "email"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 1),
        FieldCheck("lastName", 1),
        FieldCheck("email", 1),
        FieldCheck("start_date", 3),
    ],
    baseline_calls=2,  # POST employee + POST employment
)

CREATE_SUPPLIER_INVOICE = TaskDef(
    name="create_supplier_invoice",
    tier=2,
    description="Create an incoming invoice from a supplier (via ledger voucher with VAT postings)",
    gen_instruction="""\
Generate a task to create an incoming invoice (leverandørfaktura) in Tripletex.
Include ALL:
- supplier_name: company name ending in AS or Ltd
- supplier_org_number: 9-digit Norwegian org number
- invoice_date: March 2026 (YYYY-MM-DD)
- invoice_number: a vendor invoice number (e.g. "INV-2026-001")
- amount_including_vat: total amount INCLUDING VAT in NOK
- expense_account: the account number for the expense (e.g. 6590 for office services)
- vat_percentage: 25 (standard rate)

The prompt should instruct to first create the supplier, then register the supplier invoice
with the correct input VAT (inngående MVA).

Expected fields:
- supplier_name (string)
- supplier_org_number (string)
- invoice_date (string, YYYY-MM-DD)
- invoice_number (string)
- amount_including_vat (float)
- expense_account (int)
- vat_percentage (int)""",
    entity_type="voucher",
    search_fields=[],
    field_checks=[
        FieldCheck("_supplier_found", 2),
        FieldCheck("_found", 2),
        FieldCheck("supplier_name", 1),
        FieldCheck("invoice_date", 2),
        FieldCheck("invoice_number", 2),
    ],
    baseline_calls=3,
)

CREATE_TRAVEL_EXPENSE_WITH_COSTS = TaskDef(
    name="create_travel_expense_with_costs",
    tier=2,
    description="Create a travel expense with cost lines",
    gen_instruction="""\
Generate a task to create a travel expense report WITH cost/expense items.
Include ALL:
- employee_firstName, employee_lastName, employee_email
- title: travel expense title
- departure_date: a date in 2026 (YYYY-MM-DD)
- return_date: 1-5 days after departure (YYYY-MM-DD)
- cost_amount: amount for a cost item (100-5000 NOK)
- cost_description: what the cost was for (e.g. "Hotell", "Taxi", "Fly")

Expected fields:
- employee_firstName (string)
- employee_lastName (string)
- title (string)
- departure_date (string, YYYY-MM-DD)
- return_date (string, YYYY-MM-DD)
- cost_amount (number)""",
    entity_type="travelExpense",
    search_fields=["title"],
    field_checks=[
        FieldCheck("_employee_found", 1),
        FieldCheck("_found", 2),
        FieldCheck("title", 1),
        FieldCheck("departure_date", 2),
        FieldCheck("return_date", 2),
        FieldCheck("_cost_found", 2),
    ],
    baseline_calls=3,
)

CREATE_PROJECT_WITH_PM = TaskDef(
    name="create_project_with_pm",
    tier=2,
    description="Create a project with a new project manager",
    gen_instruction="""\
Generate a task to create a project with a specific project manager.
Include ALL:
- project_name: realistic project name
- customer_name: company name ending in AS
- pm_firstName: project manager's first name
- pm_lastName: project manager's last name
- pm_email: project manager's email
- start_date: a date in 2026 (YYYY-MM-DD)

The prompt should instruct to create the customer, create the project manager as an employee,
and then create the project with them as project manager (prosjektleder).

Expected fields:
- project_name (string)
- customer_name (string)
- pm_firstName (string)
- pm_lastName (string)
- start_date (string, YYYY-MM-DD)""",
    entity_type="project",
    search_fields=["project_name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("project_name", 2),
        FieldCheck("customer_name", 1),
        FieldCheck("_pm_found", 2),
        FieldCheck("start_date", 2),
    ],
    baseline_calls=3,
)

PROJECT_INVOICE = TaskDef(
    name="project_invoice",
    tier=2,
    description="Create a fixed-price project and invoice for it",
    gen_instruction="""\
Generate a task to create a project linked to a customer and then invoice it.
Include ALL:
- customer_name: company name ending in AS
- customer_email: email for the customer
- project_name: realistic project name (e.g. "Nettsideutvikling", "IT-konsulentprosjekt")
- product_name: product/service name for the invoice line
- product_price: price excluding VAT (5000-50000)
- invoice_date: March 2026 (YYYY-MM-DD)
- due_date: 14-30 days after invoice_date (YYYY-MM-DD)

The prompt should instruct to create customer, employee (project manager), project,
product, then create an invoice for the project.

Expected fields:
- customer_name (string)
- customer_email (string)
- project_name (string)
- product_name (string)
- invoice_date (string, YYYY-MM-DD)
- due_date (string, YYYY-MM-DD)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_project_found", 2),
        FieldCheck("_invoice_found", 2),
        FieldCheck("customer_name", 1),
        FieldCheck("invoice_date", 2),
        FieldCheck("due_date", 2),
    ],
    baseline_calls=6,  # customer + employee + project + product + order + invoice
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Advanced tasks (×3 multiplier)
# ═══════════════════════════════════════════════════════════════════════

DELETE_INVOICE = TaskDef(
    name="delete_invoice",
    tier=3,
    description="Credit/reverse an invoice",
    gen_instruction="""\
Generate a task to create an invoice and then credit (kreditere) it.
The task is essentially: make an invoice, then issue a credit note to cancel it.

Include ALL:
- customer_name: company name ending in AS
- customer_email: email for the customer
- product_name: product name
- product_price: price (100-5000)
- invoice_date: March 2026 (YYYY-MM-DD)

Expected fields:
- customer_name (string)
- customer_email (string)
- product_name (string)
- invoice_date (string, YYYY-MM-DD)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_customer_found", 1),
        FieldCheck("_invoice_found", 1),
        FieldCheck("_credit_note_found", 4),
        FieldCheck("customer_name", 1),
    ],
    baseline_calls=5,
)

SALARY = TaskDef(
    name="salary",
    tier=3,
    description="Run a salary transaction for an employee",
    gen_instruction="""\
Generate a task to process a salary/payroll transaction for an employee in Tripletex.
Include ALL:
- firstName: employee's first name
- lastName: employee's last name
- email: employee's email
- salary_type: type of salary (e.g. "Fastlønn", "Timelønn", "Bonus")
- year: salary year (2026)
- month: salary month (1-12)
- amount: salary amount (10000-80000 NOK)

The prompt should instruct to create the employee (with employment), then register a salary transaction.

Expected fields:
- firstName (string)
- lastName (string)
- email (string)
- year (integer)
- month (integer)""",
    entity_type="employee",
    search_fields=["firstName", "lastName", "email"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 1),
        FieldCheck("lastName", 1),
        FieldCheck("_salary_found", 4),
    ],
    baseline_calls=4,  # employee + employment + search_salary_types + salary_transaction
)

REVERSE_PAYMENT = TaskDef(
    name="reverse_payment",
    tier=3,
    description="Reverse a payment on an invoice",
    gen_instruction="""\
Generate a task to reverse (tilbakeføre) a payment that was registered on an invoice.
The prompt should say the payment was returned by the bank or needs to be reversed.

Include:
- customer_name: company name ending in AS
- reverse_reason: reason for reversal (e.g. "Betaling returnert fra bank")

Expected fields:
- customer_name (string)
- reverse_reason (string)""",
    entity_type="invoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_invoice_found", 2),
        FieldCheck("_payment_reversed", 4),
        FieldCheck("customer_name", 2),
    ],
    baseline_calls=2,  # search_invoices + register_payment(negative)
    pre_create={"type": "paid_invoice", "fields": ["customer_name"]},
)

CREATE_DIMENSION = TaskDef(
    name="create_dimension",
    tier=3,
    description="Create a custom accounting dimension with values and post a voucher linked to it",
    gen_instruction="""\
Generate a task to create a custom accounting dimension in Tripletex, add values to it,
then create a voucher posting linked to one of the dimension values.

Include ALL:
- dimension_name: name of the dimension (e.g. "Region", "Prosjekttype", "Marked")
- dimension_values: list of 2 values (e.g. ["Sør-Norge", "Nord-Norge"])
- account_number: ledger account to post to (e.g. 6340, 7140, 6860, 7300)
- amount: voucher amount in NOK (10000-50000)
- linked_value: which dimension value to link the posting to (one of dimension_values)

Expected fields:
- dimension_name (string)
- dimension_value_1 (string)
- dimension_value_2 (string)
- account_number (integer)
- amount (number)
- linked_value (string)""",
    entity_type="ledger/voucher",
    search_fields=[],
    field_checks=[
        FieldCheck("_dimension_found", 3),
        FieldCheck("dimension_name", 2),
        FieldCheck("_voucher_found", 3),
        FieldCheck("amount", 2),
    ],
    baseline_calls=4,  # create dimension + 2 values + voucher
)


# ═══════════════════════════════════════════════════════════════════════
# Registry — only confirmed competition tasks (18 types)
# ═══════════════════════════════════════════════════════════════════════

ALL_TASKS: dict[str, TaskDef] = {
    # Tier 1 — basic
    "create_customer": CREATE_CUSTOMER,
    "create_product": CREATE_PRODUCT,
    "create_department": CREATE_DEPARTMENT,
    "create_supplier": CREATE_SUPPLIER,
    # Tier 2 — multi-step
    "create_invoice": CREATE_INVOICE,
    "create_multi_line_invoice": CREATE_MULTI_LINE_INVOICE,
    "create_project": CREATE_PROJECT,
    "invoice_with_payment": INVOICE_WITH_PAYMENT,
    "create_credit_note": CREATE_CREDIT_NOTE,
    "create_employee_with_employment": CREATE_EMPLOYEE_WITH_EMPLOYMENT,
    "create_supplier_invoice": CREATE_SUPPLIER_INVOICE,
    "create_travel_expense_with_costs": CREATE_TRAVEL_EXPENSE_WITH_COSTS,
    "create_project_with_pm": CREATE_PROJECT_WITH_PM,
    "project_invoice": PROJECT_INVOICE,
    # Tier 3 — advanced
    "delete_invoice": DELETE_INVOICE,
    "salary": SALARY,
    "reverse_payment": REVERSE_PAYMENT,
    "create_dimension": CREATE_DIMENSION,
}

TIER1_TASKS = [t for t in ALL_TASKS.values() if t.tier == 1]
TIER2_TASKS = [t for t in ALL_TASKS.values() if t.tier == 2]
TIER3_TASKS = [t for t in ALL_TASKS.values() if t.tier == 3]
