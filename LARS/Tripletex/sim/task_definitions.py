"""Task type definitions for the Tripletex submission simulator.

Each task defines:
- How to generate a prompt (gen_instruction for the LLM)
- How to verify the result (entity_type, search_fields, field_checks)
- Scoring weights per field
- Baseline API call count for efficiency scoring

Competition has 30 task types across 3 tiers. We cover the likely set based
on the documented categories: employees, customers, products, invoicing,
travel expenses, projects, departments, corrections.
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

CREATE_EMPLOYEE = TaskDef(
    name="create_employee",
    tier=1,
    description="Create a new employee",
    gen_instruction="""\
Generate a task to create an employee in Tripletex.
Always include: firstName, lastName, email.
50% chance: include a mobile phone number (phoneNumberMobile, +47 format).
40% chance: make the employee an administrator/kontoadministrator (userType should be "EXTENDED").
30% chance: include dateOfBirth (YYYY-MM-DD, realistic adult age).
Use varied, uncommon Scandinavian names to avoid collisions (avoid Lars, Erik, Henrik, Silje, Johansen, Olsen, Hansen).
Examples: Tormod, Halvard, Bjarte, Solveig, Ragnhild, Astrid, Grønnli, Bergsland, Thorsnes, Kjærstad.
Email domain: example.com or test.no.

Expected fields:
- firstName (string)
- lastName (string)
- email (string)
- phoneNumberMobile (string, only if included)
- userType (string, "EXTENDED" if admin, "STANDARD" otherwise — always include this field)
- dateOfBirth (string YYYY-MM-DD, only if included)""",
    entity_type="employee",
    search_fields=["firstName", "lastName", "email"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 2),
        FieldCheck("lastName", 2),
        FieldCheck("email", 2),
        FieldCheck("phoneNumberMobile", 1),
        FieldCheck("dateOfBirth", 1),
    ],
    baseline_calls=1,
)

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

CREATE_CONTACT = TaskDef(
    name="create_contact",
    tier=1,
    description="Create a contact person for a customer",
    gen_instruction="""\
Generate a task to create a contact person linked to an existing customer in Tripletex.
The prompt should say to first create the customer, then add a contact person.

Include ALL:
- customer_name: company name ending in AS
- firstName: contact's first name
- lastName: contact's last name
- email: contact's email

Expected fields:
- customer_name (string)
- firstName (string)
- lastName (string)
- email (string)""",
    entity_type="contact",
    search_fields=["firstName", "lastName"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 2),
        FieldCheck("lastName", 2),
        FieldCheck("email", 2),
    ],
    baseline_calls=2,  # POST customer + POST contact
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Update tasks (×1 multiplier)
# ═══════════════════════════════════════════════════════════════════════

UPDATE_EMPLOYEE = TaskDef(
    name="update_employee",
    tier=1,
    description="Update an employee's phone number",
    gen_instruction="""\
Generate a task to update an existing employee's mobile phone number.
The prompt should first instruct to create the employee, then update their phone.
NOTE: Tripletex does NOT allow changing email after creation — only phone can be updated.

Include ALL:
- firstName: employee's first name
- lastName: employee's last name
- email: employee's email
- new_phoneNumberMobile: the new phone number to set (+47 8-digit format, e.g. "+4791234567")

Expected fields:
- firstName (string)
- lastName (string)
- email (string)
- new_phoneNumberMobile (string)""",
    entity_type="employee",
    search_fields=["firstName", "lastName"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 1),
        FieldCheck("lastName", 1),
        FieldCheck("new_phoneNumberMobile", 3),
    ],
    baseline_calls=4,  # GET dept + POST employee + GET employee + PUT employee
)

UPDATE_CUSTOMER = TaskDef(
    name="update_customer",
    tier=1,
    description="Update a customer's information",
    gen_instruction="""\
Generate a task to update an existing customer.
The prompt should first instruct to create the customer, then update a field.

Include ALL:
- name: customer company name ending in AS
- email: original email
- new_email: the new email to set
OR
- new_phoneNumber: the new phone number to set (+47 format)

Pick ONE field to update.

Expected fields:
- name (string)
- email (string) - original
- new_email (string, only if updating email)
- new_phoneNumber (string, only if updating phone)""",
    entity_type="customer",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("new_email", 3),
        FieldCheck("new_phoneNumber", 3),
    ],
    baseline_calls=2,  # POST + PUT
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

CREATE_TRAVEL_EXPENSE = TaskDef(
    name="create_travel_expense",
    tier=2,
    description="Create a travel expense report",
    gen_instruction="""\
Generate a task to create a travel expense report in Tripletex.
Include ALL:
- employee_firstName, employee_lastName, employee_email
- title: travel expense title (e.g. "Kundebesøk Oslo")
- departure_date: a date in 2026 (YYYY-MM-DD)
- return_date: 1-5 days after departure (YYYY-MM-DD)

Expected fields:
- employee_firstName (string)
- employee_lastName (string)
- employee_email (string)
- title (string)
- departure_date (string, YYYY-MM-DD)
- return_date (string, YYYY-MM-DD)""",
    entity_type="travelExpense",
    search_fields=["title"],
    field_checks=[
        FieldCheck("_employee_found", 1),
        FieldCheck("_found", 2),
        FieldCheck("title", 2),
        FieldCheck("departure_date", 2),
        FieldCheck("return_date", 2),
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


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Advanced tasks (×3 multiplier)
# ═══════════════════════════════════════════════════════════════════════

DELETE_TRAVEL_EXPENSE = TaskDef(
    name="delete_travel_expense",
    tier=3,
    description="Delete a travel expense report",
    gen_instruction="""\
Generate a task to delete a travel expense report from Tripletex.
Reference it by title.

Include:
- title: travel expense title to delete

Expected fields:
- title (string)""",
    entity_type="travelExpense",
    search_fields=["title"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("title", 3),
    ],
    baseline_calls=2,
    pre_create={"type": "travel_expense", "fields": ["title"]},
)

DELETE_CUSTOMER = TaskDef(
    name="delete_customer",
    tier=3,
    description="Delete a customer",
    gen_instruction="""\
Generate a task to delete a customer from Tripletex.
Reference them by company name.

Include:
- name: company name ending in AS

Expected fields:
- name (string)""",
    entity_type="customer",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("name", 3),
    ],
    baseline_calls=2,
    pre_create={"type": "customer", "fields": ["name"]},
)

CREATE_LEDGER_VOUCHER = TaskDef(
    name="create_ledger_voucher",
    tier=3,
    description="Create a correction voucher in the ledger",
    gen_instruction="""\
Generate a task to create a ledger correction voucher in Tripletex.
Include ALL:
- description: what the correction is for (e.g. "Korreksjon feilpostering mars")
- date: voucher date in March 2026 (YYYY-MM-DD)
- debit_account: account number to debit (e.g. 1920 for bank, 1500 for receivables)
- credit_account: account number to credit (e.g. 3000 for revenue, 4000 for cost)
- amount: the correction amount (100-50000)

The prompt should instruct to create a voucher with debit and credit postings.

Expected fields:
- description (string)
- date (string, YYYY-MM-DD)
- debit_account (integer)
- credit_account (integer)
- amount (number)""",
    entity_type="ledger/voucher",
    search_fields=[],
    field_checks=[
        FieldCheck("_found", 3),
        FieldCheck("description", 2),
        FieldCheck("date", 2),
        FieldCheck("amount", 3),
    ],
    baseline_calls=1,
)

REVERSE_VOUCHER = TaskDef(
    name="reverse_voucher",
    tier=3,
    description="Reverse a ledger voucher",
    gen_instruction="""\
Generate a task to reverse (tilbakeføre) a ledger voucher.
The prompt should reference a voucher by its description.

Include:
- description: voucher description to find and reverse
- reverse_date: the date for the reversal (YYYY-MM-DD, March 2026)

Expected fields:
- description (string)
- reverse_date (string, YYYY-MM-DD)""",
    entity_type="ledger/voucher",
    search_fields=[],
    field_checks=[
        FieldCheck("_reversed", 5),
        FieldCheck("description", 3),
    ],
    baseline_calls=2,  # GET search + PUT reverse
    pre_create={"type": "voucher", "fields": ["description"]},
)

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

CREATE_OPENING_BALANCE = TaskDef(
    name="create_opening_balance",
    tier=3,
    description="Set opening balance for an account",
    gen_instruction="""\
Generate a task to create an opening balance voucher in Tripletex.
This sets the starting balance for a specific ledger account.

Include ALL:
- account_number: a standard Norwegian account (e.g. 1920 for bank, 1500 for receivables)
- amount: the opening balance amount (10000-500000)
- date: January 1, 2026 (2026-01-01)
- description: e.g. "Inngående balanse bank" or "Åpningsbalanse kundefordringer"

Expected fields:
- account_number (integer)
- amount (number)
- date (string, YYYY-MM-DD)
- description (string)""",
    entity_type="ledger/voucher",
    search_fields=[],
    field_checks=[
        FieldCheck("_found", 3),
        FieldCheck("description", 2),
        FieldCheck("amount", 3),
        FieldCheck("date", 2),
    ],
    baseline_calls=1,
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Additional (update/delete for basic entities)
# ═══════════════════════════════════════════════════════════════════════

UPDATE_DEPARTMENT = TaskDef(
    name="update_department",
    tier=1,
    description="Update a department's name or number",
    gen_instruction="""\
Generate a task to update an existing department in Tripletex.
The prompt should first instruct to create the department, then update a field.

Include ALL:
- name: original department name
Pick ONE to update:
- new_name: a new department name
- new_departmentNumber: a new department number (1-3 digit string)

Expected fields:
- name (string)
- new_name (string, only if updating name)
- new_departmentNumber (string, only if updating number)""",
    entity_type="department",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("new_name", 3),
        FieldCheck("new_departmentNumber", 3),
    ],
    baseline_calls=2,
)

UPDATE_CONTACT = TaskDef(
    name="update_contact",
    tier=1,
    description="Update a contact person's information",
    gen_instruction="""\
Generate a task to update an existing contact person in Tripletex.
The prompt should first instruct to create a customer, add a contact person, then update the contact.

Include ALL:
- customer_name: company name ending in AS
- firstName: contact's first name
- lastName: contact's last name
- email: contact's original email
Pick ONE to update:
- new_email: the new email to set
- new_phoneNumberMobile: the new phone number (+47 format)

Expected fields:
- customer_name (string)
- firstName (string)
- lastName (string)
- email (string)
- new_email (string, only if updating email)
- new_phoneNumberMobile (string, only if updating phone)""",
    entity_type="contact",
    search_fields=["firstName", "lastName"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("firstName", 1),
        FieldCheck("lastName", 1),
        FieldCheck("new_email", 3),
        FieldCheck("new_phoneNumberMobile", 3),
    ],
    baseline_calls=3,  # POST customer + POST contact + PUT contact
)

UPDATE_PRODUCT = TaskDef(
    name="update_product",
    tier=1,
    description="Update a product's price or name",
    gen_instruction="""\
Generate a task to update an existing product in Tripletex.
The prompt should first instruct to create the product, then update a field.

Include ALL:
- name: original product name
- priceExcludingVatCurrency: original price (100-10000)
Pick ONE to update (50/50):
- new_name: a new product name
- new_priceExcludingVatCurrency: a new price

Expected fields:
- name (string)
- priceExcludingVatCurrency (number)
- new_name (string, only if updating name)
- new_priceExcludingVatCurrency (number, only if updating price)""",
    entity_type="product",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("new_name", 3),
        FieldCheck("new_priceExcludingVatCurrency", 3),
    ],
    baseline_calls=2,
)

UPDATE_SUPPLIER = TaskDef(
    name="update_supplier",
    tier=1,
    description="Update a supplier's information",
    gen_instruction="""\
Generate a task to update an existing supplier in Tripletex.
The prompt should first instruct to create the supplier, then update a field.

Include ALL:
- name: supplier company name ending in AS
- email: original email
Pick ONE to update:
- new_email: the new email to set
- new_phoneNumber: the new phone number (+47 format)

Expected fields:
- name (string)
- email (string)
- new_email (string, only if updating email)
- new_phoneNumber (string, only if updating phone)""",
    entity_type="supplier",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_found", 2),
        FieldCheck("name", 2),
        FieldCheck("new_email", 3),
        FieldCheck("new_phoneNumber", 3),
    ],
    baseline_calls=2,
)


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Additional multi-step tasks
# ═══════════════════════════════════════════════════════════════════════

CREATE_SUPPLIER_INVOICE = TaskDef(
    name="create_supplier_invoice",
    tier=2,
    description="Create an incoming invoice from a supplier",
    sandbox_broken=True,  # POST /supplierInvoice returns 500 in dev sandbox
    gen_instruction="""\
Generate a task to create an incoming invoice (leverandørfaktura) in Tripletex.
Include ALL:
- supplier_name: company name ending in AS
- supplier_email: email for the supplier
- invoice_date: March 2026 (YYYY-MM-DD)
- invoice_number: a vendor invoice number (e.g. "F-2026-001")

The prompt should instruct to first create the supplier, then register the incoming invoice.

Expected fields:
- supplier_name (string)
- supplier_email (string)
- invoice_date (string, YYYY-MM-DD)
- invoice_number (string)""",
    entity_type="supplierInvoice",
    search_fields=[],
    field_checks=[
        FieldCheck("_supplier_found", 2),
        FieldCheck("_found", 2),
        FieldCheck("supplier_name", 1),
        FieldCheck("invoice_date", 2),
        FieldCheck("invoice_number", 2),
    ],
    baseline_calls=2,
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


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Additional advanced tasks
# ═══════════════════════════════════════════════════════════════════════

DELETE_SUPPLIER = TaskDef(
    name="delete_supplier",
    tier=3,
    description="Delete a supplier",
    gen_instruction="""\
Generate a task to delete a supplier from Tripletex.
Reference them by company name.

Include:
- name: company name ending in AS

Expected fields:
- name (string)""",
    entity_type="supplier",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("name", 3),
    ],
    baseline_calls=2,
    pre_create={"type": "supplier", "fields": ["name"]},
)

DELETE_PRODUCT = TaskDef(
    name="delete_product",
    tier=3,
    description="Delete a product",
    gen_instruction="""\
Generate a task to delete a product from Tripletex.
Reference it by name.

Include:
- name: product name

Expected fields:
- name (string)""",
    entity_type="product",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("name", 3),
    ],
    baseline_calls=2,
    pre_create={"type": "product", "fields": ["name"]},
)

DELETE_DEPARTMENT = TaskDef(
    name="delete_department",
    tier=3,
    description="Delete a department",
    gen_instruction="""\
Generate a task to delete a department from Tripletex.
Reference it by name.

Include:
- name: department name (e.g. "Salg", "IT-avdeling")

Expected fields:
- name (string)""",
    entity_type="department",
    search_fields=["name"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("name", 3),
    ],
    baseline_calls=2,
    pre_create={"type": "department", "fields": ["name"]},
)

DELETE_CONTACT = TaskDef(
    name="delete_contact",
    tier=3,
    description="Delete a contact person",
    gen_instruction="""\
Generate a task to delete a contact person from Tripletex.
Reference them by first and last name.

Include:
- firstName: contact's first name
- lastName: contact's last name

Expected fields:
- firstName (string)
- lastName (string)""",
    entity_type="contact",
    search_fields=["firstName", "lastName"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("firstName", 2),
        FieldCheck("lastName", 1),
    ],
    baseline_calls=2,
    pre_create={"type": "contact", "fields": ["firstName", "lastName"]},
)

DELETE_EMPLOYEE = TaskDef(
    name="delete_employee",
    tier=3,
    description="Deactivate an employee",
    sandbox_broken=True,  # DELETE returns 403, employee model has no isInactive
    gen_instruction="""\
Generate a task to deactivate/remove an employee from Tripletex.
NOTE: Tripletex does not allow deleting employees with employments.
The agent should search for the employee and either delete or deactivate them.

Include:
- firstName: employee's first name
- lastName: employee's last name

Expected fields:
- firstName (string)
- lastName (string)""",
    entity_type="employee",
    search_fields=["firstName", "lastName"],
    field_checks=[
        FieldCheck("_deleted", 5),
        FieldCheck("firstName", 2),
        FieldCheck("lastName", 1),
    ],
    baseline_calls=2,
    pre_create={"type": "employee", "fields": ["firstName", "lastName"]},
)


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

ALL_TASKS: dict[str, TaskDef] = {
    # Tier 1 — basic
    "create_employee": CREATE_EMPLOYEE,
    "create_customer": CREATE_CUSTOMER,
    "create_product": CREATE_PRODUCT,
    "create_department": CREATE_DEPARTMENT,
    "create_supplier": CREATE_SUPPLIER,
    "create_contact": CREATE_CONTACT,
    "update_employee": UPDATE_EMPLOYEE,
    "update_customer": UPDATE_CUSTOMER,
    "update_product": UPDATE_PRODUCT,
    "update_supplier": UPDATE_SUPPLIER,
    "update_department": UPDATE_DEPARTMENT,
    "update_contact": UPDATE_CONTACT,
    # Tier 2 — multi-step
    "create_invoice": CREATE_INVOICE,
    "create_multi_line_invoice": CREATE_MULTI_LINE_INVOICE,
    "create_project": CREATE_PROJECT,
    "create_travel_expense": CREATE_TRAVEL_EXPENSE,
    "invoice_with_payment": INVOICE_WITH_PAYMENT,
    "create_credit_note": CREATE_CREDIT_NOTE,
    "create_employee_with_employment": CREATE_EMPLOYEE_WITH_EMPLOYMENT,
    "create_supplier_invoice": CREATE_SUPPLIER_INVOICE,
    "create_travel_expense_with_costs": CREATE_TRAVEL_EXPENSE_WITH_COSTS,
    "create_project_with_pm": CREATE_PROJECT_WITH_PM,
    # Tier 3 — advanced
    "delete_travel_expense": DELETE_TRAVEL_EXPENSE,
    "delete_customer": DELETE_CUSTOMER,
    "create_ledger_voucher": CREATE_LEDGER_VOUCHER,
    "reverse_voucher": REVERSE_VOUCHER,
    "delete_invoice": DELETE_INVOICE,
    "create_opening_balance": CREATE_OPENING_BALANCE,
    "delete_supplier": DELETE_SUPPLIER,
    "delete_product": DELETE_PRODUCT,
    "delete_department": DELETE_DEPARTMENT,
    "delete_contact": DELETE_CONTACT,
    "delete_employee": DELETE_EMPLOYEE,
}

TIER1_TASKS = [t for t in ALL_TASKS.values() if t.tier == 1]
TIER2_TASKS = [t for t in ALL_TASKS.values() if t.tier == 2]
TIER3_TASKS = [t for t in ALL_TASKS.values() if t.tier == 3]
