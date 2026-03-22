"""Static pipeline runner — deterministic task execution with LLM extraction.

Instead of running a multi-turn LLM agent (4-10 round-trips), static mode:
1. Classifies the task type (reuses tool_router.classify_task)
2. Extracts structured parameters in ONE Gemini call
3. Executes a hardcoded pipeline of tool calls deterministically
4. Falls back to agent mode on any error
"""

import json
import logging
import os
from datetime import date, timedelta

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────
EXTRACTION_MODEL = "gemini-2.5-flash"
EXTRACTION_TEMPERATURE = 0.0

# ── Exceptions ───────────────────────────────────────────────────

class ExtractionError(Exception):
    """LLM extraction failed — required field missing or invalid JSON."""


class PipelineError(Exception):
    """A tool call in the pipeline failed."""


# ── Helpers ──────────────────────────────────────────────────────

def _val(result):
    """Extract value dict from tool result, raise on error."""
    if isinstance(result, dict) and result.get("error"):
        raise PipelineError(result.get("message", str(result)))
    return result.get("value", result) if isinstance(result, dict) else result


def _id(result):
    """Get entity ID from tool result."""
    v = _val(result)
    return v["id"] if isinstance(v, dict) and "id" in v else None


def _version(result):
    """Get entity version from tool result."""
    v = _val(result)
    return v.get("version", -1) if isinstance(v, dict) else -1


def _first(result, entity="entity", name_match: str = ""):
    """Get first result from a search response, raise if empty.

    If name_match is provided, filters results by exact name (case-insensitive)
    before picking the first. Falls back to substring matching (e.g. "Leroy SARL"
    matches "Fournisseur Leroy SARL"). This is critical because Tripletex search
    APIs sometimes return ALL entities instead of filtering by name.
    """
    if isinstance(result, dict) and result.get("error"):
        raise PipelineError(result.get("message", str(result)))
    vals = result.get("values", [])
    if not vals:
        raise PipelineError(f"No {entity} found")
    if name_match:
        target = name_match.strip().lower()
        # 1. Exact match — prefer highest ID (most recently created) to avoid sandbox pollution
        exact = [v for v in vals if v.get("name", "").strip().lower() == target]
        if exact:
            return max(exact, key=lambda v: v.get("id", 0))
        # 2. Substring match (target in candidate name, or vice versa)
        substr = [v for v in vals
                  if target in v.get("name", "").strip().lower()
                  or v.get("name", "").strip().lower() in target]
        if substr:
            return max(substr, key=lambda v: v.get("id", 0))
    return vals[0]


def _today():
    return date.today().isoformat()


def _call(ctx, tools, tool_name, **kwargs):
    """Call a tool, track step, emit event, raise on error."""
    turn = len(ctx["steps"]) + 1
    # Filter None values from kwargs so tools use their defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if ctx.get("emit_fn"):
        ctx["emit_fn"]({"type": "tool_call", "request_id": ctx.get("request_id", ""),
                        "turn": turn, "tool": tool_name, "args": kwargs})
    result = tools[tool_name](**kwargs)
    is_err = isinstance(result, dict) and result.get("error")
    ctx["steps"].append({
        "tool": tool_name,
        "args": kwargs,
        "result": {"ok": not is_err, "data": str(result)[:500]},
    })
    if ctx.get("emit_fn"):
        ctx["emit_fn"]({"type": "tool_result", "request_id": ctx.get("request_id", ""),
                        "turn": turn, "tool": tool_name, "ok": not is_err,
                        "error": result.get("message", "") if is_err else ""})
    if is_err:
        raise PipelineError(f"{tool_name}: {result.get('message', result)}")
    return result


# ── Extraction Schemas ──────────────────────────────────────────
# Per task type: fields to extract + context hints for the LLM.

SCHEMAS = {
    # ── Group A: Single-call creates ────────────────────────────
    "create_employee": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
            "email": {"type": "str", "required": True},
            "userType": {"type": "str", "required": False, "default": "STANDARD",
                         "hint": "EXTENDED if kontoadministrator/administrator, NO_ACCESS if ingen tilgang"},
            "dateOfBirth": {"type": "date", "required": False, "default": ""},
            "phoneNumberMobile": {"type": "str", "required": False, "default": ""},
        },
        "context": "kontoadministrator/administrator -> userType=EXTENDED. ingen tilgang/no login -> NO_ACCESS.",
    },
    "create_customer": {
        "fields": {
            "name": {"type": "str", "required": True,
                     "hint": "Exact company name from prompt (keep AS/Ltd/GmbH suffix). Do NOT modify or expand the name."},
            "email": {"type": "str", "required": False, "default": ""},
            "organizationNumber": {"type": "str", "required": False, "default": "",
                                   "hint": "9-digit org number (organisasjonsnummer/org.nr/orgnr). Extract digits only."},
            "phoneNumber": {"type": "str", "required": False, "default": ""},
            "addressLine1": {"type": "str", "required": False, "default": "",
                            "hint": "Street address (gateadresse/adresse)"},
            "postalCode": {"type": "str", "required": False, "default": "",
                          "hint": "Postal/zip code (postnummer)"},
            "city": {"type": "str", "required": False, "default": "",
                    "hint": "City name (poststed/by)"},
        },
        "context": "Extract the EXACT company name from the prompt. Do NOT modify, expand or abbreviate it. Extract ALL address fields if present.",
    },
    "create_product": {
        "fields": {
            "name": {"type": "str", "required": True,
                    "hint": "Exact product/service name from prompt. Do NOT modify or translate."},
            "priceExcludingVatCurrency": {"type": "float", "required": True,
                                          "hint": "Price EXCLUDING VAT"},
            "productNumber": {"type": "str", "required": False, "default": ""},
            "vatPercentage": {"type": "int", "required": False, "default": 25,
                              "hint": "25=standard, 15=food/mat, 12=transport, 0=exempt"},
        },
        "context": "Extract ONLY priceExcludingVatCurrency. If only incl VAT given, divide by (1 + rate/100).",
    },
    "create_supplier": {
        "fields": {
            "name": {"type": "str", "required": True},
            "email": {"type": "str", "required": False, "default": ""},
            "organizationNumber": {"type": "str", "required": False, "default": ""},
            "phoneNumber": {"type": "str", "required": False, "default": ""},
            "addressLine1": {"type": "str", "required": False, "default": ""},
            "postalCode": {"type": "str", "required": False, "default": ""},
            "city": {"type": "str", "required": False, "default": ""},
        },
    },
    "create_department": {
        "fields": {
            "name": {"type": "str", "required": True},
            "departmentNumber": {"type": "str", "required": False, "default": "",
                                "hint": "Department number/code (avdelingsnummer/Abteilungsnummer/número de departamento/numéro de département). "
                                        "Numeric string e.g. '30', '400', '100'. Extract from prompt even if only mentioned as a number."},
        },
    },
    "create_multiple_departments": {
        "fields": {
            "departments": {"type": "list", "required": True,
                           "hint": "Array of department objects: [{name: str, departmentNumber: str}]. "
                                   "Extract ALL department names from the prompt. departmentNumber is optional."},
        },
    },

    # ── Group B: Updates + contacts ─────────────────────────────
    "create_contact": {
        "fields": {
            "customer_name": {"type": "str", "required": True, "hint": "Company/customer name"},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "firstName": {"type": "str", "required": True, "hint": "Contact person first name"},
            "lastName": {"type": "str", "required": True, "hint": "Contact person last name"},
            "email": {"type": "str", "required": True, "hint": "Contact person email"},
            "phoneNumberMobile": {"type": "str", "required": False, "default": ""},
        },
    },
    "update_employee": {
        "fields": {
            "firstName": {"type": "str", "required": True, "hint": "Employee first name"},
            "lastName": {"type": "str", "required": True, "hint": "Employee last name"},
            "email": {"type": "str", "required": True, "hint": "Employee email (for lookup)"},
            "new_phoneNumberMobile": {"type": "str", "required": False, "default": "",
                                      "hint": "New phone number to set"},
        },
    },
    "update_customer": {
        "fields": {
            "name": {"type": "str", "required": True, "hint": "Customer name (for lookup/create)"},
            "email": {"type": "str", "required": False, "default": "",
                       "hint": "Customer email (for create)"},
            "new_email": {"type": "str", "required": False, "default": ""},
            "new_phoneNumber": {"type": "str", "required": False, "default": ""},
        },
    },
    "update_product": {
        "fields": {
            "name": {"type": "str", "required": True, "hint": "Product name (for lookup/create)"},
            "priceExcludingVatCurrency": {"type": "float", "required": False, "default": 0,
                                          "hint": "Original price excl VAT for create"},
            "new_name": {"type": "str", "required": False, "default": ""},
            "new_priceExcludingVatCurrency": {"type": "float", "required": False, "default": 0,
                                               "hint": "New price excl VAT"},
        },
    },
    "update_supplier": {
        "fields": {
            "name": {"type": "str", "required": True, "hint": "Supplier name (for lookup/create)"},
            "email": {"type": "str", "required": False, "default": ""},
            "new_email": {"type": "str", "required": False, "default": ""},
            "new_phoneNumber": {"type": "str", "required": False, "default": ""},
        },
    },
    "update_department": {
        "fields": {
            "name": {"type": "str", "required": True, "hint": "Department name (for lookup/create)"},
            "departmentNumber": {"type": "str", "required": False, "default": ""},
            "new_name": {"type": "str", "required": False, "default": ""},
            "new_departmentNumber": {"type": "str", "required": False, "default": ""},
        },
    },
    "update_contact": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "firstName": {"type": "str", "required": True, "hint": "Contact first name"},
            "lastName": {"type": "str", "required": True, "hint": "Contact last name"},
            "email": {"type": "str", "required": True, "hint": "Contact email (for create)"},
            "new_email": {"type": "str", "required": False, "default": ""},
            "new_phoneNumberMobile": {"type": "str", "required": False, "default": ""},
        },
    },

    # ── Group C: Invoice workflows ──────────────────────────────
    "create_invoice": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": True},
            "product_number": {"type": "str", "required": False, "default": "",
                              "hint": "Product number/SKU if given."},
            "product_price": {"type": "float", "required": True, "hint": "Price EXCLUDING VAT"},
            "quantity": {"type": "int", "required": False, "default": 1},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date (fakturadato). Norwegian: '5. mars 2026' = 2026-03-05. "
                                    "Months: januar=01, februar=02, mars=03, april=04, mai=05, juni=06, "
                                    "juli=07, august=08, september=09, oktober=10, november=11, desember=12. Default today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Due date (forfallsdato/forfall). Norwegian: '26. mars 2026' = 2026-03-26. Default = invoice_date."},
            "send_invoice": {"type": "bool", "required": False, "default": False,
                            "hint": "True if prompt says send/og send/enviar/senden/envoyer"},
        },
        "context": "VAT: 25=standard, 15=food/mat, 12=transport, 0=exempt. Only extract priceExcludingVat.",
    },
    "create_multi_line_invoice": {
        "fields": {
            "customer_name": {"type": "str", "required": True,
                             "hint": "Customer company name. Extract EXACTLY as written (preserve spaces, AS/ANS/DA suffix)."},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "products": {"type": "list", "required": True,
                         "hint": "Array of ALL products listed in prompt: [{name: str, number: str (product number/SKU if given), "
                                 "price: float (excl VAT), quantity: int, vat_percentage: int}]. "
                                 "Extract EVERY product mentioned. Count carefully — do not miss any."},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date (fakturadato). Norwegian: '5. mars 2026' = 2026-03-05. Default today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Due date (forfallsdato/forfall). Default = invoice_date."},
            "send_invoice": {"type": "bool", "required": False, "default": False,
                            "hint": "True if prompt says send/og send/enviar/senden/envoyer"},
        },
        "context": "VAT: 25=standard, 15=food/mat, 12=transport, 0=exempt. Only extract priceExcludingVat per product.",
    },
    "invoice_with_payment": {
        "fields": {
            "is_existing_invoice": {"type": "bool", "required": True,
                                    "hint": "True if prompt says customer HAS an unpaid invoice. False if creating new."},
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": False, "default": "",
                            "hint": "Single product name. Ignore if multiple products (use 'products' list instead)."},
            "product_number": {"type": "str", "required": False, "default": "",
                              "hint": "Product number/SKU if given."},
            "product_price": {"type": "float", "required": False, "default": 0,
                            "hint": "Per-unit price EXCLUDING VAT. Extract the exact price number from the prompt. "
                                    "Look for: pris/price/precio/prix/Preis + 'eks mva'/'ex VAT'. This is the price for ONE unit."},
            "products": {"type": "list", "required": False, "default": None,
                        "hint": "Array of product objects when prompt lists MULTIPLE products: [{name, number, price, quantity, vat_percentage}]. null if single product."},
            "quantity": {"type": "int", "required": False, "default": 1,
                        "hint": "Number of units ordered. Look for: antall/stk/units/quantity."},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date. Look for: fakturadato/invoice date/fecha/date de facture/Rechnungsdatum. "
                                    "Norwegian date format: '2. mars 2026' = 2026-03-02. Default today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Payment due date. Look for: forfallsdato/due date/forfall/fecha de vencimiento/Fälligkeitsdatum/date d'échéance. "
                                "Default = invoice_date."},
            "payment_date": {"type": "date", "required": False,
                            "hint": "When payment was made. Look for: betalingsdato/payment date. Default today."},
            "payment_amount": {"type": "float", "required": False, "default": 0,
                              "hint": "Specific payment amount if stated (e.g. 'betaler 9000 kr'/'pays 9000'). 0 = pay full invoice amount."},
            "foreign_currency": {"type": "str", "required": False, "default": "",
                                "hint": "Currency code (EUR/USD/GBP) if prompt involves foreign currency. Empty if NOK only."},
            "foreign_amount": {"type": "float", "required": False, "default": 0,
                              "hint": "Invoice amount in foreign currency (e.g. 12301 for '12301 EUR'). 0 if NOK."},
            "old_exchange_rate": {"type": "float", "required": False, "default": 0,
                                 "hint": "Exchange rate at invoice time (e.g. 10.83 for 'kursen var 10.83 NOK/EUR'). 0 if NOK."},
            "new_exchange_rate": {"type": "float", "required": False, "default": 0,
                                 "hint": "Exchange rate at payment time (e.g. 11.83 for 'kursen er 11.83 NOK/EUR'). 0 if NOK."},
        },
        "context": "If prompt says 'has unpaid invoice'/'har en ubetalt faktura'/'facture impayee' -> is_existing_invoice=true. "
                   "If prompt lists MULTIPLE products, use the 'products' array and leave product_name/product_price empty. "
                   "VAT: 25=standard, 15=food/mat, 12=transport, 0=exempt. Extract priceExcludingVat (the number BEFORE VAT). "
                   "product_price is ALWAYS the per-unit amount before VAT, NOT the total line amount. "
                   "CURRENCY: If prompt mentions EUR/USD/GBP + exchange rates + agio/disagio/valutadifferanse, "
                   "extract foreign_currency, foreign_amount, old_exchange_rate, new_exchange_rate.",
    },
    "order_to_invoice_with_payment": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": False, "default": ""},
            "product_number": {"type": "str", "required": False, "default": ""},
            "product_price": {"type": "float", "required": False, "default": 0,
                            "hint": "Per-unit price EXCLUDING VAT. Extract the exact price number from the prompt."},
            "products": {"type": "list", "required": False, "default": None,
                        "hint": "Array of ALL product objects when prompt lists MULTIPLE products: "
                                "[{name, number, price, quantity, vat_percentage}]. null if single product. "
                                "Extract EVERY product. price = per-unit price EXCLUDING VAT."},
            "quantity": {"type": "int", "required": False, "default": 1},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date (fakturadato). Norwegian: '5. mars 2026' = 2026-03-05. Default today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Due date (forfallsdato/forfall). Default = invoice_date."},
            "payment_date": {"type": "date", "required": False, "hint": "Default today"},
        },
        "context": "Order-to-invoice workflow: create order, convert to invoice, register payment. "
                   "VAT: 25=standard, 15=food/mat, 12=transport, 0=exempt. Extract priceExcludingVat (the number BEFORE VAT). "
                   "product_price and products[].price are ALWAYS per-unit amounts before VAT, NOT total line amounts.",
    },
    "create_credit_note": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": True},
            "product_price": {"type": "float", "required": True},
            "quantity": {"type": "int", "required": False, "default": 1},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date (fakturadato). Norwegian: '10. mars 2026' = 2026-03-10. Default today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Due date (forfallsdato). Default = invoice_date."},
        },
    },
    "delete_invoice": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": True},
            "product_price": {"type": "float", "required": True},
            "quantity": {"type": "int", "required": False, "default": 1},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False},
            "due_date": {"type": "date", "required": False},
        },
    },

    # ── Group D: Employment/travel/supplier ─────────────────────
    "create_travel_expense": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
            "email": {"type": "str", "required": True},
            "title": {"type": "str", "required": True, "hint": "Travel expense title/description"},
            "departureDate": {"type": "date", "required": True},
            "returnDate": {"type": "date", "required": True},
        },
    },
    "create_travel_expense_with_costs": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
            "email": {"type": "str", "required": True},
            "title": {"type": "str", "required": True},
            "departureDate": {"type": "date", "required": True},
            "returnDate": {"type": "date", "required": True},
            "per_diem_rate": {"type": "float", "required": False, "default": 0,
                              "hint": "Daily allowance rate (diett/Tagegeld/per diem)"},
            "costs": {"type": "list", "required": False, "default": [],
                       "hint": "Array of {amount: float, category: str, comments: str, date: str}"},
            "mileage_km": {"type": "float", "required": False, "default": 0,
                           "hint": "Kilometers for mileage allowance (kjoeregodtgjoerelse)"},
            "accommodation_nights": {"type": "int", "required": False, "default": 0,
                                      "hint": "Number of nights for fixed-rate nattillegg"},
            "accommodation_location": {"type": "str", "required": False, "default": "",
                                        "hint": "Location for nattillegg (e.g. Oslo)"},
        },
        "context": "Tagegeld/diett = per diem. Auslagen/utlegg = expense costs. ONE per diem call covers the full trip. "
                   "overnatting/hotell with receipt = cost with category=accommodation. nattillegg = accommodation_allowance.",
    },
    "create_employee_with_employment": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
            "email": {"type": "str", "required": True},
            "dateOfBirth": {"type": "date", "required": False, "default": "",
                           "hint": "Date of birth (fodselsdato/né le/geboren am/fecha de nacimiento) in YYYY-MM-DD"},
            "nationalIdentityNumber": {"type": "str", "required": False, "default": "",
                                       "hint": "National identity number (personnummer/fodselsnummer/numero de identidad/DNI/RUT). 11-digit Norwegian or foreign ID."},
            "bankAccountNumber": {"type": "str", "required": False, "default": "",
                                  "hint": "Bank account number (bankkonto/kontonummer/cuenta bancaria). Norwegian 11-digit."},
            "department_name": {"type": "str", "required": False, "default": "",
                               "hint": "Department name (avdeling/departamento/Abteilung) from contract"},
            "startDate": {"type": "date", "required": True,
                         "hint": "Employment start date (tiltredelsesdato/startdato/ansettelsesdato/fecha de inicio/Startdatum/date de début). "
                                 "Convert to YYYY-MM-DD. Norwegian dates: '12. mars 2026' = 2026-03-12. "
                                 "Look for: starter/begynner/tiltrer/starts/commences/empieza/commence/beginnt."},
            "annualSalary": {"type": "float", "required": False, "default": 0,
                            "hint": "Annual salary (aarslonn/salario anual/Jahresgehalt)"},
            "percentageOfFullTimeEquivalent": {"type": "float", "required": False, "default": 100,
                                               "hint": "Employment percentage (stillingsprosent/porcentaje de empleo). 100 = full time."},
            "occupationCode": {"type": "str", "required": False, "default": "",
                              "hint": "Occupation/position code (yrkeskode/stillingskode/codigo de ocupacion/STYRK). Numeric code e.g. '3323', '2411'."},
            "hoursPerDay": {"type": "float", "required": False, "default": 0},
            "leave_startDate": {"type": "date", "required": False, "default": ""},
            "leave_endDate": {"type": "date", "required": False, "default": ""},
            "leave_type": {"type": "str", "required": False, "default": "",
                          "hint": "MILITARY_SERVICE, PARENTAL_LEAVE, EDUCATION, COMPASSIONATE, FURLOUGH, OTHER"},
            "leave_percentage": {"type": "float", "required": False, "default": 100},
            "jobTitle": {"type": "str", "required": False, "default": ""},
        },
    },
    "create_supplier_invoice": {
        "fields": {
            "supplier_name": {"type": "str", "required": True},
            "supplier_org_number": {"type": "str", "required": False, "default": ""},
            "supplier_bank_account": {"type": "str", "required": False, "default": "",
                                      "hint": "Supplier bank account number e.g. 19048571614"},
            "supplier_address": {"type": "str", "required": False, "default": "",
                                 "hint": "Street address e.g. Storgata 1"},
            "supplier_postal_code": {"type": "str", "required": False, "default": ""},
            "supplier_city": {"type": "str", "required": False, "default": ""},
            "invoice_number": {"type": "str", "required": False, "default": ""},
            "amount_including_vat": {"type": "float", "required": True},
            "expense_account": {"type": "int", "required": True,
                                "hint": "Expense account number e.g. 6590, 4000, 6300, 6800"},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": True},
            "due_date": {"type": "date", "required": False, "default": "",
                         "hint": "Payment due date (forfallsdato) YYYY-MM-DD"},
            "department_name": {"type": "str", "required": False, "default": "",
                                "hint": "Department name e.g. Økonomi, Salg, IT"},
            "line_description": {"type": "str", "required": False, "default": "",
                                 "hint": "What the invoice is for e.g. Kontorstoler, Nettverkstjenester"},
        },
        "context": "Common accounts: 4000=varekostnad, 6300=leie, 6590=annet driftsmateriale, 6800=kontorrekvisita.",
    },

    # ── Group E: Delete tasks ───────────────────────────────────
    "delete_customer": {
        "fields": {"name": {"type": "str", "required": True, "hint": "Customer name to delete"}},
    },
    "delete_supplier": {
        "fields": {"name": {"type": "str", "required": True, "hint": "Supplier name to delete"}},
    },
    "delete_product": {
        "fields": {"name": {"type": "str", "required": True, "hint": "Product name to delete"}},
    },
    "delete_department": {
        "fields": {"name": {"type": "str", "required": True, "hint": "Department name to delete"}},
    },
    "delete_contact": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
        },
    },
    "delete_employee": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
        },
    },
    "delete_travel_expense": {
        "fields": {
            "employee_firstName": {"type": "str", "required": False, "default": ""},
            "employee_lastName": {"type": "str", "required": False, "default": ""},
            "title": {"type": "str", "required": False, "default": "",
                       "hint": "Travel expense title to identify it"},
        },
    },

    # ── Group F: Complex tasks ──────────────────────────────────
    "correct_ledger_errors": {
        "fields": {
            "date_from": {"type": "date", "required": True, "hint": "Start of period to check (e.g. 2026-01-01)"},
            "date_to": {"type": "date", "required": True, "hint": "End of period to check (e.g. 2026-02-28)"},
            "errors": {"type": "list", "required": True,
                       "hint": """Array of error objects. EACH error = one object. Format:
[{type: "wrong_account", wrong_account: int, correct_account: int, amount: float},
 {type: "duplicate", account: int, amount: float},
 {type: "missing_vat", expense_account: int, vat_account: int, amount_excl_vat: float},
 {type: "wrong_amount", account: int, recorded_amount: float, correct_amount: float}]"""},
        },
        "context": """Extract EACH error as a separate object. Error types:
- wrong_account: account X used instead of Y, amount Z
- duplicate: duplicate posting on account X, amount Z
- missing_vat: expense on account X, amount excl VAT Z, missing VAT on account V
- wrong_amount: account X, recorded amount Z instead of correct amount W
Always extract amounts as positive numbers.""",
    },
    "create_ledger_voucher": {
        "fields": {
            "date": {"type": "date", "required": True, "hint": "Voucher date. Use the last day of the period mentioned (e.g. Feb -> 2026-02-28)."},
            "corrections": {"type": "list", "required": True,
                            "hint": "Array of correction objects. EACH error = one SEPARATE object. "
                                    "Format: [{description: str, postings: [{accountNumber: str, amount: float}]}]. "
                                    "Each object's postings must balance (sum=0). Positive=debit, negative=credit. "
                                    "CRITICAL: N errors in the prompt = N objects in this array. NEVER combine."},
        },
        "context": """CRITICAL: Each error becomes ONE separate correction object. N errors = N objects in the array.

CORRECTION PATTERNS (apply these rules to compute postings):

1. WRONG ACCOUNT (e.g. "Konto 7300 statt 7000, Betrag 7300"):
   The amount was booked to wrong account, should be on correct account.
   Postings: [{accountNumber: CORRECT, amount: +AMOUNT}, {accountNumber: WRONG, amount: -AMOUNT}]
   Example: [{accountNumber: "7000", amount: 7300}, {accountNumber: "7300", amount: -7300}]
   Description: "Korreksjon: feil konto WRONG→CORRECT"

2. DUPLICATE ENTRY (e.g. "doppelter Beleg, Konto 7000, Betrag 4600"):
   An entry was booked twice, reverse the duplicate.
   Postings: [{accountNumber: EXPENSE, amount: -AMOUNT}, {accountNumber: "1920", amount: +AMOUNT}]
   Example: [{accountNumber: "7000", amount: -4600}, {accountNumber: "1920", amount: 4600}]
   Description: "Korreksjon: duplikat konto EXPENSE"

3. MISSING VAT LINE (e.g. "fehlende MwSt, Konto 6540, Betrag ohne MwSt 15600, fehlende MwSt auf 2710"):
   Net amount was booked but the VAT line (25%) is missing. Bank was underpaid by VAT amount.
   VAT = net_amount * 0.25
   Postings: [{accountNumber: "2710", amount: +VAT}, {accountNumber: "1920", amount: -VAT}]
   Example (net=15600, VAT=3900): [{accountNumber: "2710", amount: 3900}, {accountNumber: "1920", amount: -3900}]
   Description: "Korreksjon: manglende MVA konto EXPENSE"

4. WRONG AMOUNT (e.g. "falscher Betrag, Konto 6540, 15200 gebucht statt 13400"):
   Too much was booked. Reverse the excess (booked - correct).
   EXCESS = booked_amount - correct_amount
   Postings: [{accountNumber: EXPENSE, amount: -EXCESS}, {accountNumber: "1920", amount: +EXCESS}]
   Example (15200-13400=1800): [{accountNumber: "6540", amount: -1800}, {accountNumber: "1920", amount: 1800}]
   Description: "Korreksjon: feil beløp konto EXPENSE"

Counter-account is usually 1920 (bank). Use 1500 for receivables, 2400 for payables.""",
    },
    "reverse_voucher": {
        "fields": {
            "search_description": {"type": "str", "required": False, "default": "",
                                   "hint": "Voucher description to search for"},
            "search_dateFrom": {"type": "date", "required": False, "default": ""},
            "search_dateTo": {"type": "date", "required": False, "default": ""},
            "reversal_date": {"type": "date", "required": False, "hint": "Date for the reversal"},
        },
    },
    "reverse_payment": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "payment_date": {"type": "date", "required": False, "hint": "Date for the reversal"},
        },
        "context": "Reverse by registering NEGATIVE payment amount. Search invoices to find the one to reverse.",
    },
    "create_opening_balance": {
        "fields": {
            "date": {"type": "date", "required": True, "hint": "Opening balance date (first day of month)"},
            "postings": {"type": "list", "required": True,
                         "hint": "Array of {accountNumber: str, amount: float}"},
        },
    },
    "create_dimension": {
        "fields": {
            "dimension_name": {"type": "str", "required": True, "hint": "Name of the accounting dimension (e.g. Kostsenter, Region, Marked, Satsingsområde)"},
            "values": {"type": "list", "required": True,
                       "hint": "Array of dimension value names (strings), e.g. [\"IT\", \"HR\"]"},
            "account_number": {"type": "str", "required": False, "default": "",
                              "hint": "Expense account number for the voucher (e.g. 6340, 7140, 6590, 7300). "
                                      "Extract from prompt — look for 'konto', 'account', 'Konto', 'compte'."},
            "amount": {"type": "float", "required": True,
                      "hint": "Voucher amount in NOK for the posting linked to the dimension. "
                              "Extract the monetary value from the prompt."},
            "linked_value": {"type": "str", "required": False, "default": "",
                            "hint": "Which dimension value to link the voucher posting to. "
                                    "Must be one of the values in the 'values' array."},
            "voucher_date": {"type": "date", "required": False, "default": "",
                            "hint": "Date for the voucher (today if not specified)"},
            "voucher_description": {"type": "str", "required": False, "default": ""},
        },
        "context": "Fri rekneskapsdimensjon/custom accounting dimension/dimension comptable personnalisée. "
                   "Create dimension, then values, then a voucher linked to a value. "
                   "IMPORTANT: Extract the account_number, amount, and linked_value from the prompt. "
                   "If no linked_value is specified, use the first value from the values list.",
    },
    "create_project": {
        "fields": {
            "project_name": {"type": "str", "required": True,
                            "hint": "Exact project name from prompt. Do NOT modify."},
            "customer_name": {"type": "str", "required": True,
                            "hint": "The COMPANY/customer name (ending in AS/DA/ANS/Ltd/GmbH/SA). "
                                    "This is NOT the project name. Look for 'kunde'/'customer'/'klient'/'client' or a company name with legal suffix."},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "pm_firstName": {"type": "str", "required": False, "default": ""},
            "pm_lastName": {"type": "str", "required": False, "default": ""},
            "pm_email": {"type": "str", "required": False, "default": ""},
            "startDate": {"type": "date", "required": True,
                         "hint": "Project start date in YYYY-MM-DD. MUST extract from prompt. "
                                 "Look for: startdato/oppstart/start date/fecha de inicio/Startdatum/commence. "
                                 "This is a SPECIFIC date in the prompt, NOT today's date."},
            "fixedPriceAmount": {"type": "float", "required": False, "default": 0},
            "description": {"type": "str", "required": True,
                           "hint": "Project description text. MUST extract from prompt. "
                                   "Look for: beskrivelse/description/descripción/Beschreibung. "
                                   "Copy the FULL description sentence(s) exactly from the prompt. "
                                   "If no explicit description, use the project context/purpose mentioned."},
            "isInternal": {"type": "bool", "required": False, "default": False,
                           "hint": "True if 'internal project'/'internt prosjekt' mentioned"},
        },
        "context": "IMPORTANT: Extract ALL fields carefully from the prompt. "
                   "customer_name = the company/business name (with AS/DA/ANS/Ltd suffix) — distinct from the project name. "
                   "startDate = the specific project start date mentioned in the prompt (a concrete date, NOT today). "
                   "description = copy the project description text exactly as written in the prompt.",
    },
    "create_project_with_pm": {
        "fields": {
            "project_name": {"type": "str", "required": True,
                            "hint": "Exact project name from prompt. Do NOT modify."},
            "customer_name": {"type": "str", "required": True,
                            "hint": "The COMPANY/customer name (ending in AS/DA/ANS/Ltd/GmbH/SA). "
                                    "This is NOT the project name and NOT the PM's name."},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "pm_firstName": {"type": "str", "required": True,
                           "hint": "Project manager's first name. Look for: prosjektleder/PM/project manager."},
            "pm_lastName": {"type": "str", "required": True,
                          "hint": "Project manager's last name."},
            "pm_email": {"type": "str", "required": True,
                        "hint": "Project manager's email address."},
            "pm_phoneNumberMobile": {"type": "str", "required": False, "default": "",
                                    "hint": "PM mobile phone number if mentioned in prompt."},
            "startDate": {"type": "date", "required": False,
                         "hint": "Project start date in YYYY-MM-DD. Look for: startdato/oppstart/start date/fecha de inicio/Startdatum. "
                                 "This is a SPECIFIC date in the prompt, NOT today's date."},
            "fixedPriceAmount": {"type": "float", "required": False, "default": 0},
            "description": {"type": "str", "required": False, "default": "",
                           "hint": "Project description text. Copy FULL text exactly from prompt."},
            "isInternal": {"type": "bool", "required": False, "default": False,
                           "hint": "True if 'internal project'/'internt prosjekt' mentioned"},
        },
        "context": "IMPORTANT: Extract ALL fields. customer_name = company name (with AS/Ltd suffix). "
                   "PM = the person named as prosjektleder/project manager. "
                   "startDate = specific project start date from prompt (NOT today).",
    },
    "project_invoice": {
        "fields": {
            "project_name": {"type": "str", "required": True,
                            "hint": "Exact project name from prompt. Do NOT modify."},
            "customer_name": {"type": "str", "required": True,
                            "hint": "The COMPANY/customer name (ending in AS/DA/ANS/Ltd/GmbH/SA)."},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "pm_firstName": {"type": "str", "required": False, "default": ""},
            "pm_lastName": {"type": "str", "required": False, "default": ""},
            "pm_email": {"type": "str", "required": False, "default": ""},
            "startDate": {"type": "date", "required": False,
                         "hint": "Project start date (startdato/oppstart). YYYY-MM-DD. NOT the invoice date."},
            "fixedPriceAmount": {"type": "float", "required": True,
                                "hint": "Fixed price amount for the project (fastpris/prix fixe/Festpreis/precio fijo). "
                                        "Extract the total project price/budget/invoice amount. "
                                        "If no explicit fixed price, use the product_price or total amount to invoice."},
            "isInternal": {"type": "bool", "required": False, "default": False,
                           "hint": "True if 'internal project'/'internt prosjekt' mentioned"},
            "product_name": {"type": "str", "required": False, "default": "",
                            "hint": "Invoice line item name. If not explicit, derive from project name."},
            "product_price": {"type": "float", "required": False, "default": 0,
                             "hint": "Invoice amount excluding VAT. Use fixedPriceAmount if no separate product price."},
            "invoice_date": {"type": "date", "required": False,
                            "hint": "Invoice date (fakturadato/date de facturation/Rechnungsdatum). YYYY-MM-DD. "
                                    "Look for the SPECIFIC date in the prompt — do NOT default to today."},
            "due_date": {"type": "date", "required": False,
                        "hint": "Invoice due date (forfallsdato/betalingsfrist/echeance). YYYY-MM-DD. "
                                "Look for: 'forfaller'/'due date'/'betales innen'. Calculate from invoice_date + N days if relative."},
            "send_invoice": {"type": "bool", "required": False, "default": False},
            "hourly_rate": {"type": "float", "required": False, "default": 0,
                            "hint": "Hourly rate / timepris / taux horaire / Stundensatz"},
            "hours": {"type": "float", "required": False, "default": 0,
                      "hint": "Number of hours to log / timer / heures / Stunden"},
            "employee_firstName": {"type": "str", "required": False, "default": "",
                                   "hint": "Employee who logs hours (may differ from PM)"},
            "employee_lastName": {"type": "str", "required": False, "default": ""},
            "employee_email": {"type": "str", "required": False, "default": ""},
            "activity_name": {"type": "str", "required": False, "default": "",
                              "hint": "Activity name for timesheet (e.g. Design, Development, Consulting)"},
            "milestone_percentage": {"type": "float", "required": False, "default": 0,
                                     "hint": "Percentage of fixed price to invoice (e.g. 25 for 25%)"},
        },
    },
    "project_lifecycle": {
        "fields": {
            "project_name": {"type": "str", "required": True},
            "customer_name": {"type": "str", "required": True},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "pm_firstName": {"type": "str", "required": False, "default": ""},
            "pm_lastName": {"type": "str", "required": False, "default": ""},
            "pm_email": {"type": "str", "required": False, "default": ""},
            "budget": {"type": "float", "required": False, "default": 0},
            "employees": {"type": "list", "required": False, "default": [],
                          "hint": "Array of {firstName, lastName, email, hours}"},
            "supplier_name": {"type": "str", "required": False, "default": ""},
            "supplier_org_number": {"type": "str", "required": False, "default": ""},
            "supplier_cost": {"type": "float", "required": False, "default": 0},
            "create_customer_invoice": {"type": "bool", "required": False, "default": False},
        },
    },
    "create_project_with_billing": {
        "fields": {
            "project_name": {"type": "str", "required": True},
            "customer_name": {"type": "str", "required": True},
            "customer_org_number": {"type": "str", "required": False, "default": ""},
            "pm_firstName": {"type": "str", "required": False, "default": "",
                             "hint": "Project manager first name. Identify from role: Projektleiter/project manager/prosjektleder"},
            "pm_lastName": {"type": "str", "required": False, "default": ""},
            "pm_email": {"type": "str", "required": False, "default": ""},
            "budget": {"type": "float", "required": False, "default": 0,
                       "hint": "Project budget amount in NOK"},
            "employees": {"type": "list", "required": False, "default": [],
                          "hint": "Array of ALL employees with hours: [{firstName, lastName, email, hours, role}]. "
                                  "Include the PM here too if they have hours."},
            "supplier_name": {"type": "str", "required": False, "default": ""},
            "supplier_org_number": {"type": "str", "required": False, "default": ""},
            "supplier_cost": {"type": "float", "required": False, "default": 0,
                              "hint": "Supplier cost amount in NOK (including VAT)"},
            "create_customer_invoice": {"type": "bool", "required": False, "default": False,
                                        "hint": "True if prompt asks for customer invoice / Kundenrechnung / kundefaktura"},
        },
        "context": "Extract PM from role mentions: Projektleiter/project manager/prosjektleder/chef de projet/jefe de proyecto. "
                   "Budget is the project fixed price amount. Supplier cost is amount INCLUDING VAT. "
                   "Set create_customer_invoice=True if prompt mentions Kundenrechnung/customer invoice/kundefaktura/facture client.",
    },
    "salary_with_bonus": {
        "fields": {
            "firstName": {"type": "str", "required": True},
            "lastName": {"type": "str", "required": True},
            "email": {"type": "str", "required": False, "default": "",
                      "hint": "Employee email. Generate as firstname.lastname@example.com if not given"},
            "dateOfBirth": {"type": "date", "required": False, "default": "",
                            "hint": "Date of birth YYYY-MM-DD"},
            "department_name": {"type": "str", "required": False, "default": "",
                                "hint": "Department/avdeling name"},
            "startDate": {"type": "date", "required": False, "default": "",
                          "hint": "Employment start date (tiltredelse) YYYY-MM-DD"},
            "annualSalary": {"type": "float", "required": False, "default": 0,
                             "hint": "Annual salary (arslonn/Jahresgehalt/salario anual)"},
            "percentageOfFullTimeEquivalent": {"type": "float", "required": False, "default": 100,
                                               "hint": "FTE percentage (stillingsprosent)"},
            "hoursPerDay": {"type": "float", "required": False, "default": 0,
                            "hint": "Standard working hours per day (arbeidstid)"},
            "year": {"type": "int", "required": True,
                     "hint": "Salary year — derive from startDate or current year (2026)"},
            "month": {"type": "int", "required": True,
                      "hint": "Salary month 1-12 — derive from startDate month or current month"},
            "base_salary": {"type": "float", "required": False, "default": 0,
                            "hint": "Monthly base salary (Fastlonn/Grundgehalt/salaire de base). "
                                    "If prompt mentions a single salary amount, put it here."},
            "bonus": {"type": "float", "required": False, "default": 0,
                      "hint": "Bonus amount (bonus/tillegg/Zulage/prime/bonificación)"},
            "amount": {"type": "float", "required": False, "default": 0,
                       "hint": "Generic salary amount if not clearly base_salary or bonus. Fallback field."},
        },
        "context": "Extract employee data from offer letter (tilbudsbrev) or salary prompt. "
                   "If email is missing, generate as firstname.lastname@example.com (lowercase). "
                   "year and month: extract from explicit date, or from 'this month'/'diesen Monat'/'ce mois'/'este mes'/'este mês' = current month/year. "
                   "IMPORTANT: The salary amount MUST go into base_salary (for Fastlønn/fixed salary) or bonus (for bonus). "
                   "If the prompt mentions a general salary amount without specifying type, put it in base_salary. "
                   "If both base salary and bonus are mentioned, extract each separately.",
    },
    "year_end": {
        "fields": {
            "voucher_date": {"type": "date", "required": True, "hint": "Year-end date e.g. 2025-12-31"},
            "is_monthly": {"type": "bool", "required": False, "default": False,
                          "hint": "True if monthly closing (clôture mensuelle/Monatsabschluss/månedlig avslutning). False for annual."},
            "assets": {"type": "list", "required": False, "default": [],
                       "hint": "Array of {name, cost, years, expense_account, depreciation_account}"},
            "prepaid_expenses": {"type": "list", "required": False, "default": [],
                                 "hint": "Array of {amount, prepaid_account, expense_account}"},
            "salary_provisions": {"type": "list", "required": False, "default": [],
                                  "hint": "Array of {amount, expense_account, payable_account}. Salary accruals (lønnsavsetning/provision pour salaires/Gehaltsrückstellung)."},
            "tax_rate": {"type": "float", "required": False, "default": 0.22},
            "tax_expense_account": {"type": "str", "required": False, "default": "8700"},
            "tax_liability_account": {"type": "str", "required": False, "default": "2920"},
            "taxable_profit": {"type": "float", "required": False, "default": 0},
            "year_end_note": {"type": "str", "required": False, "default": ""},
            "verify_balance": {"type": "bool", "required": False, "default": False,
                              "hint": "True if prompt says to check/verify trial balance (kontroller/vérifiez/prüfen/balanse)"},
        },
        "context": "Year-end or monthly closing task. Monthly closing = is_monthly true, depreciation divided by 12. "
                   "Salary provision: if prompt mentions salary accrual but no amount, use 45000 NOK. "
                   "Account numbers: extract from prompt. If not specified, set to empty string (pipeline will look up).",
    },
    "reminder_fee": {
        "fields": {
            "fee_amount": {"type": "float", "required": True, "hint": "Reminder fee amount in NOK (e.g. 35)"},
            "debit_account": {"type": "str", "required": False, "default": "1500",
                              "hint": "Debit account number (e.g. 1500 kundefordringer)"},
            "credit_account": {"type": "str", "required": False, "default": "3400",
                               "hint": "Credit account number (e.g. 3400 purregebyr-inntekt)"},
            "partial_payment_amount": {"type": "float", "required": False, "default": 0,
                                       "hint": "Partial payment amount on the overdue invoice (0 if none)"},
            "send_invoice": {"type": "bool", "required": False, "default": False,
                            "hint": "True if prompt says send/envoyer/senden/enviar"},
        },
        "context": "Reminder fee task. Extract fee amount and account numbers. "
                   "partial_payment_amount = any partial payment on the overdue invoice (delbetaling/paiement partiel). "
                   "send_invoice = true if prompt says to send the reminder fee invoice.",
    },
    "bank_reconciliation": {
        "fields": {
            "filename": {"type": "str", "required": True,
                         "hint": "Filename of the CSV/PDF bank statement attachment"},
        },
        "context": "Extract ONLY the filename of the bank statement attachment. "
                   "The pipeline will read and parse the CSV content itself.",
    },
    "process_invoice_file": {
        "fields": {
            "customer_name": {"type": "str", "required": True},
            "customer_email": {"type": "str", "required": False, "default": ""},
            "product_name": {"type": "str", "required": True},
            "product_price": {"type": "float", "required": True},
            "quantity": {"type": "int", "required": False, "default": 1},
            "vat_percentage": {"type": "int", "required": False, "default": 25},
            "invoice_date": {"type": "date", "required": False},
            "due_date": {"type": "date", "required": False},
        },
    },
    "register_expense_receipt": {
        "fields": {
            "amount_including_vat": {"type": "float", "required": True,
                                     "hint": "Total receipt amount INCLUDING VAT"},
            "expense_account": {"type": "int", "required": True,
                                "hint": "Expense account number: 6300=leie, 6500=verktøy, 6590=kontor, 6800=kontorrekvisita, 6900=telefon, 7100=bil, 7140=reise, 7350=representasjon"},
            "vat_percentage": {"type": "int", "required": False, "default": 25,
                               "hint": "VAT rate: 25=standard, 15=food, 12=transport, 0=exempt"},
            "receipt_date": {"type": "date", "required": True, "hint": "Receipt date YYYY-MM-DD"},
            "description": {"type": "str", "required": True, "hint": "What was purchased"},
            "payment_account": {"type": "int", "required": False, "default": 1920,
                                "hint": "1920=bank (betalt med kort), 1900=cash (betalt kontant)"},
            "department_name": {"type": "str", "required": False, "default": "",
                                "hint": "Department name if expense should be posted to a department"},
        },
        "context": "Expense receipt (kvittering/utlegg). Payment account: 1920=bank/kort, 1900=kontant/cash. "
                   "Common expense accounts: 6300=leie, 6500=verktøy, 6590=kontor, 6800=kontorrekvisita, "
                   "6900=telefon, 7100=bil, 7140=reise, 7350=representasjon.",
    },
}


# ── Parameter Extraction (ONE Gemini call) ──────────────────────

_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client()
    return _genai_client


def extract_params(prompt: str, task_type: str) -> dict:
    """Extract structured parameters from a prompt using Gemini Flash.

    Returns dict of parameter values. Raises ExtractionError on failure.
    """
    schema = SCHEMAS.get(task_type)
    if not schema:
        raise ExtractionError(f"No schema for task type: {task_type}")

    today = _today()

    # Build field descriptions for the extraction prompt
    field_lines = []
    for name, spec in schema["fields"].items():
        line = f'  "{name}": {spec["type"]}'
        if spec.get("hint"):
            line += f'  // {spec["hint"]}'
        if spec.get("required"):
            line += " [REQUIRED]"
        else:
            default = spec.get("default")
            if default is not None:
                line += f" [default: {json.dumps(default)}]"
        field_lines.append(line)

    context = schema.get("context", "")

    system_prompt = f"""You extract structured data from Norwegian/multilingual accounting task prompts.
Return ONLY a JSON object with these fields:
{{
{chr(10).join(field_lines)}
}}

{context}

NORWEGIAN DATES: Convert '5. mars 2026' → '2026-03-05'. Months: januar=01, februar=02, mars=03, april=04, mai=05, juni=06, juli=07, august=08, september=09, oktober=10, november=11, desember=12.
CUSTOMER NAMES: Extract EXACTLY as written including spaces and suffixes (AS, ANS, DA).
PRICES: Always extract as positive numbers (never negative). Price is per-unit EXCLUDING VAT.

RULES:
- Extract values EXACTLY as written in the prompt (preserve Norwegian characters)
- Dates: YYYY-MM-DD format. If no date given, use "{today}"
- Numbers: extract as numbers (not strings)
- Booleans: true/false
- For missing optional fields, use the default value
- For missing required fields, set to null
- Return ONLY a valid JSON object, nothing else"""

    from google import genai as _genai
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout

    try:
        client = _get_genai_client()
        def _do_extract():
            return client.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=prompt,
                config=_genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=EXTRACTION_TEMPERATURE,
                ),
            )
        with ThreadPoolExecutor(1) as pool:
            response = pool.submit(_do_extract).result(timeout=60)
        raw = response.text
        params = json.loads(raw)
    except _FutureTimeout:
        raise ExtractionError("Extraction timed out after 60s")
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Invalid JSON from extraction: {e}")
    except Exception as e:
        raise ExtractionError(f"Extraction call failed: {e}")

    # Post-process: apply defaults and coerce types
    for name, spec in schema["fields"].items():
        val = params.get(name)

        # Apply defaults for missing/null fields
        if val is None or val == "":
            if spec.get("required"):
                raise ExtractionError(f"Required field '{name}' not found in prompt")
            params[name] = spec.get("default")
            continue

        # Coerce types
        try:
            if spec["type"] == "int" and not isinstance(val, int):
                params[name] = int(float(val))
            elif spec["type"] == "float" and not isinstance(val, (int, float)):
                params[name] = float(val)
            elif spec["type"] == "bool" and isinstance(val, str):
                params[name] = val.lower() in ("true", "1", "yes", "ja")
            elif spec["type"] == "date" and isinstance(val, str):
                if val.lower() in ("today", "i dag", "hoy", "aujourd'hui", "heute", "hoje"):
                    params[name] = today
        except (ValueError, TypeError):
            pass  # keep original value, pipeline will fail if it's wrong

    # Fill in date defaults
    for name, spec in schema["fields"].items():
        if spec["type"] == "date" and not params.get(name):
            params[name] = today

    return params


# ── Pipeline Functions ──────────────────────────────────────────

# --- Group A: Single-call creates ---

def _pipeline_create_employee(tools, p, ctx):
    return _call(ctx, tools, "create_employee",
                 firstName=p["firstName"], lastName=p["lastName"],
                 email=p["email"], userType=p.get("userType") or "STANDARD",
                 dateOfBirth=p.get("dateOfBirth") or "",
                 phoneNumberMobile=p.get("phoneNumberMobile") or "")


def _pipeline_create_customer(tools, p, ctx):
    return _call(ctx, tools, "create_customer",
                 name=p["name"], email=p.get("email") or "",
                 organizationNumber=p.get("organizationNumber") or "",
                 phoneNumber=p.get("phoneNumber") or "",
                 addressLine1=p.get("addressLine1") or "",
                 postalCode=p.get("postalCode") or "",
                 city=p.get("city") or "")


def _pipeline_create_product(tools, p, ctx):
    return _call(ctx, tools, "create_product",
                 name=p["name"],
                 priceExcludingVatCurrency=p["priceExcludingVatCurrency"],
                 productNumber=p.get("productNumber") or "",
                 vatPercentage=p.get("vat_percentage") if "vat_percentage" in p else p.get("vatPercentage", 25))


def _pipeline_create_supplier(tools, p, ctx):
    return _call(ctx, tools, "create_supplier",
                 name=p["name"], email=p.get("email") or "",
                 organizationNumber=p.get("organizationNumber") or "",
                 phoneNumber=p.get("phoneNumber") or "",
                 addressLine1=p.get("addressLine1") or "",
                 postalCode=p.get("postalCode") or "",
                 city=p.get("city") or "")


def _pipeline_create_department(tools, p, ctx):
    return _call(ctx, tools, "create_department",
                 name=p["name"], departmentNumber=p.get("departmentNumber") or "")


def _pipeline_create_multiple_departments(tools, p, ctx):
    """Create multiple departments in a single pipeline."""
    departments = p.get("departments") or []
    last_result = None
    for dept in departments:
        name = dept.get("name") or dept if isinstance(dept, str) else dept.get("name", "")
        if not name:
            continue
        dept_num = dept.get("departmentNumber", "") if isinstance(dept, dict) else ""
        last_result = _call(ctx, tools, "create_department",
                            name=name, departmentNumber=dept_num)
    return last_result


# --- Group B: Updates + contacts ---

def _pipeline_create_contact(tools, p, ctx):
    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"], email=p.get("customer_email") or "")
    cust_id = _id(cust)
    return _call(ctx, tools, "create_contact",
                 firstName=p["firstName"], lastName=p["lastName"],
                 email=p["email"], customer_id=cust_id,
                 phoneNumberMobile=p.get("phoneNumberMobile") or "")


def _pipeline_update_employee(tools, p, ctx):
    emp = _call(ctx, tools, "create_employee",
                firstName=p["firstName"], lastName=p["lastName"], email=p["email"])
    emp_id = _id(emp)
    ver = _version(emp)
    return _call(ctx, tools, "update_employee",
                 employee_id=emp_id, firstName=p["firstName"], lastName=p["lastName"],
                 phoneNumberMobile=p.get("new_phoneNumberMobile") or "",
                 version=ver)


def _pipeline_update_customer(tools, p, ctx):
    cust = _call(ctx, tools, "create_customer",
                 name=p["name"], email=p.get("email") or "")
    cust_id = _id(cust)
    ver = _version(cust)
    return _call(ctx, tools, "update_customer",
                 customer_id=cust_id, name=p["name"],
                 email=p.get("new_email") or "",
                 phoneNumber=p.get("new_phoneNumber") or "",
                 version=ver)


def _pipeline_update_product(tools, p, ctx):
    prod = _call(ctx, tools, "create_product",
                 name=p["name"],
                 priceExcludingVatCurrency=p.get("priceExcludingVatCurrency") or 0)
    prod_id = _id(prod)
    ver = _version(prod)
    return _call(ctx, tools, "update_product",
                 product_id=prod_id, name=p.get("new_name") or p["name"],
                 priceExcludingVatCurrency=p.get("new_priceExcludingVatCurrency") or 0,
                 version=ver)


def _pipeline_update_supplier(tools, p, ctx):
    sup = _call(ctx, tools, "create_supplier",
                name=p["name"], email=p.get("email") or "")
    sup_id = _id(sup)
    ver = _version(sup)
    return _call(ctx, tools, "update_supplier",
                 supplier_id=sup_id, name=p["name"],
                 email=p.get("new_email") or "",
                 phoneNumber=p.get("new_phoneNumber") or "",
                 version=ver)


def _pipeline_update_department(tools, p, ctx):
    dept = _call(ctx, tools, "create_department",
                 name=p["name"], departmentNumber=p.get("departmentNumber") or "")
    dept_id = _id(dept)
    ver = _version(dept)
    return _call(ctx, tools, "update_department",
                 department_id=dept_id,
                 name=p.get("new_name") or p["name"],
                 departmentNumber=p.get("new_departmentNumber") or "",
                 version=ver)


def _pipeline_update_contact(tools, p, ctx):
    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"], email=p.get("customer_email") or "")
    cust_id = _id(cust)
    contact = _call(ctx, tools, "create_contact",
                    firstName=p["firstName"], lastName=p["lastName"],
                    email=p["email"], customer_id=cust_id)
    contact_id = _id(contact)
    ver = _version(contact)
    return _call(ctx, tools, "update_contact",
                 contact_id=contact_id,
                 firstName=p["firstName"], lastName=p["lastName"],
                 email=p.get("new_email") or "",
                 phoneNumberMobile=p.get("new_phoneNumberMobile") or "",
                 version=ver, customer_id=cust_id)


# --- Group C: Invoice workflows ---

def _create_invoice_common(tools, p, ctx):
    """Shared logic: create customer + product + order + invoice. Returns invoice result."""
    today = _today()
    inv_date = p.get("invoice_date") or today
    due_date = p.get("due_date") or inv_date

    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"], email=p.get("customer_email") or "",
                 organizationNumber=p.get("customer_org_number") or "")
    cust_id = _id(cust)

    prod = _call(ctx, tools, "create_product",
                 name=p["product_name"],
                 priceExcludingVatCurrency=p["product_price"],
                 vatPercentage=p.get("vat_percentage", 25),
                 productNumber=p.get("product_number") or "")
    prod_id = _id(prod)

    order_lines = json.dumps([{"product_id": prod_id, "count": p.get("quantity", 1)}])
    order = _call(ctx, tools, "create_order",
                  customer_id=cust_id, deliveryDate=inv_date, orderLines=order_lines)
    order_id = _id(order)

    inv = _call(ctx, tools, "create_invoice",
                invoiceDate=inv_date, invoiceDueDate=due_date, order_id=order_id)
    return inv


def _build_products_json(p):
    """Build products JSON for process_invoice from extracted params.

    Handles both single-product (create_invoice) and multi-product (create_multi_line_invoice)
    extraction schemas.
    """
    # Multi-product: products list already available
    if p.get("products"):
        products = p["products"]
        return json.dumps([{
            "name": pr.get("name", "Product"),
            "price": abs(pr.get("price", 0) or 0),
            "quantity": max(pr.get("quantity", 1) or 1, 1),
            "vatPercentage": pr.get("vat_percentage", 25),
            "productNumber": pr.get("number") or "",
        } for pr in products])

    # Single-product: create_invoice schema
    return json.dumps([{
        "name": p.get("product_name", "Product"),
        "price": abs(p.get("product_price", 0) or 0),
        "quantity": max(p.get("quantity", 1) or 1, 1),
        "vatPercentage": p.get("vat_percentage", 25),
        "productNumber": p.get("product_number") or "",
    }])


def _pipeline_create_invoice(tools, p, ctx):
    products_json = _build_products_json(p)
    result = _call(ctx, tools, "process_invoice",
                   customer_name=p["customer_name"],
                   customer_email=p.get("customer_email") or "",
                   customer_org_number=p.get("customer_org_number") or "",
                   products=products_json,
                   invoiceDate=p.get("invoice_date") or "",
                   invoiceDueDate=p.get("due_date") or "",
                   send_invoice=bool(p.get("send_invoice")))
    return result


def _pipeline_create_multi_line_invoice(tools, p, ctx):
    products_json = _build_products_json(p)
    result = _call(ctx, tools, "process_invoice",
                   customer_name=p["customer_name"],
                   customer_email=p.get("customer_email") or "",
                   customer_org_number=p.get("customer_org_number") or "",
                   products=products_json,
                   invoiceDate=p.get("invoice_date") or "",
                   invoiceDueDate=p.get("due_date") or "",
                   send_invoice=bool(p.get("send_invoice")))
    return result


def _create_multi_product_invoice(tools, p, ctx):
    """Create an invoice with multiple products. Returns invoice result."""
    today = _today()
    inv_date = p.get("invoice_date") or today
    due_date = p.get("due_date") or inv_date

    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"], email=p.get("customer_email") or "",
                 organizationNumber=p.get("customer_org_number") or "")
    cust_id = _id(cust)

    order_lines = []
    for prod_spec in p["products"]:
        prod = _call(ctx, tools, "create_product",
                     name=prod_spec.get("name", ""),
                     priceExcludingVatCurrency=prod_spec.get("price", 0),
                     vatPercentage=prod_spec.get("vat_percentage", 25),
                     productNumber=prod_spec.get("number", ""))
        prod_id = _id(prod)
        order_lines.append({"product_id": prod_id, "count": prod_spec.get("quantity", 1)})

    order = _call(ctx, tools, "create_order",
                  customer_id=cust_id, deliveryDate=inv_date,
                  orderLines=json.dumps(order_lines))
    order_id = _id(order)

    return _call(ctx, tools, "create_invoice",
                 invoiceDate=inv_date, invoiceDueDate=due_date, order_id=order_id)


def _pipeline_invoice_with_payment(tools, p, ctx):
    today = _today()
    payment_date = p.get("payment_date") or today

    # Detect foreign currency from extracted params OR from regex pre-computation
    foreign_amount = p.get("foreign_amount", 0) or 0
    old_rate = p.get("old_exchange_rate", 0) or 0
    new_rate = p.get("new_exchange_rate", 0) or 0
    has_currency = foreign_amount > 0 and old_rate > 0 and new_rate > 0

    # Also check ctx for pre-computed currency info (from tool_router.extract_currency_info)
    if not has_currency and ctx.get("currency_info"):
        ci = ctx["currency_info"]
        foreign_amount = ci["amount"]
        old_rate = ci["old_rate"]
        new_rate = ci["new_rate"]
        has_currency = True

    if p.get("is_existing_invoice"):
        # Existing invoice flow: search → pay (optionally with currency/agio)
        cust_result = _call(ctx, tools, "search_customers", name=p["customer_name"])
        cust = _first(cust_result, "customer", name_match=p["customer_name"])
        cust_id = cust["id"]

        inv_result = _call(ctx, tools, "search_invoices",
                           invoiceDateFrom="2000-01-01", invoiceDateTo="2030-12-31",
                           customerId=cust_id)
        invoices = inv_result.get("values", [])
        # Find unpaid invoice — pick newest (highest ID) to avoid sandbox pollution
        unpaid_list = [i for i in invoices if float(i.get("amountOutstanding", 0) or 0) > 0]
        unpaid = max(unpaid_list, key=lambda x: x.get("id", 0)) if unpaid_list else None
        if not unpaid:
            raise PipelineError("No unpaid invoice found for customer")

        if has_currency:
            # Foreign currency flow: pay at new rate + book agio/disagio
            invoice_nok = round(foreign_amount * old_rate, 2)
            payment_nok = round(foreign_amount * new_rate, 2)
            diff = round(payment_nok - invoice_nok, 2)

            # Use amountOutstanding as invoiceNOK if it doesn't match our calculation
            # (handles VAT differences in the system)
            amount_outstanding = float(unpaid.get("amountOutstanding", 0))
            if abs(amount_outstanding - invoice_nok) > 1.0 and amount_outstanding > 0:
                # System amount differs — recalculate using the system's outstanding
                ratio = new_rate / old_rate if old_rate else 1
                payment_nok = round(amount_outstanding * ratio, 2)
                diff = round(payment_nok - amount_outstanding, 2)

            _call(ctx, tools, "register_payment",
                  invoice_id=unpaid["id"], amount=payment_nok,
                  paymentDate=payment_date, paidAmountCurrency=foreign_amount)

            # Book agio/disagio voucher
            if abs(diff) > 0.01:
                if diff > 0:
                    # Agio (exchange gain): debit 1500, credit 8060
                    postings = json.dumps([
                        {"accountNumber": "1500", "amount": diff, "customerId": cust_id},
                        {"accountNumber": "8060", "amount": -diff},
                    ])
                    desc = "Agio valutadifferanse"
                else:
                    # Disagio (exchange loss): debit 8160, credit 1500
                    postings = json.dumps([
                        {"accountNumber": "8160", "amount": abs(diff)},
                        {"accountNumber": "1500", "amount": -abs(diff), "customerId": cust_id},
                    ])
                    desc = "Disagio valutadifferanse"
                return _call(ctx, tools, "create_voucher",
                             date=payment_date, description=desc, postings=postings)
            return None
        else:
            # Simple NOK payment — use extracted payment_amount if specified
            pay_amount = p.get("payment_amount", 0) or 0
            if not pay_amount:
                pay_amount = float(unpaid.get("amountOutstanding", 0) or 0)
            return _call(ctx, tools, "register_payment",
                         invoice_id=unpaid["id"], amount=pay_amount, paymentDate=payment_date)
    else:
        # New invoice flow: create → pay via compound tool
        products_json = _build_products_json(p)
        inv = _call(ctx, tools, "process_invoice",
                    customer_name=p["customer_name"],
                    customer_email=p.get("customer_email") or "",
                    customer_org_number=p.get("customer_org_number") or "",
                    products=products_json,
                    invoiceDate=p.get("invoice_date") or "",
                    invoiceDueDate=p.get("due_date") or "")

        inv_id = inv.get("invoice_id") if isinstance(inv, dict) else _id(inv)
        if not inv_id:
            raise PipelineError(f"process_invoice did not return invoice_id: {inv}")

        # Use extracted payment_amount if specified, otherwise pay full invoice
        pay_amount = p.get("payment_amount", 0) or 0
        if not pay_amount:
            # Fetch the actual invoice to get the amount
            inv_data = _call(ctx, tools, "get_entity_by_id",
                             entity_type="invoice", entity_id=inv_id)
            inv_val = _val(inv_data) if isinstance(inv_data, dict) and "value" in inv_data else inv_data
            if isinstance(inv_val, dict):
                pay_amount = inv_val.get("amount", 0) or inv_val.get("amountOutstanding", 0)
            if not pay_amount:
                # Fallback: calculate from extracted params
                vat = p.get("vat_percentage", 25) or 25
                pay_amount = p.get("product_price", 0) * p.get("quantity", 1) * (1 + vat / 100)
        # Guard: payment amount must be positive
        pay_amount = abs(pay_amount) if pay_amount else pay_amount
        return _call(ctx, tools, "register_payment",
                     invoice_id=inv_id, amount=round(pay_amount, 2), paymentDate=payment_date)


def _pipeline_order_to_invoice_with_payment(tools, p, ctx):
    """Order → Invoice → Payment via compound tool (1 call)."""
    today = _today()

    # Build products list if multi-product
    products_json = "[]"
    if p.get("products") and isinstance(p["products"], list) and len(p["products"]) > 0:
        products_json = json.dumps(p["products"])

    return _call(ctx, tools, "process_order_to_invoice_with_payment",
                 customer_name=p["customer_name"],
                 customer_email=p.get("customer_email") or "",
                 customer_org_number=p.get("customer_org_number") or "",
                 product_name=p.get("product_name") or "",
                 product_price=p.get("product_price", 0),
                 product_number=p.get("product_number") or "",
                 quantity=p.get("quantity", 1),
                 vat_percentage=p.get("vat_percentage", 25),
                 products=products_json,
                 invoiceDate=p.get("invoice_date") or today,
                 invoiceDueDate=p.get("due_date") or p.get("invoice_date") or today,
                 paymentDate=p.get("payment_date") or today)


def _pipeline_create_credit_note(tools, p, ctx):
    products_json = _build_products_json(p)
    result = _call(ctx, tools, "process_invoice",
                   customer_name=p["customer_name"],
                   customer_email=p.get("customer_email") or "",
                   customer_org_number=p.get("customer_org_number") or "",
                   products=products_json,
                   invoiceDate=p.get("invoice_date") or "",
                   invoiceDueDate=p.get("due_date") or "",
                   create_credit_note=True)
    return result


def _pipeline_delete_invoice(tools, p, ctx):
    products_json = _build_products_json(p)
    result = _call(ctx, tools, "process_invoice",
                   customer_name=p["customer_name"],
                   customer_email=p.get("customer_email") or "",
                   customer_org_number=p.get("customer_org_number") or "",
                   products=products_json,
                   invoiceDate=p.get("invoice_date") or "",
                   invoiceDueDate=p.get("due_date") or "",
                   create_credit_note=True)
    return result


# --- Group D: Employment/travel/supplier ---

def _pipeline_create_travel_expense(tools, p, ctx):
    emp = _call(ctx, tools, "create_employee",
                firstName=p["firstName"], lastName=p["lastName"], email=p["email"])
    emp_id = _id(emp)
    return _call(ctx, tools, "create_travel_expense",
                 employee_id=emp_id, title=p["title"],
                 departureDate=p["departureDate"], returnDate=p["returnDate"])


def _pipeline_create_travel_expense_with_costs(tools, p, ctx):
    emp = _call(ctx, tools, "create_employee",
                firstName=p["firstName"], lastName=p["lastName"], email=p["email"])
    emp_id = _id(emp)
    te = _call(ctx, tools, "create_travel_expense",
               employee_id=emp_id, title=p["title"],
               departureDate=p["departureDate"], returnDate=p["returnDate"])
    te_id = _id(te)

    # Per diem: one call covers the full travel period
    if p.get("per_diem_rate") and p.get("per_diem_rate") > 0:
        _call(ctx, tools, "create_per_diem_compensation",
              travel_expense_id=te_id, location="Norge")

    # Additional costs (hotel, taxi, etc.)
    for cost in (p.get("costs") or []):
        cost_kwargs = dict(
            travel_expense_id=te_id,
            amount=cost.get("amount", 0),
            category=cost.get("category", ""),
        )
        if cost.get("comments"):
            cost_kwargs["comments"] = cost["comments"]
        if cost.get("date"):
            cost_kwargs["date"] = cost["date"]
        _call(ctx, tools, "create_travel_expense_cost", **cost_kwargs)

    # Accommodation allowance (fixed-rate nattillegg)
    if p.get("accommodation_nights") and p["accommodation_nights"] > 0:
        acc_kwargs = dict(travel_expense_id=te_id, count=p["accommodation_nights"])
        if p.get("accommodation_location"):
            acc_kwargs["location"] = p["accommodation_location"]
        _call(ctx, tools, "create_accommodation_allowance", **acc_kwargs)

    # Mileage allowance
    if p.get("mileage_km") and p["mileage_km"] > 0:
        _call(ctx, tools, "create_mileage_allowance",
              travel_expense_id=te_id, date=p["departureDate"],
              km=p["mileage_km"])
    return te


def _pipeline_create_employee_with_employment(tools, p, ctx):
    emp_kwargs = dict(firstName=p["firstName"], lastName=p["lastName"], email=p["email"])
    # Always provide dateOfBirth — saves 1-2 API calls in create_employment
    emp_kwargs["dateOfBirth"] = p.get("dateOfBirth") or "1990-01-01"
    if p.get("nationalIdentityNumber"):
        emp_kwargs["nationalIdentityNumber"] = p["nationalIdentityNumber"]
    if p.get("bankAccountNumber"):
        emp_kwargs["bankAccountNumber"] = p["bankAccountNumber"]
    if p.get("department_name"):
        emp_kwargs["department_name"] = p["department_name"]
    emp = _call(ctx, tools, "create_employee", **emp_kwargs)
    emp_id = _id(emp)

    # Pass salary, FTE%, and occupationCode directly to create_employment
    # (it creates initial employment details automatically — avoids duplicate 422)
    empl_kwargs = dict(employee_id=emp_id, startDate=p["startDate"],
                       _skip_checks=True)  # employee just created — skip existence/DOB checks
    if p.get("annualSalary") and p["annualSalary"] > 0:
        empl_kwargs["annualSalary"] = p["annualSalary"]
    if p.get("percentageOfFullTimeEquivalent") and p["percentageOfFullTimeEquivalent"] != 100:
        empl_kwargs["percentageOfFullTimeEquivalent"] = p["percentageOfFullTimeEquivalent"]
    if p.get("occupationCode"):
        empl_kwargs["occupationCode"] = p["occupationCode"]
    employment = _call(ctx, tools, "create_employment", **empl_kwargs)
    employment_id = _id(employment)

    # Optional: standard time
    if p.get("hoursPerDay") and p["hoursPerDay"] > 0:
        _call(ctx, tools, "create_standard_time",
              employee_id=emp_id, fromDate=p["startDate"],
              hoursPerDay=p["hoursPerDay"])

    # Optional: leave of absence
    if p.get("leave_type") and p.get("leave_startDate"):
        _call(ctx, tools, "create_leave_of_absence",
              employment_id=employment_id,
              startDate=p["leave_startDate"],
              endDate=p.get("leave_endDate") or "",
              leaveType=p["leave_type"],
              percentage=p.get("leave_percentage", 100))

    return employment


def _pipeline_create_supplier_invoice(tools, p, ctx):
    # If files are present, try to extract content first
    for fname in ctx.get("file_names") or []:
        if "extract_file_content" in tools:
            try:
                _call(ctx, tools, "extract_file_content", filename=fname)
            except PipelineError:
                pass  # file extraction is best-effort; data may be in prompt

    kwargs = {
        "supplier_name": p["supplier_name"],
        "amountIncludingVat": p["amount_including_vat"],
        "expenseAccountNumber": p["expense_account"],
        "invoiceDate": p["invoice_date"],
    }
    if p.get("supplier_org_number"):
        kwargs["supplierOrgNumber"] = p["supplier_org_number"]
    if p.get("supplier_bank_account"):
        kwargs["supplierBankAccount"] = p["supplier_bank_account"]
    if p.get("supplier_address"):
        kwargs["supplierAddress"] = p["supplier_address"]
    if p.get("supplier_postal_code"):
        kwargs["supplierPostalCode"] = p["supplier_postal_code"]
    if p.get("supplier_city"):
        kwargs["supplierCity"] = p["supplier_city"]
    if p.get("invoice_number"):
        kwargs["invoiceNumber"] = p["invoice_number"]
    if p.get("vat_percentage") is not None:
        kwargs["vatPercentage"] = p.get("vat_percentage", 25)
    if p.get("due_date"):
        kwargs["dueDate"] = p["due_date"]
    if p.get("line_description"):
        kwargs["lineDescription"] = p["line_description"]
    if p.get("department_name"):
        kwargs["departmentName"] = p["department_name"]

    return _call(ctx, tools, "process_supplier_invoice", **kwargs)


def _pipeline_register_expense_receipt(tools, p, ctx):
    dept_id = 0
    if p.get("department_name"):
        dept = _call(ctx, tools, "create_department", name=p["department_name"])
        dept_id = _id(dept)

    kwargs = {
        "amountIncludingVat": p["amount_including_vat"],
        "expenseAccountNumber": p["expense_account"],
        "vatPercentage": p.get("vat_percentage", 25),
        "receiptDate": p.get("receipt_date") or _today(),
        "description": p.get("description", ""),
        "paymentAccountNumber": p.get("payment_account", 1920),
    }
    if dept_id:
        kwargs["departmentId"] = dept_id

    return _call(ctx, tools, "register_expense_receipt", **kwargs)


# --- Group E: Delete tasks ---

def _pipeline_delete_customer(tools, p, ctx):
    result = _call(ctx, tools, "search_customers", name=p["name"])
    entity = _first(result, "customer", name_match=p["name"])
    return _call(ctx, tools, "delete_customer", customer_id=entity["id"])


def _pipeline_delete_supplier(tools, p, ctx):
    result = _call(ctx, tools, "search_suppliers", name=p["name"])
    entity = _first(result, "supplier", name_match=p["name"])
    return _call(ctx, tools, "delete_supplier", supplier_id=entity["id"])


def _pipeline_delete_product(tools, p, ctx):
    result = _call(ctx, tools, "search_products", name=p["name"])
    entity = _first(result, "product", name_match=p["name"])
    return _call(ctx, tools, "delete_product", product_id=entity["id"])


def _pipeline_delete_department(tools, p, ctx):
    result = _call(ctx, tools, "search_departments", name=p["name"])
    entity = _first(result, "department", name_match=p["name"])
    return _call(ctx, tools, "delete_department", department_id=entity["id"])


def _pipeline_delete_contact(tools, p, ctx):
    result = _call(ctx, tools, "search_contacts",
                   firstName=p["firstName"], lastName=p["lastName"])
    entity = _first(result, "contact")
    try:
        return _call(ctx, tools, "delete_contact", contact_id=entity["id"])
    except PipelineError:
        # Fallback: deactivate if delete fails
        return _call(ctx, tools, "update_contact",
                     contact_id=entity["id"], isInactive=True)


def _pipeline_delete_employee(tools, p, ctx):
    result = _call(ctx, tools, "search_employees",
                   firstName=p["firstName"], lastName=p["lastName"])
    entity = _first(result, "employee")
    return _call(ctx, tools, "update_employee",
                 employee_id=entity["id"], isInactive=True)


def _pipeline_delete_travel_expense(tools, p, ctx):
    result = _call(ctx, tools, "search_travel_expenses")
    entities = result.get("values", [])
    if not entities:
        raise PipelineError("No travel expenses found")
    # Try to match by title or employee name
    target = entities[0]  # default to first
    title = (p.get("title") or "").lower()
    fname = (p.get("employee_firstName") or "").lower()
    lname = (p.get("employee_lastName") or "").lower()
    for e in entities:
        if title and title in (e.get("title") or "").lower():
            target = e
            break
        emp = e.get("employee", {})
        if fname and fname in (emp.get("firstName") or "").lower():
            target = e
            break
    return _call(ctx, tools, "delete_travel_expense", travel_expense_id=target["id"])


# --- Group F: Complex tasks ---

def _pipeline_correct_ledger_errors(tools, p, ctx):
    """Correct ledger errors by searching for original vouchers and creating corrections."""
    date_from = p["date_from"]
    date_to = p["date_to"]
    correction_date = date_to  # Date corrections at end of period

    # STEP 0: Fetch all vouchers in the period to find counterpart accounts + customer/supplier IDs
    acct_counter = {}    # (debit_acct_number, abs_amount) -> credit_acct_number
    expense_counter = {} # expense_acct_number -> counter_acct_number (latest)
    acct_customer = {}   # acct_number -> customerId (from 1500 postings)
    acct_supplier = {}   # acct_number -> supplierId (from 2400 postings)
    client = ctx.get("client")
    if client:
        try:
            raw = client.get("/ledger/voucher", params={
                "dateFrom": date_from,
                "dateTo": date_to,
                "fields": "id,postings(account(id,number),amount,customer(id),supplier(id))",
                "count": 200,
            })
            for v in (raw.get("values") or []):
                plist = v.get("postings") or []
                # Track customer/supplier IDs from all postings
                for pp in plist:
                    acct_num = str((pp.get("account") or {}).get("number", ""))
                    cust = pp.get("customer") or {}
                    sup = pp.get("supplier") or {}
                    if cust.get("id") and acct_num:
                        acct_customer[acct_num] = cust["id"]
                    if sup.get("id") and acct_num:
                        acct_supplier[acct_num] = sup["id"]
                if len(plist) != 2:
                    continue
                a0 = str((plist[0].get("account") or {}).get("number", ""))
                a1 = str((plist[1].get("account") or {}).get("number", ""))
                m0 = float(plist[0].get("amount") or 0)
                m1 = float(plist[1].get("amount") or 0)
                if not a0 or not a1:
                    continue
                if m0 > 0:
                    debit_acct, credit_acct, debit_amt = a0, a1, abs(m0)
                else:
                    debit_acct, credit_acct, debit_amt = a1, a0, abs(m1)
                acct_counter[(debit_acct, round(debit_amt, 2))] = credit_acct
                expense_counter[debit_acct] = credit_acct
        except Exception:
            pass

    def _find_counter(acct: str, amount: float) -> str:
        """Look up actual counterpart, fall back to 1920."""
        c = acct_counter.get((acct, round(amount, 2)))
        if c and c != acct:
            return c
        c = expense_counter.get(acct)
        if c and c != acct:
            return c
        return "1920"

    def _enrich_posting(posting: dict):
        """Add customerId/supplierId for 1500/2400 accounts."""
        acct = str(posting.get("accountNumber", ""))
        if acct == "1500" and acct in acct_customer:
            posting["customerId"] = acct_customer[acct]
        elif acct == "2400" and acct in acct_supplier:
            posting["supplierId"] = acct_supplier[acct]

    # STEP 1: Create one corrective voucher per error
    last = None
    for error in p.get("errors", []):
        etype = error.get("type", "")
        postings = []
        desc = ""

        if etype == "wrong_account":
            wrong = str(error["wrong_account"])
            correct = str(error["correct_account"])
            amount = float(error["amount"])
            postings = [
                {"accountNumber": wrong, "amount": -amount},
                {"accountNumber": correct, "amount": amount},
            ]
            desc = f"Korreksjon: feil konto {wrong} -> {correct}"

        elif etype == "duplicate":
            acct = str(error["account"])
            amount = float(error["amount"])
            counter = _find_counter(acct, amount)
            postings = [
                {"accountNumber": acct, "amount": -amount},
                {"accountNumber": counter, "amount": amount},
            ]
            desc = f"Korreksjon: duplikat konto {acct}"

        elif etype == "missing_vat":
            expense_acct = str(error["expense_account"])
            vat_acct = str(error["vat_account"])
            amount_excl = float(error["amount_excl_vat"])
            vat_amount = round(amount_excl * 0.25, 2)
            counter = _find_counter(expense_acct, amount_excl)
            postings = [
                {"accountNumber": vat_acct, "amount": vat_amount},
                {"accountNumber": counter, "amount": -vat_amount},
            ]
            desc = f"Korreksjon: manglende MVA konto {expense_acct}"

        elif etype == "wrong_amount":
            acct = str(error["account"])
            recorded = float(error["recorded_amount"])
            correct_amt = float(error["correct_amount"])
            diff = round(recorded - correct_amt, 2)
            counter = _find_counter(acct, recorded)
            postings = [
                {"accountNumber": acct, "amount": -diff},
                {"accountNumber": counter, "amount": diff},
            ]
            desc = f"Korreksjon: feil beløp konto {acct}"

        if postings:
            # Enrich postings with customer/supplier IDs for 1500/2400
            for posting in postings:
                _enrich_posting(posting)
            last = _call(ctx, tools, "create_voucher",
                         date=correction_date, description=desc,
                         postings=json.dumps(postings))
    return last


def _pipeline_create_ledger_voucher(tools, p, ctx):
    import re as _re

    corrections = p.get("corrections", [])
    # Backward compat: if old-style flat postings, wrap in single correction
    if not corrections and p.get("postings"):
        corrections = [{"description": p.get("description", "Correction"), "postings": p["postings"]}]
    vdate = p["date"]
    year = vdate[:4]

    # ── STEP 0: Search existing vouchers to find actual counter-accounts + customer/supplier IDs ──
    # The extraction assumes 1920 (bank) as counter, but the original entries
    # may use 2400 (payables), etc.  Search + fix before creating corrections.
    acct_counter = {}   # (debit_acct_number, abs_amount) -> credit_acct_number
    expense_counter = {}  # expense_acct_number -> counter_acct_number (latest)
    acct_customer = {}   # acct_number -> customerId (from 1500 postings)
    acct_supplier = {}   # acct_number -> supplierId (from 2400 postings)
    client = ctx.get("client")
    if client:
        try:
            raw = client.get("/ledger/voucher", params={
                "dateFrom": f"{year}-01-01",
                "dateTo": vdate,
                "fields": "id,postings(account(id,number),amount,customer(id),supplier(id))",
                "count": 200,
            })
            for v in (raw.get("values") or []):
                plist = v.get("postings") or []
                # Track customer/supplier IDs from all postings
                for pp in plist:
                    acct_num = str((pp.get("account") or {}).get("number", ""))
                    cust = pp.get("customer") or {}
                    sup = pp.get("supplier") or {}
                    if cust.get("id") and acct_num:
                        acct_customer[acct_num] = cust["id"]
                    if sup.get("id") and acct_num:
                        acct_supplier[acct_num] = sup["id"]
                if len(plist) != 2:
                    continue
                a0 = str((plist[0].get("account") or {}).get("number", ""))
                a1 = str((plist[1].get("account") or {}).get("number", ""))
                m0 = float(plist[0].get("amount") or 0)
                m1 = float(plist[1].get("amount") or 0)
                if not a0 or not a1:
                    continue
                # debit is positive, credit is negative
                if m0 > 0:
                    debit_acct, credit_acct, debit_amt = a0, a1, abs(m0)
                else:
                    debit_acct, credit_acct, debit_amt = a1, a0, abs(m1)
                acct_counter[(debit_acct, round(debit_amt, 2))] = credit_acct
                expense_counter[debit_acct] = credit_acct
        except Exception:
            pass

    # ── STEP 1: Fix counter-accounts in extracted corrections ──
    for corr in corrections:
        posts = corr.get("postings") or []
        if len(posts) != 2:
            continue
        # Identify the assumed-counter (1920) and the main posting
        p_counter = p_main = None
        for pp in posts:
            if str(pp.get("accountNumber", "")) == "1920":
                p_counter = pp
            else:
                p_main = pp
        if not p_counter or not p_main:
            continue

        main_acct = str(p_main.get("accountNumber", ""))
        main_amt = round(abs(float(p_main.get("amount", 0))), 2)

        # Case A: Direct match — duplicate or wrong-amount reversal
        real = acct_counter.get((main_acct, main_amt))
        if real and real != main_acct:
            p_counter["accountNumber"] = real
            continue

        # Case B: MVA correction (main acct is 2710)
        if main_acct == "2710":
            # Find the expense account from the description (e.g. "konto 7000")
            desc = corr.get("description", "")
            m = _re.search(r'konto\s+(\d{4})', desc, _re.IGNORECASE)
            if m:
                expense_acct = m.group(1)
                # Check if the original was booked net or gross
                net_amount = main_amt * 4   # MVA 25% → net = MVA × 4
                gross_amount = net_amount * 1.25
                # Look for original posting on the expense account
                for (ea, ea_amt), counter in acct_counter.items():
                    if ea == expense_acct:
                        if abs(ea_amt - gross_amount) < 1:
                            # Gross was booked to expense → move MVA from expense to 2710
                            p_counter["accountNumber"] = expense_acct
                        else:
                            # Net was booked, bank underpaid → adjust counter
                            p_counter["accountNumber"] = counter
                        break
                else:
                    # Expense account not found in map, try generic lookup
                    if expense_acct in expense_counter:
                        p_counter["accountNumber"] = expense_counter[expense_acct]
            continue

        # Case C: Wrong-amount correction where amount doesn't match exactly
        # (the correction amount is the EXCESS, not the original amount)
        # Try to find the expense account's counter from any posting
        if main_acct in expense_counter:
            real = expense_counter[main_acct]
            if real != main_acct:
                p_counter["accountNumber"] = real

    # ── STEP 2: Create each correction voucher ──
    last = None
    for corr in corrections:
        desc = corr.get("description", "Correction")
        postings = corr.get("postings", [])
        if not postings:
            continue
        # Enrich postings with customer/supplier IDs for 1500/2400
        for posting in postings:
            acct = str(posting.get("accountNumber", ""))
            if acct == "1500" and acct in acct_customer and "customerId" not in posting:
                posting["customerId"] = acct_customer[acct]
            elif acct == "2400" and acct in acct_supplier and "supplierId" not in posting:
                posting["supplierId"] = acct_supplier[acct]
        last = _call(ctx, tools, "create_voucher",
                     date=vdate, description=desc,
                     postings=json.dumps(postings))
    return last


def _pipeline_reverse_voucher(tools, p, ctx):
    result = _call(ctx, tools, "search_vouchers",
                   dateFrom=p.get("search_dateFrom") or "",
                   dateTo=p.get("search_dateTo") or "")
    vouchers = result.get("values", [])
    if not vouchers:
        raise PipelineError("No vouchers found")
    # Match by description if provided
    target = vouchers[0]
    desc = (p.get("search_description") or "").lower()
    if desc:
        for v in vouchers:
            if desc in (v.get("description") or "").lower():
                target = v
                break
    return _call(ctx, tools, "reverse_voucher",
                 voucher_id=target["id"], date=p.get("reversal_date") or "")


def _pipeline_reverse_payment(tools, p, ctx):
    """Reverse payment: try negative payment first (1 call), fallback to voucher reversal (3 calls)."""
    today = _today()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    customer_name = p["customer_name"].strip().lower()
    reversal_date = p.get("payment_date") or today

    # 1. Search all invoices (includes voucher(id) field) — 1 API call
    inv_result = _call(ctx, tools, "search_invoices",
                       invoiceDateFrom="2000-01-01", invoiceDateTo="2030-12-31")
    invoices = inv_result.get("values", [])

    # Find paid invoices matching customer name, pick newest (highest ID)
    matching_paid = []
    for inv in invoices:
        cust = inv.get("customer") or {}
        cust_name = (cust.get("name") or "").strip().lower()
        inv_amount = float(inv.get("amount", 0) or 0)
        inv_outstanding = float(inv.get("amountOutstanding", 0) or 0)
        if cust_name == customer_name and inv_amount > 0 and inv_outstanding < inv_amount:
            matching_paid.append(inv)

    if not matching_paid:
        raise PipelineError(f"No paid invoice found for customer '{p['customer_name']}'")

    invoice = max(matching_paid, key=lambda x: x.get("id", 0))
    invoice_id = invoice["id"]
    inv_amount = float(invoice.get("amount", 0) or 0)
    inv_outstanding = float(invoice.get("amountOutstanding", 0) or 0)
    paid_amount = inv_amount - inv_outstanding

    # 2. Strategy A: Use process_reverse_payment compound tool if available (handles all strategies)
    if "process_reverse_payment" in tools:
        result = _call(ctx, tools, "process_reverse_payment",
                       customer_name=p["customer_name"],
                       paymentDate=reversal_date,
                       amount=paid_amount)
        if isinstance(result, dict) and result.get("success"):
            return result

    # 3. Strategy B: Voucher reversal via ledger postings
    invoice_voucher_id = 0
    v = invoice.get("voucher") or {}
    if isinstance(v, dict):
        invoice_voucher_id = v.get("id", 0)

    postings_result = _call(ctx, tools, "get_ledger_postings",
                            dateFrom="2025-01-01", dateTo=tomorrow,
                            accountNumber="1500")
    postings = postings_result.get("values", [])

    # Find the payment posting: credit on 1500 matching invoice amount
    candidates = []
    for posting in postings:
        amt = float(posting.get("amount", 0) or 0)
        pv = posting.get("voucher") or {}
        vid = pv.get("id") if isinstance(pv, dict) else None
        if vid and abs(amt + inv_amount) < 1.0:
            if invoice_voucher_id and vid > invoice_voucher_id:
                candidates.append(vid)
            elif not invoice_voucher_id:
                candidates.append(vid)

    if not candidates:
        raise PipelineError(f"No payment voucher found for invoice {invoice_id} (amount={inv_amount})")

    payment_voucher_id = min(candidates) if invoice_voucher_id else max(candidates)

    _call(ctx, tools, "reverse_voucher",
          voucher_id=payment_voucher_id,
          date=reversal_date)

    return {"success": True, "invoice_id": invoice_id}


def _pipeline_reminder_fee(tools, p, ctx):
    today = _today()

    # 1. Find overdue invoice (due date < today, amountOutstanding > 0)
    inv_result = _call(ctx, tools, "search_invoices",
                       invoiceDateFrom="2000-01-01", invoiceDateTo="2030-12-31")
    invoices = inv_result.get("values", [])
    overdue = None
    for inv in invoices:
        due = inv.get("invoiceDueDate", "9999-12-31")
        outstanding = inv.get("amountOutstanding", 0)
        if due < today and outstanding > 0:
            if overdue is None or due < overdue.get("invoiceDueDate", "9999-12-31"):
                overdue = inv
    if not overdue:
        raise PipelineError("No overdue invoice found")

    cust_id = overdue["customer"]["id"]
    overdue_id = overdue["id"]
    fee = p["fee_amount"]
    debit = p.get("debit_account") or "1500"
    credit = p.get("credit_account") or "3400"

    # 2. Book reminder fee voucher
    postings = json.dumps([
        {"accountNumber": debit, "amount": fee, "customerId": cust_id},
        {"accountNumber": credit, "amount": -fee},
    ])
    _call(ctx, tools, "create_voucher",
          date=today, description="Purregebyr", postings=postings)

    # 3. Create reminder fee product + order + invoice
    prod = _call(ctx, tools, "create_product",
                 name="Purregebyr", priceExcludingVatCurrency=fee, vatPercentage=0)
    prod_id = _id(prod)

    order_lines = json.dumps([{"product_id": prod_id, "count": 1}])
    order = _call(ctx, tools, "create_order",
                  customer_id=cust_id, deliveryDate=today, orderLines=order_lines)
    order_id = _id(order)

    inv = _call(ctx, tools, "create_invoice",
                invoiceDate=today, invoiceDueDate=today, order_id=order_id)
    inv_id = _id(inv)

    # 4. Send invoice if requested
    if p.get("send_invoice"):
        _call(ctx, tools, "send_invoice", invoice_id=inv_id)

    # 5. Register partial payment on overdue invoice if requested
    if p.get("partial_payment_amount") and p["partial_payment_amount"] > 0:
        _call(ctx, tools, "register_payment",
              invoice_id=overdue_id, amount=p["partial_payment_amount"],
              paymentDate=today)

    return inv


def _pipeline_create_opening_balance(tools, p, ctx):
    postings_json = json.dumps(p["postings"])
    return _call(ctx, tools, "create_opening_balance",
                 voucherDate=p["date"], balancePostings=postings_json)


def _pipeline_create_dimension(tools, p, ctx):
    dim = _call(ctx, tools, "create_accounting_dimension", name=p["dimension_name"])
    dim_val = _val(dim)
    dim_index = dim_val.get("dimensionIndex", 1) if isinstance(dim_val, dict) else 1

    value_ids = {}
    for val_name in p["values"]:
        dv = _call(ctx, tools, "create_dimension_value",
                   dimensionIndex=dim_index, name=val_name)
        value_ids[val_name] = _id(dv)

    # ── Create voucher linked to dimension ──
    # Auto-construct postings from simple fields (amount, account_number, linked_value)
    amount = float(p.get("amount", 0) or 0)
    account_number = str(p.get("account_number", "") or "").strip()
    linked_value = (p.get("linked_value", "") or "").strip()
    voucher_date = p.get("voucher_date") or _today()
    voucher_desc = p.get("voucher_description") or f"Bilag {p['dimension_name']}"

    # Handle legacy voucher_postings format too
    if p.get("voucher_postings") and not amount:
        postings = []
        for posting in p["voucher_postings"]:
            entry = {
                "accountNumber": str(posting["accountNumber"]),
                "amount": posting["amount"],
            }
            dv_name = posting.get("dimensionValueName")
            if dv_name and dv_name in value_ids:
                entry["dimensionValueId"] = value_ids[dv_name]
                entry["dimensionIndex"] = dim_index
            postings.append(entry)
        _call(ctx, tools, "create_voucher",
              date=voucher_date, description=voucher_desc,
              postings=json.dumps(postings))
    elif amount > 0:
        # Auto-construct balanced postings: debit expense, credit bank (1920)
        if not account_number:
            account_number = "6340"  # Default expense account

        # Determine which dimension value to link
        if not linked_value and p.get("values"):
            linked_value = p["values"][0]  # Use first value as default

        dim_value_id = value_ids.get(linked_value)
        # If exact name not found, try case-insensitive match
        if not dim_value_id:
            target = linked_value.lower().strip()
            for vname, vid in value_ids.items():
                if vname.lower().strip() == target:
                    dim_value_id = vid
                    break
        # Still not found? Use first value
        if not dim_value_id and value_ids:
            dim_value_id = list(value_ids.values())[0]

        postings = [
            {
                "accountNumber": account_number,
                "amount": amount,
            },
            {
                "accountNumber": "1920",
                "amount": -amount,
            },
        ]
        # Add dimension reference to the expense posting
        if dim_value_id:
            postings[0]["dimensionValueId"] = dim_value_id
            postings[0]["dimensionIndex"] = dim_index

        _call(ctx, tools, "create_voucher",
              date=voucher_date, description=voucher_desc,
              postings=json.dumps(postings))

    return dim


def _pipeline_create_project(tools, p, ctx):
    today = _today()
    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"],
                 organizationNumber=p.get("customer_org_number") or "")
    cust_id = _id(cust)

    pm_id = 0
    pm_fresh = False
    if p.get("pm_email"):
        emp_kwargs = dict(
            firstName=p["pm_firstName"], lastName=p["pm_lastName"],
            email=p["pm_email"], userType="EXTENDED",
            dateOfBirth="1990-01-01")
        if p.get("pm_phoneNumberMobile"):
            emp_kwargs["phoneNumberMobile"] = p["pm_phoneNumberMobile"]
        emp = _call(ctx, tools, "create_employee", **emp_kwargs)
        pm_id = _id(emp)
        pm_fresh = True

    proj_kwargs = dict(
        name=p["project_name"], customer_id=cust_id,
        projectManagerId=pm_id,
        startDate=p.get("startDate") or today,
        fixedPriceAmount=p.get("fixedPriceAmount") or 0,
        description=p.get("description") or "",
        _pm_fresh_create=pm_fresh,
    )
    if p.get("isInternal"):
        proj_kwargs["isInternal"] = True
    return _call(ctx, tools, "create_project", **proj_kwargs)


def _pipeline_project_invoice(tools, p, ctx):
    """Project invoice via compound tool (1 call)."""
    kwargs = {
        "customer_name": p["customer_name"],
        "customer_org_number": p.get("customer_org_number") or "",
        "project_name": p["project_name"],
        "startDate": p.get("startDate") or _today(),
        "invoiceDate": p.get("invoice_date") or _today(),
        "invoiceDueDate": p.get("due_date") or p.get("invoice_date") or _today(),
    }
    # Employee who logs hours may be specified separately from PM
    pm_first = p.get("pm_firstName") or p.get("employee_firstName") or ""
    pm_last = p.get("pm_lastName") or p.get("employee_lastName") or ""
    pm_email = p.get("pm_email") or p.get("employee_email") or ""
    if pm_first:
        kwargs["pm_firstName"] = pm_first
    if pm_last:
        kwargs["pm_lastName"] = pm_last
    if pm_email:
        kwargs["pm_email"] = pm_email
    # Ensure we always have a price for invoicing
    fixed_price = p.get("fixedPriceAmount") or p.get("product_price") or 0
    if fixed_price:
        kwargs["fixedPriceAmount"] = fixed_price
    if p.get("milestone_percentage"):
        kwargs["milestonePercentage"] = p["milestone_percentage"]
    if p.get("product_name"):
        kwargs["product_name"] = p["product_name"]
    if p.get("hourly_rate"):
        kwargs["hourlyRate"] = p["hourly_rate"]
    if p.get("hours"):
        kwargs["hours"] = p["hours"]
    if p.get("activity_name"):
        kwargs["activity_name"] = p["activity_name"]
    if p.get("vatPercentage") is not None:
        kwargs["vatPercentage"] = p["vatPercentage"]
    if p.get("send_invoice"):
        kwargs["send_invoice"] = True

    return _call(ctx, tools, "process_project_invoice", **kwargs)


def _pipeline_project_lifecycle(tools, p, ctx):
    today = _today()

    # 1. Create customer
    cust = _call(ctx, tools, "create_customer",
                 name=p["customer_name"],
                 organizationNumber=p.get("customer_org_number") or "")
    cust_id = _id(cust)

    # 2. Create PM (if specified)
    pm_id = 0
    if p.get("pm_email"):
        pm = _call(ctx, tools, "create_employee",
                   firstName=p["pm_firstName"], lastName=p["pm_lastName"],
                   email=p["pm_email"], userType="EXTENDED")
        pm_id = _id(pm)

    # 3. Create project
    proj_kwargs = dict(
        name=p["project_name"], customer_id=cust_id,
        projectManagerId=pm_id,
        startDate=today,
        fixedPriceAmount=p.get("budget") or p.get("fixedPriceAmount") or 0,
    )
    if p.get("isInternal"):
        proj_kwargs["isInternal"] = True
    proj = _call(ctx, tools, "create_project", **proj_kwargs)
    proj_id = _id(proj)

    # 4. Register employee hours
    for emp_spec in (p.get("employees") or []):
        # Skip PM (already created)
        if emp_spec.get("email") == p.get("pm_email") and pm_id:
            emp_id = pm_id
        else:
            emp = _call(ctx, tools, "create_employee",
                        firstName=emp_spec["firstName"],
                        lastName=emp_spec["lastName"],
                        email=emp_spec.get("email", ""))
            emp_id = _id(emp)
            # Non-PM employees need employment record
            _call(ctx, tools, "create_employment",
                  employee_id=emp_id, startDate="2026-01-01")
            # Add non-PM employees as project participants
            if "create_project_participant" in tools:
                try:
                    _call(ctx, tools, "create_project_participant",
                          project_id=proj_id, employee_id=emp_id)
                except Exception:
                    pass  # best-effort, don't fail pipeline

        _call(ctx, tools, "create_timesheet_entry",
              employee_id=emp_id, date=today,
              hours=emp_spec.get("hours", 0), project_id=proj_id)

    # 5. Register supplier cost (25% MVA default for Norwegian invoices)
    if p.get("supplier_name") and p.get("supplier_cost"):
        sup = _call(ctx, tools, "create_supplier",
                    name=p["supplier_name"],
                    organizationNumber=p.get("supplier_org_number") or "")
        sup_id = _id(sup)
        _call(ctx, tools, "create_incoming_invoice",
              supplierId=sup_id, invoiceNumber=f"PROJECT_COST_{today}",
              amountIncludingVat=p["supplier_cost"],
              expenseAccountNumber=4000, vatPercentage=25,
              invoiceDate=today, projectId=proj_id)

    # 6. Customer invoice for the project
    if p.get("create_customer_invoice") and p.get("budget"):
        prod = _call(ctx, tools, "create_product",
                     name=f"Project Services - {p['project_name']}",
                     priceExcludingVatCurrency=p["budget"],
                     vatPercentage=25)
        prod_id = _id(prod)
        order_lines = json.dumps([{"product_id": prod_id, "count": 1}])
        order = _call(ctx, tools, "create_order",
                      customer_id=cust_id, deliveryDate=today,
                      orderLines=order_lines, project_id=proj_id)
        order_id = _id(order)
        _call(ctx, tools, "create_invoice",
              invoiceDate=today, invoiceDueDate=today, order_id=order_id)

    return proj


def _pipeline_salary(tools, p, ctx):
    # Generate email if missing
    email = p.get("email") or ""
    if not email:
        fn = p["firstName"].lower().replace(" ", ".")
        ln = p["lastName"].lower().replace(" ", ".")
        email = f"{fn}.{ln}@example.com"

    # Resolve salary amounts — handle the "amount" fallback field
    base_salary = float(p.get("base_salary", 0) or 0)
    bonus = float(p.get("bonus", 0) or 0)
    amount = float(p.get("amount", 0) or 0)
    annual_salary = float(p.get("annualSalary", 0) or 0)

    # If base_salary and bonus are both 0 but amount is set, use amount as base_salary
    if base_salary == 0 and bonus == 0 and amount > 0:
        base_salary = amount

    # If still no base_salary but annualSalary is set, derive monthly
    if base_salary == 0 and annual_salary > 0:
        base_salary = round(annual_salary / 12, 2)

    # Use compound process_salary tool — handles employee, division, employment, salary
    return _call(ctx, tools, "process_salary",
                 firstName=p["firstName"],
                 lastName=p["lastName"],
                 email=email,
                 year=p["year"],
                 month=p["month"],
                 base_salary=base_salary,
                 bonus=bonus,
                 dateOfBirth=p.get("dateOfBirth") or "",
                 department_name=p.get("department_name") or "",
                 startDate=p.get("startDate") or "",
                 hoursPerDay=p.get("hoursPerDay", 0),
                 percentageOfFullTimeEquivalent=p.get("percentageOfFullTimeEquivalent", 100),
                 annualSalary=annual_salary)


def _pipeline_year_end(tools, p, ctx):
    vdate = p["voucher_date"]
    is_monthly = p.get("is_monthly", False)

    # Helper: resolve account number — look up if empty/missing
    def _resolve_account(acct_str: str, prefix: str = "") -> str:
        if acct_str and acct_str.strip():
            return acct_str.strip()
        if prefix and "get_ledger_accounts" in tools:
            try:
                result = _call(ctx, tools, "get_ledger_accounts", number=prefix)
                accounts = result.get("values", []) if isinstance(result, dict) else []
                if accounts:
                    return str(accounts[0].get("number", ""))
            except PipelineError:
                pass
        return acct_str

    # 1. Depreciation vouchers
    for asset in (p.get("assets") or []):
        if is_monthly:
            depr = round(asset["cost"] / asset["years"] / 12)
        else:
            depr = round(asset["cost"] / asset["years"])
        expense_acct = _resolve_account(str(asset.get("expense_account", "")), "60")
        depr_acct = _resolve_account(str(asset.get("depreciation_account", "")), "10")
        postings = json.dumps([
            {"accountNumber": expense_acct, "amount": depr},
            {"accountNumber": depr_acct, "amount": -depr},
        ])
        _call(ctx, tools, "create_voucher",
              date=vdate, description=f"Depreciation - {asset['name']}",
              postings=postings)

    # 2. Prepaid expense reversals
    for prepaid in (p.get("prepaid_expenses") or []):
        expense_acct = _resolve_account(str(prepaid.get("expense_account", "")), "69")
        postings = json.dumps([
            {"accountNumber": expense_acct, "amount": prepaid["amount"]},
            {"accountNumber": str(prepaid["prepaid_account"]), "amount": -prepaid["amount"]},
        ])
        _call(ctx, tools, "create_voucher",
              date=vdate, description="Prepaid expense reversal",
              postings=postings)

    # 3. Salary provisions
    for sal in (p.get("salary_provisions") or []):
        sal_amount = sal.get("amount", 0)
        if not sal_amount:
            sal_amount = 45000  # Default monthly salary accrual
        expense_acct = _resolve_account(str(sal.get("expense_account", "")), "50")
        payable_acct = _resolve_account(str(sal.get("payable_account", "")), "29")
        postings = json.dumps([
            {"accountNumber": expense_acct, "amount": sal_amount},
            {"accountNumber": payable_acct, "amount": -sal_amount},
        ])
        _call(ctx, tools, "create_voucher",
              date=vdate, description="Salary provision",
              postings=postings)

    # 4. Tax provision (annual closing only, or when explicitly requested)
    taxable = p.get("taxable_profit") or 0
    tax_rate = p.get("tax_rate") or 0.22
    if not taxable and not is_monthly and "get_result_before_tax" in tools:
        year = vdate[:4]
        result = _call(ctx, tools, "get_result_before_tax",
                       dateFrom=f"{year}-01-01", dateTo=f"{year}-12-31")
        taxable = result.get("result_before_tax", 0) if isinstance(result, dict) else 0
    if taxable > 0:
        tax_amount = round(taxable * tax_rate, 2)
        postings = json.dumps([
            {"accountNumber": str(p.get("tax_expense_account", "8700")), "amount": tax_amount},
            {"accountNumber": str(p.get("tax_liability_account", "2920")), "amount": -tax_amount},
        ])
        _call(ctx, tools, "create_voucher",
              date=vdate, description="Tax provision",
              postings=postings)

    # 5. Verify trial balance (if requested)
    if p.get("verify_balance") and "get_result_before_tax" in tools:
        year = vdate[:4]
        _call(ctx, tools, "get_result_before_tax",
              dateFrom=f"{year}-01-01", dateTo=vdate)

    # 6. Year-end note
    if p.get("year_end_note"):
        _call(ctx, tools, "create_year_end_note", note=p["year_end_note"])


def _parse_bank_csv(csv_text: str) -> list[dict]:
    """Parse a bank statement CSV into transaction dicts.

    Handles semicolon and comma separators.
    Returns list of {date, description, amount (positive=in, negative=out)}.
    """
    import re
    lines = csv_text.strip().splitlines()
    if not lines:
        return []

    # Detect separator
    sep = ";" if ";" in lines[0] else ","

    # Parse header to find columns
    header = [h.strip().lower() for h in lines[0].split(sep)]
    col_date = col_desc = col_in = col_out = col_amount = -1
    for i, h in enumerate(header):
        if h in ("dato", "date", "fecha", "data", "datum"):
            col_date = i
        elif h in ("forklaring", "beskrivelse", "description", "descripción", "descrição", "tekst",
                    "beschreibung", "libellé", "libelle"):
            col_desc = i
        elif h in ("inn", "innskudd", "credit", "crédito", "créditos", "entrada",
                    "eingang", "haben", "crédit"):
            col_in = i
        elif h in ("ut", "uttak", "debit", "débito", "saída",
                    "ausgang", "soll", "débit"):
            col_out = i
        elif h in ("beløp", "belop", "amount", "monto", "valor", "betrag", "montant"):
            col_amount = i

    transactions = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(sep)

        # Date
        txn_date = cols[col_date].strip() if col_date >= 0 and col_date < len(cols) else ""

        # Description
        desc = cols[col_desc].strip() if col_desc >= 0 and col_desc < len(cols) else ""

        # Amount: try Inn/Ut columns, then single amount column
        amount = 0.0
        if col_in >= 0 and col_out >= 0:
            inn = cols[col_in].strip().replace(",", ".").replace(" ", "") if col_in < len(cols) else ""
            ut = cols[col_out].strip().replace(",", ".").replace(" ", "") if col_out < len(cols) else ""
            if inn and inn.lstrip("-").replace(".", "", 1).isdigit():
                amount = float(inn)
            elif ut and ut.lstrip("-").replace(".", "", 1).isdigit():
                amount = float(ut) if float(ut) < 0 else -float(ut)
        elif col_amount >= 0 and col_amount < len(cols):
            raw = cols[col_amount].strip().replace(",", ".").replace(" ", "")
            if raw and raw.lstrip("-").replace(".", "", 1).isdigit():
                amount = float(raw)

        if not amount:
            continue

        # Classify transaction
        desc_lower = desc.lower()
        txn = {"date": txn_date, "description": desc, "amount": amount}

        # Customer payment: "Innbetaling fra X / Faktura Y" (+ French/German variants)
        m = re.search(r'(?:innbetaling fra|pago de|pagamento de|payment from|paiement de|encaissement de|virement de|zahlung von|einzahlung von)\s+(.+?)\s*/\s*(?:faktura|factura|invoice|fatura|facture|rechnung)\s*(\d+)', desc, re.IGNORECASE)
        if m and amount > 0:
            txn["type"] = "customer_payment"
            txn["customer_name"] = m.group(1).strip()
            txn["invoice_ref"] = m.group(2).strip()
        # Supplier payment: "Betaling/Paiement/Zahlung Fornecedor/Leverandør/Fournisseur/Proveedor/Lieferant X"
        elif re.search(r'(?:betaling|paiement|zahlung|pago|pagamento)\s+(?:til\s+|à\s+|a\s+)?(?:fornecedor|leverandør|leverandor|fournisseur|proveedor|supplier|lieferant)', desc, re.IGNORECASE):
            # Capture the FULL supplier name INCLUDING the prefix word (e.g. "Fournisseur Leroy SARL")
            # because Tripletex supplier names typically include the prefix.
            m2 = re.search(r'((?:fornecedor|leverandør|leverandor|fournisseur|proveedor|supplier|lieferant)\s+.+)', desc, re.IGNORECASE)
            txn["type"] = "supplier_payment"
            txn["supplier_name"] = m2.group(1).strip() if m2 else desc
        # Tax: Skattetrekk / Impôt / Steuer
        elif any(kw in desc_lower for kw in ("skattetrekk", "skatt", "impuesto", "tax", "imposto",
                                              "impôt", "retenue", "steuer", "lohnsteuer")):
            txn["type"] = "tax"
        # Bank fee: Bankgebyr / Frais bancaires / Bankgebühren
        elif any(kw in desc_lower for kw in ("bankgebyr", "bank fee", "comisión", "comision", "tarifa",
                                              "frais bancaire", "frais de banque", "bankgebühr")):
            txn["type"] = "bank_fee"
        # Interest income: Rente / Intérêts / Zinsen
        elif any(kw in desc_lower for kw in ("renteinntekt", "rente", "interest", "juros", "intereses",
                                              "intérêt", "interet", "zinsen", "zinsertrag")):
            txn["type"] = "interest"
        # Salary: Lønn / Salaire / Gehalt
        elif any(kw in desc_lower for kw in ("lønn", "lonn", "lønnsutbetaling", "salary", "salario",
                                              "salaire", "gehalt", "lohn", "nómina", "nomina")):
            txn["type"] = "salary"
        # VAT/MVA: Merverdiavgift / TVA / USt
        elif any(kw in desc_lower for kw in ("mva", "merverdiavgift", "vat", "iva", "tva",
                                              "umsatzsteuer", "mehrwertsteuer", "ust")):
            txn["type"] = "vat"
        else:
            txn["type"] = "other"

        transactions.append(txn)

    return transactions


def _pipeline_bank_reconciliation(tools, p, ctx):
    # 1. Read and parse the CSV file — reuse pre-extracted content if available
    filename = p.get("filename", "")
    csv_text = ctx.get("pre_extracted_files", {}).get(filename, "")

    if not csv_text and filename and "extract_file_content" in tools:
        try:
            result = _call(ctx, tools, "extract_file_content", filename=filename)
            csv_text = result.get("text", "") if isinstance(result, dict) else ""
        except PipelineError:
            pass

    if not csv_text:
        raise PipelineError("Could not read bank statement file")

    transactions = _parse_bank_csv(csv_text)
    if not transactions:
        raise PipelineError("No transactions found in bank statement")

    # 2. Cache: search all unique customers/suppliers up front to save API calls
    customer_cache = {}  # name -> [id1, id2, ...] (handles duplicate names)
    supplier_cache = {}  # name -> id

    for txn in transactions:
        txn_type = txn.get("type", "other")
        txn_date = txn.get("date", _today())
        txn_desc = txn.get("description", "Bank transaction")
        amount = txn.get("amount", 0)
        if not amount:
            continue

        # ── customer_payment: search customer → search invoices → register_payment ──
        if txn_type == "customer_payment":
            cust_name = txn.get("customer_name", "")
            inv_ref = txn.get("invoice_ref", "")
            try:
                # Find ALL customers with this name (handles duplicates)
                if cust_name not in customer_cache:
                    cust_result = _call(ctx, tools, "search_customers", name=cust_name)
                    all_custs = cust_result.get("values", [])
                    target_lower = cust_name.strip().lower()
                    matching_ids = [c["id"] for c in all_custs
                                    if c.get("name", "").strip().lower() == target_lower]
                    if not matching_ids and all_custs:
                        matching_ids = [all_custs[0]["id"]]
                    customer_cache[cust_name] = matching_ids

                cust_ids = customer_cache[cust_name]

                # Try each customer's invoices until we find a match
                target_inv = None
                bank_amt = abs(amount)
                for cust_id in cust_ids:
                    inv_result = _call(ctx, tools, "search_invoices", customerId=cust_id)
                    invoices = inv_result.get("values", [])

                    # Strategy 1: Match by amount (most reliable for full payments)
                    for inv in invoices:
                        outstanding = inv.get("amountOutstanding", 0) or 0
                        if outstanding > 0 and abs(outstanding - bank_amt) < 0.01:
                            target_inv = inv
                            break
                    if target_inv:
                        break

                    # Strategy 2: Match by invoice number (exact or suffix match)
                    # Bank refs like "1004" may correspond to Tripletex invoiceNumber "4"
                    if inv_ref:
                        for inv in invoices:
                            inv_num = str(inv.get("invoiceNumber", ""))
                            # Exact match or inv_ref ends with invoiceNumber (min 1 digit match)
                            matched = (inv_num == inv_ref
                                       or (len(inv_num) >= 1 and inv_ref.endswith(inv_num)
                                           and len(inv_ref) > len(inv_num)))
                            if matched and (inv.get("amountOutstanding", 0) or 0) > 0:
                                target_inv = inv
                                break
                    if target_inv:
                        break

                    # Strategy 3: Any invoice with outstanding > 0 (for partial payments)
                    for inv in invoices:
                        if (inv.get("amountOutstanding", 0) or 0) > 0:
                            target_inv = inv
                            break
                    if target_inv:
                        break

                if target_inv:
                    # Use EXACT bank amount (handles partial payments correctly)
                    _call(ctx, tools, "register_payment",
                          invoice_id=target_inv["id"],
                          amount=bank_amt,
                          paymentDate=txn_date)
                    continue
            except PipelineError:
                pass  # Fall through to generic voucher

        # ── supplier_payment: search supplier → try supplier invoices → fallback to voucher ──
        if txn_type == "supplier_payment":
            supplier_name = txn.get("supplier_name", "")
            supplier_id = None
            # Find supplier (cache lookup)
            if supplier_name and "search_suppliers" in tools:
                if supplier_name not in supplier_cache:
                    try:
                        sup_result = _call(ctx, tools, "search_suppliers", name=supplier_name)
                        sup = _first(sup_result, "supplier", name_match=supplier_name)
                        supplier_cache[supplier_name] = sup["id"]
                    except PipelineError:
                        pass
                supplier_id = supplier_cache.get(supplier_name)

            pay_amt = abs(amount)

            # Try to find and pay a matching supplier invoice first
            if supplier_id and "search_supplier_invoices" in tools:
                try:
                    si_result = _call(ctx, tools, "search_supplier_invoices",
                                      supplierId=supplier_id)
                    sup_invoices = si_result.get("values", [])
                    matched_si = None
                    # Match by amount
                    for si in sup_invoices:
                        si_amt = abs(si.get("amount", 0) or 0)
                        if si_amt > 0 and abs(si_amt - pay_amt) < 0.01:
                            matched_si = si
                            break
                    # Fallback: any open supplier invoice
                    if not matched_si and sup_invoices:
                        matched_si = sup_invoices[0]

                    if matched_si:
                        si_voucher_id = matched_si.get("voucher", {}).get("id") or matched_si.get("id")
                        if si_voucher_id and "add_supplier_invoice_payment" in tools:
                            _call(ctx, tools, "add_supplier_invoice_payment",
                                  invoice_id=si_voucher_id)
                            continue
                except PipelineError:
                    pass  # Fall through to voucher

            # Fallback: create voucher (debit 2400 Accounts Payable, credit 1920 Bank)
            postings = [
                {"accountNumber": "2400", "amount": pay_amt},
                {"accountNumber": "1920", "amount": -pay_amt},
            ]
            if supplier_id:
                postings[0]["supplierId"] = supplier_id
            _call(ctx, tools, "create_voucher",
                  date=txn_date,
                  description=txn_desc,
                  postings=json.dumps(postings))
            continue

        # ── bank_fee: debit 7770 (bankgebyr), credit 1920 (bank) ──
        if txn_type == "bank_fee":
            fee = abs(amount)
            _call(ctx, tools, "create_voucher",
                  date=txn_date, description=txn_desc,
                  postings=json.dumps([
                      {"accountNumber": "7770", "amount": fee},
                      {"accountNumber": "1920", "amount": -fee},
                  ]))
            continue

        # ── tax (Skattetrekk): incoming = debit 1920, credit 2600 ──
        if txn_type == "tax":
            if amount > 0:
                # Incoming tax refund/adjustment
                _call(ctx, tools, "create_voucher",
                      date=txn_date, description=txn_desc,
                      postings=json.dumps([
                          {"accountNumber": "1920", "amount": amount},
                          {"accountNumber": "2600", "amount": -amount},
                      ]))
            else:
                # Outgoing tax deduction
                tax_amt = abs(amount)
                _call(ctx, tools, "create_voucher",
                      date=txn_date, description=txn_desc,
                      postings=json.dumps([
                          {"accountNumber": "2600", "amount": tax_amt},
                          {"accountNumber": "1920", "amount": -tax_amt},
                      ]))
            continue

        # ── interest income: debit 1920, credit 8040 ──
        if txn_type == "interest":
            if amount > 0:
                _call(ctx, tools, "create_voucher",
                      date=txn_date, description=txn_desc,
                      postings=json.dumps([
                          {"accountNumber": "1920", "amount": amount},
                          {"accountNumber": "8040", "amount": -amount},
                      ]))
            else:
                int_amt = abs(amount)
                _call(ctx, tools, "create_voucher",
                      date=txn_date, description=txn_desc,
                      postings=json.dumps([
                          {"accountNumber": "8040", "amount": int_amt},
                          {"accountNumber": "1920", "amount": -int_amt},
                      ]))
            continue

        # ── salary: debit 5000, credit 1920 ──
        if txn_type == "salary":
            sal_amt = abs(amount)
            _call(ctx, tools, "create_voucher",
                  date=txn_date, description=txn_desc,
                  postings=json.dumps([
                      {"accountNumber": "5000", "amount": sal_amt},
                      {"accountNumber": "1920", "amount": -sal_amt},
                  ]))
            continue

        # ── vat/mva: debit 2740, credit 1920 ──
        if txn_type == "vat":
            vat_amt = abs(amount)
            _call(ctx, tools, "create_voucher",
                  date=txn_date, description=txn_desc,
                  postings=json.dumps([
                      {"accountNumber": "2740", "amount": vat_amt},
                      {"accountNumber": "1920", "amount": -vat_amt},
                  ]))
            continue

        # ── other / fallback: generic voucher ──
        if amount > 0:
            _call(ctx, tools, "create_voucher",
                  date=txn_date, description=txn_desc,
                  postings=json.dumps([
                      {"accountNumber": "1920", "amount": amount},
                      {"accountNumber": "3000", "amount": -amount},
                  ]))
        else:
            out_amt = abs(amount)
            _call(ctx, tools, "create_voucher",
                  date=txn_date, description=txn_desc,
                  postings=json.dumps([
                      {"accountNumber": "7700", "amount": out_amt},
                      {"accountNumber": "1920", "amount": -out_amt},
                  ]))


def _pipeline_process_invoice_file(tools, p, ctx):
    # Same as create_invoice but may need file extraction first
    for fname in ctx.get("file_names") or []:
        if "extract_file_content" in tools:
            try:
                _call(ctx, tools, "extract_file_content", filename=fname)
            except PipelineError:
                pass
    return _pipeline_create_invoice(tools, p, ctx)


# ── Pipeline Registry ───────────────────────────────────────────

PIPELINES = {
    # Group A
    "create_employee": _pipeline_create_employee,
    "create_customer": _pipeline_create_customer,
    "create_product": _pipeline_create_product,
    "create_supplier": _pipeline_create_supplier,
    "create_department": _pipeline_create_department,
    "create_multiple_departments": _pipeline_create_multiple_departments,
    # Group B
    "create_contact": _pipeline_create_contact,
    "update_employee": _pipeline_update_employee,
    "update_customer": _pipeline_update_customer,
    "update_product": _pipeline_update_product,
    "update_supplier": _pipeline_update_supplier,
    "update_department": _pipeline_update_department,
    "update_contact": _pipeline_update_contact,
    # Group C
    "create_invoice": _pipeline_create_invoice,
    "create_multi_line_invoice": _pipeline_create_multi_line_invoice,
    "invoice_with_payment": _pipeline_invoice_with_payment,
    "order_to_invoice_with_payment": _pipeline_order_to_invoice_with_payment,
    "create_credit_note": _pipeline_create_credit_note,
    "delete_invoice": _pipeline_delete_invoice,
    # Group D
    "create_travel_expense": _pipeline_create_travel_expense,
    "create_travel_expense_with_costs": _pipeline_create_travel_expense_with_costs,
    "create_employee_with_employment": _pipeline_create_employee_with_employment,
    "create_supplier_invoice": _pipeline_create_supplier_invoice,
    "register_expense_receipt": _pipeline_register_expense_receipt,
    # Group E
    "delete_customer": _pipeline_delete_customer,
    "delete_supplier": _pipeline_delete_supplier,
    "delete_product": _pipeline_delete_product,
    "delete_department": _pipeline_delete_department,
    "delete_contact": _pipeline_delete_contact,
    "delete_employee": _pipeline_delete_employee,
    "delete_travel_expense": _pipeline_delete_travel_expense,
    # Group F
    "correct_ledger_errors": _pipeline_correct_ledger_errors,
    "create_ledger_voucher": _pipeline_create_ledger_voucher,
    "reverse_voucher": _pipeline_reverse_voucher,
    "reverse_payment": _pipeline_reverse_payment,
    "reminder_fee": _pipeline_reminder_fee,
    "create_opening_balance": _pipeline_create_opening_balance,
    "create_dimension": _pipeline_create_dimension,
    "create_project": _pipeline_create_project,
    "create_project_with_pm": _pipeline_create_project,
    "project_invoice": _pipeline_project_invoice,
    "project_lifecycle": _pipeline_project_lifecycle,
    "create_project_with_billing": _pipeline_project_lifecycle,
    "salary_with_bonus": _pipeline_salary,
    "year_end": _pipeline_year_end,
    "bank_reconciliation": _pipeline_bank_reconciliation,
    "process_invoice_file": _pipeline_process_invoice_file,
}


# ── Public API ──────────────────────────────────────────────────

def has_pipeline(task_type: str) -> bool:
    """Check if a static pipeline exists for this task type."""
    return task_type in PIPELINES and task_type in SCHEMAS


def run_static(task_type: str, prompt: str, all_tools_dict: dict,
               client, emit_fn=None, request_id: str = "",
               file_names: list[str] | None = None) -> dict:
    """Run a static pipeline for the given task type.

    Returns dict compatible with _run_agent() output.
    Raises ExtractionError or PipelineError on failure (caller should fallback to agent).
    """
    import time as _time

    t_start = _time.time()

    if emit_fn:
        emit_fn({"type": "static_start", "request_id": request_id,
                 "task_type": task_type})

    # Pre-extract file content so LLM can see PDF/CSV data during extraction
    # Also store in pre_extracted_files so pipelines can reuse without a second API call
    extraction_prompt = prompt
    pre_extracted_files = {}
    if file_names and "extract_file_content" in all_tools_dict:
        for fname in file_names:
            try:
                file_result = all_tools_dict["extract_file_content"](filename=fname)
                if isinstance(file_result, dict) and file_result.get("text"):
                    pre_extracted_files[fname] = file_result["text"]
                    extraction_prompt += f"\n\n--- File: {fname} ---\n{file_result['text']}"
                    log.info(f"[{request_id[:8]}] static: pre-extracted {fname} ({len(file_result['text'])} chars)")
            except Exception as e:
                log.warning(f"[{request_id[:8]}] static: file extraction failed for {fname}: {e}")
                extraction_prompt += f"\n\nAttached files: {', '.join(file_names)}"
                break
    elif file_names:
        extraction_prompt += f"\n\nAttached files: {', '.join(file_names)}"

    # 1. Extract parameters (ONE Gemini call)
    log.info(f"[{request_id[:8]}] static: extracting params for {task_type}")
    params = extract_params(extraction_prompt, task_type)
    log.info(f"[{request_id[:8]}] static: extracted {len(params)} params")

    # Override filename with actual attached file if extraction missed it
    if file_names and task_type in ("bank_reconciliation", "process_invoice_file"):
        extracted_fn = params.get("filename", "")
        if not extracted_fn or extracted_fn not in file_names:
            params["filename"] = file_names[0]

    if emit_fn:
        emit_fn({"type": "static_extracted", "request_id": request_id,
                 "params": {k: str(v)[:100] for k, v in params.items()}})

    # 2. Pre-compute currency info (regex-based, no LLM call) for agio tasks
    from tool_router import extract_currency_info
    currency_info = extract_currency_info(prompt) if task_type == "invoice_with_payment" else None

    # 2. Run pipeline
    ctx = {"steps": [], "emit_fn": emit_fn, "request_id": request_id, "client": client,
           "file_names": file_names or [], "pre_extracted_files": pre_extracted_files}
    if currency_info:
        ctx["currency_info"] = currency_info
        log.info(f"[{request_id[:8]}] static: currency detected: {currency_info}")
    pipeline_fn = PIPELINES[task_type]
    pipeline_fn(all_tools_dict, params, ctx)

    elapsed = _time.time() - t_start
    log.info(f"[{request_id[:8]}] static: done in {elapsed:.1f}s, "
             f"{len(ctx['steps'])} tool calls, {client._call_count} API calls")

    if emit_fn:
        emit_fn({"type": "static_done", "request_id": request_id,
                 "elapsed": round(elapsed, 2),
                 "tool_calls": len(ctx["steps"]),
                 "api_calls": client._call_count})

    return {
        "agent_response": "Done.",
        "tool_calls": ctx["steps"],
        "api_calls": client._call_count,
        "api_errors": client._error_count,
        "api_log": client._call_log,
        "elapsed": round(elapsed, 2),
        "task_types": [task_type],
        "mode": "static",
    }
