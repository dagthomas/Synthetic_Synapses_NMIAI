"""Deterministic task classifier and tool selector for Tripletex agent.

Classifies incoming prompts into known task types and returns only the
tools needed for that specific task. Falls back to all tools for
unrecognized prompts (no regression).
"""

import re
import logging

log = logging.getLogger(__name__)

# ── Task type → required tool names ──────────────────────────────────
# Every set also gets get_entity_by_id as a universal safety net.

TASK_TOOL_MAP: dict[str, list[str]] = {
    # Tier 1 — basic entity
    "create_employee":    ["create_employee"],
    "create_customer":    ["create_customer"],
    "create_product":     ["create_product"],
    "create_department":  ["create_department"],
    "create_supplier":    ["create_supplier"],
    "create_contact":     ["create_customer", "create_contact"],
    "update_employee":    ["create_employee", "update_employee", "search_employees"],
    "update_customer":    ["create_customer", "update_customer"],
    "update_product":     ["create_product", "update_product"],
    "update_supplier":    ["create_supplier", "update_supplier"],
    "update_department":  ["create_department", "update_department"],
    "update_contact":     ["create_customer", "create_contact", "update_contact"],
    # Tier 2 — multi-step workflows
    "create_invoice":              ["create_customer", "create_product", "create_order", "create_invoice"],
    "create_multi_line_invoice":   ["create_customer", "create_product", "create_order", "create_invoice"],
    "create_project":              ["create_customer", "create_employee", "create_project"],
    "create_travel_expense":       ["create_employee", "create_travel_expense"],
    "travel_expense_with_costs":   ["create_employee", "create_travel_expense", "create_travel_expense_cost",
                                    "create_mileage_allowance", "create_per_diem_compensation", "update_travel_expense"],
    "invoice_with_payment":        ["create_customer", "create_product", "create_order", "create_invoice",
                                    "register_payment"],
    "create_credit_note":          ["create_customer", "create_product", "create_order", "create_invoice",
                                    "create_credit_note"],
    "create_employee_with_employment": ["create_employee", "create_employment", "create_employment_details",
                                       "create_standard_time", "create_leave_of_absence"],
    "supplier_invoice":            ["create_supplier", "create_incoming_invoice"],
    # Tier 3 — complex
    "delete_travel_expense":  ["search_travel_expenses", "delete_travel_expense"],
    "delete_customer":        ["search_customers", "delete_customer"],
    "delete_supplier":        ["search_suppliers", "delete_supplier"],
    "delete_product":         ["search_products", "delete_product"],
    "delete_department":      ["search_departments", "delete_department"],
    "delete_contact":         ["search_contacts", "delete_contact", "update_contact"],
    "delete_employee":        ["search_employees", "update_employee"],
    "create_ledger_voucher":  ["create_voucher", "get_ledger_accounts"],
    "reverse_voucher":        ["search_vouchers", "reverse_voucher"],
    "delete_invoice":         ["create_customer", "create_product", "create_order", "create_invoice",
                               "create_credit_note"],
    "create_opening_balance": ["create_opening_balance", "get_ledger_accounts"],
    "bank_reconciliation":    ["extract_file_content", "search_bank_accounts", "create_voucher"],
    "process_invoice_file":   ["extract_file_content", "create_customer", "create_product", "create_order",
                               "create_invoice"],
    "year_end":               ["search_year_ends", "search_year_end_annexes", "create_year_end_note"],
    "salary":                 ["search_salary_types", "create_salary_transaction"],
}

# Universal tool always included
_UNIVERSAL_TOOLS = ["get_entity_by_id"]


# ── Keyword patterns for classification ──────────────────────────────
# Each entry: (task_type, positive_keywords, negative_keywords, bonus_score)
# Score = count of matched positive keywords - if any negative matches, skip.
# Multi-word patterns matched first, then single words.

_PATTERNS: list[tuple[str, list[str], list[str], int]] = [
    # ── Delete tasks (check first — "slett" is strong signal) ──
    ("delete_travel_expense", ["slett reiseregning", "delete travel expense", "fjern reiseregning",
                               "slett reise"], [], 10),
    ("delete_customer", ["slett kunde", "delete customer", "fjern kunde",
                         "slett kunden", "eliminar cliente", "supprimer client",
                         "kunde löschen", "eliminar o cliente"], [], 10),
    ("delete_supplier", ["slett leverandør", "delete supplier", "fjern leverandør",
                         "slett leverandøren", "eliminar proveedor", "supprimer fournisseur",
                         "lieferant löschen", "eliminar o fornecedor"], [], 10),
    ("delete_product", ["slett produkt", "delete product", "fjern produkt",
                        "slett produktet", "eliminar producto", "supprimer produit",
                        "produkt löschen", "eliminar o produto"], [], 10),
    ("delete_department", ["slett avdeling", "delete department", "fjern avdeling",
                           "slett avdelingen", "eliminar departamento", "supprimer département",
                           "abteilung löschen", "eliminar o departamento"], [], 10),
    ("delete_contact", ["slett kontakt", "delete contact", "fjern kontakt",
                        "slett kontaktperson", "slett kontaktpersonen",
                        "eliminar contacto", "supprimer contact",
                        "kontakt löschen", "eliminar o contato"], [], 10),
    ("delete_employee", ["slett ansatt", "delete employee", "fjern ansatt",
                         "slett den ansatte", "deaktiver ansatt", "deactivate employee",
                         "eliminar empleado", "supprimer employé",
                         "mitarbeiter löschen", "eliminar o funcionário"], [], 10),

    # ── Reverse voucher ──
    ("reverse_voucher", ["tilbakeføre bilag", "reversere bilag", "reverse voucher",
                         "tilbakefør bilag", "tilbakeføre bilaget",
                         "tilbakeføre", "reversere", "reverse"], [], 8),

    # ── Credit note (before invoice — "kreditnota" overrides "faktura") ──
    ("create_credit_note", ["kreditnota", "kreditere", "credit note", "kreditér",
                            "nota de crédito", "nota de crédito", "note de crédit",
                            "gutschrift", "kreditere fakturaen"], [], 8),

    # ── Delete/credit invoice ──
    ("delete_invoice", ["slett faktura", "delete invoice", "slett fakturaen",
                        "kanseller faktura", "annuller faktura"], [], 10),

    # ── Invoice with payment ──
    ("invoice_with_payment", ["faktura betaling", "invoice payment", "registrer betaling",
                              "register payment", "betal faktura",
                              "faktura og betaling", "invoice and payment",
                              "registrar pago", "enregistrer paiement",
                              "zahlung registrieren", "registrar pagamento"], [], 9),

    # ── Multi-line invoice ──
    ("create_multi_line_invoice", ["flere produkter", "multiple products", "flere linjer",
                                   "multiple lines", "multi-line", "multiline",
                                   "varios productos", "plusieurs produits",
                                   "mehrere produkte", "vários produtos",
                                   "fleire produkt"], [], 9),

    # ── Process invoice from file ──
    ("process_invoice_file", ["vedlagt faktura", "attached invoice", "se vedlegg",
                              "fra vedlegg", "from attachment", "from file",
                              "factura adjunta", "facture jointe",
                              "angehängte rechnung", "fatura anexa"], [], 9),

    # ── Bank reconciliation ──
    ("bank_reconciliation", ["bankavstemming", "bank reconciliation", "avstemming",
                             "reconciliación bancaria", "rapprochement bancaire",
                             "bankabstimmung", "reconciliação bancária"], [], 8),

    # ── Opening balance ──
    ("create_opening_balance", ["åpningsbalanse", "opening balance", "inngående balanse",
                                "balance de apertura", "solde d'ouverture",
                                "eröffnungsbilanz", "saldo de abertura",
                                "opningsbalanse"], [], 8),

    # ── Year-end ──
    ("year_end", ["årsoppgjør", "year-end", "year end", "årsavslutning",
                  "cierre de año", "clôture annuelle",
                  "jahresabschluss", "encerramento do ano"], [], 8),

    # ── Salary ──
    ("salary", ["lønnskjøring", "payroll", "lønnstransaksjon", "salary transaction",
                "lønnsslipp", "salary slip", "lønn", "salary",
                "nómina", "fiche de paie", "gehaltsabrechnung", "folha de pagamento"],
     ["ansatt", "employee", "ansettelse"], 6),

    # ── Travel expense with costs ──
    ("travel_expense_with_costs", ["reiseregning med utlegg", "travel expense with costs",
                                   "kjøregodtgjørelse", "mileage allowance",
                                   "diett", "per diem", "reiseutlegg",
                                   "gastos de viaje con costos",
                                   "note de frais avec coûts",
                                   "reisekosten mit kosten",
                                   "despesas de viagem com custos"], [], 9),

    # ── Supplier invoice ──
    ("supplier_invoice", ["leverandørfaktura", "supplier invoice", "inngående faktura",
                          "incoming invoice", "factura de proveedor",
                          "facture fournisseur", "lieferantenrechnung",
                          "fatura de fornecedor"], [], 8),

    # ── Employee with employment ──
    ("create_employee_with_employment", ["ansettelsesforhold", "employment", "arbeidsforhold",
                                         "ansettelse", "employment details",
                                         "relación laboral", "contrat de travail",
                                         "arbeitsverhältnis", "relação de emprego",
                                         "permisjon", "leave of absence",
                                         "arbeidstid", "working hours"], [], 7),

    # ── Create project ──
    ("create_project", ["prosjekt", "project", "proyecto", "projet", "projekt", "projeto",
                        "prosjektleder", "project manager"],
     ["faktura", "invoice"], 5),

    # ── Travel expense (simple — after "with costs" check) ──
    ("create_travel_expense", ["reiseregning", "travel expense", "reise",
                               "gastos de viaje", "note de frais",
                               "reisekosten", "despesas de viagem",
                               "reiserekning"],
     ["slett", "delete", "fjern", "utlegg", "kjøregodtgjørelse", "diett"], 5),

    # ── Simple invoice (after all invoice variants checked) ──
    ("create_invoice", ["faktura", "invoice", "factura", "facture", "rechnung", "fatura"],
     ["slett", "delete", "kreditnota", "kreditere", "credit note",
      "betaling", "payment", "vedlagt", "attached", "leverandør", "supplier",
      "inngående", "incoming"], 4),

    # ── Contact ──
    ("create_contact", ["kontaktperson", "contact person", "kontakt",
                        "persona de contacto", "personne de contact",
                        "ansprechpartner", "pessoa de contato"], [], 6),

    # ── Update tasks ──
    ("update_employee", ["oppdater ansatt", "update employee", "endre ansatt",
                         "oppdatere ansatt", "actualizar empleado",
                         "mettre à jour employé", "mitarbeiter aktualisieren",
                         "atualizar funcionário",
                         "oppdatere mobilnummer", "oppdater mobilnummer",
                         "endre mobilnummer", "oppdatere telefon", "endre telefon",
                         "nytt mobilnummer", "bytte telefon"],
     ["slett", "delete"], 8),
    ("update_customer", ["oppdater kunde", "update customer", "endre kunde",
                         "oppdatere kunde", "actualizar cliente",
                         "mettre à jour client", "kunde aktualisieren",
                         "atualizar cliente",
                         "oppdatere telefon", "endre telefonnummer",
                         "nytt telefonnummer", "oppdatere e-post", "endre e-post"],
     ["slett", "delete", "ansatt", "employee"], 8),
    ("update_product", ["oppdater produkt", "update product", "endre produkt",
                        "oppdatere produkt", "endre pris", "update price",
                        "actualizar producto", "mettre à jour produit",
                        "produkt aktualisieren", "atualizar produto",
                        "oppdatere prisen", "endre prisen", "oppdater pris",
                        "ny pris", "justere pris", "øke prisen", "senke prisen"],
     ["slett", "delete"], 8),
    ("update_supplier", ["oppdater leverandør", "update supplier", "endre leverandør",
                         "oppdatere leverandør", "actualizar proveedor",
                         "mettre à jour fournisseur", "lieferant aktualisieren",
                         "atualizar fornecedor"],
     ["slett", "delete"], 8),
    ("update_department", ["oppdater avdeling", "update department", "endre avdeling",
                           "oppdatere avdeling", "actualizar departamento",
                           "mettre à jour département", "abteilung aktualisieren",
                           "atualizar departamento"],
     ["slett", "delete"], 8),
    ("update_contact", ["oppdater kontaktperson", "update contact", "endre kontaktperson",
                         "oppdatere kontakt", "actualizar contacto",
                         "mettre à jour contact", "kontakt aktualisieren",
                         "atualizar contato"],
     ["slett", "delete"], 8),

    # ── Ledger voucher ──
    ("create_ledger_voucher", ["bilag", "voucher", "korrigeringsbilag", "correction voucher",
                               "korreksjon", "correction", "postering",
                               "comprobante", "pièce comptable", "buchungsbeleg",
                               "comprovante"],
     ["tilbakeføre", "reversere", "reverse", "slett", "delete",
      "åpningsbalanse", "opening balance"], 5),

    # ── Basic entity creation (lowest priority) ──
    ("create_supplier", ["leverandør", "supplier", "proveedor", "fournisseur",
                         "lieferant", "fornecedor"],
     ["slett", "delete", "faktura", "invoice", "inngående"], 4),
    ("create_department", ["avdeling", "department", "departamento", "département",
                           "abteilung", "departamento"],
     ["slett", "delete"], 4),
    ("create_employee", ["ansatt", "employee", "empleado", "employé",
                         "mitarbeiter", "funcionário", "tilsett",
                         "kollega", "medarbeider", "ny kollega", "new colleague"],
     ["slett", "delete", "oppdater", "update", "endre", "reise", "travel",
      "prosjekt", "project", "ansettelse", "employment", "permisjon", "leave",
      "arbeidstid", "lønn", "salary", "deaktiver", "deactivate"], 3),
    ("create_customer", ["kunde", "customer", "cliente", "client", "klient",
                         "kunden"],
     ["slett", "delete", "oppdater", "update", "endre", "faktura", "invoice",
      "kontakt", "contact", "prosjekt", "project"], 3),
    ("create_product", ["produkt", "product", "producto", "produit", "produkt",
                        "produto"],
     ["slett", "delete", "oppdater", "update", "endre", "faktura", "invoice"], 3),
]


def classify_task(prompt: str) -> str | None:
    """Classify a prompt into a known task type.

    Returns task type string if confident, None for fallback to all tools.
    """
    text = prompt.lower().strip()
    # Strip email addresses to prevent false positives (e.g. "faktura@firma.no" → "invoice")
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    best_type: str | None = None
    best_score: int = 0

    for task_type, positives, negatives, bonus in _PATTERNS:
        # Check negative keywords first — if any match, skip this pattern
        neg_hit = False
        for neg in negatives:
            if neg in text:
                neg_hit = True
                break
        if neg_hit:
            continue

        # Score positive keyword matches
        score = 0
        for pos in positives:
            if pos in text:
                # Multi-word patterns score higher
                word_count = len(pos.split())
                score += bonus * word_count

        if score > best_score:
            best_score = score
            best_type = task_type

    if best_score > 0:
        log.debug(f"Classified as '{best_type}' (score={best_score})")
        return best_type

    log.debug("Could not classify task, falling back to all tools")
    return None


# ── Category-level fallback (between exact classification and all-tools) ──
# When classify_task() returns None, detect broad categories to narrow from 118 to ~10-25 tools.

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "employee": ["ansatt", "employee", "kollega", "medarbeider", "empleado", "employé",
                 "mitarbeiter", "funcionário", "tilsett", "deaktiver ansatt"],
    "customer": ["kunde", "customer", "klient", "cliente", "client", "kunden"],
    "invoice":  ["faktura", "invoice", "kreditnota", "credit note", "factura", "rechnung",
                 "fatura", "facture"],
    "supplier": ["leverandør", "supplier", "proveedor", "fournisseur", "lieferant", "fornecedor"],
    "product":  ["produkt", "product", "producto", "produit", "produto", "vare"],
    "travel":   ["reiseregning", "travel expense", "reise", "diett", "kjøregodtgjørelse",
                 "mileage", "per diem"],
    "project":  ["prosjekt", "project", "proyecto", "projet", "projekt", "projeto"],
    "ledger":   ["bilag", "voucher", "postering", "korreksjon", "correction",
                 "åpningsbalanse", "opening balance"],
    "department": ["avdeling", "department", "departamento", "département", "abteilung"],
    "salary":   ["lønn", "salary", "lønnskjøring", "payroll", "nómina", "gehaltsabrechnung"],
    "contact":  ["kontaktperson", "contact person", "kontakt", "ansprechpartner"],
}

CATEGORY_TOOLS: dict[str, list[str]] = {
    "employee": ["create_employee", "update_employee", "search_employees",
                 "create_employment", "create_employment_details",
                 "create_standard_time", "create_leave_of_absence"],
    "customer": ["create_customer", "update_customer", "search_customers",
                 "delete_customer"],
    "invoice":  ["create_customer", "create_product", "create_order", "create_invoice",
                 "create_credit_note", "register_payment", "search_invoices"],
    "supplier": ["create_supplier", "update_supplier", "search_suppliers",
                 "delete_supplier", "create_incoming_invoice"],
    "product":  ["create_product", "update_product", "search_products",
                 "delete_product"],
    "travel":   ["create_employee", "create_travel_expense", "search_travel_expenses",
                 "delete_travel_expense", "create_travel_expense_cost",
                 "create_mileage_allowance", "create_per_diem_compensation",
                 "update_travel_expense"],
    "project":  ["create_customer", "create_employee", "create_project"],
    "ledger":   ["create_voucher", "get_ledger_accounts", "search_vouchers",
                 "reverse_voucher", "create_opening_balance"],
    "department": ["create_department", "update_department", "search_departments",
                   "delete_department"],
    "salary":   ["search_salary_types", "create_salary_transaction", "create_employee"],
    "contact":  ["create_customer", "create_contact", "update_contact",
                 "search_contacts", "delete_contact"],
}


def detect_categories(prompt: str) -> list[str]:
    """Detect broad entity categories mentioned in a prompt.

    Returns list of matching category names (e.g. ["employee", "customer"]).
    """
    text = prompt.lower().strip()
    text = re.sub(r'\S+@\S+\.\S+', '', text)  # strip emails
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(category)
                break
    return matched


def select_tools(task_type: str | None, all_tools_dict: dict, has_files: bool = False,
                 prompt: str = "") -> list:
    """Select tools for a given task type.

    Args:
        task_type: Classified task type, or None for all tools.
        all_tools_dict: Dict of {tool_name: tool_function}.
        has_files: Whether the request has file attachments.
        prompt: Original prompt text (used for category fallback when task_type is None).

    Returns:
        List of tool functions to pass to the agent.
    """
    if task_type is None:
        # Try category-level fallback before returning all tools
        if prompt:
            categories = detect_categories(prompt)
            if categories:
                required_names = set()
                for cat in categories:
                    required_names.update(CATEGORY_TOOLS.get(cat, []))
                # Add universal tools
                for name in _UNIVERSAL_TOOLS:
                    required_names.add(name)
                if has_files:
                    required_names.add("extract_file_content")
                selected = [all_tools_dict[n] for n in required_names if n in all_tools_dict]
                if selected:
                    log.info(f"Category fallback: {categories} -> {len(selected)} tools")
                    return selected
        return list(all_tools_dict.values())

    # Get required tool names for this task type
    required_names = set(TASK_TOOL_MAP.get(task_type, []))

    # Add universal tools
    for name in _UNIVERSAL_TOOLS:
        required_names.add(name)

    # Add file tools if attachments present
    if has_files:
        required_names.add("extract_file_content")

    # Resolve names to actual tool functions
    selected = []
    for name in required_names:
        if name in all_tools_dict:
            selected.append(all_tools_dict[name])
        else:
            log.warning(f"Tool '{name}' not found in tools dict")

    # Safety: if we resolved very few tools, fall back to all
    if len(selected) < 1:
        log.warning(f"No tools resolved for task_type='{task_type}', falling back to all")
        return list(all_tools_dict.values())

    return selected
