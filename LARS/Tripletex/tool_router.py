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
    # no/nn + en + es + pt + de + fr
    ("delete_travel_expense", ["slett reiseregning", "delete travel expense", "fjern reiseregning",
                               "slett reise", "slett reiserekning",
                               "eliminar gastos de viaje", "excluir despesas de viagem",
                               "supprimer note de frais", "reisekosten löschen"], [], 10),
    ("delete_customer", ["slett kunde", "delete customer", "fjern kunde", "slett kunden",
                         "eliminar cliente", "eliminar el cliente", "eliminar al cliente",
                         "excluir cliente", "excluir o cliente",
                         "supprimer client", "supprimer le client",
                         "kunde löschen", "kunden löschen"], [], 10),
    ("delete_supplier", ["slett leverandør", "delete supplier", "fjern leverandør", "slett leverandøren",
                         "eliminar proveedor", "eliminar el proveedor", "eliminar al proveedor",
                         "excluir fornecedor", "excluir o fornecedor",
                         "supprimer fournisseur", "supprimer le fournisseur",
                         "lieferant löschen", "lieferanten löschen"], [], 10),
    ("delete_product", ["slett produkt", "delete product", "fjern produkt", "slett produktet",
                        "eliminar producto", "eliminar el producto",
                        "excluir produto", "excluir o produto",
                        "supprimer produit", "supprimer le produit",
                        "produkt löschen"], [], 10),
    ("delete_department", ["slett avdeling", "delete department", "fjern avdeling", "slett avdelingen",
                           "eliminar departamento", "eliminar el departamento",
                           "excluir departamento", "excluir o departamento",
                           "supprimer département", "supprimer le département",
                           "abteilung löschen"], [], 10),
    ("delete_contact", ["slett kontakt", "delete contact", "fjern kontakt",
                        "slett kontaktperson", "slett kontaktpersonen",
                        "eliminar contacto", "eliminar el contacto",
                        "excluir contato", "excluir o contato",
                        "supprimer contact", "supprimer le contact",
                        "kontakt löschen", "kontaktperson löschen"], [], 10),
    ("delete_employee", ["slett ansatt", "delete employee", "fjern ansatt",
                         "slett den ansatte", "deaktiver ansatt", "deactivate employee",
                         "eliminar empleado", "eliminar al empleado",
                         "excluir funcionário", "excluir o funcionário",
                         "desativar funcionário", "desactivar empleado",
                         "supprimer employé", "supprimer l'employé", "désactiver employé",
                         "mitarbeiter löschen", "mitarbeiter deaktivieren"], [], 10),

    # ── Reverse voucher ──
    ("reverse_voucher", ["tilbakeføre bilag", "reversere bilag", "reverse voucher",
                         "tilbakefør bilag", "tilbakeføre bilaget",
                         "tilbakeføre", "reversere",
                         "revertir comprobante", "reverter comprovante", "estornar",
                         "contrepasser", "extourner", "stornieren",
                         "tilbakeføre bilaget"], [], 8),

    # ── Credit note (before invoice — "kreditnota" overrides "faktura") ──
    ("create_credit_note", ["kreditnota", "kreditere", "credit note", "kreditér",
                            "nota de crédito", "note de crédit", "gutschrift",
                            "kreditere fakturaen", "kreditera",
                            "nota de crédito", "nota crediticia"], [], 8),

    # ── Delete/credit invoice ──
    ("delete_invoice", ["slett faktura", "delete invoice", "slett fakturaen",
                        "kanseller faktura", "annuller faktura",
                        "cancelar factura", "cancelar fatura", "annuler facture",
                        "rechnung stornieren", "rechnung löschen"], [], 10),

    # ── Invoice with payment ──
    ("invoice_with_payment", ["faktura betaling", "invoice payment", "registrer betaling",
                              "register payment", "betal faktura",
                              "faktura og betaling", "invoice and payment",
                              "registrar pago", "factura y pago",
                              "registrar pagamento", "fatura e pagamento",
                              "enregistrer paiement", "facture et paiement",
                              "zahlung registrieren", "rechnung und zahlung"], [], 9),

    # ── Multi-line invoice ──
    ("create_multi_line_invoice", ["flere produkter", "multiple products", "flere linjer",
                                   "multiple lines", "multi-line", "multiline",
                                   "varios productos", "múltiples productos",
                                   "vários produtos", "múltiplos produtos",
                                   "plusieurs produits", "plusieurs lignes",
                                   "mehrere produkte", "mehrere positionen",
                                   "fleire produkt"], [], 9),

    # ── Process invoice from file ──
    ("process_invoice_file", ["vedlagt faktura", "attached invoice", "se vedlegg",
                              "fra vedlegg", "from attachment", "from file",
                              "factura adjunta", "factura del archivo",
                              "fatura anexa", "fatura do arquivo",
                              "facture jointe", "facture du fichier",
                              "angehängte rechnung", "rechnung aus datei"], [], 9),

    # ── Bank reconciliation ──
    ("bank_reconciliation", ["bankavstemming", "bank reconciliation", "avstemming",
                             "reconciliación bancaria", "conciliación bancaria",
                             "reconciliação bancária", "conciliação bancária",
                             "rapprochement bancaire", "réconciliation bancaire",
                             "bankabstimmung", "kontoavstemming"], [], 8),

    # ── Opening balance ──
    ("create_opening_balance", ["åpningsbalanse", "opening balance", "inngående balanse",
                                "balance de apertura", "saldo inicial",
                                "saldo de abertura", "balanço de abertura",
                                "solde d'ouverture", "bilan d'ouverture",
                                "eröffnungsbilanz", "anfangssaldo",
                                "opningsbalanse"], [], 8),

    # ── Year-end ──
    ("year_end", ["årsoppgjør", "year-end", "year end", "årsavslutning",
                  "cierre de año", "cierre del ejercicio",
                  "encerramento do ano", "fechamento do exercício",
                  "clôture annuelle", "clôture de l'exercice",
                  "jahresabschluss", "jahresende"], [], 8),

    # ── Salary ──
    ("salary", ["lønnskjøring", "payroll", "lønnstransaksjon", "salary transaction",
                "lønnsslipp", "salary slip", "lønn", "salary",
                "salário", "processar salário", "processamento salarial", "bónus salarial",
                "salario", "procesar salario", "procesar nómina", "nómina",
                "salaire", "traitement salarial", "fiche de paie", "bulletin de paie",
                "gehalt", "gehaltsabrechnung", "lohnabrechnung", "lohn",
                "folha de pagamento", "folha salarial",
                "lønskjøring", "lønstransaksjon"],
     ["opprett ansatt", "create employee", "crear empleado", "criar funcionário",
      "créer employé", "mitarbeiter erstellen"], 6),

    # ── Travel expense with costs ──
    ("travel_expense_with_costs", ["reiseregning med utlegg", "travel expense with costs",
                                   "kjøregodtgjørelse", "mileage allowance",
                                   "diett", "per diem", "reiseutlegg",
                                   "gastos de viaje con costos", "gastos de viaje con gastos",
                                   "despesas de viagem com custos",
                                   "note de frais avec coûts", "note de frais avec dépenses",
                                   "reisekosten mit kosten", "reisekosten mit ausgaben",
                                   "reiserekning med utlegg"], [], 9),

    # ── Supplier invoice ──
    ("supplier_invoice", ["leverandørfaktura", "supplier invoice", "inngående faktura",
                          "incoming invoice",
                          "factura de proveedor", "factura del proveedor",
                          "fatura de fornecedor", "fatura do fornecedor",
                          "facture fournisseur", "facture du fournisseur",
                          "lieferantenrechnung", "eingangsrechnung",
                          "leverandørfaktura"], [], 8),

    # ── Employee with employment ──
    ("create_employee_with_employment", ["ansettelsesforhold", "employment", "arbeidsforhold",
                                         "ansettelse", "employment details",
                                         "relación laboral", "contrato de trabajo", "contrato laboral",
                                         "relação de emprego", "contrato de trabalho",
                                         "contrat de travail", "relation de travail",
                                         "arbeitsverhältnis", "arbeitsvertrag",
                                         "tilsetjingsforhold",
                                         "permisjon", "leave of absence",
                                         "arbeidstid", "working hours"], [], 7),

    # ── Create project ──
    ("create_project", ["prosjekt", "project", "proyecto", "projet", "projekt", "projeto",
                        "prosjektleder", "project manager",
                        "jefe de proyecto", "gerente de projeto", "chef de projet", "projektleiter"],
     ["faktura", "invoice", "factura", "fatura", "facture", "rechnung"], 5),

    # ── Travel expense (simple — after "with costs" check) ──
    ("create_travel_expense", ["reiseregning", "travel expense", "reise",
                               "gastos de viaje", "informe de gastos",
                               "despesas de viagem", "relatório de viagem",
                               "note de frais", "frais de déplacement",
                               "reisekosten", "reisekostenabrechnung",
                               "reiserekning"],
     ["slett", "delete", "fjern", "utlegg", "kjøregodtgjørelse", "diett",
      "eliminar", "excluir", "supprimer", "löschen"], 5),

    # ── Simple invoice (after all invoice variants checked) ──
    ("create_invoice", ["faktura", "invoice", "factura", "facture", "rechnung", "fatura"],
     ["slett", "delete", "kreditnota", "kreditere", "credit note",
      "betaling", "payment", "vedlagt", "attached", "leverandør", "supplier",
      "inngående", "incoming", "proveedor", "fornecedor", "fournisseur", "lieferant",
      "eliminar", "excluir", "supprimer", "löschen",
      "pago", "pagamento", "paiement", "zahlung"], 4),

    # ── Contact ──
    ("create_contact", ["kontaktperson", "contact person", "kontakt",
                        "persona de contacto", "contacto",
                        "pessoa de contato", "contato",
                        "personne de contact",
                        "ansprechpartner", "kontaktperson"], [], 6),

    # ── Update tasks ──
    ("update_employee", ["oppdater ansatt", "update employee", "endre ansatt",
                         "oppdatere ansatt",
                         "actualizar empleado", "modificar empleado",
                         "atualizar funcionário", "modificar funcionário",
                         "mettre à jour employé", "modifier employé",
                         "mitarbeiter aktualisieren", "mitarbeiter ändern",
                         "oppdatere mobilnummer", "oppdater mobilnummer",
                         "endre mobilnummer", "oppdatere telefon", "endre telefon",
                         "nytt mobilnummer", "bytte telefon",
                         "cambiar teléfono", "alterar telefone", "changer téléphone",
                         "oppdater den ansatte", "endre den ansatte"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_customer", ["oppdater kunde", "update customer", "endre kunde", "oppdatere kunde",
                         "actualizar cliente", "modificar cliente",
                         "atualizar cliente", "modificar cliente",
                         "mettre à jour client", "modifier client",
                         "kunde aktualisieren", "kunden ändern",
                         "oppdatere telefon", "endre telefonnummer",
                         "nytt telefonnummer", "oppdatere e-post", "endre e-post",
                         "oppdater kunden", "endre kunden"],
     ["slett", "delete", "ansatt", "employee", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_product", ["oppdater produkt", "update product", "endre produkt", "oppdatere produkt",
                        "endre pris", "update price",
                        "actualizar producto", "modificar producto", "cambiar precio",
                        "atualizar produto", "modificar produto", "alterar preço",
                        "mettre à jour produit", "modifier produit", "changer prix",
                        "produkt aktualisieren", "produkt ändern", "preis ändern",
                        "oppdatere prisen", "endre prisen", "oppdater pris",
                        "ny pris", "justere pris", "øke prisen", "senke prisen",
                        "oppdater produktet", "endre produktet"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_supplier", ["oppdater leverandør", "update supplier", "endre leverandør", "oppdatere leverandør",
                         "actualizar proveedor", "modificar proveedor",
                         "atualizar fornecedor", "modificar fornecedor",
                         "mettre à jour fournisseur", "modifier fournisseur",
                         "lieferant aktualisieren", "lieferant ändern",
                         "oppdater leverandøren", "endre leverandøren"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_department", ["oppdater avdeling", "update department", "endre avdeling", "oppdatere avdeling",
                           "actualizar departamento", "modificar departamento",
                           "atualizar departamento", "modificar departamento",
                           "mettre à jour département", "modifier département",
                           "abteilung aktualisieren", "abteilung ändern",
                           "oppdater avdelingen", "endre avdelingen",
                           "gi nytt navn", "endre navn"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_contact", ["oppdater kontaktperson", "update contact", "endre kontaktperson", "oppdatere kontakt",
                         "actualizar contacto", "modificar contacto",
                         "atualizar contato", "modificar contato",
                         "mettre à jour contact", "modifier contact",
                         "kontakt aktualisieren", "kontakt ändern",
                         "oppdater kontakten", "endre kontakten"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),

    # ── Ledger voucher ──
    ("create_ledger_voucher", ["bilag", "voucher", "korrigeringsbilag", "correction voucher",
                               "korreksjon", "correction", "postering",
                               "comprobante", "asiento contable",
                               "comprovante", "lançamento contábil",
                               "pièce comptable", "écriture comptable",
                               "buchungsbeleg", "buchung", "korrekturbuchung"],
     ["tilbakeføre", "reversere", "reverse", "slett", "delete",
      "åpningsbalanse", "opening balance", "revertir", "reverter",
      "contrepasser", "stornieren"], 5),

    # ── Basic entity creation (lowest priority) ──
    ("create_supplier", ["leverandør", "supplier", "proveedor", "fournisseur",
                         "lieferant", "fornecedor", "leverandøren"],
     ["slett", "delete", "faktura", "invoice", "inngående",
      "eliminar", "excluir", "supprimer", "löschen", "oppdater", "update", "endre",
      "actualizar", "atualizar", "modifier", "ändern"], 4),
    ("create_department", ["avdeling", "department", "departamento", "département",
                           "abteilung", "avdelinga"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen",
      "oppdater", "update", "endre", "actualizar", "atualizar", "modifier", "ändern"], 4),
    ("create_employee", ["ansatt", "employee", "empleado", "employé",
                         "mitarbeiter", "funcionário", "tilsett",
                         "kollega", "medarbeider", "ny kollega", "new colleague",
                         "empregado"],
     ["slett", "delete", "oppdater", "update", "endre", "reise", "travel",
      "prosjekt", "project", "ansettelse", "employment", "permisjon", "leave",
      "arbeidstid", "lønn", "salary", "deaktiver", "deactivate",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern",
      "salário", "salario", "salaire", "gehalt", "lohn", "payroll", "nómina"], 3),
    ("create_customer", ["kunde", "customer", "cliente", "client", "klient", "kunden"],
     ["slett", "delete", "oppdater", "update", "endre",
      "faktura", "invoice", "fatura", "factura", "facture", "rechnung",
      "kontakt", "contact", "contato", "contacto",
      "prosjekt", "project", "proyecto", "projeto", "projet", "projekt",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern"], 3),
    ("create_product", ["produkt", "product", "producto", "produit", "produto", "vare"],
     ["slett", "delete", "oppdater", "update", "endre", "faktura", "invoice",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern"], 3),
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
    "employee": ["ansatt", "employee", "kollega", "medarbeider", "tilsett",
                 "empleado", "empleada", "funcionário", "funcionária", "empregado",
                 "employé", "employée", "mitarbeiter", "mitarbeiterin",
                 "deaktiver ansatt"],
    "customer": ["kunde", "customer", "klient", "kunden",
                 "cliente", "client", "cliente"],
    "invoice":  ["faktura", "invoice", "kreditnota", "credit note",
                 "factura", "rechnung", "fatura", "facture",
                 "nota de crédito", "note de crédit", "gutschrift"],
    "supplier": ["leverandør", "supplier", "leverandøren",
                 "proveedor", "fournisseur", "lieferant", "fornecedor"],
    "product":  ["produkt", "product", "producto", "produit", "produto", "vare"],
    "travel":   ["reiseregning", "travel expense", "reise", "diett", "kjøregodtgjørelse",
                 "mileage", "per diem", "reiserekning",
                 "gastos de viaje", "despesas de viagem", "note de frais",
                 "reisekosten", "frais de déplacement"],
    "project":  ["prosjekt", "project", "proyecto", "projet", "projekt", "projeto"],
    "ledger":   ["bilag", "voucher", "postering", "korreksjon", "correction",
                 "åpningsbalanse", "opening balance",
                 "comprobante", "comprovante", "pièce comptable", "buchungsbeleg",
                 "asiento", "lançamento"],
    "department": ["avdeling", "department", "departamento", "département", "abteilung", "avdelinga"],
    "salary":   ["lønn", "salary", "lønnskjøring", "payroll",
                 "salário", "salario", "salaire", "gehalt", "lohn",
                 "nómina", "gehaltsabrechnung", "folha de pagamento",
                 "fiche de paie", "bulletin de paie", "lohnabrechnung",
                 "bónus", "bonus"],
    "contact":  ["kontaktperson", "contact person", "kontakt",
                 "persona de contacto", "pessoa de contato", "contato",
                 "personne de contact", "ansprechpartner"],
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
