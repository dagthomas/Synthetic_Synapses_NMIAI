"""Deterministic task classifier and tool selector for Tripletex agent.

Classifies incoming prompts into known task types and returns only the
tools needed for that specific task. Falls back to all tools for
unrecognized prompts (no regression).
"""

import re
import logging
import unicodedata
from typing import NamedTuple

log = logging.getLogger(__name__)


class ToolSelection(NamedTuple):
    """Result of select_tools(): the tools list + how we classified."""
    tools: list
    classification_level: str  # "exact" | "category" | "fallback"


def _strip_accents(s: str) -> str:
    """Strip diacritical marks (é→e, ü→u, ñ→n, à→a) but preserve Nordic ø/æ.

    Uses NFKD decomposition which splits accented chars into base + combining mark,
    then removes combining marks. Nordic ø and æ have no NFKD decomposition so they
    are preserved. Norwegian å decomposes to a + ring, but since we normalize BOTH
    patterns and text, matching still works.
    """
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

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
    "create_invoice":              ["create_customer", "create_product", "create_order", "create_invoice", "send_invoice"],
    "create_multi_line_invoice":   ["create_customer", "create_product", "create_order", "create_invoice", "send_invoice"],
    "create_project":              ["create_customer", "create_employee", "create_project"],
    "create_travel_expense":       ["create_employee", "create_travel_expense"],
    "create_travel_expense_with_costs": ["create_employee", "create_travel_expense", "create_travel_expense_cost",
                                        "create_mileage_allowance", "create_per_diem_compensation", "update_travel_expense"],
    "invoice_with_payment":        ["search_customers", "search_invoices", "create_customer", "create_product",
                                    "create_order", "create_invoice", "register_payment"],
    "create_credit_note":          ["create_customer", "create_product", "create_order", "create_invoice",
                                    "create_credit_note"],
    "create_employee_with_employment": ["create_employee", "create_employment", "create_employment_details",
                                       "create_standard_time", "create_leave_of_absence"],
    "create_supplier_invoice":     ["create_supplier", "create_incoming_invoice"],
    "project_invoice":             ["create_customer", "create_employee", "create_project",
                                    "create_product", "create_order", "create_invoice", "send_invoice"],
    "create_project_with_pm":      ["create_customer", "create_employee", "create_project",
                                    "create_product", "create_order", "create_invoice", "send_invoice"],
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
    "reverse_payment":        ["search_invoices", "register_payment"],
    "delete_invoice":         ["create_customer", "create_product", "create_order", "create_invoice",
                               "create_credit_note"],
    "create_opening_balance": ["create_opening_balance", "get_ledger_accounts"],
    "create_dimension":       ["create_accounting_dimension", "create_dimension_value",
                               "search_accounting_dimensions", "search_dimension_values",
                               "create_voucher", "get_ledger_accounts"],
    "bank_reconciliation":    ["extract_file_content", "search_bank_accounts", "create_voucher"],
    "process_invoice_file":   ["extract_file_content", "create_customer", "create_product", "create_order",
                               "create_invoice"],
    "year_end":               ["search_year_ends", "search_year_end_annexes", "create_year_end_note"],
    "salary":                 ["create_employee", "create_employment", "search_salary_types", "create_salary_transaction"],
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
                               "slette reiseregning", "slette reiserekning",
                               "slette denne reiseregningen", "fjerne denne reiseregningen",
                               "eliminar gastos de viaje", "excluir despesas de viagem",
                               "supprimer note de frais", "reisekosten löschen"], [], 10),
    ("delete_customer", ["slett kunde", "delete customer", "fjern kunde", "slett kunden",
                         "slette kunde", "slette kunden", "fjerne kunde", "fjerne kunden",
                         "sletter kunde", "sletter kunden",
                         "slett denne kunden", "slette denne kunden",
                         "fjerne denne kunden", "sletter denne kunden",
                         "kunderegisteret", "fra kunderegisteret",
                         "eliminar cliente", "eliminar el cliente", "eliminar al cliente",
                         "excluir cliente", "excluir o cliente",
                         "supprimer client", "supprimer le client",
                         "kunde löschen", "kunden löschen"], [], 10),
    ("delete_supplier", ["slett leverandør", "delete supplier", "fjern leverandør", "slett leverandøren",
                         "slette leverandør", "slette leverandøren", "fjerne leverandør",
                         "sletter leverandør", "sletter leverandøren",
                         "slett denne leverandøren", "slett denne leverandør",
                         "slette denne leverandøren", "fjerne denne leverandøren",
                         "slette denne leverandør", "fjerne denne leverandør",
                         "sletter denne leverandøren",
                         "eliminar proveedor", "eliminar el proveedor", "eliminar al proveedor",
                         "excluir fornecedor", "excluir o fornecedor",
                         "supprimer fournisseur", "supprimer le fournisseur",
                         "lieferant löschen", "lieferanten löschen"], [], 10),
    ("delete_product", ["slett produkt", "delete product", "fjern produkt", "slett produktet",
                        "slette produkt", "slette produktet", "fjerne produkt",
                        "sletter produkt", "sletter produktet",
                        "slett dette produktet", "slett dette produkt",
                        "fjern dette produktet", "fjern dette produkt",
                        "slette dette produktet", "fjerne dette produktet",
                        "sletter dette produktet", "sletter dette produkt",
                        "slette dette produkt", "fjerne dette produkt",
                        "sortimentet", "fra sortimentet",
                        "eliminar producto", "eliminar el producto",
                        "excluir produto", "excluir o produto",
                        "supprimer produit", "supprimer le produit",
                        "produkt löschen"], [], 10),
    ("delete_department", ["slett avdeling", "delete department", "fjern avdeling", "slett avdelingen",
                           "slette avdeling", "slette avdelingen", "fjerne avdeling",
                           "slette denne avdelingen", "fjerne denne avdelingen",
                           "eliminar departamento", "eliminar el departamento",
                           "excluir departamento", "excluir o departamento",
                           "supprimer département", "supprimer le département",
                           "abteilung löschen"], [], 10),
    ("delete_contact", ["slett kontakt", "delete contact", "fjern kontakt",
                        "slett kontaktperson", "slett kontaktpersonen",
                        "slette kontakt", "slette kontaktperson", "slette kontaktpersonen",
                        "sletter kontaktperson", "sletter kontaktpersonen",
                        "slettes som kontaktperson",
                        "henne som kontaktperson", "ham som kontaktperson",
                        "fjerne kontakt", "fjerne kontaktperson",
                        "slette denne kontaktpersonen", "fjerne denne kontaktpersonen",
                        "slette denne kontakten", "fjerne denne kontakten",
                        "eliminar contacto", "eliminar el contacto",
                        "excluir contato", "excluir o contato",
                        "supprimer contact", "supprimer le contact",
                        "kontakt löschen", "kontaktperson löschen"], [], 10),
    ("delete_employee", ["slett ansatt", "delete employee", "fjern ansatt",
                         "slett den ansatte", "deaktiver ansatt", "deactivate employee",
                         "slette ansatt", "fjerne ansatt", "deaktivere ansatt",
                         "sletter ansatt", "sletter den ansatte",
                         "slette denne ansatte", "fjerne denne ansatte",
                         "slette profilen", "deaktivere profilen",
                         "fjerne ham", "fjerne henne",
                         "deaktivere ham", "deaktivere henne",
                         "ansattregisteret", "fra ansattregisteret",
                         "rydde opp i ansatt", "rydde i ansatt",
                         "jobber ikke her lenger", "ikke lenger ansatt",
                         "sluttet hos oss", "sluttet i selskapet",
                         "har sluttet", "sagt opp",
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
                         "tilbakeføre bilaget"],
     ["faktura", "invoice", "factura", "fatura", "facture", "rechnung"], 8),

    # ── Reverse payment (returned/bounced payment) ──
    ("reverse_payment", ["tilbakeføre betaling", "reversere betaling", "reverser betaling",
                          "reverse payment", "revert payment", "payment returned", "payment bounced",
                          "betaling returnert", "betaling avvist", "betaling retur",
                          "reverter pagamento", "pagamento devolvido", "estornar pagamento",
                          "devolvido pelo banco", "pagamento retornado",
                          "revertir pago", "pago devuelto", "anular pago",
                          "pago rechazado", "pago retornado",
                          "contrepasser paiement", "paiement retourné", "annuler paiement",
                          "paiement rejeté",
                          "zahlung zurückbuchen", "zahlung stornieren", "rücklastschrift",
                          "zahlung zurückgewiesen",
                          "reverta o pagamento", "reverter o pagamento"], [], 10),

    # ── Credit note (before invoice — "kreditnota" overrides "faktura") ──
    ("create_credit_note", ["kreditnota", "kreditere", "credit note", "kreditér",
                            "nota de crédito", "note de crédit", "gutschrift",
                            "kreditere fakturaen", "kreditera",
                            "nota de crédito", "nota crediticia"],
     ["kreditere konto", "debitere konto", "debitere", "bilag", "hovedbok",
      "korreksjonsbilag", "postering", "feilpostering",
      "kanseller faktura", "annuller faktura",
      "slett faktura", "slett fakturaen"], 8),

    # ── Delete/credit invoice ──
    ("delete_invoice", ["slett faktura", "delete invoice", "slett fakturaen",
                        "slette fakturaen", "slette denne fakturaen",
                        "slett denne fakturaen", "fjerne fakturaen",
                        "kanseller faktura", "kansellere faktura", "kansellere fakturaen",
                        "annuller faktura", "annullere faktura", "annullere fakturaen",
                        "umiddelbart kreditere", "korrigere en feil",
                        "tilbakeføre fakturaen", "reversere fakturaen",
                        "cancelar factura", "cancelar fatura", "annuler facture",
                        "cancelled", "canceled",
                        "storniert", "stornieren",
                        "rechnung stornieren", "rechnung löschen"], [], 10),

    # ── Invoice with payment ──
    ("invoice_with_payment", ["faktura betaling", "invoice payment", "registrer betaling",
                              "register payment", "betal faktura",
                              "faktura og betaling", "invoice and payment",
                              "registrere betaling", "registrere innbetaling",
                              "registrer innbetaling", "innbetalingen", "innbetaling",
                              "registrer full betaling", "full betaling", "betaling",
                              "registrar pago", "registra el pago", "registra pago",
                              "pago completo", "factura y pago",
                              "registrar pagamento", "registra o pagamento",
                              "fatura e pagamento",
                              "enregistrer paiement", "enregistrez paiement",
                              "enregistrez le paiement", "paiement intégral",
                              "paiement complet", "facture et paiement",
                              "zahlung registrieren", "rechnung und zahlung",
                              "paiement"], [], 9),

    # ── Multi-line invoice ──
    ("create_multi_line_invoice", ["flere produkter", "multiple products", "flere linjer",
                                   "multiple lines", "multi-line", "multiline",
                                   "tre produkter", "to produkter", "fire produkter",
                                   "følgende produkter", "følgende varer",
                                   "fakturalinjer", "følgende linjer", "følgende poster",
                                   "varios productos", "múltiples productos",
                                   "vários produtos", "múltiplos produtos",
                                   "plusieurs produits", "plusieurs lignes",
                                   "mehrere produkte", "mehrere positionen",
                                   "fleire produkt",
                                   # Count-based patterns (três/tres/three/two/zwei/deux/dos/duas lines)
                                   "linhas de produto", "líneas de producto", "lignes de produit",
                                   "produktlinjer", "produktzeilen",
                                   "três linhas", "tres linhas", "tres líneas", "três líneas",
                                   "three lines", "three products",
                                   "two lines", "two products",
                                   "zwei produkte", "deux produits",
                                   "dos productos", "duas linhas", "dos líneas",
                                   "deux lignes", "zwei positionen"], [], 9),

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
    ("create_travel_expense_with_costs", ["reiseregning med utlegg", "travel expense with costs",
                                   "kjøregodtgjørelse", "mileage allowance",
                                   "diett", "per diem", "reiseutlegg",
                                   "kostnadslinje", "cost line", "kostnad",
                                   "hotellutgift", "hotellkostnad", "hotellovernatting",
                                   "overnatting", "hotell",
                                   "gastos de viaje con costos", "gastos de viaje con gastos",
                                   "despesas de viagem com custos",
                                   "note de frais avec coûts", "note de frais avec dépenses",
                                   "reisekosten mit kosten", "reisekosten mit ausgaben",
                                   "reiserekning med utlegg",
                                   "tagegeld", "tagessatz", "auslagen",
                                   "flugticket", "taxi", "taxikosten",
                                   "verpflegungspauschale", "spesen",
                                   "ajuda de custo", "diária", "diárias",
                                   "indemnité journalière", "frais de taxi",
                                   "viáticos", "gastos diarios"],
     ["leverandør", "supplier", "leverandørfaktura", "inngående faktura",
      "proveedor", "fornecedor", "fournisseur", "lieferant"], 9),

    # ── Supplier invoice ──
    ("create_supplier_invoice", ["leverandørfaktura", "supplier invoice", "inngående faktura",
                                "incoming invoice",
                          "factura de proveedor", "factura del proveedor",
                          "fatura de fornecedor", "fatura do fornecedor",
                          "facture fournisseur", "facture du fournisseur",
                          "lieferantenrechnung", "eingangsrechnung",
                          "bokføre leverandør", "bokfør leverandør",
                          "regning fra leverandør", "mottatt regning",
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
                                         "arbeidstid", "working hours",
                                         # Start date / tiltredelse keywords (critical for competition)
                                         "start date", "startdato", "start dato",
                                         "tiltredelse", "tiltredelsesdato", "tiltrer",
                                         "startdatum", "eintrittsdatum", "arbeitsbeginn",
                                         "date d'embauche", "date de début", "date de debut",
                                         "fecha de inicio", "fecha de incorporación", "fecha de incorporacion",
                                         "data de início", "data de inicio",
                                         "stillingsprosent", "stillingstittel",
                                         "job title", "position title",
                                         "occupation code", "yrkeskode"],
     ["prosjekt", "project", "proyecto", "projeto", "projet", "projekt"], 7),

    # ── Project + invoice (before pure project — matches project + invoice keywords) ──
    ("project_invoice", ["prosjekt faktura", "project invoice", "proyecto factura",
                         "projet facture", "projekt rechnung", "projeto fatura",
                         "factura de proyecto", "factura del proyecto",
                         "fatura de projeto", "fatura do projeto",
                         "facture de projet", "facture du projet",
                         "projektrechnung", "rechnung für projekt",
                         "fastpris", "fast pris", "fixed price", "precio fijo", "preço fixo",
                         "prix fixe", "festpreis",
                         "delbetaling", "partial payment", "pago parcial", "pagamento parcial",
                         "paiement partiel", "teilzahlung",
                         "fakturere prosjekt", "invoice project", "facturar proyecto",
                         "facturar projeto", "facturer projet",
                         "facturar el proyecto", "faturar o projeto",
                         "facturer le projet"], [], 10),

    # ── Create project with PM ──
    ("create_project_with_pm", ["prosjektleder", "project manager",
                                "jefe de proyecto", "gerente de projeto", "chef de projet", "projektleiter"],
     [], 10),

    # ── Create project ──
    ("create_project", ["prosjekt", "project", "proyecto", "projet", "projekt", "projeto"],
     ["faktura", "invoice", "factura", "fatura", "facture", "rechnung"], 5),

    # ── Travel expense (simple — after "with costs" check) ──
    ("create_travel_expense", ["reiseregning", "travel expense", "reise",
                               "gastos de viaje", "informe de gastos",
                               "despesas de viagem", "relatório de viagem",
                               "note de frais", "frais de déplacement",
                               "reisekosten", "reisekostenabrechnung",
                               "reiserekning"],
     ["slett", "delete", "fjern", "utlegg", "kjøregodtgjørelse", "diett",
      "kostnadslinje", "kostnad", "mileage", "per diem",
      "hotell", "overnatting",
      "tagegeld", "tagessatz", "auslagen", "flugticket", "spesen",
      "taxi", "taxikosten", "verpflegungspauschale",
      "ajuda de custo", "diária", "diárias",
      "indemnité journalière", "frais de taxi",
      "viáticos", "gastos diarios",
      "eliminar", "excluir", "supprimer", "löschen"], 5),

    # ── Simple invoice (after all invoice variants checked) ──
    ("create_invoice", ["opprett faktura", "opprette faktura", "ny faktura",
                        "lag faktura", "lage faktura", "send faktura",
                        "create invoice", "new invoice",
                        "crear factura", "nueva factura",
                        "criar fatura", "nova fatura",
                        "créer facture", "nouvelle facture",
                        "rechnung erstellen", "neue rechnung",
                        "faktura", "invoice", "factura", "facture", "rechnung", "fatura",
                        "fakturaen", "fakturaer"],
     ["slett", "delete", "kreditnota", "kreditere", "credit note",
      "registrer betaling", "registrere betaling", "registrer innbetaling",
      "innbetaling", "full betaling",
      "vedlagt", "attached", "leverandør", "supplier",
      "inngående", "incoming", "proveedor", "fornecedor", "fournisseur", "lieferant",
      "eliminar", "excluir", "supprimer", "löschen",
      "registrar pago", "registra el pago", "registra pago", "pago completo",
      "registrar pagamento", "registra o pagamento",
      "enregistrer paiement",
      "zahlung registrieren"], 4),

    # ── Contact ──
    ("create_contact", ["kontaktperson", "contact person",
                        "persona de contacto", "contacto",
                        "pessoa de contato", "contato",
                        "personne de contact",
                        "ansprechpartner",
                        "legge til kontaktperson", "legg til kontaktperson",
                        "opprett kontaktperson", "opprette kontaktperson",
                        "legge til en kontaktperson", "legg til en kontaktperson",
                        "legge til kontakt", "legg til kontakt",
                        "opprett kontakt", "opprette kontakt"],
     ["oppdater kontakt", "oppdatere kontakt", "endre kontakt",
      "oppdater kontaktperson", "oppdatere kontaktperson", "endre kontaktperson",
      "update contact", "modify contact",
      "kontaktinformasjon",
      "actualizar contacto", "atualizar contato", "modifier contact",
      "aktualisieren kontakt", "kontakt ändern",
      "slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 6),

    # ── Update tasks ──
    ("update_employee", ["oppdater ansatt", "update employee", "endre ansatt",
                         "oppdatere ansatt",
                         "actualizar empleado", "modificar empleado",
                         "atualizar funcionário", "modificar funcionário",
                         "mettre à jour employé", "modifier employé",
                         "mettre à jour l'employé",
                         "mitarbeiter aktualisieren", "mitarbeiter ändern",
                         "aktualisieren sie den mitarbeiter", "aktualisieren sie mitarbeiter",
                         "oppdatere mobilnummer", "oppdater mobilnummer",
                         "endre mobilnummer", "oppdatere telefon", "endre telefon",
                         "nytt mobilnummer", "bytte telefon",
                         "cambiar teléfono", "alterar telefone", "changer téléphone",
                         "oppdater den ansatte", "endre den ansatte",
                         "ansattkortet", "ansattkortet hans", "ansattkortet hennes",
                         "oppdater telefonnummeret", "oppdatere telefonnummeret",
                         "registrere mobilnummeret"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_customer", ["oppdater kunde", "update customer", "endre kunde", "oppdatere kunde",
                         "actualizar cliente", "modificar cliente",
                         "atualizar cliente", "modificar cliente",
                         "mettre à jour client", "modifier client",
                         "mettre à jour le client",
                         "kunde aktualisieren", "kunden ändern",
                         "kunden aktualisieren",
                         "aktualisieren sie den kunden", "aktualisieren sie kunden",
                         "oppdater telefon", "oppdatere telefon",
                         "endre telefonnummer", "oppdater telefonnummer",
                         "oppdatere telefonnummer", "oppdatere telefonnummeret",
                         "nytt telefonnummer",
                         "oppdater e-post", "oppdatere e-post",
                         "endre e-post", "endre e-postadressen",
                         "oppdater e-postadressen", "oppdatere e-postadressen",
                         "oppdater kunden", "endre kunden",
                         "oppdatere denne kunden", "endre denne kunden"],
     ["slett", "delete", "ansatt", "employee", "kontaktperson", "contact person",
      "opprett kontakt", "legg til kontakt", "legge til kontakt",
      "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_product", ["oppdater produkt", "update product", "endre produkt", "oppdatere produkt",
                        "endre pris", "update price", "prisendring", "price change",
                        "actualizar producto", "modificar producto", "cambiar precio",
                        "atualizar produto", "modificar produto", "alterar preço",
                        "mettre à jour produit", "modifier produit", "changer prix",
                        "mettre à jour le produit",
                        "produkt aktualisieren", "produkt ändern", "preis ändern",
                        "aktualisieren sie das produkt", "aktualisieren sie produkt",
                        "oppdater pris", "oppdater prisen", "oppdatere prisen",
                        "endre prisen", "justere pris", "justere prisen",
                        "justere utsalgsprisen", "utsalgsprisen",
                        "ny pris", "øke prisen", "senke prisen",
                        "oppdater produktet", "endre produktet",
                        "prisen opp", "prisen ned", "prisøkning", "prisreduksjon"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_supplier", ["oppdater leverandør", "update supplier", "endre leverandør", "oppdatere leverandør",
                         "oppdatere leverandøren",
                         "actualizar proveedor", "modificar proveedor",
                         "atualizar fornecedor", "modificar fornecedor",
                         "mettre à jour fournisseur", "modifier fournisseur",
                         "mettre à jour le fournisseur",
                         "lieferant aktualisieren", "lieferant ändern",
                         "aktualisieren sie den lieferanten", "aktualisieren sie lieferanten",
                         "oppdater leverandøren", "endre leverandøren"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_department", ["oppdater avdeling", "update department", "endre avdeling", "oppdatere avdeling",
                           "actualizar departamento", "modificar departamento",
                           "atualizar departamento", "modificar departamento",
                           "mettre à jour département", "modifier département",
                           "mettre à jour le département",
                           "abteilung aktualisieren", "abteilung ändern",
                           "aktualisieren sie die abteilung", "aktualisieren sie abteilung",
                           "oppdater avdelingen", "endre avdelingen",
                           "oppdatere avdelingen", "oppdatere den",
                           "oppdatere denne avdelingen", "endre denne avdelingen",
                           "gi nytt navn", "endre navn", "endre navnet",
                           "nytt navn", "nye navnet", "det nye navnet",
                           "avdelingsnavnet", "oppdater avdelingsnavnet",
                           "endre avdelingsnavnet", "nytt avdelingsnavn",
                           "avdelingen skal hete", "skal hete",
                           "rename department", "umbenennen"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),
    ("update_contact", ["oppdater kontaktperson", "update contact", "endre kontaktperson",
                         "oppdatere kontakt", "oppdater kontakt",
                         "oppdatere kontaktperson", "oppdatere kontaktpersonen",
                         "endre kontaktpersonen",
                         "oppdatere profilen", "oppdater profilen",
                         "actualizar contacto", "modificar contacto",
                         "actualizar persona de contacto",
                         "atualizar contato", "modificar contato",
                         "atualizar pessoa de contato",
                         "mettre à jour contact", "modifier contact",
                         "mettre à jour personne de contact",
                         "kontakt aktualisieren", "kontakt ändern",
                         "kontaktperson aktualisieren",
                         "aktualisieren sie die kontaktperson", "aktualisieren sie kontaktperson",
                         "aktualisieren sie kontakt",
                         "oppdater kontakten", "endre kontakten"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 8),

    # ── Custom accounting dimension ──
    ("create_dimension", ["buchhaltungsdimension", "regnskapsdimensjon", "accounting dimension",
                          "custom dimension", "benutzerdefinierte dimension",
                          "prosjekttype", "project type", "projekttyp",
                          "dimensjonsverdier", "dimensionswert", "dimension value",
                          "dimensjonsverdi", "type de projet", "tipo de proyecto",
                          "tipo de projeto", "kontodimensjon",
                          "egendefinert dimensjon", "tilpasset dimensjon"], [], 12),

    # ── Ledger voucher ──
    ("create_ledger_voucher", ["bilag", "voucher", "korrigeringsbilag", "korreksjonsbilag",
                               "correction voucher",
                               "korreksjon", "correction", "postering",
                               "debitere", "hovedbok", "hovedboken",
                               "feilpostering", "feilføring",
                               "comprobante", "asiento contable",
                               "comprovante", "lançamento contábil",
                               "pièce comptable", "écriture comptable",
                               "buchungsbeleg", "buchung", "korrekturbuchung"],
     ["tilbakeføre", "reversere", "reverse", "slett", "delete",
      "åpningsbalanse", "opening balance", "revertir", "reverter",
      "contrepasser", "stornieren"], 5),

    # ── "New hire" compound words (before basic create_employee) ──
    ("create_employee", ["nyansettelse", "ny ansettelse", "nyansatt", "nytilsett",
                         "nueva contratación", "nova contratação", "nouvelle embauche",
                         "neueinstellung"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen"], 10),

    # ── Basic entity creation (lowest priority) ──
    ("create_supplier", ["leverandør", "supplier", "proveedor", "fournisseur",
                         "lieferant", "fornecedor", "leverandøren",
                         "ny leverandør", "registrere leverandør", "opprett leverandør"],
     ["slett", "delete", "leverandørfaktura", "inngående faktura",
      "supplier invoice", "incoming invoice", "inngående",
      "eliminar", "excluir", "supprimer", "löschen", "oppdater", "update", "endre",
      "actualizar", "atualizar", "modifier", "ändern", "aktualisieren",
      "mettre a jour", "mettre à jour"], 4),
    ("create_department", ["opprett avdeling", "opprette avdeling", "ny avdeling",
                           "create department", "new department",
                           "crear departamento", "nuevo departamento",
                           "criar departamento", "novo departamento",
                           "créer département", "nouveau département",
                           "abteilung erstellen", "neue abteilung",
                           "avdeling", "department", "departamento", "département",
                           "abteilung", "avdelinga"],
     ["slett", "delete", "eliminar", "excluir", "supprimer", "löschen",
      "oppdater", "update", "endre", "actualizar", "atualizar", "modifier", "ändern",
      "aktualisieren", "mettre a jour", "mettre à jour",
      "ansatt", "employee", "empleado", "funcionário", "mitarbeiter"], 4),
    ("create_employee", ["ansatt", "employee", "empleado", "employé",
                         "mitarbeiter", "funcionário", "tilsett",
                         "kollega", "medarbeider", "ny kollega", "new colleague",
                         "empregado"],
     ["slett", "delete", "oppdater", "update", "endre", "reise", "travel",
      "prosjekt", "project", "ansettelsesforhold", "arbeidsforhold", "employment",
      "permisjon", "leave", "arbeidstid", "lønn", "salary", "deaktiver", "deactivate",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern", "aktualisieren",
      "mettre a jour", "mettre à jour",
      "salário", "salario", "salaire", "gehalt", "lohn", "payroll", "nómina",
      "start date", "startdato", "tiltredelse", "tiltredelsesdato",
      "stillingsprosent", "stillingstittel", "yrkeskode", "occupation code",
      "startdatum", "eintrittsdatum", "date d'embauche", "fecha de inicio",
      "data de início", "data de inicio"], 3),
    ("create_customer", ["kunde", "customer", "cliente", "client", "klient", "kunden"],
     ["slett", "delete", "oppdater", "update", "endre",
      "faktura", "invoice", "fatura", "factura", "facture", "rechnung",
      "kontakt", "contact", "contato", "contacto",
      "prosjekt", "project", "proyecto", "projeto", "projet", "projekt",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern", "aktualisieren",
      "mettre a jour", "mettre à jour"], 3),
    ("create_product", ["produkt", "product", "producto", "produit", "produto", "vare"],
     ["slett", "delete", "oppdater", "update", "endre", "faktura", "invoice",
      "factura", "facture", "rechnung", "fatura",
      "betaling", "payment", "paiement", "pago", "pagamento", "zahlung",
      "eliminar", "excluir", "supprimer", "löschen",
      "actualizar", "atualizar", "modifier", "ändern", "aktualisieren",
      "mettre a jour", "mettre à jour"], 3),
]

# ── Normalize patterns: strip accents so both patterns and text match uniformly ──
_PATTERNS = [
    (task_type, [_strip_accents(p) for p in positives],
     [_strip_accents(n) for n in negatives], bonus)
    for task_type, positives, negatives, bonus in _PATTERNS
]


def classify_task(prompt: str) -> str | None:
    """Classify a prompt into a known task type.

    Returns task type string if confident, None for fallback to all tools.
    """
    text = _strip_accents(prompt.lower().strip())
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
    "dimension": ["dimensjon", "dimension", "prosjekttype", "project type",
                  "buchhaltungsdimension", "kontodimensjon",
                  "accounting dimension", "custom dimension"],
}

# Normalize CATEGORY_KEYWORDS for accent-insensitive matching
CATEGORY_KEYWORDS = {
    cat: [_strip_accents(kw) for kw in keywords]
    for cat, keywords in CATEGORY_KEYWORDS.items()
}

CATEGORY_TOOLS: dict[str, list[str]] = {
    "employee": ["create_employee", "update_employee", "search_employees",
                 "create_employment", "create_employment_details",
                 "create_standard_time", "create_leave_of_absence"],
    "customer": ["create_customer", "update_customer", "search_customers",
                 "delete_customer"],
    "invoice":  ["create_customer", "create_product", "create_order", "create_invoice",
                 "create_credit_note", "register_payment", "search_invoices", "send_invoice"],
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
    "dimension": ["create_project_category", "search_project_categories",
                  "create_voucher", "get_ledger_accounts"],
}


def detect_categories(prompt: str) -> list[str]:
    """Detect broad entity categories mentioned in a prompt.

    Returns list of matching category names (e.g. ["employee", "customer"]).
    """
    text = _strip_accents(prompt.lower().strip())
    text = re.sub(r'\S+@\S+\.\S+', '', text)  # strip emails
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(category)
                break
    return matched


def select_tools(task_type: str | None, all_tools_dict: dict, has_files: bool = False,
                 prompt: str = "") -> ToolSelection:
    """Select tools for a given task type.

    Args:
        task_type: Classified task type, or None for all tools.
        all_tools_dict: Dict of {tool_name: tool_function}.
        has_files: Whether the request has file attachments.
        prompt: Original prompt text (used for category fallback when task_type is None).

    Returns:
        ToolSelection(tools, classification_level).
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
                    return ToolSelection(selected, "category")
        return ToolSelection(list(all_tools_dict.values()), "fallback")

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
        return ToolSelection(list(all_tools_dict.values()), "fallback")

    return ToolSelection(selected, "exact")
