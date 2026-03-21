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
    classification_level: str  # "exact" | "multi" | "category" | "fallback"
    task_types: list = []
    missing_tools: list = []


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
    "create_invoice":              ["process_invoice"],
    "create_multi_line_invoice":   ["process_invoice"],
    "create_project":              ["create_customer", "create_employee", "create_project",
                                    "get_ledger_postings", "get_ledger_accounts",
                                    "analyze_ledger_changes", "create_activity"],
    "create_travel_expense":       ["process_travel_expense"],
    "create_travel_expense_with_costs": ["process_travel_expense",
                                        "create_employee", "create_travel_expense",
                                        "create_per_diem_compensation", "create_travel_expense_cost",
                                        "create_accommodation_allowance", "create_mileage_allowance"],
    "order_to_invoice_with_payment": ["process_order_to_invoice_with_payment"],
    "invoice_with_payment":        ["search_customers", "search_invoices", "create_customer", "create_product",
                                    "create_order", "create_invoice", "register_payment",
                                    "create_voucher", "get_ledger_accounts"],
    "create_credit_note":          ["process_invoice"],
    "create_employee_with_employment": ["create_employee", "create_employment", "create_employment_details",
                                       "create_standard_time", "create_leave_of_absence",
                                       "extract_file_content"],
    "create_supplier_invoice":     ["process_supplier_invoice", "extract_file_content"],
    "project_invoice":             ["create_customer", "create_employee", "create_project",
                                    "create_product", "create_order", "create_invoice", "send_invoice",
                                    "create_activity", "create_timesheet_entry", "create_employment",
                                    "create_project_participant", "create_hourly_cost_and_rate"],
    "create_project_with_pm":      ["create_customer", "create_employee", "create_project"],
    "project_lifecycle":           ["execute_project_lifecycle"],
    "create_project_with_billing": ["create_project_with_billing"],
    # Tier 3 — complex
    "delete_travel_expense":  ["search_travel_expenses", "delete_travel_expense"],
    "delete_customer":        ["search_customers", "delete_customer"],
    "delete_supplier":        ["search_suppliers", "delete_supplier"],
    "delete_product":         ["search_products", "delete_product"],
    "delete_department":      ["search_departments", "delete_department"],
    "delete_contact":         ["search_contacts", "delete_contact", "update_contact"],
    "delete_employee":        ["search_employees", "update_employee"],
    "correct_ledger_errors":  ["get_ledger_postings", "create_voucher", "get_ledger_accounts",
                               "search_vouchers"],
    "create_ledger_voucher":  ["create_voucher", "get_ledger_accounts", "get_ledger_postings", "search_vouchers",
                               "create_department", "extract_file_content", "process_supplier_invoice"],
    "reverse_voucher":        ["search_vouchers", "reverse_voucher"],
    "reverse_payment":        ["search_customers", "search_invoices", "register_payment"],
    "reminder_fee":           ["search_invoices", "search_customers", "create_voucher", "get_ledger_accounts",
                               "create_product", "create_order", "create_invoice", "send_invoice",
                               "register_payment"],
    "delete_invoice":         ["process_invoice"],
    "create_opening_balance": ["create_opening_balance", "get_ledger_accounts"],
    "create_dimension":       ["create_accounting_dimension", "create_dimension_value",
                               "search_accounting_dimensions", "search_dimension_values",
                               "create_voucher", "get_ledger_accounts"],
    "bank_reconciliation":    ["extract_file_content", "search_bank_accounts", "create_voucher",
                               "get_ledger_accounts",
                               "search_customers", "search_invoices", "register_payment",
                               "search_suppliers", "search_supplier_invoices",
                               "add_supplier_invoice_payment",
                               "create_incoming_invoice",
                               "create_supplier", "create_customer"],
    "process_invoice_file":   ["extract_file_content", "create_customer", "create_product", "create_order",
                               "create_invoice"],
    "year_end":               ["create_voucher", "get_ledger_accounts", "get_ledger_postings",
                               "get_result_before_tax", "create_ledger_account",
                               "search_year_end_annexes", "create_year_end_note"],
    "salary_with_bonus":      ["process_salary"],
    "register_expense_receipt": ["register_expense_receipt", "create_department",
                                 "search_departments", "get_ledger_accounts"],
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
                          "tilbakefør betalingen", "reverser betalingen", "reversere betalingen",
                          "betaling ble returnert", "returnert av banken",
                          "reverse payment", "revert payment", "payment returned", "payment bounced",
                          "returned by the bank", "bounced by the bank",
                          "betaling returnert", "betaling avvist", "betaling retur",
                          "reverter pagamento", "pagamento devolvido", "estornar pagamento",
                          "devolvido pelo banco", "pagamento retornado",
                          "revertir pago", "pago devuelto", "anular pago",
                          "pago rechazado", "pago retornado",
                          "devuelto por el banco", "revierta el pago", "fue devuelto",
                          "contrepasser paiement", "paiement retourné", "annuler paiement",
                          "paiement rejeté", "paiement rejeté par la banque",
                          "retourné par la banque", "rejeté par la banque",
                          "zahlung zurückbuchen", "zahlung stornieren", "rücklastschrift",
                          "zahlung zurückgewiesen", "von der bank zurückgewiesen",
                          "reverta o pagamento", "reverter o pagamento",
                          "devolvido pelo banco"], [], 10),

    # ── Reminder fee / purregebyr (before invoice_with_payment — compound workflow) ──
    ("reminder_fee", ["purregebyr", "forfalt faktura", "forfalte fakturaen", "forfalt fakturaen",
                      "overdue invoice", "reminder fee", "late fee", "late payment fee",
                      "betalingspaminnelse", "inkassogebyr", "inkassovarsel",
                      "purring", "purregebyr pa",
                      "frais de rappel", "frais de retard",
                      "recargo por mora", "cargo por mora",
                      "mahngebühr", "mahnkosten", "zahlungserinnerung",
                      "taxa de lembrete", "multa por atraso"],
     [], 10),

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

    # ── Order to invoice with payment (before invoice_with_payment — "pedido"/"bestilling" overrides) ──
    ("order_to_invoice_with_payment", [
        # Multi-word patterns (highest priority — word_count × bonus)
        "convierte el pedido en factura", "convertir pedido en factura",
        "convierte el pedido", "convertir el pedido",
        "pedido en factura", "pedido a factura",
        "convert order to invoice", "order to invoice",
        "bestilling til faktura", "konverter bestilling til faktura",
        "konverter bestillingen", "gjor om bestilling",
        "gjor om bestillingen til faktura",
        "commande en facture", "convertir commande en facture",
        "convertir la commande", "transformer commande en facture",
        "bestellung in rechnung", "bestellung umwandeln",
        "bestellung zur rechnung",
        "pedido para fatura", "converter pedido em fatura",
        "converter o pedido em fatura",
        # Single-word signals (order-first workflow)
        "pedido", "bestilling",
    ], [], 10),

    # ── Invoice with payment ──
    ("invoice_with_payment", ["faktura betaling", "invoice payment", "registrer betaling",
                              "register payment", "betal faktura",
                              "faktura og betaling", "invoice and payment",
                              "registrere betaling", "registrere innbetaling",
                              "registrer innbetaling", "innbetalingen", "innbetaling",
                              "registrer full betaling", "full betaling", "betaling",
                              "full payment", "register full payment", "payment",
                              "registrar pago", "registra el pago", "registra pago",
                              "pago completo", "factura y pago",
                              "registrar pagamento", "registra o pagamento",
                              "fatura e pagamento",
                              "enregistrer paiement", "enregistrez paiement",
                              "enregistrez le paiement", "paiement intégral",
                              "paiement complet", "facture et paiement",
                              "zahlung registrieren", "rechnung und zahlung",
                              "vollstandige zahlung", "zahlung",
                              "paiement",
                              # Currency / agio keywords
                              "agio", "disagio", "agiogevinst", "agiotap",
                              "valutadifferanse", "valutagevinst", "valutatap",
                              "exchange rate", "exchange gain", "exchange loss",
                              "kurs", "kursdifferanse",
                              "gain de change", "perte de change",
                              "diferencia cambiaria", "diferença cambial",
                              "wechselkursdifferenz", "kursgewinn", "kursverlust"],
     # Negative: order-first signals belong to order_to_invoice_with_payment;
     # reverse payment signals must not match invoice_with_payment
     ["pedido", "bestilling",
      "devuelto", "devuelto por el banco", "revierta", "fue devuelto",
      "devolvido", "devolvido pelo banco", "estornar",
      "retourné", "retourné par la banque", "contrepasser",
      "zurückgewiesen", "rücklastschrift", "zurückbuchen",
      "returnert av banken", "betaling ble returnert",
      "tilbakeføre betaling", "reversere betaling",
      "payment returned", "payment bounced", "reverse payment", "revert payment"], 9),

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
                             "concilia el extracto", "extracto bancario",
                             "conciliar el extracto", "conciliar extracto",
                             "concilia", "estado de cuenta bancario",
                             "reconciliação bancária", "conciliação bancária",
                             "conciliar o extrato", "extrato bancário", "extrato bancario",
                             "rapprochement bancaire", "réconciliation bancaire",
                             "relevé bancaire", "rapprocher le relevé",
                             "bankabstimmung", "kontoavstemming",
                             "kontoauszug", "bankkontoauszug",
                             "bank statement", "reconcile bank",
                             "bankutskrift", "kontoutskrift",
                             # Partial payment in bank reconciliation context
                             "teilzahlung", "delbetaling", "partial payment",
                             "pago parcial", "pagamento parcial", "paiement partiel",
                             # Match payment to invoice keywords
                             "zuordnen", "abgleichen", "abgleich",
                             "eingehende zahlungen", "ausgehende zahlungen"], [], 8),

    # ── Opening balance ──
    ("create_opening_balance", ["åpningsbalanse", "opening balance", "inngående balanse",
                                "balance de apertura", "saldo inicial",
                                "saldo de abertura", "balanço de abertura",
                                "solde d'ouverture", "bilan d'ouverture",
                                "eröffnungsbilanz", "anfangssaldo",
                                "opningsbalanse"], [], 8),

    # ── Year-end ──
    ("year_end", ["årsoppgjør", "årsoppgjer", "year-end", "year end",
                  "årsavslutning", "årsavslutt",
                  "avskrivning", "avskrivningar", "avskrivningar",
                  "avskrivninger", "avskrivingar", "depreciation",
                  "forskuddsbetalt", "forskuddsbetalte", "forskotsbetalt",
                  "forskotsbetaling", "prepaid expenses",
                  "skattekostnad", "tax provision", "tax expense",
                  "akkumulerte avskrivingar", "akkumulerte avskrivninger",
                  "month-end closing", "month-end", "month end",
                  "trial balance", "salary accrual", "accrual reversal",
                  "accrual", "periodeavslutning", "periodelukning",
                  "lønnsperiodisering", "salary provision",
                  "cierre de año", "cierre del ejercicio",
                  "cierre simplificado", "cierre anual", "cierre fiscal",
                  "fin de año", "ejercicio fiscal",
                  "provisión de impuestos", "provision de impuestos",
                  "gastos prepagados", "gastos anticipados",
                  "resultado imponible", "resultado antes de impuestos",
                  "amortización", "depreciación anual",
                  "encerramento do ano", "fechamento do exercício",
                  "clôture annuelle", "clôture de l'exercice",
                  "clôture mensuelle", "clôture du mois",
                  "monatsabschluss", "månedsslutt", "månedsavslutning",
                  "monthly closing", "monthly close", "cierre mensual",
                  "fechamento mensal", "encerramento mensal",
                  "régularisation", "periodisering", "rechnungsabgrenzung",
                  "provision pour salaires", "lønnsavsetning", "gehaltsrückstellung",
                  "jahresabschluss", "jahresende",
                  "abschreibung", "amortissement", "depreciación", "depreciação"],
     ["opprett ansatt", "create employee", "faktura", "invoice", "reiseregning"], 10),

    # ── Salary with bonus ──
    ("salary_with_bonus", ["lønnskjøring", "payroll", "lønnstransaksjon", "salary transaction",
                "lønnsslipp", "salary slip", "lønn", "løn", "salary",
                "grunnløn", "grunnlønn", "bonus", "eingongsbonus",
                "salário", "processar salário", "processamento salarial", "bónus salarial",
                "salario", "procesar salario", "procesar nómina", "nómina",
                "salaire", "traitement salarial", "fiche de paie", "bulletin de paie",
                "gehalt", "gehaltsabrechnung", "lohnabrechnung", "lohn",
                "folha de pagamento", "folha salarial",
                "lønskjøring", "lønstransaksjon"],
     ["opprett ansatt", "create employee", "crear empleado", "criar funcionário",
      "créer employé", "mitarbeiter erstellen",
      # Exclude ledger/voucher terms to prevent misclassification
      "bilag", "voucher", "buchungsbeleg", "buchung", "postering",
      "konto", "abschreibung", "rückstellung", "rechnungsabgrenzung",
      "saldenbilanz", "monatsabschluss", "kontenrahmen",
      "debitere", "kreditere", "hauptbuch", "journal entries",
      "écriture comptable", "asiento contable", "lançamento contábil"], 6),

    # ── Travel expense with costs ──
    ("create_travel_expense_with_costs", ["reiseregning med utlegg", "travel expense with costs",
                                   "kjøregodtgjørelse", "mileage allowance",
                                   "diett", "per diem", "reiseutlegg",
                                   "kostnadslinje", "cost line", "kostnad",
                                   "hotellutgift", "hotellkostnad", "hotellovernatting",
                                   "overnatting", "hotell", "nattillegg",
                                   "accommodation allowance", "overnight allowance",
                                   "overnattingsutgift", "overnattingskostnad",
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
      "proveedor", "fornecedor", "fournisseur", "lieferant",
      "avskriv", "årsoppgj", "depreciation", "year-end", "year end",
      "skattekostnad", "bilag", "voucher", "periodisering"], 9),

    # ── Register expense receipt (kvittering — paid expense, no supplier) ──
    ("register_expense_receipt", [
        # Multi-word patterns (high score due to word count × bonus)
        "registrer utgift fra kvittering", "registrere utgift fra kvittering",
        "bokfør utgift fra kvittering", "bokføre utgift fra kvittering",
        "utgift fra kvittering", "kvittering bokføres",
        "registrer utgift", "registrere utgift", "register expense",
        "bokfør kvittering", "bokføre kvittering", "bokfør utgift", "bokføre utgift",
        "expense receipt", "expense from receipt", "utgiftskvittering",
        "fra kvittering", "kvittering fra", "kvittering på", "kvittering for",
        "registrer kvittering", "registrere kvittering",
        "register receipt", "book receipt", "book expense",
        "enregistrer dépense", "enregistrer la dépense", "reçu de dépense",
        "registrar gasto", "registrar el gasto", "recibo de gasto",
        "registrar despesa", "registar despesa", "recibo de despesa",
        "ausgabe erfassen", "ausgabe buchen", "quittung buchen",
        "beleg erfassen", "beleg buchen",
        "utlegg fra", "utlegg på", "utlegg for",
        "betalt med kort", "betalt med bank", "betalt kontant",
        "paid by card", "paid by bank", "paid in cash",
    ], ["leverandør", "supplier", "proveedor", "fornecedor", "fournisseur", "lieferant",
        "leverandørfaktura", "inngående faktura", "supplier invoice", "incoming invoice",
        "reiseregning", "travel expense", "reise"], 10),

    # ── Supplier invoice / expense receipt ──
    ("create_supplier_invoice", ["leverandørfaktura", "supplier invoice", "inngående faktura",
                                "incoming invoice",
                          "factura de proveedor", "factura del proveedor",
                          "fatura de fornecedor", "fatura do fornecedor",
                          "facture fournisseur", "facture du fournisseur",
                          "lieferantenrechnung", "eingangsrechnung",
                          "bokføre leverandør", "bokfør leverandør",
                          "regning fra leverandør", "mottatt regning",
                          "leverandørfaktura",
                          # English expense/receipt keywords
                          "expense from this receipt", "expense account",
                          "posted to department", "posted to avdeling",
                          "bokføre utgift", "bokfør utgift",
                          "receipt posted", "expense posted",
                          "vat treatment", "mva-behandling",
                          "utgift fra kvittering", "kvittering bokføres",
                          # Generic expense-to-department patterns
                          "expense from", "utgift fra"],
     ["depreciation", "avskrivning", "month-end", "trial balance",
      "accrual reversal", "salary accrual", "accrual", "monthly closing",
      "årsoppgjør", "årsavslutning", "periodisering"], 8),

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

    # ── Project with billing (lifecycle + customer invoice — before project_lifecycle) ──
    ("create_project_with_billing", [
        # Lifecycle keywords (same as project_lifecycle)
        "ciclo de vida", "project lifecycle", "prosjektlivssyklus",
        "prosjektets livsløp", "cycle de vie du projet",
        "ciclo de vida del proyecto", "lebenszyklus des projekts",
        "projektzyklus", "vollständigen projektzyklus",
        "vollständiger projektzyklus", "gesamten projektzyklus",
        "custo de fornecedor", "supplier cost", "leverandørkostnad",
        "coût fournisseur", "coste de proveedor", "lieferantenkosten",
        "registe horas", "registrer timer", "register hours",
        "registrar horas", "enregistrer les heures", "stunden registrieren",
        "registre horas", "registre timer",
        "horas", "timesheet", "timeregistrering", "timeforing",
        "taxa horaria", "hourly rate", "timepris", "taux horaire",
        "stundensatz", "tarifa por hora",
        # Billing-specific keywords (differentiator from plain lifecycle)
        "kundenrechnung", "customer invoice", "kundefaktura",
        "facture client", "factura de cliente", "fatura do cliente",
        "project billing", "prosjektfakturering", "projektabrechnung",
        "erfassen sie stunden", "stunden erfassen",
        "budget",
    ],
     [], 13),  # Higher bonus than project_lifecycle (12) to win on matching prompts

    # ── Project lifecycle (before project_invoice — matches lifecycle + hours + supplier cost) ──
    ("project_lifecycle", ["ciclo de vida", "project lifecycle", "prosjektlivssyklus",
                           "prosjektets livsløp", "cycle de vie du projet",
                           "ciclo de vida del proyecto", "lebenszyklus des projekts",
                           "projektzyklus", "vollständigen projektzyklus",
                           "vollständiger projektzyklus", "gesamten projektzyklus",
                           "custo de fornecedor", "supplier cost", "leverandørkostnad",
                           "coût fournisseur", "coste de proveedor", "lieferantenkosten",
                           "registe horas", "registrer timer", "register hours",
                           "registrar horas", "enregistrer les heures", "stunden registrieren",
                           "registre horas", "registre timer",
                           # Individual keywords for "registe N horas" patterns (number between words)
                           "horas", "timesheet", "timeregistrering", "timeforing",
                           "taxa horaria", "hourly rate", "timepris", "taux horaire",
                           "stundensatz", "tarifa por hora"],
     [], 12),  # No negatives — lifecycle tasks legitimately include invoice creation

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
                         "facturer le projet"],
     # Negative: bank reconciliation context — "teilzahlung" is shared but bank statement keywords override
     ["kontoauszug", "bankabstimmung", "bankavstemming", "kontoavstemming",
      "bank reconciliation", "bank statement", "bankutskrift",
      "extracto bancario", "extrato bancário", "extrato bancario",
      "relevé bancaire", "rapprochement bancaire",
      "reconciliación bancaria", "conciliação bancária",
      "avstemming"], 10),

    # ── Create project with PM ──
    ("create_project_with_pm", ["prosjektleder", "project manager",
                         "jefe de proyecto", "director del proyecto", "director de proyecto",
                         "gerente de projeto", "gerente do projeto",
                         "chef de projet", "directeur du projet", "responsable du projet",
                         "projektleiter", "projektleder"],
     ["faktura", "invoice", "factura", "fatura", "facture", "rechnung",
      "horas", "hours", "timer", "heures", "stunden", "timesheet"], 10),

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
      "nota de credito", "note de credit", "gutschrift",
      "registrer betaling", "registrere betaling", "registrer innbetaling",
      "innbetaling", "full betaling",
      "vedlagt", "attached", "leverandør", "supplier",
      "inngående", "incoming", "proveedor", "fornecedor", "fournisseur", "lieferant",
      "eliminar", "excluir", "supprimer", "löschen",
      "registrar pago", "registra el pago", "registra pago", "pago completo",
      "registrar pagamento", "registra o pagamento",
      "enregistrer paiement",
      "zahlung registrieren", "zahlung", "vollstandige zahlung",
      # Reverse payment signals — prevent invoice creation on reversal prompts
      "devuelto", "devuelto por el banco", "revierta", "revertir pago",
      "devolvido", "devolvido pelo banco", "estornar", "reverter pagamento",
      "retourné", "retourné par la banque", "contrepasser",
      "zurückgewiesen", "rücklastschrift", "zurückbuchen",
      "tilbakeføre betaling", "reversere betaling", "betaling returnert",
      "payment returned", "payment bounced", "reverse payment", "revert payment"], 4),

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
                          "egendefinert dimensjon", "tilpasset dimensjon",
                          # French
                          "dimension comptable", "dimension personnalisée", "dimension personnalisee",
                          "valeur de dimension", "valeurs de dimension",
                          "créez une dimension", "creez une dimension",
                          # Spanish
                          "dimensión contable", "dimension contable",
                          "dimensión personalizada", "dimension personalizada",
                          "valor de dimensión", "valor de dimension",
                          # Portuguese
                          "dimensão contábil", "dimensao contabil",
                          "dimensão personalizada", "dimensao personalizada",
                          "valor de dimensão", "valor de dimensao",
                          # English extras
                          "dimension with values", "kostsenter", "cost center", "cost centre"], [], 12),

    # ── Correct ledger errors (BEFORE generic ledger voucher) ──
    ("correct_ledger_errors", [
        "feil i hovedboken", "feil i hovedbok", "feilene i hovedboken",
        "errors in the general ledger", "errors in the ledger",
        "erreurs dans le grand livre", "erreurs dans le livre",
        "fehler im hauptbuch", "fehler in der buchhaltung",
        "errores en el libro mayor", "errores en el diario",
        "erros no razão", "erros no livro razão",
        "duplikatpostering", "duplikat postering", "dobbeltpostering",
        "feilpostering i hovedbok", "korrigere feil",
        "duplicate posting", "duplicate entry",
        "écriture en double", "pièce en double",
        "doppelte buchung", "doppelbuchung",
        "manglende mva-linje", "manglende mva linje", "missing vat line",
        "ligne de tva manquante", "tva manquante",
        "fehlende umsatzsteuer", "fehlende mwst",
        "feil beløp", "feil konto", "wrong account", "wrong amount",
        "mauvais compte", "montant incorrect",
        "falscher betrag", "falsches konto",
    ], [], 15),

    # ── Ledger voucher ──
    ("create_ledger_voucher", ["bilag", "voucher", "korrigeringsbilag", "korreksjonsbilag",
                               "correction voucher",
                               "korreksjon", "correction", "postering",
                               "debitere", "hovedbok", "hovedboken",
                               "feilpostering", "feilføring",
                               "comprobante", "asiento contable",
                               "comprovante", "lançamento contábil",
                               "pièce comptable", "écriture comptable",
                               "buchungsbeleg", "buchung", "korrekturbuchung",
                               # German accounting terms that indicate journal entries
                               "rückstellung", "abschreibung", "rechnungsabgrenzung",
                               "saldenbilanz", "monatsabschluss", "kontenrahmen",
                               "hauptbuch", "gegenkonto", "sollkonto", "habenkonto",
                               "journal entries", "journal entry",
                               "kontieren", "verbuchen",
                               # Norwegian expense/receipt keywords
                               "kvittering", "utgift", "utgiftskonto",
                               "bokføre utgift", "bokfør utgift",
                               # Portuguese expense keywords
                               "despesa", "conta de despesas", "recibo",
                               "registar despesa",
                               # Spanish expense keywords
                               "gasto", "cuenta de gastos",
                               "registrar gasto", "registrar el gasto",
                               # French expense keywords
                               "dépense", "compte de charges", "reçu",
                               "enregistrer la dépense",
                               # German expense keywords
                               "aufwand", "aufwandskonto", "quittung",
                               "ausgabe", "ausgabenkonto"],
     ["tilbakeføre", "reversere", "reverse", "slett", "delete",
      "åpningsbalanse", "opening balance", "revertir", "reverter",
      "contrepasser", "stornieren",
      # Year-end discriminators — prevent year-end tasks from matching ledger voucher
      "årsoppgjør", "årsavslutt", "year-end", "year end",
      "avskrivning", "depreciation", "depreciación", "amortissement", "abschreibung",
      "cierre simplificado", "cierre anual", "cierre fiscal", "fin de año",
      "cierre de año", "cierre del ejercicio",
      "provisión de impuestos", "tax provision", "skattekostnad",
      "clôture annuelle", "jahresabschluss",
      "gastos prepagados", "forskuddsbetalt", "prepaid expenses"], 7),

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
      "mettre a jour", "mettre à jour",
      # Prevent misclassification of complex multi-step prompts
      "horas", "hours", "timer", "heures", "stunden", "timesheet",
      "orçamento", "budget", "budsjett",
      "ciclo de vida", "lifecycle", "livsløp"], 4),
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
      "ansatt", "employee", "empleado", "funcionário", "mitarbeiter",
      # Prevent expense/receipt prompts from matching create_department
      "expense", "receipt", "utgift", "kvittering", "mva", "vat",
      "posted to", "bokføre", "bokfør", "expense account", "utgiftskonto",
      "regning", "faktura", "invoice"], 4),
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


# ── Multi-intent splitting ──────────────────────────────────────────
# Conservative patterns: only split on strong multi-intent signals.
# We do NOT split on simple "og"/"and" because those often join sub-parts
# of a single intent (e.g., "name and email").

_INTENT_SPLIT_RE = re.compile(
    r'(?:\.\s+(?=[A-ZÆØÅÜÖÄÉÈ]))'   # Period + capital letter (new sentence)
    r'|(?:\n\s*\n)'                    # Double newline (paragraph break)
    r'|(?:\n\s*[-•*]\s+)'             # Bullet points
    r'|(?:\n\s*\d+[\.\)]\s+)',        # Numbered list items
    re.UNICODE
)


def classify_tasks(prompt: str) -> list[str]:
    """Classify a prompt into one or more task types (multi-intent aware).

    Uses sentence splitting to detect prompts with multiple independent tasks.
    Returns list of matched task types. Empty list means fallback to all tools.
    """
    # Primary classification on full prompt (existing behavior)
    primary = classify_task(prompt)

    # Split into segments for multi-intent detection
    segments = _INTENT_SPLIT_RE.split(prompt)
    segments = [s.strip() for s in segments if s and len(s.strip()) >= 15]

    if len(segments) <= 1:
        return [primary] if primary else []

    # Task types whose workflow already includes sub-tasks (suppress false multi-intent)
    _SUBSUMES: dict[str, set[str]] = {
        "create_credit_note": {"create_invoice", "create_customer", "create_product"},
        "delete_invoice": {"create_invoice", "create_customer", "create_product"},
        "invoice_with_payment": {"create_invoice", "create_customer", "create_product"},
        "order_to_invoice_with_payment": {"create_invoice", "create_customer", "create_product"},
        "create_multi_line_invoice": {"create_invoice", "create_customer", "create_product"},
        "project_lifecycle": {"create_project", "create_invoice", "create_customer",
                              "create_supplier", "create_employee"},
        "create_project_with_billing": {"create_project", "create_invoice", "create_customer",
                              "create_supplier", "create_employee", "project_lifecycle"},
        "project_invoice": {"create_project", "create_invoice", "create_customer",
                              "project_lifecycle", "create_project_with_billing"},
        "create_project_with_pm": {"create_project", "create_customer", "create_employee"},
    }

    # Classify each segment independently
    types: list[str] = []
    if primary:
        types.append(primary)

    subsumed = _SUBSUMES.get(primary, set()) if primary else set()
    for seg in segments:
        tt = classify_task(seg)
        if tt and tt not in types and tt not in subsumed:
            types.append(tt)

    if types:
        if len(types) > 1:
            log.info(f"Multi-intent detected: {types}")
        return types

    return [primary] if primary else []


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
    "salary_with_bonus": ["lønn", "løn", "salary", "lønnskjøring", "payroll",
                 "grunnløn", "grunnlønn", "bonus", "eingongsbonus",
                 "salário", "salario", "salaire", "gehalt", "lohn",
                 "nómina", "gehaltsabrechnung", "folha de pagamento",
                 "fiche de paie", "bulletin de paie", "lohnabrechnung",
                 "bónus"],
    "contact":  ["kontaktperson", "contact person", "kontakt",
                 "persona de contacto", "pessoa de contato", "contato",
                 "personne de contact", "ansprechpartner"],
    "dimension": ["dimensjon", "dimension", "prosjekttype", "project type",
                  "buchhaltungsdimension", "kontodimensjon",
                  "accounting dimension", "custom dimension",
                  "dimension comptable", "dimension personnalisée",
                  "dimensión contable", "dimensão contábil",
                  "kostsenter", "cost center", "cost centre"],
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
    "project":  ["create_customer", "create_employee", "create_project",
                 "create_employment", "create_timesheet_entry", "create_activity",
                 "search_projects"],
    "ledger":   ["create_voucher", "get_ledger_accounts", "search_vouchers",
                 "reverse_voucher", "create_opening_balance"],
    "department": ["create_department", "update_department", "search_departments",
                   "delete_department"],
    "salary_with_bonus": ["process_salary"],
    "contact":  ["create_customer", "create_contact", "update_contact",
                 "search_contacts", "delete_contact"],
    "dimension": ["create_accounting_dimension", "create_dimension_value",
                  "search_accounting_dimensions", "search_dimension_values",
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


def select_tools(task_types: list[str] | str | None, all_tools_dict: dict,
                 has_files: bool = False, prompt: str = "") -> ToolSelection:
    """Select tools for given task type(s).

    Args:
        task_types: Classified task type(s) — str, list[str], or None for fallback.
        all_tools_dict: Dict of {tool_name: tool_function}.
        has_files: Whether the request has file attachments.
        prompt: Original prompt text (used for category fallback when no types matched).

    Returns:
        ToolSelection(tools, classification_level, task_types, missing_tools).
    """
    # Normalize to list
    if isinstance(task_types, str):
        task_types = [task_types]
    elif task_types is None:
        task_types = []

    if not task_types:
        # Try category-level fallback before returning all tools
        if prompt:
            categories = detect_categories(prompt)
            if categories:
                required_names = set()
                for cat in categories:
                    required_names.update(CATEGORY_TOOLS.get(cat, []))
                for name in _UNIVERSAL_TOOLS:
                    required_names.add(name)
                if has_files:
                    required_names.add("extract_file_content")
                selected = [all_tools_dict[n] for n in required_names if n in all_tools_dict]
                if selected:
                    log.info(f"Category fallback: {categories} -> {len(selected)} tools")
                    return ToolSelection(selected, "category", [], [])
        return ToolSelection(list(all_tools_dict.values()), "fallback", [], [])

    # Union tools from all task types
    required_names: set[str] = set()
    for tt in task_types:
        required_names.update(TASK_TOOL_MAP.get(tt, []))

    # Add universal tools
    for name in _UNIVERSAL_TOOLS:
        required_names.add(name)

    # Add file tools if attachments present
    if has_files:
        required_names.add("extract_file_content")

    # Resolve names to actual tool functions, tracking missing ones
    selected = []
    missing = []
    for name in sorted(required_names):
        if name in all_tools_dict:
            selected.append(all_tools_dict[name])
        else:
            missing.append(name)
            log.warning(f"MISSING TOOL: '{name}' required by {task_types} but not found in tools dict")

    # Safety: if we resolved very few tools, fall back to all
    if len(selected) < 1:
        log.warning(f"No tools resolved for task_types={task_types}, falling back to all")
        return ToolSelection(list(all_tools_dict.values()), "fallback", task_types, missing)

    level = "exact" if len(task_types) == 1 else "multi"
    return ToolSelection(selected, level, task_types, missing)


# ── Currency / Agio pre-computation ──────────────────────────────

_CURRENCY_CODES = r"(?:EUR|USD|GBP|CHF|SEK|DKK)"

def extract_currency_info(prompt: str) -> dict | None:
    """Parse foreign currency amount and exchange rates from prompt text.

    Returns dict with pre-computed values if currency signals found, else None.
    Keys: currency, amount, old_rate, new_rate, invoice_nok, payment_nok,
          diff, is_agio, agio_account.
    """
    text = _strip_accents(prompt.lower())

    # Must have currency signals
    has_currency = bool(re.search(r'\b(?:eur|usd|gbp|chf|sek|dkk)\b', text))
    has_rate = bool(re.search(r'kurs|exchange.?rate|taux|wechselkurs|cambio|agio|disagio|valutadifferanse', text))
    if not has_currency or not has_rate:
        return None

    # Extract currency code (from original prompt to preserve case)
    m = re.search(r'\b(' + _CURRENCY_CODES + r')\b', prompt)
    if not m:
        return None
    currency = m.group(1)

    # Extract currency amount: "<number> EUR" or "EUR <number>"
    amount = None
    # Pattern: number before currency code
    for pat in [
        r'([\d]+[\d\s]*(?:[.,]\d+)?)\s*' + currency,
        currency + r'\s*([\d]+[\d\s]*(?:[.,]\d+)?)',
    ]:
        m = re.search(pat, prompt)
        if m:
            raw = m.group(1).replace(' ', '').replace(',', '.')
            try:
                amount = float(raw)
                break
            except ValueError:
                continue
    if not amount:
        return None

    # Extract exchange rates — find all decimal numbers near rate keywords
    # Look for patterns like "kursen var 10.83", "kurs 10.83", "10.83 NOK/EUR"
    rates = []
    # Pattern: rate keyword + up to 2 optional words + number (handles "kursen er nå 9.80", "Wechselkurs war 11.20")
    for m in re.finditer(
        r'(?:kurs(?:en)?|rate|taux|wechselkurs|cambio|tipo\s+de\s+cambio)'
        r'\s+(?:\w+\s+){0,2}(\d+[.,]\d+)',
        text
    ):
        try:
            val = float(m.group(1).replace(',', '.'))
            if val not in rates:
                rates.append(val)
        except ValueError:
            pass
    # Pattern: number NOK/CUR
    for m in re.finditer(r'(\d+[.,]\d+)\s*(?:nok[/\\]|kr[/\\])', text):
        try:
            val = float(m.group(1).replace(',', '.'))
            if val not in rates:
                rates.append(val)
        except ValueError:
            pass
    # Pattern: standalone decimals near "var/was/war" and "er/is/ist" (multi-language)
    if len(rates) < 2:
        for m in re.finditer(r'(?:var|was|war|etait|era)\s+(\d+[.,]\d+)', text):
            try:
                r = float(m.group(1).replace(',', '.'))
                if r not in rates:
                    rates.append(r)
            except ValueError:
                pass
        for m in re.finditer(r'(?:\ber\b|\bis\b|\bist\b|\best\b|\bes\b)\s+(?:\w+\s+)?(\d+[.,]\d+)', text):
            try:
                r = float(m.group(1).replace(',', '.'))
                if r not in rates:
                    rates.append(r)
            except ValueError:
                pass

    if len(rates) < 2:
        return None

    old_rate = rates[0]  # First mentioned = invoice-time rate
    new_rate = rates[1]  # Second mentioned = payment-time rate

    invoice_nok = round(amount * old_rate, 2)
    payment_nok = round(amount * new_rate, 2)
    diff = round(payment_nok - invoice_nok, 2)

    return {
        "currency": currency,
        "amount": amount,
        "old_rate": old_rate,
        "new_rate": new_rate,
        "invoice_nok": invoice_nok,
        "payment_nok": payment_nok,
        "diff": diff,
        "is_agio": diff > 0,
        "agio_account": "8060" if diff > 0 else "8160",
    }


# ── LLM-based independent classifier ──────────────────────────────
# Uses Gemini Flash to classify a prompt independently of the keyword
# classifier.  This is used to detect misroutes: if the LLM disagrees
# with the keyword classifier, we likely have a routing bug.

_LLM_TASK_LIST = sorted(TASK_TOOL_MAP.keys())

_LLM_CLASSIFY_PROMPT = """You are a task classifier for a Norwegian accounting system (Tripletex).
Given a user prompt (may be in Norwegian, English, Spanish, Portuguese, French, or German),
classify it into exactly ONE of these task types:

{task_types}

Rules:
- Return ONLY the task type name, nothing else.
- If the prompt clearly asks to DELETE an entity, use the delete_* type.
- If the prompt asks to UPDATE/CHANGE an entity, use the update_* type.
- If the prompt involves creating an invoice AND registering payment, use invoice_with_payment.
- If the prompt involves a credit note, use create_credit_note.
- If the prompt involves multiple products/lines on an invoice, use create_multi_line_invoice.
- If the prompt involves a project with hours/timesheet/supplier cost, use project_lifecycle.
- If the prompt involves a project + invoice, use project_invoice.
- If the prompt involves travel expense with costs/mileage/per diem/hotel, use create_travel_expense_with_costs.
- If the prompt involves salary/payroll/bonus, use salary_with_bonus.
- If the prompt involves year-end/depreciation/tax provision, use year_end.
- If the prompt involves bank reconciliation/bank statement, use bank_reconciliation.
- If the prompt involves ledger correction/voucher/journal entry, use create_ledger_voucher.
- If the prompt involves opening balance, use create_opening_balance.
- If the prompt involves an expense receipt (kvittering), use register_expense_receipt.
- If the prompt involves a supplier invoice, use create_supplier_invoice.
- If the prompt involves accounting dimensions, use create_dimension.
- If none match well, return "unknown".

Prompt: "{prompt}"
"""


def llm_classify_task(prompt: str) -> str | None:
    """Classify a prompt using Gemini Flash, independent of keyword classifier.

    Returns task type string or None on failure.  Non-blocking safe to call
    from a background thread.
    """
    try:
        from google import genai
        from config import GOOGLE_API_KEY

        if not GOOGLE_API_KEY:
            return None

        client = genai.Client(api_key=GOOGLE_API_KEY)

        task_types_str = "\n".join(f"- {t}" for t in _LLM_TASK_LIST)
        full_prompt = _LLM_CLASSIFY_PROMPT.format(
            task_types=task_types_str,
            prompt=prompt[:1000],  # truncate very long prompts
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )

        # Extract text from response (handle thinking model returning None for .text)
        raw_text = response.text
        if not raw_text and response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    raw_text = part.text
                    break
        if not raw_text:
            return None

        result = raw_text.strip().lower().replace(" ", "_")
        # Validate it's a known task type
        if result in TASK_TOOL_MAP or result == "unknown":
            return result
        # Try to find closest match (e.g. "create_employee" vs "create employee")
        for t in _LLM_TASK_LIST:
            if t in result or result in t:
                return t
        return result  # return as-is even if not in list

    except Exception as e:
        log.warning(f"LLM classify failed: {e}")
        return None
