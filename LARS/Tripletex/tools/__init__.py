from tripletex_client import TripletexClient
from tools.employees import build_employee_tools
from tools.customers import build_customer_tools
from tools.products import build_product_tools
from tools.invoicing import build_invoicing_tools
from tools.travel import build_travel_tools
from tools.projects import build_project_tools
from tools.departments import build_department_tools
from tools.ledger import build_ledger_tools
from tools.contacts import build_contact_tools
from tools.employment import build_employment_tools
from tools.bank import build_bank_tools
from tools.supplier import build_supplier_tools
from tools.address import build_address_tools
from tools.balance import build_balance_tools
from tools.common import build_common_tools
from tools.files import build_file_tools
from tools.activity import build_activity_tools
from tools.company import build_company_tools
from tools.division import build_division_tools
from tools.order import build_order_tools
from tools.timesheet import build_timesheet_tools
from tools.salary import build_salary_tools
from tools.supplier_invoice import build_supplier_invoice_tools
from tools.year_end import build_year_end_tools
from tools.employee_extras import build_employee_extras_tools
from tools.travel_extras import build_travel_extras_tools
from tools.incoming_invoice import build_incoming_invoice_tools


def build_tools_dict(client: TripletexClient, files_dir: str = "") -> dict:
    """Build all tools, return as {name: function} dict."""
    all_tools = {}
    all_tools.update(build_employee_tools(client))
    all_tools.update(build_customer_tools(client))
    all_tools.update(build_product_tools(client))
    all_tools.update(build_invoicing_tools(client))
    all_tools.update(build_travel_tools(client))
    all_tools.update(build_project_tools(client))
    all_tools.update(build_department_tools(client))
    all_tools.update(build_ledger_tools(client))
    all_tools.update(build_contact_tools(client))
    all_tools.update(build_employment_tools(client))
    all_tools.update(build_bank_tools(client))
    all_tools.update(build_supplier_tools(client))
    all_tools.update(build_address_tools(client))
    all_tools.update(build_balance_tools(client))
    all_tools.update(build_common_tools(client))
    all_tools.update(build_activity_tools(client))
    all_tools.update(build_company_tools(client))
    all_tools.update(build_division_tools(client))
    all_tools.update(build_order_tools(client))
    all_tools.update(build_timesheet_tools(client))
    all_tools.update(build_salary_tools(client))
    all_tools.update(build_supplier_invoice_tools(client))
    all_tools.update(build_year_end_tools(client))
    all_tools.update(build_employee_extras_tools(client))
    all_tools.update(build_travel_extras_tools(client))
    all_tools.update(build_incoming_invoice_tools(client))
    if files_dir:
        all_tools.update(build_file_tools(files_dir))
    return all_tools


def build_all_tools(client: TripletexClient, files_dir: str = "") -> list:
    """Build all tool functions as closures over the given client."""
    return list(build_tools_dict(client, files_dir).values())
