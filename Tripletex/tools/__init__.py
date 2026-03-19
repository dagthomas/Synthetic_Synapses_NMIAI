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
    all_tools.update(build_contact_tools(client))
    all_tools.update(build_employment_tools(client))
    all_tools.update(build_bank_tools(client))
    all_tools.update(build_supplier_tools(client))
    all_tools.update(build_address_tools(client))
    all_tools.update(build_balance_tools(client))
    all_tools.update(build_common_tools(client))
    if files_dir:
        all_tools.update(build_file_tools(files_dir))
    return list(all_tools.values())
