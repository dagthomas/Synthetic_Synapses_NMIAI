# E2E Test Report — 2026-03-20

**Pass rate: 6/6 (100%)**
**Total: 73s | API calls: 13 | API errors: 1**

| # | Test | Status | Time | Tools | API Calls | API Errors |
|---|------|--------|------|-------|-----------|------------|
| 1 | Create employee | PASS | 16s | 1 | 1 | 0 |
| 2 | Create customer | PASS | 9s | 1 | 1 | 0 |
| 3 | Create invoice | PASS | 18s | 4 | 4 | 0 |
| 4 | Create travel expense | PASS | 13s | 2 | 3 | 1 |
| 5 | Create voucher | PASS | 10s | 1 | 3 | 0 |
| 6 | Create supplier | PASS | 8s | 1 | 1 | 0 |

## 1. Create employee — PASS

**Prompt:** Opprett en ansatt med navn Kari Nordmann og epostadresse kari.e2e.1774006262@example.org. Hun skal være kontoadministrator.

**Tool calls:**
1. `create_employee` [PASS]
   - Args: `{"lastName": "Nordmann", "userType": "EXTENDED", "email": "kari.e2e.1774006262@example.org", "firstName": "Kari"}`
   - Response: `{'value': {'id': 18563152, 'version': 1, 'url': 'kkpqfuj-amager.tripletex.dev/v2/employee/18563152', 'firstName': 'Kari', 'lastName': 'Nordmann', 'displayName': 'Kari Nordmann', 'employeeNumber': '', 'dateOfBirth': None, 'email': 'kari.e2e.1774006262@example.org', 'phoneNumberMobileCountry': {'id': `

**Agent response:** Done.

## 2. Create customer — PASS

**Prompt:** Opprett kunden Fjordkraft E2E 1774006262 med epost fjordkraft@example.org og organisasjonsnummer 998877665.

**Tool calls:**
1. `create_customer` [PASS]
   - Args: `{"organizationNumber": "998877665", "isCustomer": true, "name": "Fjordkraft E2E 1774006262", "email": "fjordkraft@example.org"}`
   - Response: `{'value': {'id': 108241934, 'version': 1, 'url': 'kkpqfuj-amager.tripletex.dev/v2/customer/108241934', 'name': 'Fjordkraft E2E 1774006262', 'organizationNumber': '998877665', 'globalLocationNumber': 0, 'supplierNumber': 0, 'customerNumber': 10240, 'isSupplier': False, 'isCustomer': True, 'isInactive`

**Agent response:** Done.

## 3. Create invoice — PASS

**Prompt:** Opprett en faktura for kunden Fjordkraft E2E 1774006262. Produktet heter 'Konsulenttime E2E 1774006262' og koster 1500 kr eks. mva. Kunden bestiller 10 timer. Fakturadato er 2026-03-20, forfallsdato 2026-04-20.

**Tool calls:**
1. `create_customer` [PASS]
   - Args: `{"name": "Fjordkraft E2E 1774006262"}`
   - Response: `{'value': {'id': 108241936, 'version': 1, 'url': 'kkpqfuj-amager.tripletex.dev/v2/customer/108241936', 'name': 'Fjordkraft E2E 1774006262', 'organizationNumber': '', 'globalLocationNumber': 0, 'supplierNumber': 0, 'customerNumber': 10241, 'isSupplier': False, 'isCustomer': True, 'isInactive': False,`
2. `create_product` [PASS]
   - Args: `{"name": "Konsulenttime E2E 1774006262", "priceExcludingVatCurrency": 1500}`
   - Response: `{'value': {'id': 84384617, 'version': 0, 'url': 'kkpqfuj-amager.tripletex.dev/v2/product/84384617', 'name': 'Konsulenttime E2E 1774006262', 'number': '', 'displayNumber': '', 'description': '', 'orderLineDescription': '', 'costExcludingVatCurrency': 0, 'costPrice': 0, 'priceExcludingVatCurrency': 15`
3. `create_order` [PASS]
   - Args: `{"deliveryDate": "2026-03-20", "orderLines": "[{\"product_id\": 84384617, \"count\": 10}]", "customer_id": 108241936}`
   - Response: `{'value': {'id': 401955186, 'version': 0, 'url': 'kkpqfuj-amager.tripletex.dev/v2/order/401955186', 'customer': {'id': 108241936, 'url': 'kkpqfuj-amager.tripletex.dev/v2/customer/108241936'}, 'contact': None, 'attn': None, 'displayName': '183 Ordre', 'receiverEmail': '', 'overdueNoticeEmail': '', 'n`
4. `create_invoice` [PASS]
   - Args: `{"order_id": 401955186, "invoiceDate": "2026-03-20", "invoiceDueDate": "2026-04-20"}`
   - Response: `{'value': {'id': 2147523768, 'version': 3, 'url': 'kkpqfuj-amager.tripletex.dev/v2/invoice/2147523768', 'invoiceNumber': 180, 'invoiceDate': '2026-03-20', 'customer': {'id': 108241936, 'url': 'kkpqfuj-amager.tripletex.dev/v2/customer/108241936'}, 'creditedInvoice': 0, 'isCredited': False, 'invoiceDu`

**Agent response:** Done.

## 4. Create travel expense — PASS

**Prompt:** Opprett en reiseregning for Kari Nordmann (kari.e2e.1774006262@example.org). Turen heter 'Kundebesøk Bergen', avreise 2026-04-01, retur 2026-04-03.

**Tool calls:**
1. `create_employee` [PASS]
   - Args: `{"email": "kari.e2e.1774006262@example.org", "firstName": "Kari", "lastName": "Nordmann"}`
   - Response: `{'value': {'id': 18563152, 'firstName': 'Kari', 'lastName': 'Nordmann', 'email': 'kari.e2e.1774006262@example.org'}, '_note': 'Employee already existed, returning existing.'}`
2. `create_travel_expense` [PASS]
   - Args: `{"returnDate": "2026-04-03", "employee_id": 18563152, "title": "Kundebesøk Bergen", "departureDate": "2026-04-01"}`
   - Response: `{'value': {'id': 11142986, 'version': 2, 'url': 'kkpqfuj-amager.tripletex.dev/v2/travelExpense/11142986', 'attestationSteps': [], 'attestation': None, 'project': None, 'employee': {'id': 18563152, 'url': 'kkpqfuj-amager.tripletex.dev/v2/employee/18563152'}, 'approvedBy': None, 'completedBy': None, '`

**Failed API calls:**
- `POST https://kkpqfuj-amager.tripletex.dev/v2/employee` -> **422**: Validering feilet. [email: Det finnes allerede en bruker med denne e-postadressen.]

**Agent response:** Done.

## 5. Create voucher — PASS

**Prompt:** Opprett et korrigeringsbilag datert 2026-03-20 med beskrivelse 'Korreksjon bankinnskudd'. Debet konto 1920 (Bankinnskudd) 5000 kr, kredit konto 7700 5000 kr.

**Tool calls:**
1. `create_voucher` [PASS]
   - Args: `{"postings": "[{\"accountNumber\": \"1920\", \"amount\": 5000}, {\"accountNumber\": \"7700\", \"amount\": -5000}]", "description": "Korreksjon bankinnskudd", "date": "2026-03-20"}`
   - Response: `{'value': {'id': 608822191, 'version': 5, 'url': 'kkpqfuj-amager.tripletex.dev/v2/ledger/voucher/608822191', 'date': '2026-03-20', 'number': 117, 'tempNumber': 0, 'year': 2026, 'description': 'Korreksjon bankinnskudd', 'voucherType': None, 'reverseVoucher': None, 'postings': [{'id': 3844909097, 'ver`

**Agent response:** Done.

## 6. Create supplier — PASS

**Prompt:** Opprett leverandøren 'Leverandør E2E 1774006262' med epost lev@example.org og telefon 99887766.

**Tool calls:**
1. `create_supplier` [PASS]
   - Args: `{"phoneNumber": "99887766", "email": "lev@example.org", "name": "Leverandør E2E 1774006262"}`
   - Response: `{'value': {'id': 108241952, 'version': 1, 'url': 'kkpqfuj-amager.tripletex.dev/v2/supplier/108241952', 'name': 'Leverandør E2E 1774006262', 'organizationNumber': '', 'supplierNumber': 20064, 'customerNumber': 0, 'isSupplier': True, 'isCustomer': False, 'isInactive': False, 'email': 'lev@example.org'`

**Agent response:** Done.
