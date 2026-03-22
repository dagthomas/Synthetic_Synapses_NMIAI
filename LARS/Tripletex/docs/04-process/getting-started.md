# Getting Started

Setup and operation guide for the Tripletex AI accounting agent.

---

## Prerequisites

- Python 3.10+
- Google API key (for Gemini 2.5 Pro)
- Tripletex sandbox credentials (base_url + session_token)

---

## Setup

```bash
cd LARS/Tripletex
pip install -r requirements.txt

# Create .env from template
cp .env.example .env
# Fill in: GOOGLE_API_KEY, TRIPLETEX_BASE_URL, TRIPLETEX_SESSION_TOKEN
```

---

## Running

### Agent Server (Competition Mode)

```bash
# Start on port 8000
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

The `/solve` endpoint is now ready to receive competition requests.

### Dashboard

```bash
# Start dashboard on port 8001
uvicorn dashboard.app:app --port 8001

# Or build the React frontend first
cd dashboard/frontend && npm install && npm run build && cd ../..
uvicorn dashboard.app:app --port 8001
```

### Simulator

```bash
# Run all task types
python simulator.py --batch 30

# Specific task in Norwegian
python simulator.py --task create_employee --lang no

# List available tasks
python simulator.py --list
```

### Auto-Fixer

```bash
# Analyze random competition failure
python auto_fixer.py

# Fix specific tasks and auto-apply
python auto_fixer.py --tasks create_invoice update_employee --apply --retries 2
```

---

## Testing

```bash
# Unit tests
pytest tests/

# E2E tests (requires agent running on port 8003)
python test_e2e.py

# Tool API tests (direct, no agent)
python test_all_tools.py
```

---

## Key Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `TRIPLETEX_BASE_URL` | For testing | Sandbox API URL |
| `TRIPLETEX_SESSION_TOKEN` | For testing | Sandbox session token |
| `AGENT_API_KEY` | No | Bearer token to protect /solve |
| `GEMINI_MODEL` | No | Default: gemini-2.5-pro |

---

## Common Operations

| Goal | Command |
|------|---------|
| Run competition server | `python main.py` |
| Run dashboard | `uvicorn dashboard.app:app --port 8001` |
| Test a task type | `python simulator.py --task create_employee` |
| Fix failing tasks | `python auto_fixer.py --tasks update_customer --apply` |
| Debug a submission | `curl -X POST http://localhost:8000/solve-debug -d '...'` |
