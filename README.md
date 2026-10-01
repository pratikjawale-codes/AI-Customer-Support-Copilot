# Virtual Assistant Customer Support

A FastAPI support-agent demo with ticket triage, local policy search, mock
order/payment tools, PII masking, safety checks, human review, feedback, and
request metrics. Integrations and in-memory metrics are illustrative; configure
real services and persistent storage before production use.

## Run locally (PowerShell)

```powershell
uv venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\python.exe -m app.main
```

Open <http://localhost:8000>. Copy `.env.example` to `.env` to configure an
optional OpenAI key; without one, the app uses local fallback behavior.

## Useful endpoints

- `/api/health` — service and provider mode
- `/api/knowledge-base` — indexed local documents
- `/api/observability/stats` — process-local request metrics
- `/metrics` — Prometheus scrape endpoint
- `/docs` — interactive API documentation

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```
