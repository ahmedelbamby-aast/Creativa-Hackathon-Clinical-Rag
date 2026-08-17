# System Quickstart — Local Development

This path runs the Python/Gradio application on the host and PostgreSQL with
pgvector in Docker. It is the fastest setup for editing and debugging on
Windows, while remaining equivalent on macOS and Linux.

## 1. Prerequisites

- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop with Docker Compose v2
- A Gemini API key from Google AI Studio (not an OAuth access token)

Verify the tools:

```powershell
python --version
uv --version
docker version
docker compose version
```

## 2. Configure the environment

Copy the development template:

```powershell
Copy-Item .env.development.example .env.development
```

On macOS/Linux, use `cp .env.development.example .env.development`. Add or
replace the Gemini key without changing the deterministic local defaults:

```dotenv
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_PROVIDER=local
EMBEDDING_DIMENSION=384
EMBEDDING_NAMESPACE=local_384
```

The default local database connection is:

```text
Host: localhost
Port: 5432
Database: creativa_diabetes
Username: creativa
Password: creativa-local
URL: postgresql://creativa:creativa-local@localhost:5432/creativa_diabetes
```

These credentials are intentionally local-demo values. Do not reuse them in a
public or production environment.

## 3. Install CPU-only dependencies

```powershell
uv sync --extra local
```

The lockfile pins the PyTorch CPU index. CUDA, NVIDIA, and Triton runtimes are
not installed or required.

## 4. Start PostgreSQL

Start only the database because Gradio will run on the host:

```powershell
docker compose up -d postgres
docker compose ps postgres
```

Wait for the database status to become `healthy`, then prepare pgvector and the
local embedding namespace:

```powershell
uv run python scripts/bootstrap.py
```

The first bootstrap downloads the multilingual embedding model and can take
several minutes. Later runs use the local cache.

## 5. Ingest a deterministic reference document

```powershell
uv run python scripts/ingest.py --file "data/rew_data/books/An Overview of Diabetes Mellitus in Egypt and the Significance of Integrating Preventive Cardiology in Diabetes Management.pdf" --force
```

Expected result: six pages processed and approximately 70 chunks stored in the
`local_384` namespace.

## 6. Run Gradio

```powershell
uv run python app.py
```

Open <http://localhost:7860>. As a patient, ask:

> What role do preventive cardiologists have in diabetes care?

Confirm that the response is grounded in the overview PDF and includes a page
citation. Open **Developer diagnostics** to review stage timings and run the
retrieval benchmark twice; the page retains benchmark history for comparison.

## 7. Verify the system

Keep Gradio running, open a second terminal in the repository, and run:

```powershell
uv run python -m pytest
uv run python scripts/dry_run.py
uv run python scripts/database_smoke.py
.\scripts\test_system.ps1
```

The last command validates configuration, pgvector data, deterministic
retrieval invariants, Gradio APIs, and one live Gemini request. To avoid that
billable request, use:

```powershell
uv run python scripts/system_consistency.py --gradio-url http://localhost:7860 --live-gradio
```

On macOS/Linux, replace `test_system.ps1` with `./scripts/test_system.sh`; the
other commands are unchanged.

## 8. Stop the system

Stop Gradio with `Ctrl+C`, then stop PostgreSQL while preserving its data:

```powershell
docker compose stop postgres
```

To remove only this project's stopped containers and network while preserving
volumes, run `docker compose down`. Avoid `docker compose down -v` unless you
intend to delete the indexed documents and model-independent database state.

## Troubleshooting

- **Port 5432 is occupied:** set `POSTGRES_PORT` and update `DATABASE_URL` in
  `.env.development` to the same host port before starting PostgreSQL.
- **Port 7860 is occupied:** stop the conflicting process before starting the
  host application; the current local entry point binds to port 7860.
- **Embedding model download stalls:** retry bootstrap; optionally set an
  `HF_TOKEN` in `.env.development` for higher Hugging Face rate limits.
- **Gemini request fails:** confirm the key and
  `GEMINI_MODEL=gemini-3.6-flash`, then restart Gradio.
- **`401 ACCESS_TOKEN_TYPE_UNSUPPORTED`:** replace `GEMINI_API_KEY` with an API
  key created in Google AI Studio; OAuth access tokens are not accepted by this
  client configuration.
- **No citations appear:** rerun the ingestion command with `--force`, then
  inspect `uv run python scripts/ingest.py --stats`.
