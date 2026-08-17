# Docker Quickstart — End to End

This path runs PostgreSQL 16, pgvector, the local multilingual embedding model,
Gemini generation, and the Gradio application in Docker.

## 1. Prerequisites

- Docker Desktop with Docker Compose v2
- A Gemini API key
- At least 6 GB of free disk space for images and the cached embedding model

The application image uses the PyTorch CPU wheel only. CUDA, NVIDIA, and
Triton runtimes are not installed or required.

Verify Docker:

```bash
docker version
docker compose version
```

## 2. Configure the demo

The repository contains a demo `.env`. Set or replace this line:

```dotenv
GEMINI_API_KEY=your-key-here
```

Keep these deterministic defaults:

```dotenv
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_PROVIDER=local
EMBEDDING_DIMENSION=384
EMBEDDING_NAMESPACE=local_384
```

If ports `5432` or `7860` are already occupied, set alternatives before
starting. PowerShell:

```powershell
$env:POSTGRES_PORT = "5433"
$env:GRADIO_PORT = "7861"
```

macOS/Linux:

```bash
export POSTGRES_PORT=5433
export GRADIO_PORT=7861
```

## 3. Build and start

```bash
docker compose up --build -d
docker compose ps
```

The first image build and local-model download can take 5–20 minutes on a
constrained connection; later starts reuse both caches. The app health check
allows a 10-minute bootstrap grace period. Watch startup until both services
are healthy:

```bash
docker compose logs -f app
```

Press `Ctrl+C` after Gradio reports its local URL. Open:

- Default: <http://localhost:7860>
- With `GRADIO_PORT=7861`: <http://localhost:7861>

## 4. Ingest one reference PDF

Use the small text-based publication for a fast deterministic demo:

```bash
docker compose exec app uv run python scripts/ingest.py \
  --file "data/rew_data/books/An Overview of Diabetes Mellitus in Egypt and the Significance of Integrating Preventive Cardiology in Diabetes Management.pdf" \
  --force
```

Expected result: six pages processed and approximately 70 chunks in the
`local_384` namespace.

## 5. Verify the full stack

```bash
docker compose exec app uv run python scripts/dry_run.py
docker compose exec app uv run python scripts/database_smoke.py
docker compose exec app uv run python scripts/system_consistency.py \
  --gradio-url http://localhost:7860 --live-gemini --live-gradio
```

Then ask this question in Gradio:

> What role do preventive cardiologists have in diabetes care?

Confirm that the answer cites the overview PDF and page 4. Expand **Developer
diagnostics** to inspect stage timings and run the retrieval benchmark twice.

## 6. Operations

```bash
# Service status
docker compose ps

# Recent logs
docker compose logs --tail=100 app postgres

# Restart only Gradio
docker compose restart app

# Stop while preserving data/model caches
docker compose down

# Remove containers, indexed data, and model cache
docker compose down -v
```

Do not use `down -v` unless you intend to delete the local pgvector index and
downloaded model cache.

## Troubleshooting

- **Port already allocated:** set `POSTGRES_PORT` or `GRADIO_PORT` as shown above.
- **App remains unhealthy:** run `docker compose logs app`; first model download can take several minutes.
- **Hugging Face rate limits:** optionally add `HF_TOKEN` to `.env`; the default configuration disables Xet and uses standard HTTP downloads.
- **No citations:** ingest the PDF in step 4 and verify `docker compose exec app uv run python scripts/ingest.py --stats`.
- **Gemini error:** confirm the key and `GEMINI_MODEL=gemini-3.6-flash`, then run `docker compose restart app`.
