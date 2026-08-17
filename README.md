# Diabetes RAG Assistant

A bilingual English/Arabic RAG application that answers diabetes questions from the medical documents in this repository. It supports local sentence-transformer embeddings for development and Gemini API embeddings for serverless deployment. PostgreSQL with pgvector stores and searches all embeddings.

> This is a research and educational tool, not a substitute for professional medical advice.

## Architecture

```text
Documents
  -> structure-aware parsing
  -> language and category classification
  -> chunking and quality filtering
  -> local or Gemini embeddings
  -> PostgreSQL + pgvector

Question
  -> medical safety check
  -> follow-up query rewriting
  -> treatment/prevention/nutrition routing
  -> pgvector cosine search
  -> grounded Gemini prompt
  -> answer, disclaimer, and citations
```

Embedding spaces are isolated with PostgreSQL partitions. The defaults create `local_384` and `gemini_384` namespaces, each with its own cosine HNSW index. This prevents vectors from different models from being compared accidentally.

## Main modules

```text
app.py                      Gradio chat application
compose.yaml                Local PostgreSQL 16 + pgvector service
database/schema.sql         Partitioned vector schema
pyproject.toml              UV dependency definitions
uv.lock                     Reproducible dependency lock

scripts/
  bootstrap.py              Prepare the model and pgvector namespace
  dry_run.py                Validate modules, embeddings, database, and UI
  ingest.py                 Parse and ingest source documents
  evaluate.py               Run retrieval and response evaluations

src/
  config.py                 Environment-based configuration
  embeddings.py             Local/Gemini embedding providers
  scoring.py                Cosine-distance conversion
  vector_store.py           PostgreSQL/pgvector storage and search
  retriever.py              Similarity filtering and result mapping
  router.py                 Query category routing
  rewriter.py               Follow-up query contextualization
  safety.py                 Medical-risk classification and disclaimers
  prompts.py                Grounded Gemini prompts
  generator.py              Gemini generation client and retries
  citations.py              Page- and section-aware citations
  memory.py                 Bounded per-session conversation memory
  ingestion/                Parsing, classification, chunking, and filtering

tests/                      Focused pytest tests
data/rew_data/books/        Default source-document directory
```

## Requirements

- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/)
- PostgreSQL with the pgvector extension
- A Gemini API key for generation and online embeddings

## Local setup

Install the locked dependencies including the local embedding model runtime:

```bash
uv sync --extra local
```

Create local configuration:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Never commit `.env`.

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
```

Download the local embedding model and create its database partition/index:

```bash
uv run python scripts/bootstrap.py
```

## Online embeddings

For API-based embeddings, set:

```dotenv
EMBEDDING_PROVIDER=gemini
ONLINE_EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=384
```

The serverless environment does not need the local model extra:

```bash
uv sync
uv run python scripts/bootstrap.py --verify-online
```

The online provider adds retrieval instructions to document and query inputs, requests 384-dimensional vectors, and normalizes the returned vectors before pgvector storage.

Changing providers selects a separate database namespace. Run ingestion once for each provider you intend to use.

## Tests and consistency checks

Run pytest:

```bash
uv run python -m pytest
```

The default test command runs in parallel, randomizes test order, reports
failures immediately, applies per-test timeouts, and enforces 100% statement
and branch coverage for the measured chunking and security modules. The suite
uses `pytest-instafail`; `pytest-installfail` is not a published package.

Run the integration dry run:

```bash
uv run python scripts/dry_run.py
```

Exercise real pgvector writes, nearest-neighbor search, and cleanup:

```bash
uv run python scripts/database_smoke.py
```

The dry run validates:

- Required files and environment keys
- Runtime library imports
- Routing, rewriting, safety, classification, and chunking
- Active local or Gemini embeddings
- PostgreSQL connectivity and pgvector availability
- The active namespace and cosine query path
- Gradio UI construction

It does not ingest documents or call Gemini generation. When `EMBEDDING_PROVIDER=gemini`, it does make embedding API requests.

## Ingest documents

Place PDF, DOCX, or TXT files in `data/rew_data/books/`, then run:

```bash
uv run python scripts/ingest.py
```

Options:

```text
--data-dir PATH    Use another document directory
--file PATH        Ingest one document
--force            Replace an already-ingested document
--reset            Delete chunks in the active embedding namespace
--stats            Print category counts without ingesting
--verbose, -v      Enable debug logging
```

Without `--force`, an existing document is skipped. With `--force`, old chunks are removed only after parsing, chunking, and embedding succeed.

## Run the application

```bash
uv run python app.py
```

Open [http://localhost:7860](http://localhost:7860).

## Evaluation

```bash
# Full retrieval and generation evaluation
uv run python scripts/evaluate.py

# Retrieval only; no Gemini generation
uv run python scripts/evaluate.py --no-generate

# One category
uv run python scripts/evaluate.py --category nutrition
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | local Compose URL | PostgreSQL connection URL |
| `EMBEDDING_PROVIDER` | `local` | `local` or `gemini` |
| `EMBEDDING_MODEL` | multilingual MiniLM | Local sentence-transformer |
| `ONLINE_EMBEDDING_MODEL` | `gemini-embedding-2` | Hosted embedding model |
| `EMBEDDING_DIMENSION` | `384` | Shared pgvector dimension |
| `EMBEDDING_NAMESPACE` | provider-derived | Optional database partition namespace |
| `GEMINI_API_KEY` | empty | Generation and hosted embeddings |
| `GEMINI_MODEL` | `gemini-2.5-flash` in `.env.example` | Generation model |
| `TOP_K` | `5` | Final retrieved chunks |
| `SIMILARITY_THRESHOLD` | `0.30` | Minimum cosine similarity |
| `CHUNK_SIZE` | `2000` | Maximum chunk characters |
| `CHUNK_OVERLAP` | `200` | Adjacent chunk overlap |
| `DATA_DIR` | `data/rew_data/books` | Source documents |
| `DEBUG` | `false` | Show retrieval diagnostics |
| `MAX_MEMORY_TURNS` | `6` | Conversation turns retained |

## Future Vercel deployment

Use `EMBEDDING_PROVIDER=gemini` on Vercel so the deployment does not need PyTorch or local model files. Set `DATABASE_URL` to a managed PostgreSQL service with pgvector enabled, such as a compatible Neon or Supabase database.

Run `scripts/bootstrap.py --verify-online` against the production database during provisioning, not on every request. The storage and embedding layers are serverless-compatible; packaging the current Gradio UI as a Vercel entry point remains a separate deployment step.

## Safety

The pre-generation safety layer classifies queries as emergency, high-risk, diagnosis, or informational. Emergency questions bypass retrieval and generation. High-risk and diagnosis questions receive mandatory disclaimers.

This keyword-based guardrail is not a clinical diagnostic system.
