# SugerFree — Intelligent Semantic Retrieval & Grounded AI

SugerFree is a bilingual (English/Arabic) RAG (Retrieval-Augmented Generation) application designed to answer diabetes-related questions strictly using certified medical documents in this repository. It features local sentence-transformer embeddings for offline development and Gemini API embeddings for serverless deployment. PostgreSQL with the `pgvector` extension stores and searches all vectors using list partitioning and HNSW graph indexing.

> ⚠️ **Disclaimer:** This is a research and educational engineering tool, not a substitute for professional clinical medical advice or direct diagnosis.

---

## 📖 System Architecture Overview

For a detailed technical deep-dive into internal modules, data flows, database schemas, and ER diagrams, see:
👉 **[docs/system-architecture.md](docs/system-architecture.md)**

Below are high-level overviews of the main system components and logic flows.

### Running System and External Boundaries

```mermaid
flowchart LR
    user["User<br/>English or Arabic"]
    ui["Bilingual chat UI<br/>Gradio local · HTML/FastAPI on Vercel"]
    pipeline["Request orchestration<br/>rag_pipeline()"]
    controls["Safety · rewrite · route<br/>src/safety.py · rewriter.py · router.py"]
    retrieval["Semantic retrieval<br/>src/retriever.py"]
    provider{"EMBEDDING_PROVIDER"}
    local["Local sentence-transformer<br/>multilingual MiniLM"]
    online["Gemini embeddings API"]
    vectors[("PostgreSQL 16 + pgvector<br/>partitioned vector store")]
    prompt["Grounded prompt<br/>src/prompts.py"]
    generation["Extractive evidence by default<br/>optional Gemini or AI Gateway"]
    output["Answer + disclaimer<br/>metadata-built source list"]
    telemetry[("Local JSONL diagnostics<br/>.runtime/")]

    user --> ui --> pipeline --> controls --> retrieval
    retrieval --> provider
    provider -->|local| local --> vectors
    provider -->|gemini| online --> vectors
    vectors --> retrieval --> prompt --> generation --> output --> ui --> user
    pipeline --> telemetry
```

The embedding provider is resolved from environment configurations. Document and query vectors use compatible vector spaces and dimensions. Consecutive PDF blocks are aggregated into page-scoped semantic units so citations remain exact while tiny layout fragments do not become weak standalone embeddings. Gemini is used for primary answer generation, with automatic fallback configurations.

### Document Ingestion and Index Construction

```mermaid
flowchart TD
    command["scripts/ingest.py"]
    files{"Input extension"}
    pdf["PDF"]
    docx["DOCX"]
    txt["TXT"]
    fitz["PyMuPDF<br/>blocks, headings, tables, pages"]
    fallback["pypdf fallback<br/>only when PyMuPDF yields no elements"]
    docxParser["python-docx paragraphs<br/>heading styles"]
    textParser["UTF-8 / Latin-1 paragraphs"]
    elements["DocumentElement records<br/>document · page · section · type · content"]
    sections["Forward-fill section and subsection context"]
    docLanguage["Detect document language"]
    classify["Per-element category and language classification"]
    tableDecision{"Table element?"}
    tableChunk["Keep the Markdown table as one chunk"]
    chunker["SmartChunker<br/>character limit + overlap + semantic block protection"]
    quality["Quality scoring and minimum-score filter"]
    records["Chunk records<br/>ID · provenance · counts · quality"]
    embed["Batch document embeddings<br/>local MiniLM or Gemini"]
    store["Upsert chunks and vectors<br/>src/vector_store.py"]
    partition[("rag_chunks_namespace<br/>cosine HNSW index")]

    command --> files
    files -->|.pdf| pdf --> fitz
    fitz -->|no elements| fallback
    fitz -->|elements found| elements
    fallback --> elements
    files -->|.docx| docx --> docxParser --> elements
    files -->|.txt| txt --> textParser --> elements
    elements --> sections --> docLanguage --> classify --> tableDecision
    tableDecision -->|yes| tableChunk --> quality
    tableDecision -->|no| chunker --> quality
    quality --> records --> embed --> store --> partition
```

Existing document chunks are not deleted until parsing, chunking, and embedding of their replacement succeed. Medical tables bypass normal chunk splitting and are kept monolithic to preserve rows and layout.

### Question Processing and Decision Branches

```mermaid
flowchart TD
    question["Question + selected category + session memory"]
    empty{"Question empty?"}
    safety["Classify medical safety level"]
    emergency{"Emergency?"}
    emergencyReply["Return emergency guidance<br/>No retrieval or generation"]
    rewrite["Rewrite vague/follow-up query using bounded history"]
    route["Use selected category or keyword route<br/>treatment · prevention · nutrition · all"]
    queryEmbed["Embed rewritten query"]
    categoryFilter["Database category filter<br/>includes general chunks"]
    search["pgvector cosine search<br/>fetch 2 × TOP_K"]
    threshold["Drop results below SIMILARITY_THRESHOLD<br/>then keep TOP_K"]
    enough{"At least one sufficient chunk?"}
    refusal["Return bilingual no-evidence response<br/>No Gemini call"]
    citations["Build source list from retrieved metadata"]
    groundedPrompt["Label every chunk as SOURCE n<br/>add limited conversation context"]
    gemini["Generate with Gemini<br/>temperature 0.1 · max 2048 tokens · up to 3 rate-limit retries"]
    disclaimer["Append disclaimer for<br/>high-risk or diagnosis queries"]
    memory["Store user and assistant turns<br/>record timings and status"]
    response["Render answer, sources, and optional DEBUG retrieval preview"]

    question --> empty
    empty -->|yes| response
    empty -->|no| safety --> emergency
    emergency -->|yes| emergencyReply --> memory --> response
    emergency -->|no| rewrite --> route --> queryEmbed --> categoryFilter --> search --> threshold --> enough
    enough -->|no| refusal --> memory --> response
    enough -->|yes| citations --> groundedPrompt --> gemini --> disclaimer --> memory --> response
```

Citations are built dynamically from retrieved database metadata, not parsed from LLM text. When `DEBUG=true`, the interface displays detailed similarity scores and raw retrieval text.

---

## 🛠️ Codebase Structure

| Path | Actual Role |
| :--- | :--- |
| `app.py` | Local Gradio UI and shared synchronous RAG request orchestration. |
| `backend/server.py` | Stateless Vercel API entrypoint. |
| `backend/static/index.html` | Dependency-free, bilingual HTML/CSS/JS web client. |
| `src/ingestion/` | Active parsers, section propagation, classification, chunking adapters, and filters used by `scripts/ingest.py`. |
| `src/embeddings.py` | Local and Gemini embedding providers. |
| `src/vector_store.py` | PostgreSQL partitions and nearest-neighbor cosine search. |
| `src/retriever.py` | Thresholded, category-aware results retrieval. |
| `src/prompts.py` | Grounded system instructions and user prompt templates. |
| `src/generator.py` | Primary (Gemini) and fallback (Groq) LLM integration client. |
| `src/extractive.py` | Tertiary deterministic extractive synthesis fallback. |
| `src/citations.py` | Metadata-derived inline citations and debug representations. |
| `src/observability.py` | Local trace recorders and telemetry metric events logic. |
| `chunking/` | Outdated earlier JSON chunking scripts (retained for backward compatibility). |
| `example/services/` | Prototype components and security tests. |
| `src/context_builder.py` | Secondary helper (context is assembled in `src/prompts.py`). |

---

## ⚙️ Configuration Variables

The application is configured using environment variables loaded via `dotenv`. The table below lists the active settings:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://creativa:creativa-local@localhost:5433/creativa_diabetes` | Connection URL for PostgreSQL. |
| `DATABASE_URL_UNPOOLED` | `""` | Direct database connection URL (used for schema creation). |
| `APP_ENV` | `development` | Active runtime environment (`development` or `deployment`). |
| `AUTO_CREATE_SCHEMA` | `true` (locally) | Enables database schema partition generation at startup. |
| `EMBEDDING_PROVIDER` | `local` | Embedding provider (`local` or `gemini`). |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model. |
| `ONLINE_EMBEDDING_MODEL` | `gemini-embedding-2` | Remote embedding model. |
| `EMBEDDING_DIMENSION` | `384` | Target vector dimension (384, 768, 1024, 2048, 3072). |
| `EMBEDDING_NAMESPACE` | `""` | Optional override for database partitions. |
| `GEMINI_API_KEY` | `""` | API credentials for Gemini embeddings and generation. |
| `GENERATION_PROVIDER` | `auto` | Generation provider (`gemini`, `groq`, `vercel_gateway`, `extractive`, or `auto`). |
| `GENERATION_PRIMARY_PROVIDER`| `gemini` | Primary LLM model provider. |
| `GENERATION_FALLBACK_PROVIDER`| `groq` | Backup LLM model provider. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Selected Gemini generation model. |
| `GROQ_API_KEY` | `""` | API credentials for Groq fallback. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Selected Groq generation model. |
| `AI_GATEWAY_MODEL` | `google/gemini-2.5-flash` | Model path when using Vercel AI Gateway. |
| `ONLINE_EMBEDDING_BATCH_SIZE`| `16` | Maximum texts sent in a single batch request. |
| `ONLINE_EMBEDDING_RPM` | `90` | Rolling free-tier RPM limit for embeddings. |
| `OCR_LANGUAGE` / `OCR_DPI` | `eng` / `150` | Settings for Tesseract OCR scanned PDF fallback. |
| `TOP_K` | `5` | Maximum retrieved chunks passed to the LLM. |
| `SIMILARITY_THRESHOLD` | `0.30` | Cosine similarity cutoff limit. |
| `CHUNK_SIZE` | `2000` | Target chunk size (characters). |
| `CHUNK_OVERLAP` | `200` | Overlap size between adjacent chunks (characters). |
| `DATA_DIR` | `data/rew_data/books` | Folder containing source documents. |
| `DEBUG` | `false` | Enables debug panels and query diagnostic outputs. |
| `MAX_MEMORY_TURNS` | `6` | Number of previous chat history turns sent to the LLM. |

---

## ⚡ Quickstarts

### 1. Docker End-to-End Setup
Run the database, local embeddings, Gemini generation, and Gradio using docker compose:
```bash
docker compose up -d
```
See **[QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md)** for details.

### 2. Local Host Development
Run PostgreSQL in Docker while executing the Python pipeline and UI on your host system:
1. Copy environmental variables:
   ```bash
   cp .env.development.example .env.development
   ```
2. Set your `GEMINI_API_KEY` inside `.env.development`.
3. Start the database service:
   ```bash
   docker compose up -d postgres
   ```
4. Install locked dependencies:
   ```bash
   uv sync --extra local
   ```
5. Initialize the database schemas and local models:
   ```bash
   uv run python scripts/bootstrap.py
   ```
6. Start the local server:
   ```bash
   uv run python app.py
   ```
   Open [http://localhost:7860](http://localhost:7860). See **[QUICKSTART_SYSTEM.md](QUICKSTART_SYSTEM.md)** for details.

### 3. Vercel + Neon Cloud Deployment
See **[DEPLOYMENT_VERCEL.md](DEPLOYMENT_VERCEL.md)** for configuration instructions for Vercel's stateless functions and Neon database connections.

---

## 🧪 Testing and Verification

Run the test suite:
```bash
uv run python -m pytest
```

Validate RAG routing, safety rules, and database schema compliance:
```bash
uv run python scripts/dry_run.py
```

Check database connection and pgvector queries:
```bash
uv run python scripts/database_smoke.py
```

Run retrieval quality and generation accuracy evaluations:
```bash
uv run python scripts/evaluate.py
```
