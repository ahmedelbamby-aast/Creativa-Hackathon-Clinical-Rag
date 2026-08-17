# Diabetes RAG Assistant

A fully grounded, bilingual (English + Arabic) RAG application that answers diabetes questions
using **only** information retrieved from provided medical documents.

> ⚕️ **This is a research/educational tool. Not a substitute for professional medical advice.**

---

## Architecture

```
User Question (EN / AR / EGY-AR)
    → Safety Check
    → Query Rewriting (follow-up contextualisation)
    → Category Routing (treatment / prevention / nutrition / all)
    → Embedding (sentence-transformers multilingual)
    → ChromaDB Semantic Search
    → Relevance Threshold Filter
    → Gemini LLM (grounded generation, context-only)
    → Citation Builder
    → Safety Disclaimer (if applicable)
    → Final Answer + Sources
```

## Project Structure

```
├── app.py                  # Gradio UI entry point
├── requirements.txt
├── .env.example            # Configuration template
├── README.md
│
├── data/                   # Place your PDF/DOCX documents here
│   └── rew_data/books/     # Default data directory
│
├── src/
│   ├── config.py           # Central configuration (all from .env)
│   ├── embeddings.py       # sentence-transformers wrapper
│   ├── vector_store.py     # ChromaDB client (3 collections)
│   ├── retriever.py        # Semantic search + threshold filter
│   ├── router.py           # Query category routing
│   ├── rewriter.py         # Follow-up query contextualisation
│   ├── prompts.py          # Gemini system + user prompt templates
│   ├── generator.py        # Gemini API wrapper
│   ├── citations.py        # Source citation builder
│   ├── safety.py           # Safety classifier + disclaimers
│   ├── memory.py           # Conversation history
│   ├── context_builder.py  # Context formatting for LLM
│   └── ingestion/
│       ├── parser.py           # PyMuPDF structure-aware PDF parser
│       ├── chunker_adapter.py  # SmartChunker → ChunkRecord
│       ├── category_classifier.py  # Auto-assign category to chunks
│       ├── pipeline.py         # Orchestrates parse→chunk→embed→store
│       └── core/
│           ├── chunker.py      # SmartChunker (word/sentence-safe)
│           ├── language_detector.py
│           └── quality_filter.py
│
├── scripts/
│   ├── ingest.py           # CLI ingestion pipeline
│   └── evaluate.py         # RAG evaluation with test cases
│
└── chroma_db/              # ChromaDB storage (auto-created)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Place your documents

Put PDF/DOCX files into `data/rew_data/books/` (or configure `DATA_DIR` in `.env`).

### 4. Run ingestion

```bash
python scripts/ingest.py
```

Expected output:
```
Documents processed : 12
Successful          : 12
Pages processed     : 847
Chunks created      : 3241
  treatment          982
  prevention         1105
  nutrition          743
  general            411
```

### 5. Launch the application

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | sentence-transformers model |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.30` | Minimum similarity score (0–1) |
| `CHUNK_SIZE` | `2000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DATA_DIR` | `data/rew_data/books` | Document directory |
| `CHROMA_DB_DIR` | `chroma_db` | ChromaDB storage directory |
| `DEBUG` | `false` | Show retrieval debug info in UI |

---

## Evaluation

```bash
# Full evaluation (requires Gemini API key)
python scripts/evaluate.py

# Retrieval-only (no LLM calls)
python scripts/evaluate.py --no-generate

# Single category
python scripts/evaluate.py --category nutrition

# Verbose (show answer previews)
python scripts/evaluate.py --verbose
```

---

## Knowledge Categories

| Category | What it covers |
|---|---|
| **Treatment** | Medications, clinical management, HbA1c, insulin, complications |
| **Prevention** | Risk factors, lifestyle changes, screening, physical activity |
| **Nutrition** | Dietary guidance, recommended foods, glycemic index, meal planning |
| **All** | Cross-domain queries (default) |

---

## Technology Stack

| Component | Technology |
|---|---|
| PDF Parsing | PyMuPDF (fitz) — structure-aware, table detection |
| Chunking | SmartChunker — word/sentence-safe, Arabic-aware |
| Embeddings | sentence-transformers (multilingual, free, local) |
| Vector DB | ChromaDB (local, no API key needed) |
| LLM | Google Gemini API |
| UI | Gradio |

---

## Safety & Disclaimers

This system implements a four-level safety classification:

- **EMERGENCY**: Acute symptoms → immediate redirect to emergency services
- **HIGH_RISK**: Dosing/prescription requests → answer with mandatory medical disclaimer
- **DIAGNOSIS**: Personal diagnosis requests → educational answer + consult disclaimer
- **INFORMATIONAL**: Normal RAG flow

The LLM is instructed to never invent medical facts, dosages, medications, or thresholds.
All answers must be traceable to retrieved document chunks.

---

## Ingestion Script Options

```bash
python scripts/ingest.py [OPTIONS]

Options:
  --data-dir PATH    Directory containing documents
  --file PATH        Ingest a single file
  --force            Re-ingest even if already stored
  --reset            Delete all ChromaDB data before ingesting
  --stats            Print collection stats and exit
  --verbose, -v      Enable debug logging
```
