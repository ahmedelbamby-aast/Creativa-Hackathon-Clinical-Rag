# SugerFree — System Architecture & Technical Design

This document provides a detailed technical explanation of the architecture, components, data flows, and design principles implemented in the **SugerFree** clinical RAG (Retrieval-Augmented Generation) application.

---

## 1. High-Level System Architecture

SugerFree is built with a decoupled architecture containing:
1. **Frontend Layer:** A bilingual production web client (HTML/CSS/JS) and a local Gradio UI for debugging.
2. **API/Orchestration Layer:** Stateless FastAPI endpoints running as serverless functions on Vercel, paired with a python backend pipeline.
3. **Semantic Processing:** Pre-generation safety checks, conversation memory, contextual query rewriters, and domain-specific routers.
4. **Vector Store & Indexing:** A partitioned PostgreSQL 16 database with the `pgvector` extension, running on Neon serverless postgres.
5. **LLM Generation Engine:** Multi-provider API routing (Gemini primary, Groq secondary) with a fallback to purely extractive generation.

```mermaid
flowchart TB
    subgraph Clients["Frontend Layer"]
        browser["Bilingual HTML/JS Client<br/>(Vercel Web UI)"]
        gradio["Gradio UI<br/>(Local Dev & Debug)"]
    end

    subgraph API["FastAPI / Orchestrator"]
        server["backend/server.py<br/>(Stateless API Endpoints)"]
        app["app.py<br/>(Gradio Orchestration)"]
        pipeline["rag_pipeline()<br/>(Request Director)"]
    end

    subgraph Preprocessing["Intelligent Pre-processing"]
        safety["Safety Classifier<br/>(src/safety.py)"]
        rewriter["Query Rewriter / Hints<br/>(src/rewriter.py)"]
        router["Domain Category Router<br/>(src/router.py)"]
        memory["Conversation Memory<br/>(src/memory.py)"]
    end

    subgraph Storage["Vector & Persistence Layer"]
        neon[("Neon Cloud PostgreSQL 16<br/>+ pgvector")]
        partitions[("Partitioned Tables<br/>(rag_chunks_d384 / d768 / d1024...)")]
        hnsw{"HNSW Cosine Indexes"}
        metrics_db[("rag_metric_events<br/>(Telemetry DB)")]
        quota_db[("rag_embedding_runs<br/>(Embedding Runs & Events)")]
    end

    subgraph Models["Embedding & LLM APIs"]
        local_embed["Local MiniLM<br/>(384D offline)"]
        gemini_embed["Gemini Embeddings API<br/>(384D to 3072D)"]
        gemini_gen["Gemini Generation<br/>(gemini-3.6-flash)"]
        groq_gen["Groq LLM Fallback<br/>(openai/gpt-oss-120b)"]
        extractive["Extractive Synthesis<br/>(Deterministic fallback)"]
    end

    %% Client Interactions
    browser <-->|JSON API /api/chat| server
    gradio <--> app
    server --> pipeline
    app --> pipeline

    %% Pipeline Orchestration
    pipeline <--> Preprocessing
    pipeline -->|"Embed Query"| gemini_embed
    pipeline -->|"Embed Query"| local_embed
    pipeline <-->|"Retrieve & Filter Chunks"| neon
    pipeline -->|"Synthesis Prompt"| gemini_gen
    pipeline -->|"LLM Fallback"| groq_gen
    pipeline -->|"Extractive Fallback"| extractive

    %% DB Structure
    neon --> partitions --> hnsw
    pipeline -->|"Log Metrics"| metrics_db
    pipeline -->|"Track Quotas"| quota_db
```

---

## 2. Request Lifecycle

The system processes incoming user messages sequentially. The diagram below illustrates the timeline of a query from the time a user types it until the final answer is rendered.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Bilingual Frontend UI
    participant Server as FastAPI / Gradio Backend
    participant Safety as Safety Classifier (src/safety.py)
    participant Rewriter as Query Rewriter (src/rewriter.py)
    participant Router as Category Router (src/router.py)
    participant Embedder as Embedding Model (src/embeddings.py)
    participant DB as pgvector Partitioned Store
    participant LLM as Generator (src/generator.py)
    participant Citations as Citation Builder (src/citations.py)

    User->>UI: Types question (e.g. "كم جرعة الأنسولين؟")
    UI->>Server: HTTP POST /api/chat/stages (message + history + category)
    
    %% Pre-processing
    Server->>Safety: classify_safety(query)
    alt SafetyLevel == EMERGENCY
        Safety-->>Server: EMERGENCY detected
        Server-->>UI: Return Emergency Warning (Bypasses Retrieval & LLM)
        UI-->>User: Displays emergency response (Call 123)
    else SafetyLevel is HIGH_RISK, DIAGNOSIS, or INFORMATIONAL
        Server->>Rewriter: rewrite_query(query, history)
        Note over Rewriter: Expands follow-up references & maps Arabic<br/>retrieval hints to English corpus concepts
        Rewriter-->>Server: Returns rewritten_query
        
        Server->>Router: route_query(query, user_selected_category)
        Router-->>Server: Returns category route (treatment/prevention/nutrition/all)

        %% Embedding & Retrieval
        Server->>Embedder: embed_query(rewritten_query)
        alt Embedding API Online
            Embedder-->>Server: Query Vector (e.g., 768 dimensions)
            Server->>DB: Cosine similarity search query (HNSW index table partition)
        else Embedding API Outage / Rate Limit
            Server->>DB: Lexical SQL Fallback (Full-text keyword match)
        end
        DB-->>Server: Raw matching chunks (scores, metadata)
        
        Note over Server: Filters out chunks below SIMILARITY_THRESHOLD (0.30)<br/>& drops bibliography/references chunks
        
        alt No chunks match OR max similarity score < 0.25
            Server-->>UI: Returns polite "no-evidence" refusal (Prevents LLM Hallucination)
            UI-->>User: Renders refusal text
        else Adequate evidence retrieved
            %% Generation
            Server->>Citations: build_citation_list(retrieved_chunks)
            Citations-->>Server: Programmatic citation markers (SOURCE 1, 2) and source metadata
            
            Server->>Server: Build Grounded Prompt (inserts chunks & history)
            
            Server->>LLM: generate(prompt)
            alt Gemini API Online
                LLM-->>Server: Generated Answer (strictly grounded)
            else Gemini Outage (Groq Fallback)
                Server->>LLM: generate_via_groq(prompt)
                LLM-->>Server: Answer via Groq
            else All LLM APIs Out of Service (Extractive Fallback)
                Server->>Server: extract_deterministic_excerpts()
            end
            
            Server->>Safety: get_disclaimer(safety_level, is_arabic)
            Safety-->>Server: Medical / Dosage Disclaimer text
            Server->>Server: Append disclaimer & format inline citations
            
            Server->>DB: Log trace (latency, tokens, costs) to rag_metric_events
            Server-->>UI: Return final answer + sources list + debug metadata
            UI-->>User: Renders markdown answer with clickable references
        end
    end
```

---

## 3. RAG Pipeline & Document Ingestion

SugerFree maintains distinct offline (data ingestion) and online (query/inference) phases to optimize indexing quality and request latencies.

```mermaid
flowchart LR
    subgraph Ingestion["Offline Ingestion Phase (scripts/ingest.py)"]
        docs["Document Corpus<br/>(data/rew_data/books/)"]
        parser["Parser (PyMuPDF / docx)<br/>+ Tesseract OCR fallback"]
        lang["Language Detection"]
        chunker["SmartChunker<br/>(Size & Overlap limit)"]
        tables["Monolithic Tables<br/>(Markdown bypass)"]
        q_filter["Quality Filter<br/>(Min quality score 0.1)"]
        ingest_embed["Batch Document Embeddings<br/>(Local or Gemini API)"]
        db_upsert["Upsert to pgvector<br/>(namespace partition)"]
        
        docs --> parser --> lang --> chunker
        parser --> tables
        chunker --> q_filter
        tables --> q_filter
        q_filter --> ingest_embed --> db_upsert
    end

    subgraph Inference["Online Query Phase (app.py / server.py)"]
        user_query["User Query"]
        rewriter_m["Rewriter & Hints"]
        query_embed["Query Embedder"]
        vector_match["pgvector HNSW Cosine Scan<br/>(Threshold 0.30)"]
        context_prompt["Grounded Prompt Context"]
        llm_gen["Gemini / Groq LLM<br/>(Temperature 0.1)"]
        response["Grounded Response<br/>+ Citations & Disclaimer"]
        
        user_query --> rewriter_m --> query_embed --> vector_match
        vector_match --> context_prompt --> llm_gen --> response
    end

    db_upsert -.->|"Read Index"| vector_match
```

---

## 4. Component Interaction Graph

The components interact in a layered configuration, ensuring that query preprocessing is completed before retrieval, and retrieval is verified before synthesis.

```mermaid
flowchart TD
    server[backend/server.py]
    evidence_svc[src/evidence_service.py]
    safety[src/safety.py]
    rewriter[src/rewriter.py]
    router[src/router.py]
    embeddings[src/embeddings.py]
    vector_store[src/vector_store.py]
    retriever[src/retriever.py]
    generator[src/generator.py]
    citations[src/citations.py]
    extractive[src/extractive.py]
    obs[src/observability.py]

    server -->|"1. Stage Evidence"| evidence_svc
    evidence_svc -->|"2. Detect Safety Level"| safety
    evidence_svc -->|"3. Rewrite Vague Query"| rewriter
    evidence_svc -->|"4. Resolve Category Route"| router
    evidence_svc -->|"5. Convert to Query Vector"| embeddings
    evidence_svc -->|"6. Query Cosine HNSW Table"| vector_store
    evidence_svc -->|"7. Filter Threshold & Bibliography"| retriever
    
    server -->|"8. Generate Grounded Answer"| generator
    generator -->|"9. Load Prompt Templates"| prompts
    generator -->|"10. Programmatic Citation Markers"| citations
    generator -->|"11. Fallback Extractive Answer"| extractive
    generator -->|"12. Generate Medical Disclaimers"| safety
    
    server -->|"13. Log Diagnostics & Telemetry"| obs
    obs -->|"14. Write Metric Events DB"| vector_store
```

---

## 5. Database & Storage Architecture

SugerFree uses list partitioning in PostgreSQL to separate vector spaces. A separate parent table is used for each dimension to prevent dimension mismatch errors.

```mermaid
erDiagram
    RAG_CHUNKS_PARENT {
        text namespace PK
        text chunk_id PK
        text document_name
        integer page_number
        text section_title
        text subsection_title
        varchar category
        varchar content_type
        varchar language
        text source_id
        text source_url
        text content
        integer char_count
        integer word_count
        real quality_score
        vector embedding
        timestamptz created_at
        timestamptz updated_at
    }

    RAG_CHUNKS_D384_PARTITION {
        vector embedding_384D
    }

    RAG_CHUNKS_D768_PARTITION {
        vector embedding_768D
    }

    RAG_CHUNKS_D1024_PARTITION {
        vector embedding_1024D
    }

    RAG_METRIC_EVENTS {
        uuid trace_id PK
        text conversation_id
        integer turn_index
        timestamptz recorded_at
        text status
        double_precision total_ms
        double_precision retrieval_ms
        double_precision generation_ms
        integer total_tokens
        numeric estimated_cost_usd
        jsonb payload
    }

    RAG_EMBEDDING_RUNS {
        uuid run_id PK
        text namespace
        text table_family
        integer dimension
        text model
        text corpus_hash
        text status
        integer total_documents
        integer completed_documents
        text current_document
        jsonb checkpoint
        text last_error
        timestamptz created_at
        timestamptz updated_at
        timestamptz completed_at
    }

    RAG_EMBEDDING_EVENTS {
        bigint event_id PK
        uuid run_id FK
        timestamptz recorded_at
        text event_type
        text operation
        text provider
        text model
        integer dimension
        text namespace
        text table_family
        integer request_count
        integer input_tokens
        integer embedded_items
        integer retry_delay_ms
        text error_code
        jsonb metadata
    }

    RAG_CHUNKS_PARENT ||--o| RAG_CHUNKS_D384_PARTITION : "partition of (namespace='local_384')"
    RAG_CHUNKS_PARENT ||--o| RAG_CHUNKS_D768_PARTITION : "partition of (namespace='gemini_768')"
    RAG_CHUNKS_PARENT ||--o| RAG_CHUNKS_D1024_PARTITION : "partition of (namespace='gemini_1024')"
    RAG_EMBEDDING_RUNS ||--o{ RAG_EMBEDDING_EVENTS : "references"
```

### Table Roles
1. **`rag_chunks` Parent Tables:** Store the actual split chunks of document content alongside meta-information (e.g. source URLs, page numbers, quality scores). Partitioned by list using the `namespace` field. HNSW cosine indexes are built on the `embedding` vector column for each partition.
2. **`rag_metric_events` Table:** Persists telemetry data for each RAG pipeline transaction, including processing times, generated responses, estimated token costs, and internal steps for offline diagnostics.
3. **`rag_embedding_runs` & `rag_embedding_events` Tables:** Used by the ingestion pipeline to track batch indexing states, check for errors, and verify token usage against daily limits.

---

## 6. Request Lifecycle — Detailed Technical Explanation

Below is the technical data transformation log for a single query:

### 6.1. User Input & UI Intercept
* **Input:** User query string (e.g. `"ما هي أسباب السكر؟"`) and optional category choice.
* **Component:** `backend/static/index.html` (Browser) or `app.py` (Gradio).
* **Processing:** The UI intercepts the submit event, packs the message alongside browser-retained chat history, and calls the API endpoint.
* **Output:** JSON request payload matching the `ChatRequest` schema.

### 6.2. Endpoint Validation & Staging
* **Input:** `ChatRequest` model.
* **Component:** `backend/server.py` (`POST /api/chat/stages`).
* **Processing:** Validates the message size, sanitizes the category route, and resolves the requested vector dimension. It constructs a `ConversationMemory` instance and calls `src/evidence_service.py:stage_evidence`.
* **Output:** `RetrievalEnvelope` containing evidence status and metadata.

### 6.3. Pre-generation Intercepts
* **Input:** Raw query string.
* **Component:** `src/safety.py:classify_safety`, `src/rewriter.py:rewrite_query`, and `src/router.py:route_query`.
* **Processing:** 
  1. `classify_safety` uses keyword heuristics to match Arabic/English medical emergency signals. If matched, it returns `SafetyLevel.EMERGENCY` and halts execution.
  2. `rewrite_query` resolves conversational pronouns (using history) and adds bilingual retrieval hints mapping Arabic expressions to English terms.
  3. `route_query` scores the query against category keyword lists to return `treatment`, `prevention`, `nutrition`, or `general`.
* **Output:** Safe, expanded search query and active search category.

### 6.4. Embedding & Search
* **Input:** Rewritten query string.
* **Component:** `src/embeddings.py` (via `EmbeddingModel`) and `src/vector_store.py` (via `VectorStore`).
* **Processing:** Resolves the active model (local `paraphrase-multilingual-MiniLM-L12-v2` or cloud `gemini-embedding-2`), vectorizes the query, and performs a cosine-distance search (`<=>` operator) on the database partition. If the embedding provider fails, it triggers the lexical keyword SQL fallback.
* **Output:** A list of raw document chunk dictionary records.

### 6.5. Thresholding & Sufficiency Verification
* **Input:** Raw search results with cosine distance values.
* **Component:** `src/retriever.py:retrieve` & `is_retrieval_sufficient`.
* **Processing:** Converts cosine distance to similarity score: `score = 1.0 - distance`. Drops chunks with scores below `SIMILARITY_THRESHOLD` (0.30) and removes chunks representing bibliographies. If the highest score is below `0.25`, the query is flagged as unanswerable.
* **Output:** Verified list of `RetrievedChunk` objects, or an empty list if insufficient.

### 6.6. Prompts & LLM Generation
* **Input:** `RetrievedChunk` elements and conversation history.
* **Component:** `src/prompts.py` and `src/generator.py`.
* **Processing:** If the retrieval is sufficient, `build_user_prompt` constructs the LLM context, labeling retrieved texts as `SOURCE 1`, `SOURCE 2`, etc. System instructions explicitly forbid generating claims not contained in the sources. The generator sends this prompt to the active model (`gemini-3.6-flash`). If rate-limited, it falls back to Groq, and then to extractive excerpts.
* **Output:** Grounded answer string.

### 6.7. Citation Reconstruction & Telemetry Logging
* **Input:** LLM generated text, retrieved chunks, and pipeline timers.
* **Component:** `src/citations.py:ensure_inline_citations`, `src/safety.py:get_disclaimer`, and `src/observability.py:record_trace`.
* **Processing:** Formats inline markers to match retrieved database attributes, appends the appropriate medical disclaimer based on the safety level, estimates token costs, and saves the event metrics payload into `rag_metric_events`.
* **Output:** Complete JSON payload (`ChatResponse`) returned to the user interface.

---

## 7. Code-to-Architecture Mapping

| Architectural Component | File / Path | Responsibility |
| :--- | :--- | :--- |
| **API Server & Routing** | [backend/server.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/backend/server.py) | Defines the stateless serverless HTTP interface, request validation, and Vercel-compatible routers. |
| **Local Orchestrator** | [app.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/app.py) | Handles Gradio interface rendering, pipeline orchestration, and debug panels. |
| **Ingestion Entrypoint** | [scripts/ingest.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/scripts/ingest.py) | Manages corpus reading, force updates, database index verification, and chunk ingestion commands. |
| **Ingestion Pipeline** | [src/ingestion/pipeline.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/ingestion/pipeline.py) | Coordinates document parsing, page context propagation, quality filtration, embedding, and storage. |
| **PDF/Docx Parsing** | [src/ingestion/parser.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/ingestion/parser.py) | Extracts elements from source files. Uses PyMuPDF, python-docx, and Tesseract OCR for scanned images. |
| **Quality Filtering** | [src/ingestion/core/quality_filter.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/ingestion/core/quality_filter.py) | Scores chunk text density, filters out OCR noise, numbers-only blocks, and short header fragments. |
| **Text Chunking** | [src/ingestion/core/chunker.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/ingestion/core/chunker.py) | Splits parsed documents into size-bounded chunks while preserving monolithic markdown tables. |
| **Vector Database Client** | [src/vector_store.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/vector_store.py) | Manages connection pooling, index schema initialization, partition creation, and nearest-neighbor search. |
| **Embedding Facade** | [src/embeddings.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/embeddings.py) | Wraps local sentence-transformers and Gemini APIs, ensuring unit-length vectors are used for cosine search. |
| **Quota Controller** | [src/embedding_quota.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/embedding_quota.py) | Throttles ingestion requests to prevent rate limit exceptions, tracking RPM/TPM against constraints. |
| **Retrieval Filtering** | [src/retriever.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/retriever.py) | Applies similarity thresholds, category filters, and ignores bibliography sections. |
| **Safety Classifier** | [src/safety.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/safety.py) | Inspects queries for emergency or high-risk signals, managing disclaimers and bypass routing. |
| **Query Rewriter** | [src/rewriter.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/rewriter.py) | Resolves conversational context using history and adds translation hints for Arabic query retrieval. |
| **Category Router** | [src/router.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/router.py) | Matches queries to domain categories (treatment/prevention/nutrition) using keyword scoring. |
| **Prompt Construction** | [src/prompts.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/prompts.py) | Structures the grounded LLM system context, inserting history and formatted context chunks. |
| **Inference Client** | [src/generator.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/generator.py) | Manages primary (Gemini) and fallback (Groq) generative LLM calls, implementing retry logic. |
| **Extractive Answer Fallback** | [src/extractive.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/extractive.py) | Acts as a tertiary generation fallback, directly building answer summaries from retrieved texts. |
| **Citation Formatter** | [src/citations.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/citations.py) | Matches generated inline tags to verified database attributes, outputting clickable metadata links. |
| **Telemetry System** | [src/observability.py](file:///d:/CREATIVA%20RAG%20PROJECT%20-%20SugerFree/Creativa_Diabetes/src/observability.py) | Records latency breakdowns and saves event metrics to `rag_metric_events` for analysis. |
