CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    namespace text NOT NULL,
    chunk_id text NOT NULL,
    document_name text NOT NULL,
    page_number integer,
    section_title text NOT NULL DEFAULT '',
    subsection_title text NOT NULL DEFAULT '',
    category varchar(32) NOT NULL,
    content_type varchar(32) NOT NULL DEFAULT 'text',
    language varchar(16) NOT NULL DEFAULT 'en',
    source_id text NOT NULL DEFAULT '',
    source_url text NOT NULL DEFAULT '',
    content text NOT NULL,
    char_count integer NOT NULL,
    word_count integer NOT NULL,
    quality_score real,
    embedding vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, chunk_id)
) PARTITION BY LIST (namespace);

ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS source_id text NOT NULL DEFAULT '';
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS source_url text NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS rag_chunks_document_idx
    ON rag_chunks (namespace, document_name);

CREATE INDEX IF NOT EXISTS rag_chunks_category_idx
    ON rag_chunks (namespace, category);

-- Independent of the embedding namespace so index/model changes remain comparable.
CREATE TABLE IF NOT EXISTS rag_metric_events (
    trace_id uuid PRIMARY KEY,
    conversation_id text NOT NULL DEFAULT '',
    turn_index integer NOT NULL DEFAULT 0,
    recorded_at timestamptz NOT NULL,
    status text NOT NULL,
    total_ms double precision NOT NULL DEFAULT 0,
    retrieval_ms double precision,
    generation_ms double precision,
    total_tokens integer NOT NULL DEFAULT 0,
    estimated_cost_usd numeric(14, 8),
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_metric_events_recorded_at_idx ON rag_metric_events (recorded_at DESC);
CREATE INDEX IF NOT EXISTS rag_metric_events_conversation_time_idx ON rag_metric_events (conversation_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS rag_metric_events_status_idx ON rag_metric_events (status, recorded_at DESC);
