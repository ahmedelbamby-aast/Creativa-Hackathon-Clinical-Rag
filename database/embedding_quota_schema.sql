CREATE TABLE IF NOT EXISTS rag_embedding_runs (
    run_id uuid PRIMARY KEY,
    namespace text NOT NULL,
    table_family text NOT NULL,
    dimension integer NOT NULL CHECK (dimension BETWEEN 1 AND 3072),
    model text NOT NULL,
    corpus_hash text NOT NULL DEFAULT '',
    status text NOT NULL CHECK (status IN ('running', 'paused_quota', 'completed', 'failed')),
    total_documents integer NOT NULL DEFAULT 0,
    completed_documents integer NOT NULL DEFAULT 0,
    current_document text NOT NULL DEFAULT '',
    checkpoint jsonb NOT NULL DEFAULT '{}'::jsonb,
    last_error text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS rag_embedding_events (
    event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id uuid REFERENCES rag_embedding_runs(run_id) ON DELETE SET NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    event_type text NOT NULL,
    operation text NOT NULL,
    provider text NOT NULL DEFAULT 'gemini',
    model text NOT NULL,
    dimension integer NOT NULL CHECK (dimension BETWEEN 1 AND 3072),
    namespace text NOT NULL,
    table_family text NOT NULL,
    request_count integer NOT NULL DEFAULT 0,
    input_tokens integer NOT NULL DEFAULT 0,
    embedded_items integer NOT NULL DEFAULT 0,
    retry_delay_ms integer NOT NULL DEFAULT 0,
    error_code text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS rag_embedding_events_minute_idx
    ON rag_embedding_events (recorded_at DESC) WHERE event_type = 'reserved';
CREATE INDEX IF NOT EXISTS rag_embedding_events_run_idx
    ON rag_embedding_events (run_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS rag_embedding_runs_status_idx
    ON rag_embedding_runs (status, updated_at DESC);
