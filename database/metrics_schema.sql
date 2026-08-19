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
