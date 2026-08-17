BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri text UNIQUE,
    title text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    page_number integer CHECK (page_number IS NULL OR page_number > 0),
    content text NOT NULL CHECK (length(btrim(content)) > 0),
    language varchar(16),
    quality_score real CHECK (
        quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0
    ),
    embedding vector(768) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx
    ON rag_chunks (document_id);

CREATE INDEX IF NOT EXISTS rag_documents_metadata_idx
    ON rag_documents USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx
    ON rag_chunks USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS rag_chunks_search_vector_idx
    ON rag_chunks USING gin (search_vector);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE OR REPLACE FUNCTION match_rag_chunks(
    query_embedding vector(768),
    match_count integer DEFAULT 10,
    metadata_filter jsonb DEFAULT '{}'::jsonb,
    minimum_similarity double precision DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id bigint,
    document_id uuid,
    chunk_index integer,
    page_number integer,
    content text,
    language varchar(16),
    quality_score real,
    metadata jsonb,
    similarity double precision
)
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.page_number,
        c.content,
        c.language,
        c.quality_score,
        c.metadata,
        1.0 - (c.embedding <=> query_embedding) AS similarity
    FROM rag_chunks AS c
    WHERE c.metadata @> COALESCE(metadata_filter, '{}'::jsonb)
      AND 1.0 - (c.embedding <=> query_embedding) >= minimum_similarity
    ORDER BY c.embedding <=> query_embedding
    LIMIT LEAST(GREATEST(match_count, 0), 1000);
$$;

CREATE OR REPLACE FUNCTION search_rag_chunks(
    query_text text,
    match_count integer DEFAULT 10,
    metadata_filter jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    chunk_id bigint,
    document_id uuid,
    chunk_index integer,
    page_number integer,
    content text,
    language varchar(16),
    quality_score real,
    metadata jsonb,
    rank real
)
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    WITH query AS (
        SELECT websearch_to_tsquery('simple', COALESCE(query_text, '')) AS value
    )
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.page_number,
        c.content,
        c.language,
        c.quality_score,
        c.metadata,
        ts_rank_cd(c.search_vector, query.value) AS rank
    FROM rag_chunks AS c
    CROSS JOIN query
    WHERE c.search_vector @@ query.value
      AND c.metadata @> COALESCE(metadata_filter, '{}'::jsonb)
    ORDER BY rank DESC, c.id
    LIMIT LEAST(GREATEST(match_count, 0), 1000);
$$;

CREATE OR REPLACE FUNCTION hybrid_search_rag_chunks(
    query_embedding vector(768),
    query_text text,
    match_count integer DEFAULT 10,
    metadata_filter jsonb DEFAULT '{}'::jsonb,
    semantic_weight double precision DEFAULT 1.0,
    keyword_weight double precision DEFAULT 1.0,
    rrf_k integer DEFAULT 60,
    candidate_count integer DEFAULT 100
)
RETURNS TABLE (
    chunk_id bigint,
    document_id uuid,
    chunk_index integer,
    page_number integer,
    content text,
    language varchar(16),
    quality_score real,
    metadata jsonb,
    score double precision
)
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    WITH semantic_candidates AS MATERIALIZED (
        SELECT
            c.id,
            row_number() OVER (ORDER BY c.embedding <=> query_embedding) AS position
        FROM rag_chunks AS c
        WHERE c.metadata @> COALESCE(metadata_filter, '{}'::jsonb)
        ORDER BY c.embedding <=> query_embedding
        LIMIT LEAST(GREATEST(candidate_count, match_count, 1), 1000)
    ),
    text_query AS (
        SELECT websearch_to_tsquery('simple', COALESCE(query_text, '')) AS value
    ),
    keyword_candidates AS MATERIALIZED (
        SELECT
            c.id,
            row_number() OVER (
                ORDER BY ts_rank_cd(c.search_vector, text_query.value) DESC, c.id
            ) AS position
        FROM rag_chunks AS c
        CROSS JOIN text_query
        WHERE c.search_vector @@ text_query.value
          AND c.metadata @> COALESCE(metadata_filter, '{}'::jsonb)
        ORDER BY ts_rank_cd(c.search_vector, text_query.value) DESC, c.id
        LIMIT LEAST(GREATEST(candidate_count, match_count, 1), 1000)
    ),
    fused AS (
        SELECT
            COALESCE(s.id, k.id) AS id,
            COALESCE(
                semantic_weight / (GREATEST(rrf_k, 1) + s.position),
                0.0
            ) + COALESCE(
                keyword_weight / (GREATEST(rrf_k, 1) + k.position),
                0.0
            ) AS score
        FROM semantic_candidates AS s
        FULL OUTER JOIN keyword_candidates AS k USING (id)
    )
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.page_number,
        c.content,
        c.language,
        c.quality_score,
        c.metadata,
        fused.score
    FROM fused
    JOIN rag_chunks AS c ON c.id = fused.id
    ORDER BY fused.score DESC, c.id
    LIMIT LEAST(GREATEST(match_count, 0), 1000);
$$;

COMMENT ON TABLE rag_chunks IS
    'Page-aware text chunks and 768-dimensional embeddings for RAG retrieval.';
COMMENT ON FUNCTION match_rag_chunks(vector, integer, jsonb, double precision) IS
    'Cosine-similarity retrieval with optional JSONB metadata filtering.';
COMMENT ON FUNCTION search_rag_chunks(text, integer, jsonb) IS
    'PostgreSQL full-text retrieval with optional JSONB metadata filtering.';
COMMENT ON FUNCTION hybrid_search_rag_chunks(
    vector, text, integer, jsonb, double precision, double precision, integer, integer
) IS 'Hybrid semantic and keyword retrieval using reciprocal rank fusion.';

COMMIT;

