\set ON_ERROR_STOP on

DO $$
DECLARE
    embedding_dimensions integer;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'pgvector extension is not enabled';
    END IF;

    SELECT atttypmod
    INTO embedding_dimensions
    FROM pg_attribute
    WHERE attrelid = 'rag_chunks'::regclass
      AND attname = 'embedding'
      AND NOT attisdropped;

    IF embedding_dimensions <> 768 THEN
        RAISE EXCEPTION
            'rag_chunks.embedding has % dimensions, expected 768',
            embedding_dimensions;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE tablename = 'rag_chunks'
          AND indexname = 'rag_chunks_embedding_hnsw_idx'
          AND indexdef LIKE '%vector_cosine_ops%'
    ) THEN
        RAISE EXCEPTION 'cosine HNSW index is missing';
    END IF;

    IF to_regprocedure(
        'match_rag_chunks(vector,integer,jsonb,double precision)'
    ) IS NULL THEN
        RAISE EXCEPTION 'semantic search function is missing';
    END IF;

    IF to_regprocedure(
        'search_rag_chunks(text,integer,jsonb)'
    ) IS NULL THEN
        RAISE EXCEPTION 'keyword search function is missing';
    END IF;

    IF to_regprocedure(
        'hybrid_search_rag_chunks(vector,text,integer,jsonb,double precision,double precision,integer,integer)'
    ) IS NULL THEN
        RAISE EXCEPTION 'hybrid search function is missing';
    END IF;
END
$$;

SELECT
    current_setting('server_version') AS postgres_version,
    extversion AS pgvector_version
FROM pg_extension
WHERE extname = 'vector';

SELECT 'RAG database verification passed' AS result;

