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
    content text NOT NULL,
    char_count integer NOT NULL,
    word_count integer NOT NULL,
    quality_score real,
    embedding vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, chunk_id)
) PARTITION BY LIST (namespace);

CREATE INDEX IF NOT EXISTS rag_chunks_document_idx
    ON rag_chunks (namespace, document_name);

CREATE INDEX IF NOT EXISTS rag_chunks_category_idx
    ON rag_chunks (namespace, category);
