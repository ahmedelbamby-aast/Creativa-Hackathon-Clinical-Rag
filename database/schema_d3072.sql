-- Migration: 3072-dimension embedding parent table
--
-- Creates the parent table that holds 3072-d Gemini Embedding 2 vectors.
-- This is the maximum output dimensionality supported by Gemini Embedding 2.
-- The baseline rag_chunks (384-d) is left untouched for rollback safety.
-- Namespace partitions and HNSW indexes are created at runtime by
-- VectorStore.ensure_schema() when a new namespace is first used.
--
-- Apply via the direct/unpooled connection (DATABASE_URL_UNPOOLED) on an
-- isolated Neon branch before running ingestion against that branch.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks_d3072 (
    namespace          text         NOT NULL,
    chunk_id           text         NOT NULL,
    document_name      text         NOT NULL,
    page_number        integer,
    section_title      text         NOT NULL DEFAULT '',
    subsection_title   text         NOT NULL DEFAULT '',
    category           varchar(32)  NOT NULL,
    content_type       varchar(32)  NOT NULL DEFAULT 'text',
    language           varchar(16)  NOT NULL DEFAULT 'en',
    source_id          text         NOT NULL DEFAULT '',
    source_url         text         NOT NULL DEFAULT '',
    content            text         NOT NULL,
    char_count         integer      NOT NULL,
    word_count         integer      NOT NULL,
    quality_score      real,
    embedding          vector(3072) NOT NULL,
    created_at         timestamptz  NOT NULL DEFAULT now(),
    updated_at         timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, chunk_id)
) PARTITION BY LIST (namespace);

CREATE INDEX IF NOT EXISTS rag_chunks_d3072_document_idx
    ON rag_chunks_d3072 (namespace, document_name);

CREATE INDEX IF NOT EXISTS rag_chunks_d3072_category_idx
    ON rag_chunks_d3072 (namespace, category);
