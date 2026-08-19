# Local 768-Dimension Embeddings and Cloud Sync Plan

## Decision

Use `intfloat/multilingual-e5-base` with 768-dimensional vectors for the next
index generation. It supports the English and Arabic retrieval workflow while
removing the corpus-embedding dependency on Gemini.

Do not put these vectors into the existing `rag_chunks` table. Its embedding
column is `vector(384)`, and its `gemini_384` data must remain available for
rollback.

## Invariants

- Documents and user queries must be embedded with the same pinned model,
  revision, dimension, normalization method, and query/document prompt format.
- Document parsing and embedding happen locally.
- Cloud Postgres stores already-computed vectors; sync does not call Gemini.
- The current `gemini_384` production path remains unchanged until a new path
  passes preview acceptance checks.
- Generation providers receive retrieved text, not vectors. Groq/GPT-OSS-120B
  and Gemini 2.5 Flash do not need a dimensionality change.

## Target names

```text
Local table:       rag_chunks_768
Local namespace:   e5_multilingual_768
Cloud table:       rag_chunks_768
Cloud namespace:   e5_multilingual_768
Embedding model:   intfloat/multilingual-e5-base
Dimension:         768
Chunk settings:    3000 characters / 300-character overlap
```

The common chunk settings are required for an exact local-to-cloud transfer.

## Phase 1 - Add a parallel vector store

Create a source-controlled migration that creates `rag_chunks_768` with the
same metadata columns as `rag_chunks`, but with `embedding vector(768)`. Add
HNSW cosine indexes to its namespace partitions. Do not alter the current
`rag_chunks.embedding vector(384)` column.

Update the vector-store routing code so it selects a table family from the
configured dimension. The metric-event table stays shared and needs no vector
migration.

Test this only on a Neon branch using the unpooled/direct connection. Do not
run the migration ad hoc against production.

## Phase 2 - Local indexing

Create a local embedding profile that uses the E5 model and explicitly applies
the model's retrieval prompts:

```text
query: <user question>
passage: <document chunk>
```

Pin the model revision and normalize every vector before writing. Parse and
chunk all 12 PDFs with 3000/300 settings, then store the chunks and 768-vector
embeddings in the local `e5_multilingual_768` index.

Before sync, verify all of the following locally:

- 12 document names and checksums match the source corpus;
- every embedding has exactly 768 values;
- all expected chunk IDs are unique;
- the index manifest records model, revision, dimension, chunk profile, and
  corpus hash;
- English and Arabic retrieval smoke cases pass.

## Phase 3 - Local-to-cloud sync

Add a resumable sync command that reads local rows and bulk-upserts them into
the cloud `rag_chunks_768` table. It must compare the manifest first and stop
on a mismatch.

The sync key is `(namespace, chunk_id)`. It must preserve document metadata,
source provenance, text, and the already-created normalized vectors. It must
not call an embedding provider.

Use the pooled cloud connection for bulk application writes. The direct
connection is reserved for the Phase 1 schema migration.

After each document, write a local checkpoint containing its checksum and
chunk count. A rerun skips verified documents and resumes only incomplete or
mismatched ones.

## Phase 4 - Production query embeddings

Vercel cannot use Gemini query embeddings with E5 document vectors. It needs a
reachable service that runs the identical pinned E5 model for queries.

Run an authenticated local embedding gateway with one endpoint:

```text
POST /embed/query -> normalized 768-dimensional E5 query vector
```

The Vercel API calls this gateway, then runs pgvector search in the cloud
`e5_multilingual_768` namespace. The gateway must have HTTPS, authentication,
request limits, health checks, and a clear unavailable response.

This is the trade-off for truly local embeddings: the host machine and its
secure tunnel must remain online. If that is not acceptable, host the same
model in a persistent embedding service instead; corpus embedding remains
Gemini-free, but query embedding is no longer physically local.

## Phase 5 - Preview and cutover

1. Point a Vercel preview at the Neon branch and local query gateway.
2. Verify all 12 cloud documents, document-level counts, and manifest match.
3. Run English and Arabic direct questions, vague and unsupported refusals,
   citations, metrics, and provider-fallback tests.
4. Promote only the tested preview.
5. Retain `gemini_384` for rollback until the new path is stable.

## Generation-model impact

Higher embedding dimension changes retrieval storage, index size, query latency,
and which chunks are selected. It does not alter Groq/GPT-OSS-120B or Gemini
2.5 Flash APIs because both receive text context after retrieval. Regression
testing is still required because a different retrieved context can change the
answer.
