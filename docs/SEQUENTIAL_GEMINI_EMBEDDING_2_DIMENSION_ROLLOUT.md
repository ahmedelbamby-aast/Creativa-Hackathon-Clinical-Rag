# Sequential Gemini Embedding 2 Dimension Rollout

## Summary

Run controlled blue/green upgrades from the current 384-dimensional index:

```text
384 baseline -> 768 -> 1024 -> 2048 -> 3072 maximum
```

Gemini Embedding 2 supports 128 to 3072 output dimensions. Each stage fully
re-embeds all 12 PDFs and user queries using the same Gemini model and stage
dimension. Keep the existing 384 index available until the final stage is
accepted.

Actual Gemini RPM, TPM, and RPD limits are project and tier specific. Read the
active values in AI Studio before each stage; do not hard-code public limits.

## Storage and retrieval changes

The current `rag_chunks` schema is fixed at `vector(384)`. A new dimension
cannot be added by namespace alone. Create parallel table families:

```text
rag_chunks_d384
rag_chunks_d768
rag_chunks_d1024
rag_chunks_d2048
rag_chunks_d3072
```

Each table has an HNSW cosine index and uses its own matching namespace:

```text
gemini_384
gemini_768
gemini_1024
gemini_2048
gemini_3072
```

Update vector-store reads, writes, document checks, reset/delete operations,
HNSW creation, audits, and retrieval to select a table family from
`EMBEDDING_DIMENSION`. Raise configuration validation from 2000 to 3072.

Keep metrics and `rag_metric_events` shared, but record embedding model,
dimension, namespace, and table family on every trace.

## Sequential migration procedure

For 768, then 1024, then 2048, then 3072:

1. Create an isolated Neon branch from the current production state.
2. Apply the source-controlled migration for the target vector table through
   the direct/unpooled connection.
3. From a trusted local machine, parse and chunk all 12 PDFs with the
   production 3000/300 profile and create Gemini Embedding 2 vectors at the
   target dimension.
4. Store a manifest containing model, dimension, corpus hash, checksums,
   chunk profile, and per-document chunk counts.
5. Confirm all 12 documents exist in the new cloud namespace.
6. Deploy a Vercel preview against the branch and target namespace.
7. Test retrieval, citations, English and Arabic questions, refusals, metrics,
   and production-provider fallback.
8. Promote only the exact tested preview. Keep the prior dimension available
   for rollback.

### Acceptance gate

Promote a stage only when:

- all 12 PDFs and expected checksums are present;
- every vector has the target dimension;
- Hit Rate@5, Recall@5, nDCG@5, and Task Success regress by no more than 0.02
  absolute from the preceding active stage;
- retrieval p95 is no more than 2x the preceding active stage;
- provenance, citations, and refusal behavior have no regression; and
- preview and deployment-scoped logs are clean.

If a stage fails, retain the preceding production dimension, record the
comparison, and stop the sequence.

## Rate-limit detection and control

Store these protected configuration values from AI Studio:

```dotenv
GEMINI_EMBEDDING_RPM_LIMIT=
GEMINI_EMBEDDING_TPM_LIMIT=
GEMINI_EMBEDDING_RPD_LIMIT=
GEMINI_EMBEDDING_SAFETY_FACTOR=0.70
```

Implement a persistent quota controller that:

- tracks requests, input tokens, embedded items, retries, 429 responses, and
  provider retry delays;
- enforces rolling RPM and TPM budgets at 70% of the active quota;
- persists RPD usage using America/Los_Angeles daily boundaries;
- honors provider retry delays or applies exponential backoff with jitter;
- checkpoints ingestion at document boundaries and resumes without
  re-embedding completed documents; and
- fails interactive query embeddings quickly so grounded fallback behavior can
  remain responsive.

Create protected dashboard records for ingestion runs and embedding events.
The dashboard shows utilization, remaining configured quota, active dimension,
document progress, 429/retry history, and paused/resumable status. Alert at
70%, mark critical at 85%, and hard-stop background ingestion at 95%.

## Compatibility

Gemini document vectors and Gemini query vectors must use the same model and
the same output dimension. Equal vector length does not make embeddings from
different models compatible.

Groq/GPT-OSS-120B and Gemini 2.5 Flash do not require an API change: they
receive retrieved text, not vectors. Higher dimensions can change the selected
context, so retrieval and end-to-end answer metrics must be compared at every
stage.

## Test plan

- Unit-test table routing, dimension validation, vector-length rejection, and
  cross-dimension search prevention.
- Test each dimension's ingestion, manifest validation, all-12-document
  completeness, checkpoint recovery, and duplicate prevention.
- Test rolling RPM/TPM windows, Pacific-day RPD reset, threshold alerts, 429
  retry-delay parsing, jitter backoff, and resume behavior.
- For every preview, test direct English and Arabic retrieval, vague and
  unsupported refusals, citation provenance, metrics persistence, and rate
  dashboard rendering.
