# Agent Runbook: Gemini Embedding 2 Dimension Rollout

This document is for Claude, Codex, and other engineering agents working on
the sequential Gemini Embedding 2 migration plan in
`docs/SEQUENTIAL_GEMINI_EMBEDDING_2_DIMENSION_ROLLOUT.md`.

## Mission and current state

The current production retrieval index is:

```text
Provider:  Gemini Embedding 2
Dimension: 384
Namespace: gemini_384
Database:  cloud Lakebase Postgres / Neon
```

The cloud index contains all 12 PDFs. The target rollout is sequential:

```text
gemini_384 -> gemini_768 -> gemini_1024 -> gemini_2048 -> gemini_3072
```

Each new dimension is a parallel, independently verifiable index. Never
replace the currently active production index in place.

## Security and authority rules

- Never print, paste, commit, or return `DATABASE_URL`,
  `DATABASE_URL_UNPOOLED`, Gemini keys, Groq keys, Vercel tokens, or protection
  bypass values.
- `.env.deployment` is ignored by Git and contains local administrative
  connection settings. Read only the needed keys; never show its contents.
- Vercel environment variables are the production runtime source of truth.
  A local `.env.deployment` file is for trusted admin tooling only.
- Before any Vercel command, deployment, environment change, promotion, or
  alias change, read `AGENTS_DEPLPOY_VERCEL.md` completely.
- Never run a schema migration directly on production. Use a Neon branch and
  a Vercel preview first.
- Never disable Vercel Deployment Protection.

## Reaching the cloud database

Use the existing deployment environment only from a trusted local machine:

```powershell
$env:APP_ENV = 'deployment'
.\.venv\Scripts\python.exe scripts\audit_local_corpus.py
Remove-Item Env:APP_ENV
```

This uses the existing `DATABASE_URL` through `src.config`; it does not print
the connection string. The audit lists the active namespace, document names,
and chunk counts.

Use connection types correctly:

| Operation | Connection |
|---|---|
| Application requests, read audits, chunk/vector upserts | `DATABASE_URL` pooled |
| Schema migrations, partition/index creation, schema inspection | `DATABASE_URL_UNPOOLED` direct |

The current code has a fixed `rag_chunks.embedding vector(384)` column. A
new dimension requires a new table family; a new namespace alone is not enough.

## Preflight for every dimension

1. Confirm the current branch, clean working tree, and source commit.
2. Read the current Gemini RPM, TPM, and RPD quotas in AI Studio for the
   exact project and record only numerical limits in protected configuration.
3. Confirm the active production deployment is healthy with `/api/health` and
   `/api/ready`.
4. Capture the current reviewed retrieval and task metrics as the stage
   baseline.
5. Create an isolated Neon branch from production. Record its branch ID and
   connection references privately; do not include values in source control.

## Per-dimension procedure

Run this sequence for 768, 1024, 2048, and 3072. Do not start the next stage
until the current stage has been accepted or rejected.

### 1. Source-controlled migration

Add a migration that creates the target table, for example
`rag_chunks_d768`, with `embedding vector(768)` and the same metadata,
constraints, document indexes, and HNSW cosine index behavior as the 384
table. Update the vector-store router so the target table is selected by
dimension.

Apply the migration only to the Neon branch through the direct connection.
Verify the vector column dimension and HNSW index before ingesting data.

### 2. Local embedding and cloud sync

Run ingestion from the trusted local machine. Gemini Embedding 2 must use the
same model and output dimension for document chunks and query vectors.

For the target stage:

```text
EMBEDDING_PROVIDER=gemini
EMBEDDING_DIMENSION=<768|1024|2048|3072>
ACTIVE_INDEX_NAMESPACE=gemini_<dimension>
```

Parse the local 12-PDF corpus with the production 3000/300 chunk profile.
Generate the target-dimension embeddings and persist a manifest with model,
dimension, corpus hash, source checksums, document count, and chunk counts.

The planned sync command must bulk-upsert document metadata and vectors through
the pooled connection. It must use `(namespace, chunk_id)` as its idempotency
key, checkpoint after each document, and never call Gemini during a resumed
cloud upload when a local vector already exists.

### 3. Quota protection

Use protected configuration values populated from AI Studio:

```text
GEMINI_EMBEDDING_RPM_LIMIT
GEMINI_EMBEDDING_TPM_LIMIT
GEMINI_EMBEDDING_RPD_LIMIT
GEMINI_EMBEDDING_SAFETY_FACTOR=0.70
```

The quota controller must:

- enforce RPM and TPM at 70% of configured caps;
- persist RPD usage using America/Los_Angeles day boundaries;
- record request count, input tokens, 429s, retry delay, retry count, and
  per-document checkpoint state;
- honor a provider retry delay; otherwise use exponential backoff plus jitter;
- pause and checkpoint background ingestion at 95% of RPD; and
- fail interactive query embedding quickly, preserving grounded fallback.

Expose quota utilization, alerts, and resumable-run status only in the
protected administration dashboard. Do not expose any credential or database
detail in the user-facing chat UI.

### 4. Branch verification

Before a Vercel preview, prove all of the following on the Neon branch:

- all 12 documents exist exactly once in the target namespace;
- checksums and manifest match the local corpus;
- every stored vector has the target dimension;
- duplicate chunk IDs do not exist;
- English and Arabic retrieval smoke cases pass;
- refined metrics meet the stage acceptance gate.

### 5. Preview and promotion

Follow `AGENTS_DEPLPOY_VERCEL.md` without shortcuts:

1. Commit and push a clean tested source snapshot.
2. Create a Vercel preview from the exact committed snapshot.
3. Configure the preview to use the Neon branch and target namespace.
4. Test the protected preview in a real browser: UI shell, direct evidence,
   English/Arabic retrieval, vague and unsupported refusals, citations,
   metrics, and provider fallback.
5. Inspect deployment-scoped error logs.
6. Promote the exact verified preview only after all acceptance checks pass.
7. Confirm both production aliases resolve to the same READY deployment.

## Acceptance gate

A dimension is eligible for production only when:

- all 12 PDFs and their checksums are present;
- target vectors have the exact configured dimension;
- Hit Rate@5, Recall@5, nDCG@5, and Task Success regress by no more than 0.02
  absolute against the preceding active index;
- retrieval p95 is no more than twice the preceding active index;
- citations, provenance filtering, and safety refusals have no regression; and
- preview and production deployment-scoped error logs are clean.

If any check fails, leave the preceding dimension active, record the failure,
and stop the sequence. Do not delete the old table or namespace.

## Generation-model compatibility

Groq/GPT-OSS-120B and Gemini 2.5 Flash receive retrieved text, not embedding
vectors. No generation API migration is required for a dimension increase.

However, a higher-dimensional index can retrieve different chunks. Every stage
must therefore run the answer and task metrics before promotion.

## Agent handoff checklist

- State the target dimension, table, namespace, model, and source commit.
- State whether the task is read-only, a Neon-branch migration, local
  ingestion/sync, Vercel preview, or production promotion.
- Include document count, chunk count, metric comparison, quota state, and
  deployment status in handoff notes.
- Never include secrets, raw environment files, database URLs, or Vercel
  protection URLs.
