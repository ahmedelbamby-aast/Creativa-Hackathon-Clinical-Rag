# Deploy to Vercel Hobby + Neon Free

This deployment runs the existing Gradio interface as an ASGI application on
Vercel's Python runtime. Neon provides PostgreSQL and pgvector. Gemini provides
both production embeddings and answer generation, so the Vercel function does
not need Torch or a persistent model cache.

## Deployment architecture

- **Vercel request runtime:** `backend/server.py` mounts the Gradio app on
  FastAPI and exposes `/api/health` and `/api/ready`.
- **Neon database:** the application uses the pooled `DATABASE_URL`; schema
  operations use the direct `DATABASE_URL_UNPOOLED`.
- **Admin ingestion:** PDF parsing and ingestion run from a trusted developer
  machine or `.github/workflows/ingest-production.yml`, not inside a web
  request. The complete ingestion code and PDF corpus remain in the repository.
- **Function bundle:** `.vercelignore` and `vercel.json` omit raw PDFs,
  development examples, tests, and local Docker assets from the deployed
  function only.

The selected free services are appropriate for a personal/hackathon demo. Do
not store patient records or protected health information in this deployment.

## 1. Environment files

Two local files are supported and ignored by Git:

| File | Use |
|---|---|
| `.env.development` | Local Gradio, local PostgreSQL, and local embeddings. |
| `.env.deployment` | Admin access to the Vercel/Neon production environment. |

Tracked templates are `.env.development.example` and
`.env.deployment.example`. Vercel's encrypted environment variables are the
runtime source of truth; `.env.deployment` is only for local administration.

To create the development file:

```powershell
Copy-Item .env.development.example .env.development
```

## 2. Install and authenticate Vercel CLI

```powershell
npm install --global vercel@latest
vercel login
vercel link
```

Link to the `creativa-diabetes-rag` project when prompted. The repository's
`vercel.json` enables Fluid Compute, selects Frankfurt (`fra1`), gives the
Python function a 300-second ceiling, and removes non-runtime files from the
function bundle.

## 3. Provision Neon Free

Provision Neon through the Vercel Marketplace and select the **Free** plan and
a European region when offered:

```powershell
vercel integration add neon --name creativa-diabetes-db
```

Connect the resource to development, preview, and production. Neon supplies:

- `DATABASE_URL`: pooled endpoint for serverless requests.
- `DATABASE_URL_UNPOOLED`: direct endpoint for schema creation and migrations.

Pull production variables for the local admin tools:

```powershell
vercel env pull .env.deployment --environment=production
```

Add or verify the remaining Vercel variables using the keys in
`.env.deployment.example`. Production must use:

```dotenv
APP_ENV=deployment
EMBEDDING_PROVIDER=gemini
ONLINE_EMBEDDING_MODEL=gemini-embedding-2
ONLINE_EMBEDDING_RPM=90
EMBEDDING_DIMENSION=384
EMBEDDING_NAMESPACE=gemini_384
AUTO_CREATE_SCHEMA=false
DEBUG=false
```

Set `GEMINI_API_KEY` as a sensitive Vercel variable. Never paste its value into
Git, workflow YAML, build arguments, logs, or the Vercel project configuration.

## 4. Create and populate the hosted index

Run this from a trusted machine after pulling `.env.deployment`:

```powershell
$env:APP_ENV = "deployment"
uv sync
uv run python scripts/bootstrap.py --verify-online
uv run python scripts/ingest.py
Remove-Item Env:APP_ENV
```

`bootstrap.py` uses the unpooled URL for schema/extension/index operations.
Application reads and writes use the pooled URL. Gemini requests are sent in
bounded batches so the full corpus is not submitted as one API call.

For repeatable later updates, configure the GitHub `production` environment
with `GEMINI_API_KEY`, `DATABASE_URL`, and `DATABASE_URL_UNPOOLED`, then run the
**Rebuild production knowledge base** workflow. It can ingest the full corpus
or one repository-relative source file and never runs automatically on a web
request.

## 5. Deploy and verify

```powershell
vercel deploy --prod
```

Verification endpoints:

- `GET /api/health` confirms the function and required configuration loaded.
- `GET /api/ready` confirms PostgreSQL, pgvector, the active namespace, and a
  non-empty index.
- `/` serves the mounted Gradio application.

After deployment, verify one English and one Arabic query, confirm citations
contain document/page metadata, and inspect Vercel runtime logs for 5xx errors.

## 6. Rollback

Vercel keeps immutable deployments. To restore an earlier production build:

```powershell
vercel rollback <deployment-id-or-url>
```

The vector namespace is independent of the application deployment. Do not
delete or reset `gemini_384` during an application rollback.

## Free-tier constraints

- Vercel Hobby is for personal, non-commercial use and applies function usage
  limits.
- Neon Free can suspend compute after allowance exhaustion and has finite
  storage/network transfer.
- Gemini usage is governed separately by the Google API account and quota. The
  ingestion client honors provider retry delays and defaults to a rolling
  90-item/minute cap to stay below the current free-tier embedding limit.
- The local JSONL diagnostics stored by Vercel are ephemeral because serverless
  filesystems are not durable application storage.

Official references:

- [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python)
- [Vercel Function limits](https://vercel.com/docs/functions/limitations)
- [Vercel environment variables](https://vercel.com/docs/environment-variables/manage-across-environments)
- [Neon connection pooling](https://neon.com/docs/connect/connection-pooling)
- [Neon pricing](https://neon.com/pricing)
- [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)
