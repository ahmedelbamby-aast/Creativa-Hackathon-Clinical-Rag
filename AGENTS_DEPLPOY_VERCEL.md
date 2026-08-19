# Mandatory Vercel deployment instructions for agents

> **Required reading:** Every agent must read this entire file before running
> any Vercel command, changing Vercel configuration or environment variables,
> creating a preview, promoting a deployment, or touching production aliases.
>
> The filename intentionally preserves the requested spelling:
> `AGENTS_DEPLPOY_VERCEL.md`.

These instructions are specific to the Creativa Diabetes RAG project. They
capture the failures already encountered in this repository and define the
minimum safe deployment and verification procedure.

## Non-negotiable rules

1. Never deploy an uncommitted or untested working tree.
2. Never deploy directly to production before validating a preview.
3. Promote the exact tested preview; do not rebuild separately for production.
4. Never expose, print, commit, or paste Vercel, Neon, Gemini, or Groq secrets.
5. Never disable Vercel Deployment Protection to make testing easier.
6. Never run an ad hoc migration against the production Neon database.
7. Never treat `/api/ready` alone as proof that retrieval and generation work.
8. Never claim success until the deployment is `READY`, both stable aliases
   point to it, the UI has been tested, and deployment-scoped logs are clean.
9. Preserve unrelated user changes. Stop if the working tree contains changes
   that cannot safely be separated from the deployment.
10. Use non-interactive Vercel commands with `--yes` and the explicit project
    and team scope shown below.

## Fixed project identity

Verify these values against `.vercel/project.json` before every deployment:

| Setting | Required value |
|---|---|
| Project | `creativa-diabetes-rag` |
| Project ID | `prj_RtaOAVHEwLY0SiLOhD7s0QPf2Ejs` |
| Team | `POS INV Demo` |
| Team ID | `team_VxVg9p19tEIJ7pinDFnUKLga` |
| Team slug | `pos-inv-demo` |
| Production branch | `main` |
| Framework | FastAPI |
| Primary production URL | `https://creativa-diabetes-rag.vercel.app` |
| Team production URL | `https://creativa-diabetes-rag-pos-inv-demo.vercel.app` |

Do not run `vercel link` if `.vercel/project.json` already contains these
values. Relinking can silently target the wrong project or team.

## Required tools and authentication checks

Run from the repository root:

```powershell
vercel --version
vercel whoami
Get-Content -LiteralPath '.vercel\project.json' -Raw
git branch --show-current
git status --short
git fetch origin main
git rev-list --left-right --count HEAD...origin/main
```

Expected before a deployment from `main`:

- Vercel CLI is installed and authenticated.
- The current branch is `main`.
- `git status --short` is empty.
- `HEAD` and `origin/main` identify the intended source commit.
- The linked project and team match the fixed identity table.

If the user requested another commit or branch, record that exact SHA and do
not describe it as `main`.

## Test gate before creating a preview

The minimum repository test gate is:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --no-cov -p no:randomly -n 0
.\.venv\Scripts\python.exe -m compileall -q app.py backend src tests
git diff --check
```

Also run focused tests for the files or behavior being changed. For provider
routing and grounded fallback, the suite must cover at least:

- Gemini success.
- Gemini operational failure followed by Groq success.
- Gemini and Groq both failing, followed by deterministic evidence excerpts.
- Provider safety or invalid-request errors not being misrepresented as an
  availability failure.
- Vague questions being rejected before retrieval or provider calls.
- Unsupported or indirectly related evidence causing a refusal and a request
  for a clearer diabetes question.
- Certified provenance filtering and legacy database-schema compatibility.

Do not deploy if any required test fails.

## Known Vercel Git-author block

This Hobby-team project has previously returned `TEAM_ACCESS_REQUIRED` with a
reason similar to:

```text
Git author <email> must have access to the team POS INV Demo on Vercel to create deployments.
```

A normal `vercel deploy` from a Git working tree automatically attaches Git
metadata. Vercel may block the deployment before the build starts if the commit
author email is not recognized as a team member, even when the authenticated
CLI user has project access.

Do not work around this by falsifying the commit author, rewriting published
history, changing team membership, or weakening team security. Unless the team
configuration has been explicitly fixed and verified, deploy an exact committed
snapshot without the `.git` directory as described below.

## Create an exact source snapshot

The snapshot must come from a committed SHA, not the mutable working tree.
Create it under the operating-system temporary directory:

```powershell
$deploySha = git rev-parse HEAD
$snapshotRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('codex-vercel-' + [guid]::NewGuid().ToString('N'))
$sourceDir = Join-Path $snapshotRoot 'source'
$archivePath = Join-Path $snapshotRoot 'source.zip'
New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
git archive --format=zip --output="$archivePath" $deploySha
Expand-Archive -LiteralPath $archivePath -DestinationPath $sourceDir
```

Verify `git archive` succeeded and record `$snapshotRoot`, `$sourceDir`, and
`$deploySha`. The archive must contain the intended commit, including any newly
added tracked files.

Never copy `.env*`, `.vercel`, `.git`, local virtual environments, or secrets
into the snapshot.

## Deploy a preview first

Deploy the snapshot with explicit scope and neutral source metadata:

```powershell
vercel deploy "$sourceDir" `
  --yes `
  --project prj_RtaOAVHEwLY0SiLOhD7s0QPf2Ejs `
  --scope pos-inv-demo `
  --meta sourceCommit=$deploySha `
  --meta sourceBranch=main `
  --meta actor=agent
```

Record all of the following:

- Preview URL.
- Deployment ID.
- Inspector URL.
- Source SHA.
- Build result and duration.

Wait for the terminal state:

```powershell
vercel inspect <preview-url> --wait --scope pos-inv-demo
```

Do not continue if the preview is `BLOCKED`, `ERROR`, `CANCELED`, or still
`BUILDING`.

## Diagnosing a blocked deployment

If a deployment is blocked or the CLI spinner does not reflect the platform
state, inspect the deployment directly:

```powershell
vercel api /v13/deployments/<deployment-id>
```

Read `readyState`, `readyStateReason`, `seatBlock`, and `errorLink`. A blocked
deployment has no useful build logs because the build never started. Do not
wait indefinitely on the CLI after the platform reports a terminal block.

For an actual build failure, use:

```powershell
vercel inspect <deployment-url> --logs --scope pos-inv-demo
vercel logs <deployment-url>
```

## Test protected previews without disabling protection

For API checks, use `vercel curl`:

```powershell
vercel curl <preview-url>/api/health
vercel curl <preview-url>/api/ready
```

For browser/UI automation, request a temporary authenticated share URL through
the connected Vercel capability, then open that share URL in the browser. The
share URL sets the required cookie and expires automatically.

Never disable Deployment Protection and never publish the temporary share URL
in a commit, issue, or final report.

## Required preview acceptance matrix

### 1. UI shell

Verify through a real browser:

- The page is not blank.
- No framework error overlay appears.
- The question box, category selector, Send button, Clear button, evidence
  panel, and Sources panel render.
- The route label is visible as:
  `Gemini → Groq → Evidence excerpts (automatic)`.

### 2. Health and readiness

`GET /api/health` must report:

- `status: ok`.
- `generation_provider: auto`.
- The Gemini → Groq → evidence-excerpts route.
- A configured database and generation route.

`GET /api/ready` must report:

- `status: ready`.
- The expected pgvector namespace.
- A non-zero indexed chunk count.

### 3. Vague question

Submit `Tell me more` with an empty conversation. Expected behavior:

- The user is asked for a more specific diabetes question.
- Retrieval and LLM providers are not called.
- The UI explains that no provider was called.

Then verify that the same phrase is allowed as a follow-up when prior
conversation context exists.

### 4. Directly supported question

Use a question directly supported by a certified source, for example:

```text
What risk factors for gestational diabetes mellitus are listed in the source?
```

Expected behavior:

- Certified evidence appears before the generated answer.
- Every medical claim is directly supported by the displayed evidence.
- Citations include document name, section, page, and HTTPS source URL.
- The UI displays the actual successful provider.

### 5. Indirect or unsupported question

Use examples such as:

```text
What are the main risk factors for type 2 diabetes?
Who won the football World Cup in 2022?
```

If the certified evidence does not directly answer the same condition,
population, and intent, the response must say the sources are insufficient and
ask for a more specific diabetes question. It must not transfer a list from
gestational diabetes, type 1 diabetes, prediabetes, or another population.

### 6. Provider failover

The configured order must be:

```text
Gemini → Groq → deterministic certified evidence excerpts
```

Do not intentionally break production credentials merely to test failover.
Use automated tests for deterministic failure injection. If Gemini naturally
returns an operational error such as 429 or 503 during preview testing, verify
deployment logs show the switch to Groq and that the UI identifies Groq.

If both LLM providers fail, the API must still return direct excerpts from the
already-displayed certified evidence and label the response as the grounded
evidence fallback. It must not return outside knowledge.

## Database and provenance safeguards

The live Lakebase Postgres/Neon table may be older than the current schema. A
previous failure was:

```text
psycopg.errors.UndefinedColumn: column "source_id" does not exist
```

The application intentionally supports legacy read schemas and enriches source
identity from `data/retrieval_sources.json`. It discards uncertified chunks and
continues only when certified evidence remains.

Important consequences:

- `/api/ready` can pass while a real retrieval query still fails. Always test a
  real question through `/api/retrieve` and the UI.
- Do not remove legacy-schema compatibility without migrating and re-ingesting
  the deployed database first.
- Do not weaken the HTTPS provenance requirement to make a deployment pass.
- Do not run `database/schema.sql` directly against production as a quick fix.

If a schema migration is genuinely required:

1. Add the migration to source control.
2. Create an isolated Neon branch from production.
3. Use the direct/unpooled connection for the migration.
4. Run the migration and retrieval tests on that branch.
5. Review the schema diff.
6. Point a Vercel preview at the branch and run the full UI acceptance matrix.
7. Apply the migration to production only with explicit authorization.

## Inspect logs before promotion

Use deployment-scoped logs. Project-wide error groups may include old previews
or expected, handled provider failures.

Check:

```powershell
vercel logs <preview-url> --level error --since 1h
```

For this application, an expected Gemini 429 followed by a successful Groq 200
is handled failover, not a failed user request. Confirm the final response is
successful and do not hide genuine retrieval, database, or HTTP 5xx errors.

## Promote the exact verified preview

After every acceptance check passes:

```powershell
vercel promote <verified-preview-url> --yes --scope pos-inv-demo
```

Promotion can create a new production deployment that temporarily reports
`BUILDING`. Discover its deployment ID and generated URL, then wait:

```powershell
vercel inspect <generated-production-url> --wait --scope pos-inv-demo
```

Do not report production success until it is `READY`.

Verify both stable aliases resolve to the new deployment:

```powershell
vercel inspect https://creativa-diabetes-rag.vercel.app --scope pos-inv-demo
vercel inspect https://creativa-diabetes-rag-pos-inv-demo.vercel.app --scope pos-inv-demo
```

Both must report the same final deployment ID, `target: production`, and
`status: Ready`.

## Required production verification

Repeat these checks against the stable production alias:

1. Open the UI in a browser and confirm no error overlay.
2. Confirm the provider route label is visible.
3. Run `/api/health` and `/api/ready` with `vercel curl`.
4. Submit at least one real supported or intentionally limited RAG question.
5. Confirm displayed evidence and citations are certified and directly related.
6. Inspect deployment-scoped error and fatal logs.
7. Confirm Vercel deployment metadata contains the exact `sourceCommit`.
8. Confirm `HEAD`, `origin/main`, and the deployed SHA match when the requested
   source is `main`.
9. Confirm the Git working tree is clean.

## Rollback rule

If the production UI, retrieval, database access, or provider fallback is
broken after promotion, stop and roll back to the last verified production
deployment:

```powershell
vercel rollback <last-known-good-deployment-url-or-id> --scope pos-inv-demo
```

Verify the rollback reaches `READY` and repeat health, readiness, and UI checks.
Do not leave a known-broken production alias active while investigating.

## Cleanup

After deployment:

1. Close every browser automation session.
2. Resolve each snapshot path to an absolute path.
3. Verify it is below the operating-system temporary directory.
4. Verify its leaf name begins with `codex-vercel-`.
5. Remove only those verified temporary snapshot directories.
6. Confirm no temporary environment file, access URL, token, or deployment
   archive was added to Git.
7. Confirm `git status --short` is empty.

Never recursively delete a computed or unresolved path.

## Final deployment report

Every deployment handoff must include:

```text
URL: <stable production URL>
Target: production
Status: READY
Deployment ID: <dpl_...>
Commit: <full or short SHA>
Framework: FastAPI
Tests: <count and result>
UI verification: <cases checked>
Provider route observed: <Gemini/Groq/evidence fallback>
Health/readiness: <result and indexed chunk count>
Deployment-scoped error scan: <clean or summarized findings>
```

State any remaining operational limitation explicitly. In particular, expected
Gemini quota exhaustion is acceptable only when Groq or the deterministic
evidence fallback completes the user request safely.
