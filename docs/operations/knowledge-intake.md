# Automatic PDF intake and publication

Checkpoint 9 adds authorized PDF submission, durable PostgreSQL jobs and automatic
extraction/indexing/publication. Uploading through the API or operator command queues
a job. A separate worker processes it. Copying a file into docs/RAG_Docs alone does
not submit it; the existing registration/import commands remain available.

## Document behavior

| Submission | Result |
|---|---|
| Identical bytes already in this collection | Reuse the document and ready index; no extraction or embedding calls |
| Different bytes with replace_document_id | New revision of that stable document identity; publish only after complete indexing |
| Different bytes without replacement identity | Add a new document while retaining the collection's other documents |

Content hashes detect identical bytes, including renamed uploads. They do not detect
semantically similar PDFs or byte-different exports of the same text. Filenames and
titles never identify replacements or assign agents. New document IDs derive from
collection + source hash; explicit replacement preserves the old document ID.
Submitting bytes belonging to another existing document as a replacement is rejected.

Collections use complete revision snapshots. A worker preserves unchanged extracted
pages/sections, replaces or adds the requested source and rebuilds the combined
index using the existing chunker/metadata/indexer. Changed collection snapshots can
re-embed unchanged passages; this checkpoint does not add a cross-revision embedding
cache. Identical PDF reuploads against the current snapshot avoid that work entirely.
Old published revisions and original source blobs remain available; no destructive
replacement, historical row deletion or automatic source garbage collection occurs.

## Trusted deployment configuration

Use [the example](../knowledge/intake-runtime.example.json) as the structure for a
server-owned `.knowledge-intake.json` (ignored by Git). Replace its dummy IDs, domain
and embedding digest with real values. Each binding contains:

- An existing collection ID and an explicit write scope for its domain/private owner.
- Internal manager user UUIDs permitted to upload, inspect documents/jobs and retry.
- The complete trusted IndexConfig, including exact embedding provider/model/digest,
  dimension and prefixes. It must match the installed serving profile and database dimension.
- Optional initial_revision_id for an existing ready but unpublished collection.

The current retained collection has no active revision. Extending it requires
explicitly choosing its existing ready revision as initial_revision_id, so intake
cannot silently discard its three PDFs. A genuinely empty collection needs no seed.
After first publication, the active pointer becomes the baseline for later jobs.
Do not use the example's placeholder digest as a deployment profile. The existing
indexing-nomic-v2.json records the retained corpus profile; validate the serving
endpoint before using it. Configuration changes require process restart.

Manager authorization controls who changes shared knowledge. It is not an instructor
approval/review stage for extracted content. Ordinary student credentials cannot grant
themselves this capability. Model output and uploaded document text never supply scopes.
Existing trusted operators may use the CLI without a pilot bearer token; their writer
credentials and deployment configuration form that boundary.

Apply migrations through the existing command, from backend:

```powershell
.venv/Scripts/python.exe -m scripts.migrate_knowledge --embedding-dimension 768
```

Migration 004_intake_jobs is additive. It creates durable management jobs and explicitly
removes reader access to their payloads. Applied migrations 001-003 remain unchanged.
All existing publication/retrieval scope checks remain authoritative.

## HTTP API

Enable KNOWLEDGE_INTAKE_ENABLED only after configuring:

| Setting | Purpose |
|---|---|
| APPLICATION_STORAGE=postgres and existing pilot authentication | Signed, persistent identity |
| KNOWLEDGE_INTAKE_CONFIG_FILE | Trusted binding/index policy JSON |
| KNOWLEDGE_INGEST_DATABASE_URL | Protected knowledge writer DSN |
| KNOWLEDGE_DB_HOST / KNOWLEDGE_WRITER_PASSWORD | Compose alternative to the writer DSN |
| KNOWLEDGE_SOURCE_ROOT | Durable source store shared with the worker |

The flag defaults to false and is separate from RAG_ENABLED. Enabling management
intake does not enable student RAG. With intake disabled, management endpoints return
unavailable and existing chat continues unchanged.

All requests require an existing signed bearer token for a configured manager:

| Method/path | Purpose |
|---|---|
| GET /knowledge/collections/{collection_id}/documents | List stable document IDs/titles/hashes and current active revision |
| POST /knowledge/collections/{collection_id}/documents?title=Manual | Queue a new PDF; binary body, Content-Type: application/pdf |
| Same POST with &replace_document_id={document_id} | Explicitly replace that current document |
| GET /knowledge/collections/{collection_id}/jobs/{job_id} | Inspect status/result or sanitized failure code |
| POST /knowledge/collections/{collection_id}/jobs/{job_id}/retry | Requeue a failed job |

Submission returns 202 with job_id and document_id. It does not wait for inference.
Statuses are queued, processing, succeeded and failed. Source bytes, stored job
configuration and worker lease tokens are excluded from responses. Unsupported media
returns 415, oversized bodies 413, denied management 403 and conflicting operations
409. Upload size is bounded at 25 MiB; parsing has existing page/text/time limits.
Use deployment request-size limits and a resource-limited worker for untrusted files.
Encrypted, corrupt or textless/scanned PDFs fail technically; OCR is not added here.

## Operator and worker

For guided local use, run `.venv/Scripts/python.exe -m scripts.manage_knowledge`
from backend. It provisions the dedicated local operator, completes/protects the
configuration, verifies the model profile and offers add/replace/jobs menus.
See the [wizard guide](knowledge-management-wizard.md). The commands below remain
available for scripts and deployed workers.

From backend, use the same trusted configuration and source directory for all commands:

```powershell
.venv/Scripts/python.exe -m scripts.intake_knowledge --config ../.knowledge-intake.json --source-store .knowledge-sources submit --collection COLLECTION_UUID --pdf C:/path/manual.pdf --title Manual
.venv/Scripts/python.exe -m scripts.intake_knowledge --config ../.knowledge-intake.json --source-store .knowledge-sources worker
```

Use submit --replace-document DOCUMENT_UUID for replacement. status and retry take
--collection and --job. worker --once performs one polling pass over configured
collections for scripts/smoke checks. Native Windows commands select the psycopg-compatible
event loop. Credentials come from the protected existing local file or configured
writer environment; commands print IDs/status/counts, not document text or credentials.

The optional [intake Compose overlay](../../docker/compose/docker-compose.intake.yml)
extends the existing dev + knowledge + knowledge.dev + application overlays. Set
INTAKE_CONFIG_PATH to the absolute trusted file path and pass backend/.env.knowledge
as before. It mounts that file read-only and the same knowledge-sources volume into
backend and worker. The knowledge-intake profile provides knowledge-intake-worker;
enable KNOWLEDGE_INTAKE_ENABLED for the backend separately. Deploy both submission
and worker against the same database/source volume. A host CLI using a different
source directory cannot service uploads stored only inside the Docker volume.

The worker image has CPU/memory limits and restart: unless-stopped. The base running
stack does not start it automatically. No Redis dependency or new model download is
required. No fixed document filenames, classifier, semantic enrichment or review queue
are involved. Published knowledge is available through normal scoped retrieval to
whichever specialist needs it; agents do not own separate PDF copies.

## Recovery and publication

One processing job per collection serializes document-set updates. Different
collections can be handled by separate workers. PostgreSQL transactions are short;
none spans extraction or model inference. Workers renew a 60-second lease every ten
seconds. Expired work is reclaimed on a later poll, up to five interrupted attempts;
after that it becomes a failed job requiring explicit retry. Graceful shutdown cancels
work and lets its lease expire. There is no global destructive reaper.

Each claim has a new token. A stale worker cannot renew or publish after lease loss.
Publication validates source-set completeness, exact embedding profile, ready status,
document extraction and chunk provenance, then locks/rechecks the revision and active
pointer and atomically commits both the new pointer and succeeded job status.
Readers see the old or the complete new revision. No partial index becomes active.
Interrupted processing can reuse a completed index instead of embedding it again.

Concurrent additions preserve both documents. Two replacements submitted against the
same old document do not silently overwrite each other: after one succeeds, the other
fails with intake_conflict and must be resubmitted against the current document.
An external active-pointer change also prevents stale publication. A planner request
already admitted to a revision remains pinned according to the existing retrieval
contract; a later request resolves the newly active revision.

For failures, inspect the sanitized code and fix storage/serving/source issues before
retrying. Retrying an extraction timeout may re-extract the same bytes. Retrying a
permanently corrupt PDF will fail again; submit a corrected replacement instead.
Changing the trusted indexing policy while a job is queued produces a conflict;
resubmit under the new configuration. Failed updates leave current published knowledge
usable. PostgreSQL and the source volume must be backed up together. Published
revisions select separate keys in the optional [Redis retrieval cache](knowledge-cache.md).

Disable intake and stop the worker to pause new processing. Existing published
knowledge/chat/history remain intact. Do not delete volumes or roll back applied SQL
to disable this feature. Student RAG remains disabled pending model-quality acceptance;
Optional Redis caching is provided by [Checkpoint 10](knowledge-cache.md); broader
evaluation remains Checkpoint 11, and vLLM is outside Sprint 5.

## Verification

The intake suites use actual PDFs and disposable PostgreSQL databases. Set the existing
repository integration flags and run tests/knowledge/test_intake.py,
tests/knowledge/test_intake_api.py and tests/persistence/test_intake_runtime.py.
The opt-in tests/knowledge/test_intake_real_model.py additionally uses the installed
embedding endpoint when FREDI_INTAKE_REAL_MODEL=1. It records direct-query and
paraphrase outcomes separately; neither a direct-match smoke nor successful storage
establishes general retrieval quality. See the Checkpoint 9 report for actual results.
