# Registered PDF imports

Checkpoint 3 imports operator-registered PDFs into staging. It does not classify
sections, generate chunks/vectors, publish a collection, or serve student requests.

Checkpoint 5 now provides a separate [indexing command](knowledge-indexing.md)
for complete chunks, real embeddings and structural metadata. Import remains the
extraction stage and preserves its original revision identity and behavior.

## Run the initial import

Start the knowledge services and apply the dimension-768 migration as described in
[the repository runbook](knowledge-repositories.md). From the repository root:

```powershell
backend/.venv/Scripts/python.exe backend/scripts/knowledge_infra.py import
```

`make knowledge-import` runs the same helper. It builds a disposable
`knowledge-import` container, mounts `docs/RAG_Docs` at `/input` read-only,
`docs/knowledge` at `/config` read-only, and the existing `knowledge-sources` named
volume at `/sources`. The container joins the application network and connects to
`postgres:5432` with the writer login. It exits after the import; there are still
four long-running application/infrastructure services. The helper does not start
or migrate PostgreSQL automatically, or recreate backend/Ollama.

The helper prints an outcome containing revision ID, reuse flag, document/section/
diagnostic counts, and failed-document count. It suppresses Compose/build logs to
avoid displaying resolved credentials. A failed document is persisted with its
raw source and diagnostic code; the CLI exits 2. Setup/preflight failures exit 1.
Review diagnostics without extraction failures return 0 and remain unpublished.

For other operator-controlled registrations, use the same container directly,
adjusting input/config bind mounts as needed. The underlying CLI is:

```text
python -m scripts.import_knowledge --registration /config/ap40-sources.json --source-root /input --source-store /sources
```

For host development, invoke that module from `backend` with host paths. It uses
`KNOWLEDGE_INGEST_DATABASE_URL`, if configured, or the ignored local credentials
file. In the import container, it constructs connection parameters from the
internal host and writer password environment variables. Never put credentials
in registration JSON or command arguments. A host source-store directory is
independent of the Docker named volume; prefer the container for durable imports.

## Registration and identity

[ap40-sources.json](../knowledge/ap40-sources.json) is trusted operator metadata.
It declares collection/access/owner boundaries, stable document IDs, source paths,
titles, workflow authority, an embedding compatibility profile, and structural
expectations. It does not assign agents or approve extracted content.

- Filenames only locate input files inside the allowed source root. Resolved paths
  escaping that root are rejected. Missing, empty, or oversized files fail preflight.
- Preserve document IDs when renaming or updating a document. Renaming unchanged
  bytes reuses the existing revision. Keep collection IDs stable across imports.
- Source bytes are addressed by SHA-256 and copied atomically into the source volume.
  Existing copies are checked for hash corruption before reuse.
- Revision identity includes source hashes, document registration metadata and
  structural profiles, extractor version, embedding compatibility and pipeline
  versions. Paths and source-list ordering do not affect identity.
- Changed bytes or structural requirements create a new staging revision. Existing
  completed imports, including subsequent processing, are never reset by this
  command. Changing collection access metadata requires a separate explicit migration.
- A process interrupted before its atomic snapshot write leaves a version-zero
  revision that the command can resume. Concurrent identical imports converge on
  one snapshot using database uniqueness and optimistic version checks.
- A completed failed extraction is reused on identical input/configuration. Retry
  after fixing the source or changing the extractor version; automatic retry/job
  administration belongs to Checkpoint 9. Orphaned content-addressed copies after
  a failed database write are harmless; this command never deletes shared sources.

The embedding profile records compatibility with the already measured 768-dimensional
database. This import makes no embedding/model calls and does not need Ollama running.

## Extraction and technical processing

Pypdf runs in a separate process with a 60-second per-document timeout. Limits:
25 MiB input, 500 pages, 10 MiB decoded content per page, 100,000 text characters per
page, and 2,000,000 characters per document. The worker suppresses parser logs and
returns stable error codes. The import container has a 768 MiB memory limit and one
CPU; the subprocess itself is not an OS security sandbox or a hard memory limiter.
Only registered operator-controlled inputs are supported here.

Encrypted/malformed/empty PDFs receive durable failure codes. Blank pages produce
`page_without_text`; all-blank/image-only extraction also produces `ocr_required`.
OCR is not implemented, and a blank page can be intentional, so an operator decides
whether the warning needs action. A successful parser does not prove extraction is
visually complete; extraction quality is measured by tests and representative evaluation.

Structural profiles declare an identifier label and optional expected references.
Historical required_hint_levels settings are accepted but ignored; no hint ladder
is required. The extractor version is pdf-sections-v2. Logical reference records can span pages. General documents
need no course IDs or hint ladder: blank-line paragraphs are used where available,
with bounded continuous sections otherwise. Sections above 50,000 characters are
split into parts retaining their reference and provenance. This is extraction
sectioning; retrieval chunking is a later checkpoint.

Pages and section text are persisted in document/section JSON payloads. A section
stores its ordinal, optional reference, page range and exact page text offsets.
Offsets refer to extracted Unicode text, not visual PDF coordinates. Repository
validation checks that section text agrees with those spans. Page text hashes can
be computed through `ExtractedPage.text_hash`; raw source hashes remain canonical.
No SQL migration is required for these additive payload fields.

Diagnostics include missing/duplicate/unexpected references and technical extraction
warnings. Reports are capped at 1,000 entries. Structural warnings do not hold useful
text. Missing authored hint levels are no longer generated as diagnostics.

Usable text has extraction_status=extracted; empty/scanned/unreadable sources have
extraction_status=failed and actionable technical codes (OCR remains unimplemented).
Sections default to processing_status=ready, meaning usable extracted text, not
published/indexed knowledge. No classifier, content-type gate or instructor approval
is involved. Future chunking/embedding still must complete before student retrieval.

Original 001-era snapshots remain readable after 002_general_indexing. Retired
annotations/needs_review strings remain only as historical metadata, never policy.
The original staged PDFs/sections are retained. Old pipeline markers stay opaque
legacy versions so historical revision IDs and fingerprints remain reproducible.

## Tests

From `backend`:

```powershell
$env:FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION = '1'
.venv/Scripts/python.exe -m pytest tests/knowledge -q -p no:cacheprovider
```

The opt-in suite uses unique scratch databases with actual writer/reader logins;
it never clears the imported collection. Without the variable, PostgreSQL cases
skip and deterministic extraction/in-memory cases still run. Shared import contracts
cover rename/update reuse, simultaneous imports, interrupted extraction, operator
snapshot preservation, changed profiles, durable failure/OCR state and provenance.
