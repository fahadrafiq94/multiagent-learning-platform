# Knowledge indexing: chunks, metadata and real embeddings

Checkpoint 5 indexes an authorized extracted revision into a separate, complete
snapshot. Original PDFs and extraction revisions remain immutable inputs. It does
not activate a collection, expose student retrieval or add Redis caching.

## Configuration

[indexing-nomic-v2.json](../knowledge/indexing-nomic-v2.json) pins the installed
Ollama model digest, dimension, normalization and document/query prefixes. The
[official model guidance](https://registry.ollama.com/library/nomic-embed-text-v2-moe)
specifies its prefixes and 512-token input limit. No chat model is changed/downloaded.
The [embedding API](https://docs.ollama.com/api/embed) supports truncate=false;
indexing always uses it so excess input cannot silently lose its ending.

Defaults: 1000-character source chunks, up to 120 characters of overlap, a bounded
160-character document/heading context prefix, eight inputs per batch and a 20,000
chunk job limit. Character limits are not token counts. If the model rejects a batch
for input length, the job isolates oversized inputs and recursively splits their
exact source slices. Inputs that cannot fit even at the minimum size fail explicitly.
All original nonblank text remains covered; no final source sentence is truncated.

Configuration and source snapshot/version determine revision identity. Batching is
operational and does not change identity. Changed bytes create a new extraction
revision through the existing importer; changed model/prefix/chunk/metadata settings
create a separate index revision. A complete matching index is reused with zero
embedding calls. Never overwrite a previously applied SQL file or change the
database dimension argument to switch models; plan an explicit compatibility migration.
Reuse currently applies to a complete matching revision. A changed snapshot rebuilds
its embeddings; cross-revision embedding reuse is not implemented in this checkpoint.

## Run

Start the existing services and refresh the registered import if PDFs changed.
Apply the forward knowledge migration from backend:

```powershell
.venv/Scripts/python.exe -m scripts.migrate_knowledge --embedding-dimension 768
```

Then from the repository root:

```powershell
docker compose --env-file backend/.env.knowledge `
  -f docker/compose/docker-compose.dev.yml `
  -f docker/compose/docker-compose.knowledge.yml `
  -f docker/compose/docker-compose.knowledge.dev.yml `
  --profile knowledge-tools run --rm --no-deps --build knowledge-index `
  --source-revision b2abddbb-bafe-5534-8201-4d4ae81b7c44 `
  --config /config/indexing-nomic-v2.json --scope-domain course:ap40 --progress
```

Use the source revision printed by the importer for updated PDFs. The command is
an operator boundary: scope/optional --owner-user-id come from trusted configuration,
never document text or a student request. The service uses the existing writer role
and internal Ollama endpoint. Config is mounted read-only; it reads preserved pages
from PostgreSQL and does not need another copy of the source volume. The disposable
job has a 768 MiB memory limit and one CPU; it is not another persistent application
container. Larger corpora need appropriately bounded jobs/resources; automatic intake
and job scheduling remain Checkpoint 9.

For native development, run python -m scripts.index_knowledge from backend with host
paths and an explicit reachable --model-url. Database credentials come from protected
KNOWLEDGE_INGEST_DATABASE_URL or the existing ignored local file, never CLI arguments.
Use the same installed model digest as the profile; do not accidentally point a host
command at a different Ollama installation. Windows uses an explicit SelectorEventLoop.

Final JSON contains revision/source IDs, status, reuse flag, document/section/chunk/
step/diagnostic counts and embedding request count. --progress prints counts, never
source text or vectors. Exit 0 means ready; 2 means a recorded technical indexing
failure; 1 means setup/access/storage failure. No output grants content approval.

## Data and integrity

- Chunks retain exact section offsets, page offsets, text hashes, heading paths,
  structural reference IDs and an embedding-input hash. Context prefixes are separate
  from quoted evidence. Markdown/numbered headings carry into following paragraph
  sections; explicit step boundaries reset inherited heading context.
- Whole text lines/table rows are retained when they fit; oversized rows/paragraphs
  split with exact provenance. Existing extracted table text is preserved, not rebuilt
  into an inferred grid. Images/scans and unsupported visual table structures require
  later extraction/OCR work; existing OCR/blank-page diagnostics remain visible.
- Every eligible section is indexed, including introductory and mixed content without
  step IDs/hints. There are no ingestion classifiers, mandatory hint ladders, fixed
  PDF-name rules or permanent document-to-agent assignments. Authored hint wording is
  source evidence; the existing student guidance policy still chooses guidance level.
- Explicit labelled fields can supply canonical steps only from documents whose
  workflow_authority was granted in trusted registration. Metadata aliases/namespace
  are configurable. No reference label or metadata table is required for ordinary RAG.
  Each parsed field retains its original text, section/page provenance and offsets.
  Reference links retain authored qualifications; they are not an executable workflow.
  Ambiguous duplicate records or unresolved links produce diagnostics and preserve
  raw evidence instead of inventing facts or introducing an instructor review queue.
- Embedding inspection checks provider, exact installed digest and dimension before
  and after the run. Output count, finite/nonzero vectors, float32 representation,
  dimension, normalization and source coverage are validated. The source snapshot
  version is rechecked before commit. Database writes hold no model-call transaction.
- PostgreSQL stores vector(768), generated search_text/search_vector fields, a GIN
  full-text index and a cosine HNSW index. Tokenization uses simple for mixed-language
  text and identifiers. Following [pgvector's HNSW limits](https://github.com/pgvector/pgvector#hnsw),
  HNSW is created for vector dimensions up to 2000; larger
  supported storage profiles retain exact search until a suitable index is planned.
- Readers enforce access scope and revalidate complete indexed snapshots. Writer
  methods require explicit management scope, including failure-state updates.

## Completion, failure and retry

New revisions start pending with an extraction-only snapshot. Embeddings are built
outside SQL; the final complete snapshot and ready state commit atomically. No partial
batch is visible as a ready index. A completed index cannot be overwritten through
the repository. Concurrent duplicate runs can compute redundantly but converge through
identity uniqueness and version checks; compute coalescing belongs to later jobs.

Known failures record a bounded code, such as embedding_profile_mismatch,
invalid_embedding_vector, embedding_context_too_small or index_storage_unavailable.
The failed snapshot retains extraction and no partial chunks. A rerun retries it
without duplicating a completed index. A killed process or unavailable database can
leave pending state; rerun after restoring dependencies. Unreadable/scanned sources
retain extraction diagnostics. An all-unusable corpus cannot become a ready index.

Indexing readiness is separate from collection publication. indexing_status=ready
means technical completeness only; revision
status remains staging and active_revision_id is unchanged. Checkpoint 6 now provides
[scoped operator retrieval](knowledge-retrieval.md); later checkpoints attach evidence
to specialists and Checkpoint 9 connects
automatic intake/readiness/publication. Existing HTTP chat currently remains ungrounded.
