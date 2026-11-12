# Knowledge schema, repositories, and source storage

Storage foundations include Checkpoints 2-3 and the revised 4A general-indexing
contracts. Checkpoint 5 adds complete indexed revisions with real embeddings,
full-text fields and cited structural metadata; see [knowledge indexing](knowledge-indexing.md).
Collection activation, student-graph retrieval and Redis integration remain pending.
Checkpoint 6 provides scoped operator/service retrieval with the reader role; see
[knowledge retrieval](knowledge-retrieval.md). Student-graph integration remains later.
PostgreSQL student/session adapters and authenticated durable HTTP/history APIs are
implemented through 4C. See [application persistence](application-persistence.md).

## Migrations

Start the services using [the infrastructure runbook](knowledge-infrastructure.md).
Then, from backend:

```powershell
uv run python -m scripts.migrate_knowledge --embedding-dimension 768
```

The explicit dimension above is for the current local embedding model,
nomic-embed-text-v2-moe, whose actual API returned a finite nonzero 768-dimensional
vector during Checkpoint 2. It is not a universal default for future models.

The command uses the local writer credentials from backend/.env.knowledge. For a
different deployment, set KNOWLEDGE_INGEST_DATABASE_URL in the process environment;
the command then uses that secret instead. Do not put passwords into command-line
arguments or commits. The bootstrap vector extension and writer-owned knowledge
schema from Checkpoint 1 must already exist.

Migrations take a transaction-scoped advisory lock and apply transactionally. The
preflight checks require pgvector and a knowledge schema owned by
fredi_knowledge_writer with migration USAGE/CREATE permissions. Missing schema or
incorrect ownership produces an explicit bootstrap error, not a database-outage
diagnostic. Migrations do not create roles/schemas or repair privileged bootstrap.
An existing Docker volume does not rerun its initialization script automatically;
have an administrator restore the missing bootstrap configuration without deleting data.

The schema_migrations ledger records the source checksum and configured vector dimension.
Repeating the same migration is a no-op. Altered/missing applied migration files or
a changed dimension are rejected. Do not edit an applied SQL file or delete the
database volume to change dimensions; write a deliberate migration/rebuild plan.

003_chunk_index adds indexing state/error codes, generated lexical search fields,
GIN and dimension-compatible cosine HNSW indexes. Original migration checksums are
unchanged. Technically ready index snapshots are immutable through repositories;
completion and failure updates require trusted management scope. Readers validate
complete chunk coverage, provenance and embedding inputs/normalization before
returning a ready snapshot. Index readiness does not activate the collection.

001_foundation is preserved with its original checksum. 002_general_indexing
removes semantic approval columns/constraints with a transactional forward upgrade;
retired annotations remain readable in legacy_annotations. The tables are:

| Table | Purpose |
|---|---|
| knowledge_collections | Trusted access domain, private/shared ownership, future active-revision/report slots |
| knowledge_revisions | Source manifest, embedding/pipeline fingerprints, staging state, optimistic version |
| knowledge_documents | Source hash/URI, descriptive metadata, explicit workflow authority |
| knowledge_sections | Extracted text/provenance and technical processing_status |
| knowledge_chunks | Bounded typed content, support/hint metadata, content hash, evidence vector |
| process_steps | Namespaced canonical process facts and links to an authoritative document |

Relationships use revision-qualified keys. Chunks require a technically ready source
section, not semantic approval. Section READY means extracted text is usable; it
does not publish a revision or prove chunking/embedding is complete. Canonical facts
retain explicit source authority. Authored hint records may carry levels 1-4 without
requiring a course step; ordinary documents need no authored hints. Only evidence
carries a finite nonzero vector. Models normalize vector components to pgvector's
single-precision representation. No ingestion classifier or instructor review exists.

Nullable publication fields prepare the schema for later work. No publication or
status-transition API is exposed yet. A newly created revision must be staging at
version zero; future publication/evaluation code must bind evidence to the revision's
version as well as its input fingerprint and prevent editing published material.

## Repository contract

The PostgreSQL and in-memory implementations share these operations:

```text
create_collection(collection, scope=trusted_write_scope)
get_collection(collection_id, trusted_access_scope)
create_revision(revision, scope=trusted_write_scope)
get_revision(revision_id, trusted_access_scope)
write_revision_data(revision_id, snapshot, scope=trusted_write_scope, expected_version=...)
get_revision_data(revision_id, trusted_access_scope)
```

Reads require an AccessScope whose domain/user identity has been resolved by
application code. Wrong domains, wrong private owners, and absent records return
None. AccessScope itself is not an authentication system; these contracts are not
public student endpoints. Runtime authentication and course resolution remain
separate prerequisites for later restricted retrieval.

A collection has one ownership/access policy; documents inherit it. An owner-private
source belongs in an owner-private collection, not mixed into a shared course
collection. These low-level reads intentionally support staged processing and evaluation;
future student retrieval must resolve the authorized active revision first.

Every mutation requires a KnowledgeWriteScope with can_manage_knowledge=True,
matching access domain and, for private collections, the matching owner. The grant
defaults to false. Read access alone does not grant editing permission. Construct
this context only from trusted operator configuration or authenticated application
authorization; never accept the grant/domain/owner from model or student input.
PostgreSQL checks policy inside the write transaction and locks the collection row
to keep the policy stable until the write commits. The operator import CLI explicitly
creates its editing context; no student editing endpoint exists.

Writer operations manage candidate knowledge. They revalidate typed records,
source-manifest matches, technical readiness, dimensions, and step/support references. Updating
data replaces a complete staging snapshot atomically and increments its version.
A stale expected_version raises KnowledgeConflict. Concurrent writers cannot both
overwrite the same version. PostgreSQL rollback preserves the old data/version even
if insertion fails after deletion has begun.
Rows use parameterized executemany batches of at most 128, inside that same
transaction. Failure after an earlier batch still rolls back all tables.

Returned records are independently reconstructed/deep-copied. Lists have deterministic
identity order; document/step presentation order should use the explicit provenance
and process fields. PostgreSQL multi-table reads use one repeatable-read, read-only
transaction so a snapshot cannot mix parts of concurrent writes.

The database reader login can read the writer-created tables through Checkpoint 1's
default privileges and cannot write them. Collection filtering is additionally
enforced by repository methods. Direct database credentials belong to trusted
backend/administrative processes; the reader role is not a separate database tenant
or a substitute for application authorization.

## Connections and disabled behavior

KnowledgeDatabaseConfig holds a SecretStr DSN, bounded pool size, acquisition/start
timeout, and statement timeout. Constructing KnowledgeDatabase performs no network
I/O. start()/connection() open it lazily; concurrent start calls are serialized.
close() is idempotent and terminal. Context managers close resources on exit.
Transaction cancellation is allowed to propagate through the driver's transaction
context. Database error messages exposed by this layer do not include DSNs or raw
SQL/source contents.

The migration command and tests explicitly use SelectorEventLoop on Windows,
as required by [Psycopg async operation](https://www.psycopg.org/psycopg3/docs/advanced/async.html).
No global backend event-loop policy was changed. Async pool behavior follows the
[driver's pool lifecycle](https://www.psycopg.org/psycopg3/docs/advanced/pool.html).
Student request factories/lifespan will own these resources only when that later
integration checkpoint is implemented; they do not construct a knowledge pool now.

## Source and cache interfaces

FileSourceStore implements SourceStore.put(bytes)/get(hash) for an operator-managed
root directory. It stores immutable content-addressed files under SHA-256 identifiers,
returns an opaque sha256 URI, validates source size and hashes, writes by atomic
replacement, and rejects path traversal. Reopening the adapter against the same root
can read existing sources. It never selects an agent using a filename.

The storage interface does not authorize student access itself: callers must resolve
the corresponding authorized document first. Tests use isolated filesystem roots;
the ingestion worker's connection to the managed Docker source volume is later work.
Actual PDF extraction and parser validation belong to Checkpoint 3.

KnowledgeCache defines opaque bytes get/set with a positive TTL. NoKnowledgeCache
always misses and stores nothing. The Redis adapter, namespaces, and shared-cache
authorization checks belong to Checkpoint 9; the running Redis service is unchanged.

The immutable capability registry maps process reasoning to Process Coach and AP+
execution to Navigator. Structured metadata is eligible for both. Scenario and
Orchestrator retrieval remain disabled, unknown agents fail closed, and no registry
rule references a PDF filename. Classification is still Checkpoint 4 work.

## Tests

From backend, run deterministic storage/model/source tests:

```powershell
uv run pytest tests/knowledge -q
```

Real database contract tests are opt-in:

```powershell
$env:FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION = '1'
uv run pytest tests/knowledge -q
```

On a POSIX shell use FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION=1 before the command.
This flag is separate from Checkpoint 1's infrastructure test flag, so repository
tests do not restart PostgreSQL. Each real test creates a unique scratch database,
bootstraps its extension/schema, applies migrations at dimension 3, opens writer/reader
pools, and closes/removes only its own scratch database afterward. It never empties
the local 768-dimensional knowledge schema or deletes application/source volumes.

Contract cases run against both implementations. Additional real cases verify
reader permissions, migration idempotence/drift detection, SQL foreign-key enforcement,
rollback after an injected mid-write failure, and pool cleanup. Normal test runs
skip database cases unless explicitly opted in; a skip is not integration evidence.
