# Scoped PostgreSQL retrieval

Checkpoint 6 provides uncached read-only search and exact metadata lookup. It is
available through KnowledgeRetrievalService and a disposable operator command.
The retrieval service itself does not decide when to search. Checkpoint 7 now adds
the [hybrid planning layer](knowledge-planning.md) and a reader-only operator above
it. Attaching evidence to specialists, publication and Redis remain later checkpoints.
The current HTTP learning workflow and student progress/guidance remain unchanged.

## Admission and access

Every operation requires a trusted AccessScope: course/access domain and, for private
collections, the owning internal user UUID. The caller must establish authorization;
a domain supplied by a student or document is not a grant. The operator CLI uses
trusted operator arguments. A future HTTP/graph caller must bind scope to authenticated
server-side course/ownership metadata, never model-generated request metadata.

Pass a collection ID and an optional revision ID. An omitted revision resolves only
the collection's active revision. It never silently selects the newest staging data.
The pilot collection still has active_revision_id=null: use its explicit ready
revision for operator evaluation. Automatic publication remains Checkpoint 9; an
explicit revision is not an access bypass and must belong to the requested collection.

Admission requires indexing_status=ready and eligible evidence. Missing, unready,
wrong-course and wrong-owner revisions all return the same unavailable outcome.
The revision/profile/version is pinned across model calls and SQL reads. An active
pointer change does not switch an in-flight request to a new snapshot. Each read
rechecks scope/readiness and a final check precedes returning evidence. Revocation
or incompatible mutation cannot return previously read evidence under stale access.

## Operator usage

Use the running PostgreSQL and Ollama services. From the repository root:

```powershell
docker compose --env-file backend/.env.knowledge `
  -f docker/compose/docker-compose.dev.yml `
  -f docker/compose/docker-compose.knowledge.yml `
  -f docker/compose/docker-compose.knowledge.dev.yml `
  --profile knowledge-tools run --rm --no-deps --build knowledge-retrieve `
  --collection 89c7a7b4-70d8-58d4-913b-7cde21bf3a40 `
  --revision ec469aea-bef2-5b57-b631-c30340092055 `
  --scope-domain course:ap40 `
  --query "Why must goods issue be posted before creating an invoice?" `
  --config /config/retrieval-defaults.json
```

For metadata, replace --query with --step-id AP40-02 --namespace ap40. This performs
no model calls. Optional namespace disambiguates canonical step IDs; a missing or
ambiguous ID returns no_evidence instead of guessing. Use canonical IDs for exact
lookup. Semantic/alias resolution belongs to the later planner/integration.

Optional search preferences: --current-step, --namespace, --phase and --purpose.
These influence relevance only; they never restrict documents to particular agents
or exclude other steps. Explicit cross-step queries remain supported.
--top-k is 1-20; --max-chars is 1-24000 and defaults to 8000. A smaller budget omits
whole evidence items instead of cutting quotations/citations. Metadata field omissions
are reported. The service request also supports selecting particular metadata fields.

Default output includes outcome, IDs, ranking diagnostics and citations without source
quotes. --include-evidence explicitly includes bounded content/context. No query
embedding/vector is exported. Exit codes: found=0, no_evidence=2, unavailable=3,
error/setup failure=1. These are evidence outcomes, not grades or student attempts.

The tool uses only fredi_knowledge_reader and protected reader credentials, has a
512 MiB/one-CPU container limit, and exits after use. It has no source-volume mount,
writer/admin password or Redis connection. No new migration is needed after the
existing 003_chunk_index. All SQL runs in short repeatable-read, read-only transactions;
model inference holds no SQL transaction open.

For host development, use python -m scripts.retrieve_knowledge from backend, with a
reachable --model-url and host config path. KNOWLEDGE_RETRIEVAL_DATABASE_URL can
supply a protected read DSN; otherwise the existing local reader credentials are used.
Never put credentials in CLI arguments. Windows entry points use SelectorEventLoop.

## Ranking and evidence

1. Admit and pin an authorized complete revision before any model work.
2. Inspect its exact embedding provider/digest/dimension, embed the question with the
   stored query prefix and truncate=false, then recheck model compatibility. Oversized
   queries produce query_too_long; they are not silently shortened.
3. PostgreSQL computes cosine and full-text scores inside the authorized revision.
   Identifier matches and current-step candidates supplement the bounded pools.
4. Fuse vector/lexical/identifier ranks with reciprocal-rank fusion, then apply small
   step/phase/purpose preferences. Explicit known identifiers take priority and receive
   cross-step coverage when the requested count/budget permits. Preferences cannot
   make an otherwise irrelevant passage eligible.
5. Verify selected chunk hashes, embedding-input identity/normalization, exact source
   slices and page fragments. Return quotations with document/revision/section/page
   provenance and grouped source bundles. Metadata fields have stable fact IDs and
   field/page citations; they do not invent chunk IDs.
6. Merge identical quotations while preserving source citations, remove fully contained
   overlapping text, retain distinct qualifications and enforce the serialized item
   budget. Sources/scores are not evidence that the question is completely answered.

The default [retrieval configuration](../knowledge/retrieval-defaults.json) uses 40
candidates per main channel, a 0.50 vector similarity floor or lexical support with
similarity at least 0.35, reciprocal-rank constant 60 and at most eight citations per
merged item. Known exact identifiers can qualify independently of similarity; a pure
unknown identifier does not retrieve a neighboring step as an answer. These are
transparent pilot relevance gates, not calibrated probabilities or answerability
guarantees. Wider per-specialist/held-out evaluation remains Checkpoint 11.

The vector baseline is exact search over a materialized scoped candidate universe.
It avoids approximate-index post-filter recall loss for small course/private subsets.
The existing HNSW/GIN indexes remain available, but this correctness baseline does
not claim approximate-index acceleration. Approximate search/reranking require later
measurements before replacing it. See [pgvector filtering](https://github.com/pgvector/pgvector#filtering)
and [PostgreSQL text-search controls](https://www.postgresql.org/docs/16/textsearch-controls.html).

Full-text parsing uses simple and websearch_to_tsquery with bound parameters;
SQL identifiers/query syntax never come from student text. No filename-based routing,
semantic ingestion labels or instructor review are involved.

## Outcomes and validation

| Status | Meaning |
|---|---|
| found | At least one cited item meets the gates and budget; sufficiency still needs checking |
| no_evidence | No eligible match, unknown/ambiguous metadata, or budget exhaustion |
| unavailable | No authorized ready admitted revision; no source/ownership details exposed |
| error | Storage, model compatibility, query-length or source-integrity problem |

Operational failures never masquerade as no evidence. Internal calls do not persist
messages, alter context or increment struggle counters. model_calls counts logical
query-embedding attempts, not provider inspection HTTP calls/retries. Missing/invalid
provenance produces an error, not a fabricated citation. Full-corpus integrity auditing
remains available through the existing complete snapshot reader.

Regression tests use the real reader role in disposable databases. The frozen
[pilot smoke cases](../knowledge/retrieval-smoke.json) additionally exercise the actual
indexed PDFs and model: natural-language concepts/navigation, dependencies, exact and
cross-step identifiers, an unknown identifier and an unrelated question. This small
smoke is not a general retrieval-quality benchmark. See the Checkpoint 6 report.
