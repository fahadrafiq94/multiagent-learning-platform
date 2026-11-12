# Optional Redis knowledge cache

Checkpoint 10 caches validated source hydration and SQL search candidates. PostgreSQL
remains authoritative. Originals, indexed snapshots, student messages and learning
state remain in their existing durable stores. Redis contains disposable acceleration,
never the only copy of a message, state record or document.

## Cached data and isolation

| Payload | Scope and identity |
|---|---|
| Individual chunk/citation bundle | Domain, collection, private owner if applicable, revision/version/fingerprint, chunk ID |
| Individual metadata fact | Same source scope plus structured namespace, step ID and field |
| SQL hybrid search candidates | Same revision identity plus caller user, query, purpose, step, namespace, phase, references, vector, candidate limit and SQL version |

Immutable shared course chunks/facts can be reused across students after each caller's
database admission. Private content stays owner-scoped. Candidate results remain
user-scoped even for shared collections. The model's answer, guidance decisions,
conversation text, planner decisions and query embeddings are not cached.

Keys use fredi:knw:v2: plus a SHA-256 digest of structured identity. Student text,
raw query/purpose and user UUIDs do not appear in keys. Values hold source passages,
bounded citation metadata or candidate IDs/scores, so treat Redis access as sensitive.
There are no full-PDF/page dumps or student chat archives in this cache.

Values are authenticated with HMAC-SHA256, bound to their exact key. Runtime signatures
derive from the knowledge-reader password, independently of Redis storage credentials;
native/Docker DSN aliases with that same password can share entries. Passwordless
database connections use a process-local random signing key, so reuse across process
restarts is not assured for that deployment mode. Normal password rotation invalidates
old signatures and causes refill. Signatures authenticate payloads; they do not encrypt
source text. The existing private network/password controls still apply.

Read paths always recheck PostgreSQL scope, readiness and pin identity before cache
access and again after returning data from it. A hit cannot grant access or bypass a
revoked owner/course grant. Typed decoding checks source IDs, hashes, expected evidence
identity and metadata field/text provenance. Invalid signatures, wrong-key copies,
oversized data, malformed JSON and invalid decoded identities are evicted/refilled.
Ready snapshots remain immutable through the repository. Normal changes publish a
new revision; the key also includes version/fingerprint as a defensive boundary.
Do not directly edit ready source rows with SQL and expect cache consistency.

## Configuration

The feature defaults to disabled:

| Setting | Default / purpose |
|---|---|
| KNOWLEDGE_CACHE_ENABLED | false; independent from RAG_ENABLED |
| KNOWLEDGE_CACHE_REDIS_HOST | Required when enabled |
| KNOWLEDGE_CACHE_REDIS_PORT | 6379 inside Docker; local published service uses 16379 |
| KNOWLEDGE_CACHE_REDIS_PASSWORD | Protected Redis credential |
| KNOWLEDGE_CACHE_TTL_SECONDS | 3600; accepted runtime range 30-86400 |
| KNOWLEDGE_CACHE_TTL_JITTER_SECONDS | 300; positive random addition from zero through this limit |
| KNOWLEDGE_CACHE_OPERATION_TIMEOUT_SECONDS | 0.1 seconds per Redis operation, including retries |
| KNOWLEDGE_CACHE_MAX_PAYLOAD_BYTES | 262144 source payload bytes; signature adds 32 bytes |

The optional docker-compose.rag.yml overlay supplies redis:6379 and the protected
existing KNOWLEDGE_REDIS_PASSWORD. Enable KNOWLEDGE_CACHE_ENABLED explicitly if that
RAG runtime is deployed; the overlay itself does not enable RAG. The current running
backend keeps both flags disabled. It does not require Redis health before startup:
an outage must remove the speedup rather than prevent otherwise valid retrieval.
No existing secret file needs to be rewritten or copied into source configuration.
The same overlay supplies cache connection settings to the knowledge-retrieve
operator image for an explicit scripts.check_knowledge_cache entry-point override.
Its packaged Docker check reused native-process entries successfully, confirming
consistent signatures across PostgreSQL host aliases. No student feature was enabled.

Only the RAG reader runtime and the explicit operator cache check construct this
backend. Existing low-level retrieval/planning commands retain their uncached default.
Use the reader runtime's create_knowledge_cache factory for other trusted integrations
so signing and settings remain consistent. No cache resources are opened by disabled
student RAG. Lifespan closes its cache before the reader database.

## Outage, fills and updates

Redis errors, authentication failure, operation timeout, eviction, expiry or lost
server contents become misses/skipped stores. PostgreSQL errors remain authoritative
errors; cached sources never conceal unavailable database authorization.

Within a reader process, concurrent misses on the same enabled-cache key share a
fill. Waiters wait at most two seconds, then can fetch independently. Fetch/store
failure or cancellation releases the in-flight registry. This is not a distributed
lock: separate processes can perform duplicate cold fills safely. Disabled cache
uses the direct PostgreSQL path and does not introduce fill waiting.

Metadata admission releases its database connection before cache waits and source
hydration; a pool of one is supported. Cold chunk hydration uses individual primary-key
reads to maximize reuse between different top-k result sets. This can cost more cold
queries than the earlier batch read. Warm hits avoid hydration/ranking SQL, not the
database permission checks. Query embedding and planner/model latency remain.

Publishing a new active revision selects a separate keyspace. Existing admitted
requests remain pinned to their immutable revision under the established contract;
trusted explicit revision bindings also remain pinned until changed. Old entries expire
naturally. Access revocation is checked immediately through PostgreSQL and does not
wait for TTL, although inaccessible bytes may remain in the restricted cache until expiry.

The existing Redis container is an ephemeral, memory-bounded cache. Restarting it loses
cached entries and triggers refills. Back up PostgreSQL/source storage, not cache keys.
Do not use FLUSHDB/FLUSHALL on a shared service to disable this feature. Set the flag
false and restart the reader runtime; saved history and source records remain intact.

## Operator verification without enabling student RAG

From backend, using the existing protected native credentials:

```powershell
.venv/Scripts/python.exe -m scripts.check_knowledge_cache --local --collection 89c7a7b4-70d8-58d4-913b-7cde21bf3a40 --scope-domain course:ap40 --namespace ap40 --step-id AP40-02
```

This admits the current published revision using the knowledge reader, compares two
actual chunk hydrations and an optional named fact with PostgreSQL, repeats the reads,
and prints only revision/count/equivalence statistics. It writes disposable Redis
entries and no PostgreSQL data. --revision supports trusted ready-revision evaluation;
--owner-user-id supports the established private-owner scope. Without --local, supply
protected cache settings through the deployment environment. It never turns on a
student feature, generates an answer or imports/retries PDFs.

Exit 0 indicates an equivalent warm hit, 2 indicates equivalent reads without observed
hits (for example unavailable Redis), and 1 indicates a validation/operation failure.
Repository CacheStats exposes cumulative hits/misses/fills/waits for operator checks;
per-request HTTP research telemetry was not expanded in this checkpoint. A fill count
records successful authoritative fetching, even when an unavailable cache skips storing.

## Tests and remaining evaluation

Set the existing two PostgreSQL integration flags plus
FREDI_KNOWLEDGE_REDIS_INTEGRATION=1 for real service coverage. Tests use disposable
PostgreSQL databases and unique Redis test namespaces/known keys. One outage/restart
case starts and removes its own uniquely named fredi-cp10-* Redis container with the
already installed Redis image; it never stops the user's Redis or flushes shared keys.
Without the Redis integration flag, those tests skip explicitly.

Tests cover uncached equivalence, shared hydration/private isolation, scoped candidates,
revocation during hits, corruption, expiry, publication/version changes, concurrency,
single-connection metadata, lifecycle and signed chat/history under Redis authentication
failure. The live course smoke additionally verifies separate-process reuse.
See the [checkpoint report](../checkpoints/sprint-5-checkpoint-10.md) for actual results.

This checkpoint establishes cache correctness and operational fallback. It does not
certify semantic retrieval quality, planner decisions, answer completeness or concurrency
capacity. The prior model-quality/latency gate remains open. Held-out evaluation and
remaining release/operations work are Checkpoint 11; vLLM remains outside this sprint.
