# Application persistence: identities, context and conversations

Checkpoint 4B supplies PostgreSQL repositories. Checkpoint 4C connects them to HTTP
chat/history, signed local-pilot identity, response replay and safe restart recovery.
The application Compose overlay explicitly enables PostgreSQL. Base configuration
retains memory for development/tests; database failures never fall back to memory.

## Local setup

Run as the application user from backend, with the existing PostgreSQL container up:

```powershell
.venv/Scripts/python.exe -m scripts.bootstrap_application
.venv/Scripts/python.exe -m scripts.migrate_application
```

Bootstrap adds APPLICATION_MIGRATOR_PASSWORD, APPLICATION_RUNTIME_PASSWORD and
APPLICATION_PILOT_AUTH_SECRET to the
protected, ignored .env.knowledge file. It retains every existing password. A temporary
file is protected before secret data is written and replaces the original atomically.
POSIX mode/Windows owner and ACL protection reuse the established credential helper.
Concurrent bootstrap is rejected by a short-lived exclusive lock. If a process is
killed, verify it is no longer running before removing its stale bootstrap lock.
No password or resolved connection string is printed. Do not copy example placeholders
as actual credentials. Repeating successful bootstrap does not rotate passwords.

The admin bootstrap provisions fredi_app_migrator and fredi_app_runtime, CONNECT
privileges and the application schema in the existing fredi_knowledge database.
It rejects unexpected elevated roles/memberships or schema ownership. It never
deletes or modifies knowledge records. An existing role's password is not reset.

The migration command uses the dedicated migrator and a separate application
schema_migrations ledger. It requires neither pgvector nor an embedding dimension.
For non-local migration, supply APPLICATION_MIGRATIONS_DATABASE_URL through protected
process configuration. ApplicationDatabase accepts a SecretStr DSN; the future runtime
configuration uses APPLICATION_DATABASE_URL. Never put credentials in shell arguments.

From the repository root, activate durable HTTP chat after migrations:

```powershell
docker compose --env-file backend/.env.knowledge `
  -f docker/compose/docker-compose.dev.yml `
  -f docker/compose/docker-compose.knowledge.yml `
  -f docker/compose/docker-compose.knowledge.dev.yml `
  -f docker/compose/docker-compose.application.yml `
  --profile knowledge up -d --build backend postgres redis ollama
```

This retains named volumes and starts one backend worker, without reload. Production
uses its base Compose file and omits knowledge.dev.yml. Use deployment-specific
secrets. PostgreSQL must already have bootstrap/migrations applied: runtime never
creates tables or changes to memory when configuration, schema or connectivity fails.
`/health` is liveness; `/ready` checks runtime, PostgreSQL and worker ownership.
Container health requires readiness and model-provider health.

For native Windows, inject the DSN/signing secret through protected environment
configuration and run `python -m scripts.run_backend` from backend. This entry point
uses SelectorEventLoop. Do not use multiple workers or reload with durable mode.

## Pilot identity and HTTP usage

From backend, provision a student (idempotent; creates initial user context):

```powershell
.venv/Scripts/python.exe -m scripts.application_admin create-user --student-id K12345
# Use the internal UUID printed above, not the institutional student ID:
.venv/Scripts/python.exe -m scripts.application_admin token --user-id <internal-user-uuid>
```

Only the explicit token command prints a credential. Keep it private and send it
as `Authorization: Bearer <token>`. Tokens expire after one hour by default; the
operator can set --lifetime up to 86400 seconds. Signatures bind an internal UUID,
expiry and pilot audience. Every request checks the persisted identity is active;
disabled/missing users and invalid/expired tokens receive 401. No public registration
or client-supplied identity header establishes access. University SSO remains later.
Use HTTPS before sending credentials outside localhost. Signing-key rotation revokes
existing tokens; update protected deployment configuration and restart the backend.

| Operation | Endpoint |
|---|---|
| Create chat | POST /conversations, JSON {} or {"title":"AP+"} |
| List own chats | GET /conversations?limit=50&offset=0 |
| Reopen own chat/context | GET /conversations/{session_id} |
| Read chronological history | GET /conversations/{session_id}/messages?limit=50&offset=0 |
| Inspect own request outcome | GET /conversations/{session_id}/turns/{request_id} |
| Send question | POST /orchestration/invoke |

Invoke JSON retains student_id, session_id, user_message and optional metadata, and
requires a client-generated UUID request_id in durable mode. The student_id must
match the signed principal; storage scope comes from its internal UUID, never that
request field. A foreign session is 404, an identity mismatch 403. Pagination limits
are 1-100 with nonnegative offsets. Messages retain the existing 4000-character
bound. Oversized input is 422; invalid/oversized model output fails the turn explicitly
without truncation or a false success. History includes typed source_references;
these are empty until later retrieval/context checkpoints generate real citations.

## Database roles and tables

| Table | Purpose |
|---|---|
| identities | Stable user ID, institutional ID, status and timestamps |
| external_identities | Unique provider/instance/external-ID mappings |
| user_contexts | Scenario, profile, progress, guidance state and adaptations |
| conversation_sessions | Owner, title, status and activity |
| session_contexts | Step/objective/blocker, onboarding draft and summary |
| conversation_messages | Immutable message archive, role, time, route and trace |
| turns | Request identity, status, message links and bounded failure code |
| schema_migrations | Application-only checksum ledger |
| runtime_owner | Current worker token/PID; fences transactions after ownership loss |

The migrator owns schema/tables. Runtime can perform necessary reads/writes but
cannot migrate, create/drop tables, read the migration ledger, or update/delete
archived messages. External mapping replacement is transactional. SQL foreign keys
bind contexts, turns and messages to the same user/session pair.

Repositories require server-resolved internal user IDs. A database login is trusted
backend access, not per-student authentication. History endpoints need the trusted
identity boundary implemented in 4C; knowing a student_id is not authentication.
Knowledge and application roles are separate. Future runtime RAG uses the knowledge
reader independently rather than granting application runtime knowledge-writer access.

## Repository injection

```python
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.repositories.postgres import create_postgres_repositories

# dsn comes from protected runtime configuration, never from a student request.
database = ApplicationDatabase(ApplicationDatabaseConfig(dsn=dsn))
repositories = create_postgres_repositories(database)
# Pass repositories to build_orchestration_runtime(..., repositories=repositories).
# Open/close database explicitly with application lifecycle; construction does no I/O.
```

RepositoryBundle exposes the five repositories plus optional typed turn storage and
atomic session initialization. In-memory doubles use None for those SQL capabilities.
Existing services and default runtime behavior are preserved. Returned values are
reconstructed; writes revalidate nested ownership. User/session context saves use
expected_version and row locks so a stale writer cannot overwrite a newer version.
Identity/session metadata updates retain the existing last-writer contract; those
models do not have context-style version fields.

History is ordered by created_at with a database-assigned sequence as the stable
tie breaker. Sequence gaps after rollback are normal. Existing offset/limit contracts
remain; HTTP pagination is bounded by 4C. Full archived history is never implicitly
sent to a model: ContextService still selects bounded messages and summaries.

## Retry-safe turns

The existing ConversationRepository.append method retains its duplicate-message
error contract. For retry-safe requests use repositories.turns:

1. begin(user_id, session_id, request_id, content) atomically records the processing
   turn and one student message. Message IDs are derived from the scoped request.
2. If the same request/body is admitted again, created=false returns its existing
   state. A different body under that request ID is rejected. Another processing
   turn in the same conversation is rejected; other conversations remain independent.
3. Perform model calls outside a database transaction.
4. complete(..., response) atomically stores one assistant message and marks the
   turn completed. Repeating the same completion reuses it; conflicting responses
   are rejected. Message IDs/times/traces may differ on retry without duplicating it.
5. stop(..., status=failed/interrupted, error_code=...) preserves an incomplete turn
   explicitly. Use a bounded code, never raw exception text. Terminal states cannot
   be overwritten. A deliberate new attempt uses a new request ID.

The HTTP path uses this store and suppresses the graph's ordinary message append
operations for managed turns. complete stores the assistant message and full replay
response in the same transaction. A completed retry returns that response unchanged,
without generation or another student-state update. A processing retry is 409; poll
the status endpoint. A changed body under an existing ID is 409. Failed/interrupted
retries are 409; inspect partial history/context before deliberately using a new ID.
Failures and cancellation call stop. Storage/workflow failure returns sanitized 503.
If commit outcome is uncertain, retry the same ID to discover the durable outcome;
never blindly resubmit with a new ID. A database outage can prevent stop, leaving
processing state for recovery after restoration and an orderly backend restart.

## Recovery and capacity

The worker holds a dedicated PostgreSQL session advisory lock in autocommit mode,
outside a transaction. A second worker fails startup. After acquiring ownership,
startup atomically replaces runtime_owner and interrupts old processing turns with
worker_restarted. Completed/failed/interrupted turns are preserved. Each runtime
transaction checks ownership under a shared row lock; a process that loses the lock
or is superseded cannot continue saving student state or messages. Model calls hold
no SQL transaction or session-row lock. No elapsed-time reaper guesses whether a
slow model call is abandoned. Stop the backend before manual recovery:

```powershell
.venv/Scripts/python.exe -m scripts.application_admin recover
```

Recovery refuses to run while a backend owns the lock. It preserves user messages,
marks incomplete work explicitly and releases the conversation for a new attempt.
State updates committed before failure remain committed; recovery does not roll them
back or regenerate a response. This is not exactly-once model generation or an atomic
transaction across the whole learning workflow. A new deliberate attempt is a new
student turn. Older 4B-only completed turns without replay payloads cannot be replayed
automatically; their message archive remains readable.

Identity/session metadata updates are intentionally last-write-wins; contexts use
optimistic versions. Session row locks serialize short writes within one conversation,
not across students. Measure contention before changing this. Knowledge and application
schemas share local fredi_knowledge and server resources but retain separate roles.
Application pool defaults to four connections (configurable 1-32) plus one dedicated
ownership connection. Budget knowledge pools, migrations and operators against the
same PostgreSQL server limit. Separate databases on that server do not isolate capacity.
Multi-worker/replica serving is blocked until ownership/recovery is redesigned/tested.

## Verification

```powershell
$env:FREDI_APPLICATION_REPOSITORY_INTEGRATION = "1"
.venv/Scripts/python.exe -m pytest tests/persistence -q
```

Real tests require the local bootstrap and application-user access to credentials.
They create uniquely named scratch databases, then close pools and remove only
those databases. No test student records are inserted into the actual application
database. They also run common contracts against the original in-memory adapters.

Coverage includes ownership, normalized/unique identities, rollback, nested context,
version conflicts, concurrent retries, archive immutability, cancellation, separate
roles, migration drift and reopening through existing services with fresh pools.
Actual model generation is not part of this storage checkpoint.

Psycopg on Windows uses SelectorEventLoop in operator commands/tests. Its pool
connection contexts commit/roll back short transactions and return connections;
see the [Psycopg pool documentation](https://www.psycopg.org/psycopg3/docs/advanced/pool.html).
