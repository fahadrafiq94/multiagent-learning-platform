# Knowledge infrastructure: PostgreSQL/pgvector and Redis

Checkpoint 1 provides real infrastructure, not application retrieval. The backend
does not yet read knowledge or use Redis, and has no new startup dependency on them.
Keep the existing backend/Ollama stack running for the full smoke check.

## Development setup

From the repository root, using Python 3.12+ (or the existing backend virtual environment):

```powershell
python backend/scripts/knowledge_infra.py init
python backend/scripts/knowledge_infra.py config
python backend/scripts/knowledge_infra.py start
python backend/scripts/knowledge_infra.py status
python backend/scripts/knowledge_infra.py smoke
```

The helper uses the existing development Compose file plus two explicit knowledge
overlays. It starts only postgres and redis with --no-deps, without rebuilding or
recreating backend/Ollama. Docker Desktop must be running. Initial image pulls need
network access; subsequent use reuses local images.

Equivalent Make targets: knowledge-init, knowledge-config, knowledge-up,
knowledge-status, knowledge-smoke, and knowledge-smoke-persistence.

The init command generates four independent random passwords in the Git-ignored
backend/.env.knowledge file. Repeating it preserves existing values. Do not commit
that file or share resolved Compose configuration/logs containing credentials.
An editable template is provided at backend/.env.knowledge.example.

Run init as the Windows account that operates the application. It applies and
verifies a protected DACL allowing only that account, SYSTEM and administrators,
and sets the file owner to that account. New files are protected while still empty,
before credentials are written. Existing values are preserved while permissions are
repaired. POSIX hosts use mode 0600. Protection failure stops initialization; there
is no fallback to inherited Windows permissions. This requires ownership/ACL rights;
an agent sandbox may need execution as the actual application user. Existing files
created by another account may require that owner to grant the application user
ACL/ownership rights once. Do not grant broad user groups access.

Images are pinned to explicit release/distribution tags:

- pgvector/pgvector:0.8.7-pg16-bookworm.
- redis:8.2.11-bookworm.

These tags are listed by the [pgvector project](https://github.com/pgvector/pgvector#docker)
and [official Redis image](https://hub.docker.com/_/redis). A version change needs
its own compatibility and data-upgrade review; do not use latest for these services.

## Connections and storage

| Service | Inside the Compose network | Development host endpoint |
|---|---|---|
| PostgreSQL/pgvector | postgres:5432 | 127.0.0.1:15432 |
| Redis | redis:6379 | 127.0.0.1:16379 |

Only the development overlay publishes ports, bound to loopback. Change
KNOWLEDGE_POSTGRES_PORT or KNOWLEDGE_REDIS_PORT in the local knowledge environment
file if a host port is occupied. Internal service ports stay unchanged.

The database is fredi_knowledge. Roles are:

- fredi_knowledge_admin: bootstrap/database administration; not an application login.
- fredi_knowledge_writer: owns the knowledge schema; no superuser, database-create,
  or role-create privilege. Intended for later migrations/ingestion.
- fredi_knowledge_reader: connects and selects writer-created knowledge tables
  through default privileges, without insert/update/delete/create permissions.

The vector extension is enabled during empty-volume bootstrap. Checkpoint 2 adds
the [versioned schema migrations and repository implementations](knowledge-repositories.md).
These are not connected to student requests yet and do not migrate student/session
data from the existing in-memory repositories.

The named knowledge-postgres-data volume holds database files. The separate
knowledge-sources volume is reserved for managed source PDFs. A one-off
knowledge-volume-check service mounts it for verification; it does not stay running
and will later be replaced/supplemented by the ingestion worker's volume access.
Backups must cover both durable volumes and be restore-tested.

Redis uses authentication, a 128 MB cache limit, and allkeys-lru eviction, with
snapshot/AOF persistence disabled. It is disposable; database/source storage are
authoritative. Redis credentials come from the separate local environment file.

Future connection strings use the reader/writer passwords from that file, for example
postgresql://fredi_knowledge_reader:<reader-password>@postgres:5432/fredi_knowledge
and redis://:<redis-password>@redis:6379/0. These are placeholders, not working secrets.
No backend connection strings are injected or enabled in this checkpoint.

## Real-service verification

The smoke command uses the container-supplied psql and redis-cli clients, so no
native PostgreSQL/Redis tools or new Python database dependencies are needed yet.
It checks:

- Actual vector extension and known three-dimensional cosine ordering.
- Password-authenticated writer/reader connections over TCP and rejection of a reader write.
- Authenticated Redis PING, unauthenticated rejection, and expiration of a test key.
- Backend-to-postgres/redis DNS and TCP reachability on the shared Compose network.
- Source-volume content surviving between two independent one-off containers.

Scratch database/table/key/file names use unique generated IDs. Cleanup removes
only those objects. The command does not flush Redis, drop the knowledge database,
delete a volume, or modify application tables.

When rebuilding only the backend with the base Compose file, Docker can label the
knowledge containers as orphans because their optional overlay is not included in
that command. They are intentional services; do not use --remove-orphans to clear
that warning. Manage them with the documented knowledge overlay/helper instead.

To additionally verify database data across restart:

```powershell
python backend/scripts/knowledge_infra.py smoke --restart-postgres
```

This explicitly restarts only the new PostgreSQL service while its test record
exists, verifies it afterward, and removes the scratch data. Run before knowledge
traffic is enabled, not during an active ingestion or production request.

Run the opt-in automated integration test from backend:

```powershell
$env:FREDI_KNOWLEDGE_INTEGRATION = '1'
.venv/Scripts/python.exe -m pytest tests/integration/test_knowledge_infrastructure.py -q -s
```

The standard test suite skips this one test when that flag is absent. It requires
the real knowledge services plus backend and performs the PostgreSQL restart check.
Unit tests verify credential preservation, safe scratch identifiers, and sanitized
command failures without needing Docker.

## Operations and first-volume initialization

Do not run down -v to troubleshoot; that deletes durable volumes. Start/status and
the smoke helper deliberately have no data-reset operation. Existing backend and
Ollama volumes remain separate and unchanged.

PostgreSQL's init script runs only on a new empty database volume. Changing passwords
in .env.knowledge later does not rotate existing role passwords. Rotate them through
an explicit database administration operation and update configuration together.
If first initialization fails, inspect it locally and recover deliberately; do not
delete an existing volume automatically or silently rerun partially applied bootstrap.

The helper sanitizes external command failures rather than printing configuration or
raw database errors. Use Docker Desktop/service logs locally for diagnosis. Keep
credentials out of screenshots, issue reports, and commits.

## Production definition

The shared docker/compose/docker-compose.knowledge.yml overlay contains no host
ports. Use it with the production base and deployment-specific credentials; omit
docker-compose.knowledge.dev.yml. For example, after configuring that deployment:

```powershell
docker compose --env-file backend/.env.knowledge -f docker/compose/docker-compose.prod.yml -f docker/compose/docker-compose.knowledge.yml --profile knowledge up -d --no-deps --wait postgres redis
```

Use the same Compose project/network as that deployment's backend. Provision distinct
credentials and volumes per deployment. The helper's start/smoke commands deliberately
target development; production rollout and backup/restore verification remain explicit
later release work. No production containers were started by Checkpoint 1.

## Revised Sprint 5 scope

General indexing uses 001_foundation plus the forward 002_general_indexing migration.
4B provides separate application-schema repositories/roles; see
[application persistence](application-persistence.md). Runtime activation/history APIs
follow in 4C; default HTTP data is still in memory until then. Redis application
caching is included later in this sprint. vLLM is excluded. See the approved
[execution roadmap](../plans/sprint-5-rag-execution-roadmap.md).
