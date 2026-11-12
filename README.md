# FREDi – Multi-Agent Learning Platform

## Overview

FREDi is a research-oriented multi-agent learning environment in which
students work through an AP+ ERP business-process scenario with
specialized AI support. FREDi is **not a generic chatbot**: it supports
business-scenario understanding, business-process reasoning, AP+
navigation when the student is blocked, and graduated guidance instead of
immediate answers.

The project is developed as a production-quality software system while
serving as the implementation platform for academic research in Agentic
AI, Multi-Agent Systems, Retrieval-Augmented Generation (RAG), and Large
Language Models.

Development follows an incremental sprint-based methodology, where each
sprint produces a stable, tested, and deployable milestone.

---

# Project Goals

* Build a modular multi-agent learning environment
* Develop production-quality backend architecture
* Support local LLM inference through a model abstraction layer
* Keep LLM context bounded, observable, and multi-user-safe
* Persist student/session state outside the agents
* Enable future deployment on university infrastructure
* Provide a reproducible platform for AI research and experimentation
* Incorporate evaluation frameworks for both agents and RAG pipelines

---

# Current Status

**Version:** v0.5.0

**Sprint 4 — Context Engineering is complete.** All substeps (4A
foundation, 4B.1–4B.11) are implemented, tested, and wired into the
runtime. The Sprint 4 completion gate (security review, release notes,
README sync, Docker validation) is in progress on `feature/sprint-4`.

Implemented through Sprint 4:

* **Sprint 0–1** — FastAPI backend, configuration, structured logging,
  Docker, CI, model-service abstraction with provider independence
* **Sprint 2** — specialist agent foundations (Scenario, Process Coach,
  AP+ Navigator) with structured agent interfaces
* **Sprint 3** — semantic multi-agent orchestration with LangGraph
  (LLM Orchestrator with deterministic keyword fallback); FREDi does not
  rely on a static routing keyword list
* **Sprint 4** — identity and multi-user isolation, repository
  contracts with in-memory persistence (locks, optimistic versioning,
  deep copies), conversation/session state, ScenarioCard onboarding
  lifecycle, bounded `AgentContext` via ContextService, controlled
  LLM-proposed state updates, context budgeting, incremental session
  summaries, deterministic guidance policy, per-request observability
  events, and deterministic multi-user/concurrency validation (584
  tests green)

Release notes per sprint live in [`docs/releases/`](docs/releases/);
sprint plans in [`docs/plans/`](docs/plans/).

---

# Architecture

```text
HTTP request (POST /orchestration/invoke)
        |
        v
resolve student identity -> user_id
        |
        v
persist user message (full conversation archive)
        |
        v
ContextService.build()  -- bounded AgentContext:
        |                  essential state + recent history suffix
        |                  (+ session summary, guidance level)
        v
needs scenario onboarding?
   | yes                    | no
   v                        v
scenario onboarding     OrchestratorAgent
workflow (Scenario        (semantic routing,
Agent: extract /           deterministic fallback)
review / merge)                |
   +----------------------+----+
   |            |         |
   v            v         v
Scenario    Process   AP+
Agent       Coach     Navigator
                 (stateless, shared across users)
        |
        v
SpecialistAgentOutput: content
                      state_update_proposal  -> validated +
                      response_assessment        version-persisted
        |
        v
finalize_response -> HTTP response
(+ per-request observability event, best-effort)
```

Core principles:

* **Agents are stateless and reusable** — persistent state lives in
  repositories and services; the same agent instances serve all
  concurrent students.
* **Memory ≠ State ≠ Knowledge** — conversation archive, structured
  session/user state, and (later) RAG knowledge are separate.
* **Full archive, bounded prompt** — only a token-budgeted recent
  suffix (plus summary) reaches the model; the archive is never
  truncated.
* **LLMs propose, application code persists** — state updates are typed,
  allow-listed per agent role, and version-checked.
* **Provider independence** — Ollama today, vLLM later, behind the same
  model-service abstraction; context and state layers never depend on
  the provider.
* **User/session isolation is non-negotiable** — every state operation
  is keyed by `(user_id, session_id)` with ownership validation, and
  concurrency behavior is validated by deterministic barrier-forced
  tests with canary leakage checks.

---

# Technology Stack

## Backend

* Python 3.12+, FastAPI, Pydantic v2, Uvicorn
* LangGraph orchestration
* Pydantic Settings + `.env`

## Model layer

* Ollama (local development, current)
* vLLM (planned serving/batching, provider-agnostic boundary ready)

## Quality and operations

* uv (environment management)
* Ruff (formatting/linting), mypy (type checks)
* pytest / pytest-asyncio (concurrency, PostgreSQL/Redis isolation and RAG coverage;
  measured results in the Checkpoint 11 report)
* Docker + Docker Compose (dev and prod configs)
* GitHub Actions CI

## Persistence and knowledge

* PostgreSQL + pgvector (durable conversations/context and authorized knowledge retrieval)
* Optional Redis knowledge cache; database/source backups and scratch restore verification

## Planned (later sprints)

* Custom frontend (after backend behavior is stable)

---

# Repository Structure

```text
multiagent-learning-platform/
├── backend/
│   ├── app/
│   │   ├── agents/          specialist + orchestrator agents, schemas, policies
│   │   ├── api/             FastAPI routes (orchestration, model, health)
│   │   ├── config/          Pydantic settings
│   │   ├── context/         context schemas, budgeting, summaries, state updates
│   │   ├── domain/          guidance level + policy
│   │   ├── identity/        user identity schemas
│   │   ├── logging/         logging configuration
│   │   ├── observability/   per-request event schema + mapper
│   │   ├── orchestration/   LangGraph graph, nodes, service, factory
│   │   ├── repositories/    repository contracts + in-memory implementation
│   │   └── services/        conversation, context, state, summary, onboarding, model
│   ├── tests/               unit, service, orchestration, integration, concurrency
│   ├── scripts/             local smoke scripts (routing, concurrency, behavior)
│   ├── pyproject.toml
│   └── uv.lock
├── docker/
│   ├── backend/Dockerfile
│   └── compose/             docker-compose.dev.yml, docker-compose.prod.yml
├── docs/
│   ├── releases/            per-sprint release notes (v0.1.0 … v0.5.0)
│   ├── plans/               sprint implementation plans
│   └── architecture/
└── .github/workflows/       CI
```

---

# Getting Started

## Local development

```bash
cd backend
cp .env.example .env          # configure model/provider settings
uv sync
uv run uvicorn app.main:app --reload
```

Run the test suite:

```bash
uv run pytest -q
uv run ruff format .
uv run ruff check .
uv run mypy app tests
```

## Docker

```bash
# development
docker compose -f docker/compose/docker-compose.dev.yml up --build

# production
docker compose -f docker/compose/docker-compose.prod.yml up --build
```

---

## Optional knowledge infrastructure and general indexing (Sprint 5)

Real PostgreSQL/pgvector and Redis services are available through optional Compose
overlays. From the repository root:

```bash
python backend/scripts/knowledge_infra.py init
python backend/scripts/knowledge_infra.py start
python backend/scripts/knowledge_infra.py smoke
```

Credentials are generated in a Git-ignored local file. PostgreSQL/source storage
use persistent volumes; Redis is disposable. The existing backend and Ollama
containers are not recreated by this setup helper. Typed knowledge models, versioned
SQL migrations, PostgreSQL/in-memory repositories, and source-storage interfaces are
implemented. Student-facing retrieval integration and Redis caching remain pending.

Setup, real-service tests, development ports, and production definitions are
documented in [the knowledge infrastructure runbook](docs/operations/knowledge-infrastructure.md).
Migrations, storage contracts, and repository test commands are documented in
[the knowledge repository runbook](docs/operations/knowledge-repositories.md).
Registered PDFs can now be imported into staging with durable page/section
provenance. See [the import runbook](docs/operations/knowledge-imports.md).
Sprint 5 now follows the approved [general RAG and persistence roadmap](docs/plans/sprint-5-rag-execution-roadmap.md).
The obsolete classifier/instructor-review workflow has been removed in
[Checkpoint 4A](docs/checkpoints/sprint-5-checkpoint-4a.md). Automatic
indexing, hybrid request-time retrieval, PostgreSQL student/context/history storage
and real Redis caching are in scope; vLLM is excluded. Current stored content is
now indexed in a separate complete revision; specialist context integration is opt-in
and remains disabled in the running stack pending model-quality acceptance.

[Checkpoint 4B](docs/checkpoints/sprint-5-checkpoint-4b.md) adds migrated PostgreSQL
repositories for identities, student/session context and conversation history, plus
retry-safe turn storage. [Checkpoint 4C](docs/checkpoints/sprint-5-checkpoint-4c.md)
connects durable HTTP chat through the application Compose overlay, with signed pilot
credentials, owner-scoped history APIs, response replay and safe restart recovery.
The base configuration retains explicit memory mode for tests/development; database
failures never switch storage modes. See the [persistence runbook](docs/operations/application-persistence.md).
Earlier sprint behavior and adaptive guidance are preserved.

[Checkpoint 5](docs/checkpoints/sprint-5-checkpoint-5.md) adds source-linked contextual
chunks, optional structural step facts, real pinned embeddings and PostgreSQL
full-text/vector indexes. The initial three PDFs produced 264 chunks and 40 step
records. Indexing is repeatable without re-embedding unchanged input. Subsequent
operator updates published the revised collection; opt-in specialist integration is available. See the
[indexing runbook](docs/operations/knowledge-indexing.md).

[Checkpoint 6](docs/checkpoints/sprint-5-checkpoint-6.md) provides read-only scoped
metadata and vector/full-text retrieval, source citations, bounded evidence and
distinct no-evidence/error outcomes. The operator tool uses the knowledge reader
role; see the [retrieval runbook](docs/operations/knowledge-retrieval.md). Student-facing
specialist context integration is provided by opt-in Checkpoint 8.

[Checkpoint 7](docs/checkpoints/sprint-5-checkpoint-7.md) adds bounded hybrid
skip/reuse/metadata/search/clarify decisions, scoped fresh evidence reuse and read-only
execution through the operator. Application validation controls permissions, revisions
and budgets; the model interprets semantic needs. Prior chat defaults and adaptive
guidance are preserved. See the [planning runbook](docs/operations/knowledge-planning.md).
Actual model quality and latency are recorded separately from regression results;
model calibration remains required before enabling it. Optional Redis integration
is provided by Checkpoint 10.

[Checkpoint 8](docs/checkpoints/sprint-5-checkpoint-8.md) connects optional retrieval
after routing, budgets the complete specialist prompt and stores validated source
citations with durable chat responses. It preserves adaptive guidance and excludes
technical failures from student learning signals. Trusted server bindings control
course access; RAG remains disabled in the running app while model-quality and latency
issues are open. See the [integration runbook](docs/operations/rag-specialist-context.md).

[Checkpoint 9](docs/checkpoints/sprint-5-checkpoint-9.md) adds authorized PDF intake
and a durable worker for automatic extraction, embeddings and publication. Identical
uploads reuse the index; explicit replacements publish a complete new revision while
preserving earlier data. Failed updates and worker restarts are covered. The API and
worker are opt-in; see the [intake runbook](docs/operations/knowledge-intake.md).
The operator-updated collection is published with three PDFs, 242 chunks and 40 step
facts. HTTP intake and student RAG remain separate opt-in settings.

The [local management wizard](docs/operations/knowledge-management-wizard.md)
automates operator setup and provides add/replace/job-retry menus. From backend,
run `.venv/Scripts/python.exe -m scripts.manage_knowledge`. It preserves valid
configuration, uses current database document IDs for explicit replacements and
processes confirmed selections through the existing intake worker. Student RAG and
HTTP intake remain separate opt-in settings.

[Checkpoint 10](docs/checkpoints/sprint-5-checkpoint-10.md) adds optional real Redis
caching of source chunks/facts and scoped SQL search candidates. PostgreSQL still
checks permissions and revision identity for cache hits. Corrupt entries, expiry and
outages fall back to authoritative reads; shared sources can be reused across students.
The running backend keeps cache/RAG disabled. A read-only operator check verifies
warm hits without enabling chat RAG; see the [cache runbook](docs/operations/knowledge-cache.md).
[Checkpoint 11](docs/checkpoints/sprint-5-checkpoint-11.md) adds frozen retrieval/planner
evaluation, all four guidance-level probes, private database/source backups and
scratch restore/recovery checks. See the [evaluation and recovery runbook](docs/operations/rag-evaluation-and-recovery.md)
and [release checklist](docs/releases/sprint-5-release-checklist.md).
Model quality and latency acceptance remain open; student RAG stays disabled.
The [first acceptance follow-up](docs/checkpoints/sprint-5-rag-diagnostics-1.md)
fixes copied planner example IDs, adds private generation diagnostics and records
laptop model comparisons. Faster model outputs still require source/teaching checks.

The [synthetic student journey](docs/operations/student-simulation-and-observability.md)
runs real onboarding and signed HTTP conversations against isolated PostgreSQL,
exports presentation transcripts and preserves private model/state diagnostics.
Request events now support bounded rotating retention and an optional persistent
Docker log volume. This does not enable live RAG or close teaching acceptance gates.
The [saved 12-turn presentation](docs/demos/student-journey-20261106-presentation-2/README.md)
includes the actual transcript, timing chart, persistence checks and remaining issues.
The [quality-improvement checklist](docs/plans/sprint-5-quality-improvement-checklist.md)
now has [Step 1 evidence](docs/checkpoints/sprint-5-quality-step-1.md): focused
development cases, a semantic review rubric, protected reserved cases and an
offline replay baseline. [Step 2](docs/checkpoints/sprint-5-quality-step-2.md)
implements explicit onboarding fields and supervisor-interpreted current-step
updates before retrieval, with a [real HTTP demonstration](docs/demos/student-state-step2-v3-20261109/README.md).
[Step 3](docs/checkpoints/sprint-5-quality-step-3.md) adds bounded routing dialogue
and confirmed-fact previews, an interaction-purpose decision in the existing
supervisor call, and application replies for clear acknowledgements and personal
recall. Mixed external questions retain normal specialists and hybrid retrieval.
One bounded progress-only verification can recover a declaration missed while
routing its accompanying question. Explicit lookup targets stay separate from
saved progress. The [seven-turn presentation](docs/demos/student-interactions-step3-v5-20261109/README.md)
records onboarding, routing, direct replies, cross-reference retrieval and replay/restart.
[Step 5](docs/checkpoints/sprint-5-quality-step-5.md), requested next, adds internal
claim-to-passage links and one bounded support check across all three grounded
specialists. Unsupported or unverified answers become neutral abstentions with no
citations, learning signal or specialist state proposal. Exact quotations and a
model's support verdict remain fallible; broad semantic and latency acceptance stays
open. Step 4 assessment/adaptive guidance is still pending. No live flags changed.

---

# API

The API surface follows the implemented routes (see
`backend/app/api/routes/`):

* health checks,
* model/provider status endpoints,
* `POST /orchestration/invoke` — the main student interaction endpoint
  (`session_id`, `student_id`, `user_message`, plus `request_id` in durable mode);
  requires a signed pilot bearer token and returns the specialist
  response, selected route, route reason, and request metadata.
* authenticated conversation creation/listing/reopening, paginated message history
  and turn status under `/conversations` (see the persistence runbook).

Business logic lives in services and orchestration, not in route
handlers.

---

# Configuration

Configuration uses `.env` with Pydantic Settings
(`backend/app/config/settings.py`). Model names, endpoints, context
budgets (`CONTEXT_MAX_TOKENS`, `CONTEXT_MAX_RECENT_MESSAGES`, …),
summary policy, observability export (`OBSERVABILITY_LOG_FILE`), and
per-agent model overrides are all configurable — nothing
deployment-specific is hardcoded.

---

# Roadmap

| Sprint | Focus | Status |
|---|---|---|
| 0 | Development environment, FastAPI, config, logging, Docker, CI | ✅ v0.1.0 |
| 1 | Model service abstraction, Ollama integration | ✅ v0.2.0 |
| 2 | Specialist agent foundations | ✅ v0.3.0 |
| 3 | Semantic multi-agent orchestration (LangGraph) | ✅ v0.4.0 |
| 4 | Context engineering (identity, state, context, onboarding, guidance, observability, concurrency) | ✅ v0.5.0 |
| 5 | RAG / knowledge grounding (shared step metadata, AP+ Execution Guide, Process Coach Guide) | Checkpoint 11 evaluation/operations implemented; release quality/latency acceptance remains open; runtime RAG/intake/cache disabled |
| — | Persistent DB (PostgreSQL + pgvector), frontend validation/implementation, vLLM serving | later |

Release notes: [`docs/releases/`](docs/releases/).

---

# License

This project is currently under active research and development.

License information will be added before the first public release.
