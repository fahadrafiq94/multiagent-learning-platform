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
* pytest / pytest-asyncio (584 tests, incl. deterministic concurrency
  suite)
* Docker + Docker Compose (dev and prod configs)
* GitHub Actions CI

## Planned (later sprints)

* PostgreSQL + pgvector (persistence + RAG retrieval)
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

# API

The API surface follows the implemented routes (see
`backend/app/api/routes/`):

* health checks,
* model/provider status endpoints,
* `POST /orchestration/invoke` — the main student interaction endpoint
  (`session_id`, `student_id`, `user_message`); returns the specialist
  response, selected route, route reason, and request metadata.

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
| 5 | RAG / knowledge grounding (shared step metadata, AP+ Execution Guide, Process Coach Guide) | ⏳ next |
| — | Persistent DB (PostgreSQL + pgvector), frontend validation/implementation, vLLM serving | later |

Release notes: [`docs/releases/`](docs/releases/).

---

# License

This project is currently under active research and development.

License information will be added before the first public release.
