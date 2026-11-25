# FREDi Development Roadmap

> **Project:** FREDi – Multi-Agent Learning Platform
> **Purpose:** This document provides a high-level roadmap of the project. It summarizes the objectives of each sprint and records what was implemented after the sprint is completed. The roadmap is a living document and may evolve as architectural decisions change during development.

---

# Sprint 0 – Foundation ✅

**Status:** Completed

## Goal

Build a production-quality development environment and software engineering foundation for the project.

## Planned

* Setup development environment
* Configure FastAPI
* Environment configuration using Pydantic Settings
* Structured logging with structlog
* Configure Ruff
* Configure mypy
* Configure pytest
* Configure pre-commit hooks
* Dockerize the backend
* Configure Docker Compose
* Setup Git and GitHub workflow
* Configure GitHub Actions (CI)
* Create initial project documentation

## Completed

* ✅ FastAPI backend
* ✅ Configuration management
* ✅ Structured logging
* ✅ Ruff, mypy and pytest
* ✅ Pre-commit hooks
* ✅ Docker and Docker Compose
* ✅ GitHub repository
* ✅ Feature branch workflow
* ✅ GitHub Actions
* ✅ README and release documentation

---

# Sprint 1 – Model Service ✅

**Status:** Completed

## Goal

Create a unified abstraction layer between the application and any language model provider.

## Planned

* Design Model Service architecture
* Create provider interface
* Implement Ollama provider
* Define request and response models
* Support synchronous inference
* Support streaming responses
* Add logging
* Add error handling
* Add retry policies
* Add health check endpoint
* Unit tests
* Prepare architecture for future vLLM integration

## Completed

* ✅ Designed a provider-independent Model Service architecture
* ✅ Created a `ModelProvider` interface/protocol
* ✅ Implemented the Ollama provider
* ✅ Defined typed request and response models for chat, streaming, embeddings, usage, and health checks
* ✅ Added non-streaming generation support through the Model Service
* ✅ Added streaming response support using newline-delimited JSON responses
* ✅ Added embedding support through the Model Service
* ✅ Added structured logging for model calls
* ✅ Added model service error types and provider-level error handling
* ✅ Added timeout and retry handling for model provider requests
* ✅ Added a model provider health check endpoint
* ✅ Added FastAPI testing endpoints for the Model Service
* ✅ Added unit tests for schemas, exceptions, service facade, factory, provider helpers, and API routes
* ✅ Wired the backend and Ollama through Docker Compose for local development
* ✅ Configured development model defaults for `qwen3:4b` and `nomic-embed-text-v2-moe`
* ✅ Prepared the architecture for future vLLM integration without implementing vLLM yet

---

# Sprint 2 – LangGraph Foundation

**Status:** Planned

## Goal

Introduce LangGraph as the orchestration engine for the multi-agent workflow.

## Planned

* Setup LangGraph
* Build first graph
* Define graph state
* Implement nodes
* Implement edges
* Conditional routing
* Shared state management
* Graph visualization
* Connect LangGraph with Model Service

## Completed

*To be updated after Sprint 2.*

---

# Sprint 3 – Agent Framework

**Status:** Planned

## Goal

Develop reusable agents that perform specialized educational tasks.

## Planned

* BaseAgent abstraction
* Orchestrator Agent
* Scenario Agent
* Process Coach Agent
* AP+ Navigator Agent
* Prompt management
* Agent lifecycle
* Structured outputs
* Tool integration
* Logging
* Unit tests

## Completed

*To be updated after Sprint 3.*

---

# Sprint 4 – Context Engineering

**Status:** Planned

## Goal

Develop a context management system that provides each agent with relevant and efficient context.

## Planned

* Context Builder
* Active Context
* Session Summary
* Full Archive
* Checklist State
* Structured note-taking
* Conversation compaction
* Context retrieval
* Prompt assembly
* Token budgeting
* Context evaluation

## Completed

*To be updated after Sprint 4.*

---

# Sprint 5 - Knowledge base and persistence

Implementation complete. The platform stores student conversations and scenario state
in PostgreSQL, indexes course PDFs, retrieves passages and checks citations. The team
built evaluation and recovery tooling. Outstanding quality and capacity evaluation
moves to Sprint 7.

# Sprint 6 - Spark inference and the student workspace

Closed on 25 November 2026. The team integrated vLLM and independent embeddings,
connected the pilot student interface, and added onboarding assistance and confirmed
role changes. See the [Sprint 6 closeout](releases/sprint-6-closeout.md) for validation and limitations.

<a id="sprint-7"></a>

# Sprint 7 - Monitoring, deployment, student login and evaluation

| Workstream | Deliverable and completion check |
| --- | --- |
| Prometheus and Grafana | Export application metrics and scrape Spark inference metrics. Build dashboards for full-turn latency, errors, agent routing, model queues and service health. Exercise an alert in staging. |
| DGX Spark deployment | Deploy the browser and backend behind HTTPS with private inference, PostgreSQL and persistent source storage. Verify ARM64 compatibility, migrations, backup restore, restart and rollback on the target. Document deployment and recovery commands. |
| Student login service | Agree on the provider and cohort access policy. Verify the external account, map it to the existing FREDi UUID, and issue the browser session. Test sign-in, sign-out, denied access and preservation of existing conversations. |
| Deferred Sprint 5 evaluation | Evaluate retrieval and planner decisions on the intended document revision, citation support and abstention, teaching across three specialists and four guidance levels, and the full student journey. Measure full-turn latency and concurrent pilot capacity. Record a release decision against the [Sprint 5 acceptance checklist](releases/sprint-5-release-checklist.md). |

Start with the current Spark model as the baseline. Use the staging results to decide
the supported cohort size and whether to enable student-facing retrieval.

# Version History

| Version | Sprint   | Status      |
| ------- | -------- | ----------- |
| v0.1.0  | Sprint 0 | ✅ Completed |
| v0.2.0  | Sprint 1 | ✅ Completed |
| v0.3.0  | Sprint 2 | ✅ Completed |
| v0.4.0  | Sprint 3 | ✅ Completed |
| v0.5.0  | Sprint 4 | Completed |
| v0.6.0  | Sprint 5 | Implementation complete; evaluation in Sprint 7 |
| v0.7.0  | Sprint 6 | Closed (implementation milestone) |
| v1.0.0  | Sprint 7 | ⏳ Planned   |
