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

# Sprint 1 – Model Service

**Status:** Planned

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

*To be updated after Sprint 1.*

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
* Coach Agent
* Scenario Agent
* Reflection Agent
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
* Structured note-taking
* Conversation compaction
* Working memory
* Long-term memory
* Context retrieval
* Prompt assembly
* Token budgeting
* Context evaluation

## Completed

*To be updated after Sprint 4.*

---

# Sprint 5 – Knowledge Base & RAG

**Status:** Planned

## Goal

Build the knowledge layer of the platform using Retrieval-Augmented Generation (RAG).

## Planned

* PostgreSQL integration
* pgvector setup
* Document ingestion
* Chunking pipeline
* Metadata handling
* Embeddings
* Retrieval pipeline
* Hybrid search
* Citation support
* RAG evaluation with RAGAS

## Completed

*To be updated after Sprint 5.*

---

# Sprint 6 – Evaluation & Observability

**Status:** Planned

## Goal

Measure, monitor and evaluate the performance of the complete multi-agent system.

## Planned

* Agent evaluation
* DeepEval integration
* LangSmith tracing
* Experiment tracking
* Metrics dashboard
* Logging improvements
* Benchmark datasets
* Evaluation reports

## Completed

*To be updated after Sprint 6.*

---

# Sprint 7 – Deployment & Production

**Status:** Planned

## Goal

Prepare the platform for deployment on the university server.

## Planned

* Multi-container Docker deployment
* PostgreSQL container
* Redis container
* Model server deployment (Ollama)
* Future vLLM support
* Reverse proxy
* Environment configuration
* Backup strategy
* Security improvements
* Deployment documentation
* Production testing

## Completed

*To be updated after Sprint 7.*

---

# Final Architecture (Target)

```text
                    Frontend (Future)
                           │
                           ▼
                     FastAPI Backend
                           │
                           ▼
                  LangGraph Orchestrator
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
  Coach Agent       Scenario Agent      Reflection Agent
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                    Context Engine
                           │
                           ▼
                     Model Service
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
              Ollama             Future vLLM
                           │
                           ▼
                PostgreSQL • pgvector • Redis

          Evaluation: LangSmith • DeepEval • RAGAS
```

---

# Version History

| Version | Sprint   | Status      |
| ------- | -------- | ----------- |
| v0.1.0  | Sprint 0 | ✅ Completed |
| v0.2.0  | Sprint 1 | ⏳ Planned   |
| v0.3.0  | Sprint 2 | ⏳ Planned   |
| v0.4.0  | Sprint 3 | ⏳ Planned   |
| v0.5.0  | Sprint 4 | ⏳ Planned   |
| v0.6.0  | Sprint 5 | ⏳ Planned   |
| v0.7.0  | Sprint 6 | ⏳ Planned   |
| v1.0.0  | Sprint 7 | ⏳ Planned   |
