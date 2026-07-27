# FREDi – Multi-Agent Learning Platform

## Overview

FREDi is a research-oriented multi-agent learning platform designed to investigate how autonomous AI agents can support personalized learning, coaching, scenario generation, and reflective feedback.

The project is being developed as a production-quality software system while simultaneously serving as the implementation platform for academic research in Agentic AI, Multi-Agent Systems, Retrieval-Augmented Generation (RAG), and Large Language Models.

The implementation follows an incremental sprint-based development methodology, where each sprint produces a stable and deployable milestone.

---

# Project Goals

The primary objectives of the project are:

* Build a modular multi-agent learning environment
* Develop production-quality backend architecture
* Support local LLM inference through a model abstraction layer
* Enable future deployment on university infrastructure
* Provide a reproducible platform for AI research and experimentation
* Incorporate evaluation frameworks for both agents and RAG pipelines

---

# Current Status

**Version:** v0.1.0

Sprint 0 (Foundation) has been completed.

Implemented components:

* FastAPI backend
* Configuration management using Pydantic Settings
* Structured logging with structlog
* Docker containerization
* Docker Compose development environment
* Code quality using Ruff
* Static type checking with mypy
* Unit testing with pytest
* Git pre-commit hooks
* GitHub Actions Continuous Integration
* Production-ready project structure

---

# Planned Architecture

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
   Coach Agent       Scenario Agent     Reflection Agent
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                     Context Builder
                            │
                            ▼
                      Model Service
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Ollama                 vLLM
                 │
                 ▼
          Qwen / Gemma Models

PostgreSQL • pgvector • Redis

Evaluation:
LangSmith • DeepEval • RAGAS
```

---

# Technology Stack

## Backend

* FastAPI
* Pydantic Settings
* Uvicorn

## Development

* uv
* Ruff
* mypy
* pytest
* pre-commit

## Logging

* structlog

## Containerization

* Docker
* Docker Compose

## CI/CD

* GitHub Actions

## Planned AI Stack

* LangGraph
* Ollama
* vLLM
* PostgreSQL
* pgvector
* Redis
* DeepEval
* RAGAS
* LangSmith

---

# Repository Structure

```text
multiagent-learning-platform/
│
├── backend/
│   ├── app/
│   ├── tests/
│   ├── pyproject.toml
│   └── uv.lock
│
├── docker/
│   ├── backend/
│   └── compose/
│
├── docs/
│
└── .github/
    └── workflows/
```

---

# Development Workflow

The project follows a feature-branch workflow.

* Main branch always remains stable.
* Development is performed in feature branches.
* Every push automatically executes:

  * Ruff
  * mypy
  * pytest
* Code is merged into `main` only after all quality checks pass.

---

# Roadmap

## Sprint 0

* Development Environment
* FastAPI
* Configuration
* Logging
* Docker
* CI/CD

## Sprint 1

* Model Service Abstraction
* Provider Interface
* Ollama Integration
* Health Checks
* Unit Tests

## Sprint 2

* LangGraph Foundation

## Sprint 3

* Agent Framework

## Sprint 4

* Context Engineering

## Sprint 5

* Knowledge Base & RAG

## Sprint 6

* Evaluation & Observability

## Sprint 7

* Deployment & Production
---

# License

This project is currently under active research and development.

License information will be added before the first public release.
