# Phase 1 Project Context — Multi-Agent AP+ Learning Environment

Use this file as the persistent project context for AI coding sessions. It can be placed at the repository root as `PHASE1_CONTEXT.md`, or renamed/imported as `CLAUDE.md` / `AGENTS.md` depending on the coding assistant being used.

---

## 1. Project Identity

**Project name:** Agentic AI Learning Environment for AP+ ERP Education  
**Phase:** Phase 1 — API-first local multi-agent learning prototype  
**Primary goal:** Build a local, research-oriented learning environment where students can interact with specialized AI agents while working through one AP+ business process scenario.

This is not a generic chatbot project. It is a controlled multi-agent learning environment designed to support business scenario understanding, process reasoning, and AP+ navigation while preserving student thinking.

---

## 2. Research and Engineering Role

The assistant working with this repository should act as a senior agentic AI software developer and research collaborator, not as an essay writer.

Expected behavior:

- Brainstorm architecture before writing code.
- Challenge unclear assumptions instead of silently inventing details.
- Keep Phase 1 narrow and implementable.
- Prefer local, open-source tools whenever possible.
- Separate research concepts from implementation details.
- Avoid adding future-phase tools to Phase 1 unless explicitly approved.
- When uncertain, ask clarifying questions before making architectural assumptions.

---

## 3. Phase 1 Goal

Phase 1 delivers a local, API-first, multi-agent AP+ learning prototype for small student pilot testing.

The system should allow a student to:

1. Create or confirm a business scenario.
2. Explain the business problem they are solving.
3. Receive Socratic business-process coaching.
4. Request AP+-specific navigation support when blocked.
5. Mark AP+ process steps as completed, blocked, skipped, or needing review.
6. Generate local logs that can later be analyzed for agent response quality.

The goal is to validate whether controlled specialized agents can improve the learning experience without turning the system into another step-by-step tutorial.

---

## 4. Phase 1 Non-Goals

Do not implement these in Phase 1 unless explicitly requested:

- Custom React frontend.
- Production UI/dashboard.
- Student Reflection Agent.
- Agentic self-reflection/adaptation module.
- Automatic AP+ UI observation.
- Direct AP+ backend integration.
- University SSO.
- Redis event streams.
- Nginx deployment layer.
- Prometheus/Grafana/Loki monitoring stack.
- Kubernetes.
- LangSmith cloud tracing.
- Large-scale deployment.
- Multi-course support.
- Full learning analytics dashboard.

These are later-phase ideas, not Phase 1 requirements.

---

## 5. Phase 1 Container Architecture

Phase 1 uses exactly three core containers:

```text
Docker network

backend  <----HTTP---->  ollama
   |
   +----SQL/vector---->  postgres_pgvector
```

### 5.1 Backend Container

Runs the main application logic:

- FastAPI API.
- LangGraph orchestration.
- Scenario Agent.
- Process Coach Agent.
- AP+ Navigator Agent.
- Context engine.
- Checklist state management.
- Model service client.
- Database access layer.
- Local structured logging.
- Basic evaluation utilities.

### 5.2 Model Provider Container

Runs Ollama locally.

Responsibilities:

- Serve chat models.
- Serve embedding models.
- Expose Ollama API to the backend container.
- Keep model calls local.

Initial model candidates:

- Chat: Qwen3, Gemma 3, Llama 3.x, or Mistral Nemo.
- Reasoning experiments: DeepSeek-R1 Distill or Qwen Thinking.
- Embeddings: bge-m3 or nomic-embed-text.

### 5.3 Database Container

Runs PostgreSQL with pgvector.

Stores:

- Students.
- Sessions.
- Messages.
- Scenario contexts.
- Step checklist state.
- Structured AP+ step data.
- Document chunks.
- Embeddings.
- Agent events.
- Evaluation cases/results.

---

## 6. Backend Technology Stack

Required Phase 1 backend tools:

- Python 3.12+
- FastAPI
- Uvicorn
- Pydantic v2
- SQLAlchemy 2
- Alembic
- LangGraph
- LangChain only where useful
- Ollama Python/API client or HTTP client
- PostgreSQL driver such as psycopg/asyncpg
- pgvector integration
- structlog
- pytest
- pytest-asyncio
- Ruff
- mypy
- pre-commit
- uv for package management
- Bruno or Postman-like API testing

Guiding rule:

> LangGraph controls orchestration. Our own code controls memory, context, state, logging, and research data. LangChain is used selectively for prompts, output parsing, and retrieval helpers, not as the main memory architecture.

---

## 7. Backend Project Structure

Use this initial structure:

```text
backend/
  app/
    api/
      routes_chat.py
      routes_sessions.py
      routes_steps.py
      routes_scenarios.py
      routes_health.py

    orchestration/
      graph.py
      state.py
      router.py
      policies.py

    agents/
      scenario_agent.py
      process_coach_agent.py
      ap_plus_navigator_agent.py
      base.py

    context_engine/
      active_context.py
      session_summary.py
      checklist_state.py
      context_builder.py
      compaction.py

    knowledge/
      step_repository.py
      rag_retriever.py
      ingestion.py
      chunking.py

    memory/
      archive_store.py
      summary_store.py
      context_store.py

    services/
      model_service/
        base.py
        ollama_adapter.py
        factory.py

    database/
      models.py
      session.py
      repositories/
      migrations/

    evaluation/
      response_quality.py
      benchmark_runner.py
      rubrics.py

    logging/
      event_logger.py
      schemas.py

    config/
      settings.py

  tests/
  pyproject.toml
  Dockerfile
```

The Orchestrator Agent is implemented under `orchestration/`, not as a normal sub-agent under `agents/`.

---

## 8. Agent Architecture

Phase 1 agents:

1. Orchestrator Agent
2. Scenario Agent
3. Process Coach Agent
4. AP+ Navigator Agent

### 8.1 Orchestrator Agent

Responsibilities:

- Receive student message.
- Load active context, session summary, checklist state, and relevant knowledge.
- Route to the correct sub-agent.
- Decide whether AP+ navigation help is allowed.
- Control hint level.
- Compose or validate final response.
- Trigger logging and memory updates.

The Orchestrator should be deterministic where possible. Use LLM classification only when rule-based routing is insufficient.

### 8.2 Scenario Agent

Responsibilities:

- Collect company and problem context.
- Ask the student for missing scenario fields.
- Create a compact Scenario Card.
- Store scenario data in active context.

Suggested scenario fields:

- Company name.
- Industry.
- Location/address.
- Department.
- Student role.
- Business problem.
- Process goal.
- Main stakeholders.
- Expected outcome.

### 8.3 Process Coach Agent

Responsibilities:

- Help the student reason about the business process.
- Use Socratic guidance.
- Connect AP+ actions to business meaning.
- Encourage conceptual understanding before procedural help.
- Avoid direct step-by-step tutorial behavior.

Must not:

- Immediately tell the student exactly what to click.
- Reveal the full 40-step process sequence.
- Solve the task without requiring student reasoning.

### 8.4 AP+ Navigator Agent

Responsibilities:

- Provide AP+-specific help when the student is blocked.
- Use structured AP+ step data and retrieved AP+ documents.
- Give controlled hints based on hint level.
- Explain required inputs and likely AP+ areas.
- Avoid becoming a tutorial bot.

The AP+ Navigator should normally activate only when:

- The student explicitly asks for AP+ system help.
- The student is blocked.
- The Process Coach determines the student has already reasoned conceptually.
- Repeated confusion is detected.

---

## 9. Hint Policy

Use graduated support instead of direct answers.

```text
Hint Level 0: Ask the student to reason.
Hint Level 1: Give a conceptual hint.
Hint Level 2: Identify the relevant business object/process concept.
Hint Level 3: Point to the AP+ module/menu area.
Hint Level 4: Give precise AP+ procedural guidance.
Hint Level 5: Give direct corrective instruction only when necessary.
```

Default mode: Socratic guidance.  
Escalate only when the student is blocked or has already attempted reasoning.

---

## 10. Context and Memory Architecture

Phase 1 uses four memory/context layers.

### 10.1 Active Context

Small state loaded on every request.

Contains:

- student_id
- session_id
- scenario card
- current process phase
- current_step_id
- support mode
- current hint level
- recent confusion/blocker

### 10.2 Session Summary

Compact natural-language summary of important prior events.

Contains:

- completed milestones
- important student statements
- errors or blockers
- help requests
- misconception candidates
- unresolved issues

Generated by LLM summarization, but stored deterministically.

### 10.3 Full Archive

Raw database record of everything.

Contains:

- user messages
- agent responses
- selected agent
- routing decision
- context snapshot
- retrieved chunks
- hint level
- timestamps
- errors/blockers
- evaluation labels

Never load the entire archive into the prompt by default.

### 10.4 Checklist State

Student-marked progress through the AP+ 40-step process.

Each step can be:

- not_started
- in_progress
- completed
- blocked
- skipped
- needs_review

---

## 11. Knowledge Architecture

Use two knowledge layers.

### 11.1 Structured AP+ Step Data

Machine-readable JSON/YAML for the 40 AP+ steps.

Each step should include:

- step_id
- phase
- business_goal
- expected_student_understanding
- AP+ action summary
- required inputs
- common errors
- Socratic prompts
- hint policy
- related document references

Structured data controls process flow and checklist behavior.

### 11.2 RAG Documents

Use RAG to enrich explanations and AP+ help.

Sources:

- AP+ conceptual guide.
- AP+ software documentation.
- AP+ 40-step implementation guide.
- Reflection templates later.

Initial ingestion tools:

- PyMuPDF or Docling for PDF extraction.
- LangChain text splitters if useful.
- bge-m3 or nomic-embed-text embeddings.
- pgvector storage.

Rule:

> Structured AP+ data controls the process. RAG enriches the explanation.

---

## 12. API-First Interaction Design

Phase 1 is backend/API-first. Initial testing is through Bruno/Postman-like tools.

Core endpoints:

```text
GET  /health
POST /sessions
GET  /sessions/{session_id}
POST /sessions/{session_id}/scenario
POST /sessions/{session_id}/chat
GET  /sessions/{session_id}/progress
POST /sessions/{session_id}/steps/{step_id}/complete
POST /sessions/{session_id}/steps/{step_id}/blocked
POST /sessions/{session_id}/steps/{step_id}/needs-review
GET  /sessions/{session_id}/events
POST /evaluation/run-benchmark
```

Later compatibility endpoint:

```text
POST /v1/chat/completions
```

This endpoint may be added later for Open WebUI or OpenAI-compatible clients, but it must still route through the Orchestrator, context engine, logging, and model service.

---

## 13. Logging and Research Data

All logs must stay local.

Use:

- structlog for application logs.
- PostgreSQL `agent_events` table for research-grade event records.
- Optional local JSONL logs for backup/export.

Every student interaction should record:

- session_id
- student_id
- timestamp
- user message
- selected agent
- routing reason
- retrieved knowledge references
- hint level
- model name
- latency
- final response
- evaluation labels if available

---

## 14. Phase 1 Evaluation

Evaluation target: agent response quality.

Do not try to prove learning gains in Phase 1.

Evaluate:

- routing correctness
- pedagogical appropriateness
- Socratic behavior
- AP+ grounding
- hallucination avoidance
- hint-level appropriateness
- usefulness and clarity

Create a small benchmark dataset with example student messages, expected agent, expected behavior, and maximum allowed hint level.

---

## 15. Configuration Rules

Use `.env` and Pydantic Settings.

Never hardcode:

- model names
- database credentials
- ports
- file paths
- API keys
- environment names

Example settings:

```text
APP_ENV=development
DATABASE_URL=postgresql+psycopg://...
OLLAMA_BASE_URL=http://ollama:11434
CHAT_MODEL=qwen3
EMBEDDING_MODEL=bge-m3
LOG_LEVEL=INFO
```

---

## 16. Development Workflow

Default workflow:

1. Update architecture/design notes before major implementation changes.
2. Implement the smallest useful vertical slice.
3. Add tests for deterministic components.
4. Add benchmark cases for agent behavior.
5. Log all agent decisions locally.
6. Keep future-phase ideas separate from Phase 1 code.

Before adding a new dependency, ask:

- Is it open source?
- Can it run locally?
- Does it keep student data local?
- Does it reduce complexity enough to justify itself?
- Is it needed in Phase 1, or only later?

---

## 17. Coding Standards

- Prefer clear, typed Python.
- Use Pydantic models for API schemas and agent state.
- Use SQLAlchemy models for persistent data.
- Keep business logic out of FastAPI route files.
- Keep prompts versioned and stored outside random Python strings when they stabilize.
- Keep deterministic state separate from LLM-generated summaries.
- Use Ruff for linting/formatting.
- Use mypy for type checking where practical.
- Use pytest for unit and integration tests.

---

## 18. Durable Design Principles

- AI guides rather than solves.
- The system should preserve productive student thinking.
- Process understanding is more important than task completion speed.
- Structured state should control workflow; LLMs should enrich and explain.
- The AP+ Navigator is an escalation support agent, not the default tutor.
- Full archives are for research and future reflection, not prompt stuffing.
- Phase 1 must remain local-first and privacy-conscious.

---

## 19. Later Phase Parking Lot

Keep these ideas, but do not implement them in Phase 1:

- Open WebUI integration.
- Custom React frontend.
- Scenario cards UI.
- Progress visualization UI.
- Student Reflection Agent.
- Agentic self-reflection/adaptation.
- Long-term Student Model.
- Redis background jobs.
- Prometheus/Grafana/Loki.
- Nginx reverse proxy.
- University SSO.
- vLLM scaling.
- AP+ direct integration.
- Teacher dashboard.
- Multi-course support.

