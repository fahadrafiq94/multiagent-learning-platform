# FREDi Project Context — Multi-Agent AP+ Learning Environment

> **Authoritative working context for coding sessions**
>
> Use this file as the persistent project context for FREDi. It may live at the repository root as `FREDI_CONTEXT.md`, `PHASE1_CONTEXT.md`, `CLAUDE.md`, or `AGENTS.md` depending on the coding assistant in use.
>
> This document distinguishes **implemented**, **current**, **next**, and **later** work. Do not treat planned components as already implemented.

---
Work only on the specific task described below.

Rules:
1. Inspect only relevant files.
2. Work on one feature or bug at a time.
3. Read targeted sections, not entire large files.
4. Limit scope to 5 source files unless necessary.
5. Avoid large logs and unnecessary tool outputs.
6. Keep responses concise.
7. Break large tasks into independent subtasks.
8. Summarize changes and remaining work.

## 1. Project Identity

**Project name:** FREDi — Agentic AI Learning Environment for AP+ ERP Education  
**Current phase:** Phase 1 — local/API-first multi-agent learning prototype  
**Current engineering sprint:** Sprint 4 — all 4B substeps complete; Sprint 4 completion gate next (security review, docs, README, Docker, merge)  
**Current sub-step:** none — 4B.11 Multi-user/concurrency validation just completed  

### Primary goal

Build a local, research-oriented, multi-agent learning environment in which students work through an AP+ ERP business-process scenario with specialized AI support.

FREDi is **not a generic chatbot**. The system should support:

- business-scenario understanding;
- business-process reasoning;
- AP+ navigation when the student is blocked;
- graduated guidance instead of immediate answers;
- bounded and observable LLM context;
- persistent user/session state outside agents;
- future multi-user serving without redesigning the agent architecture.

The pedagogical goal is to help students reason before giving procedural help.

---

## 2. How the Coding Assistant Should Work

The assistant working with this repository should act as a senior agentic-AI software developer and research collaborator.

Expected behavior:

- Understand the existing architecture before changing it.
- Preserve already-working Sprint 1–4 behavior when adding new features.
- Prefer complete updated files when requested rather than partial patches.
- State the exact file, class, function/method, and action when proposing changes.
- Keep current-sprint work separate from later-sprint ideas.
- Do not silently invent repository methods, constructors, service names, or schema fields.
- Preserve public APIs that are already used by tests or application code.
- Prefer local/open-source components where practical.
- Keep model-provider logic separate from context/state/orchestration logic.
- Add or update deterministic tests whenever deterministic behavior changes.
- Keep agents stateless and reusable.
- **Plan first, implement second.** At the start of every sprint or material feature, first produce an implementation plan and wait for explicit user approval before changing code.
- Before implementation, inspect the existing code paths that the feature must connect to and explain how the new work coordinates with them.
- After implementation, verify that the new feature is actually wired into the existing runtime rather than existing as isolated code.
- Treat tests, security review, documentation, README updates, and Docker runtime validation as part of the sprint definition of done.
- Never merge incomplete or failing sprint work into `main`.

When repository details are uncertain, inspect the actual file before replacing it.

---

## 3. Durable Architecture Principles

These principles are already established and should guide all later work.

### 3.1 Agents are stateless and reusable

Agents must not own persistent student/session state.

```text
Agent
  receives bounded AgentContext
  performs reasoning
  returns structured output
  optionally proposes allowed state changes
```

Persistent state lives in repositories/services.

### 3.2 Memory ≠ State ≠ Knowledge

Keep these concepts separate.

```text
State
→ authoritative structured runtime information
→ current objective, process phase, current step, blocker, etc.

Memory / conversation archive
→ what happened in prior interactions
→ messages, later summaries

Knowledge
→ external AP+ / Process Coach material
→ Sprint 5 RAG
```

Do not merge them into one generic “memory” object.

### 3.3 Full history is stored; only bounded context reaches the LLM

Never load the complete conversation archive into every model request.

```text
Full conversation archive
        ↓
fetch bounded recent candidates
        ↓
ContextBudgetService
        ↓
selected recent suffix
        ↓
AgentContext
```

Older history will later be represented by a persisted session summary in Sprint 4B.8.

### 3.4 Orchestration is semantic, not keyword-only

Sprint 3 replaced static keyword-only routing with semantic orchestration.

The Orchestrator should understand normal student language and select the best specialist agent.

Deterministic rules are still appropriate for:

- ownership validation;
- onboarding lifecycle gates;
- allowed state-update fields;
- guidance constraints;
- context budgeting;
- application-side persistence.

Semantic/LLM reasoning is appropriate where intent or meaning must be interpreted.

### 3.5 LLMs propose; application code persists

Specialist agents may propose controlled session-state updates, but they do not directly mutate persistent state.

### 3.6 Provider independence

Context, state, repositories, and orchestration must not be tied to Ollama.

Current development uses Ollama, but the architecture must support later vLLM deployment and batched multi-user inference.

---

## 4. Roadmap and Current Status

### Sprint 1 — Backend & Model Foundation ✅

Foundation for the Python backend, provider abstraction, configuration, testing, and local model access.

### Sprint 2 — Specialist Agent Foundation ✅

Implemented specialist-agent foundations:

- Scenario Agent;
- Process Coach Agent;
- AP+ Navigator Agent;
- structured agent interfaces/schemas;
- provider-backed model calls.

### Sprint 3 — Semantic Multi-Agent Orchestration ✅

Implemented semantic multi-agent orchestration with LangGraph.

Important outcome:

> FREDi does not rely on a static list of routing keywords. The Orchestrator semantically interprets student requests and routes them to the most suitable specialist.

### Sprint 4 — Context Engineering 🟡

Sprint 4 builds the identity, state, conversation, context, onboarding, controlled-update, and multi-user-safe foundations needed before RAG.

#### Sprint 4A foundation ✅

Completed:

- **4.1A Identity** ✅
- **4.1B Context schemas** ✅
- cross-user/session validation ✅
- **4.2 Repository contracts** ✅
- **4.3 In-memory persistence** ✅
- **4.4 Conversation / Session Service** ✅
- **4.5 UserContextService** ✅
- **4.6A ScenarioCard management** ✅
- **4.6B Scenario onboarding lifecycle** ✅
- **4.7 SessionContext management** ✅
- **4.8 ContextService foundation** ✅

#### Sprint 4B — advanced context behavior

- **4B.1 Shared `GuidanceLevel` domain** ✅
- **4B.2 Agent-specific context policies** ✅
- **4B.3 Structured Scenario onboarding** ✅
- **4B.4 Semantic confirmation/correction** ✅
- **4B.5 LangGraph context integration** ✅
- **4B.6 Controlled State Update Proposals** ✅
- **4B.7 Context Budgeting** ✅
  - **4B.7A Token estimator + ContextBudgetService** ✅
  - **4B.7B ContextService/factory/test integration** ✅
- **4B.8 Session Summaries** ✅
- **4B.9 Guidance Policy** ✅
- **4B.10 Observability** ✅
- **4B.11 Multi-user integration / concurrency tests** ✅

### Sprint 5 — RAG / Knowledge Grounding ⏳

Starts only after Sprint 4 is stable.

Sprint 5 will ground specialist agents in AP+ and Process Coach knowledge sources rather than relying only on model knowledge.

---

## 5. Current Runtime Architecture

The current runtime should conceptually behave as follows:

```text
HTTP request
    ↓
resolve external student identity → internal user_id
    ↓
persist current user message
    ↓
ContextService.build()
    ↓
AgentContext
    ↓
Does the user need Scenario onboarding?
    ├── YES
    │     ↓
    │ scenario_onboarding LangGraph node
    │     ↓
    │ ScenarioOnboardingInteractionService
    │     ↓
    │ ScenarioAgent
    │     ├── extract_draft()
    │     ├── assess_review()
    │     └── merge_extraction()
    │
    └── NO
          ↓
      OrchestratorAgent
          ↓
      semantic routing
          ↓
      specialist agent
          ↓
   SpecialistAgentOutput
      ├── content
      └── state_update_proposal
                 ↓
          apply_state_updates
                 ↓
       SessionStateUpdateService
                 ↓
       SessionContextService
                 ↓
          finalize_response
```

Scenario onboarding is **not a separate agent**. It is a LangGraph/application workflow around the existing `ScenarioAgent`.

---

## 6. Agent Architecture

FREDi currently uses four logical agent roles.

### 6.1 Orchestrator Agent

Responsibilities:

- interpret the student request semantically;
- route to the correct specialist;
- respect onboarding and guidance constraints;
- use the bounded `AgentContext` supplied by the context layer;
- avoid duplicating context-building logic inside the orchestrator;
- support normal natural-language queries that do not contain predefined keywords.

The Orchestrator is not a simple keyword router.

### 6.2 Scenario Agent

Responsibilities after onboarding:

- support discussion of the business scenario;
- reason about scenario context;
- produce structured specialist output;
- propose only allowed session-state updates.

Additional onboarding responsibilities:

```text
ScenarioAgent.execute()
→ normal scenario-agent conversation

ScenarioAgent.extract_draft()
→ structured onboarding extraction

ScenarioAgent.assess_review()
→ semantic CONFIRM / CORRECT / REJECT / UNCLEAR assessment

ScenarioAgent.merge_extraction()
→ deterministic merge of onboarding data
```

Typical ScenarioCard information includes:

- company;
- industry;
- location if relevant;
- department;
- student role;
- business problem;
- process goal;
- stakeholders;
- expected outcome.

### 6.3 Process Coach Agent

Responsibilities:

- help the student reason about the business process;
- use Socratic guidance;
- connect actions to business meaning;
- identify misconceptions or blockers;
- reinforce conceptual understanding before giving procedural instructions.

Must not become an automatic step-by-step solution bot.

### 6.4 AP+ Navigator Agent

Responsibilities:

- provide AP+-specific support when appropriate;
- explain relevant AP+ objects, inputs, process areas, and navigation;
- give controlled procedural support based on guidance level;
- later use grounded AP+ execution knowledge from Sprint 5 RAG.

The AP+ Navigator is escalation support, not the default tutor.

---

## 7. Identity and Multi-User Isolation

Identity and session ownership are first-class constraints.

All context/state operations must preserve:

```text
user_id
session_id
```

Rules:

- a session must belong to the requesting user;
- cross-user access must be rejected;
- context from one student must never leak into another student's request;
- shared agents must remain stateless so multiple users can reuse the same agent instances safely;
- repositories/services own mutable state, not agents.

These rules become part of explicit runtime concurrency validation in Sprint 4B.11.

---

## 8. Current State and Context Model

### 8.1 UserContext

Longer-lived user-level learning context.

May include domain objects such as:

- ScenarioCard;
- learning profile;
- progress/adaptation information;
- shared guidance-related state.

Do not allow specialist agents to directly persist arbitrary UserContext changes.

### 8.2 SessionContext

Authoritative mutable session state.

Examples include:

- current objective;
- current process phase;
- current AP+ step;
- active blocker;
- unresolved questions;
- last selected agent;
- onboarding lifecycle state;
- later session summary.

### 8.3 Conversation archive

All messages are stored independently from the bounded prompt context.

The conversation archive should not be truncated just because model context is bounded.

### 8.4 AgentContext

`AgentContext` is the common bounded input assembled for agents.

Conceptually:

```text
UserContext
SessionContext
Recent conversation candidates
Current user message
Later: Session Summary
Later: retrieved knowledge
        │
        ▼
   ContextService
        │
        ▼
    AgentContext
        │
        ▼
      Agent
```

Agents should not independently reconstruct these sources.

---

## 9. Sprint 4B.6 — Controlled State Updates ✅

Sprint 4B.6 established the trust boundary between LLM reasoning and persistent application state.

```text
SpecialistAgent
    ↓
SpecialistAgentOutput
    ├── content
    └── state_update_proposal
             ↓
      apply_state_updates node
             ↓
      SessionStateUpdateService
             ↓
      SessionContextService
```

### Allowed proposal areas

Scenario Agent may propose fields such as:

- `current_objective`;
- `active_blocker`;
- `unresolved_questions`.

Process Coach may propose fields such as:

- `current_objective`;
- `current_process_phase`;
- `active_blocker`;
- `unresolved_questions`.

AP+ Navigator may propose fields such as:

- `current_step_id`;
- `current_process_phase`;
- `active_blocker`;
- `unresolved_questions`.

### Protected state

Agents must not directly control protected fields such as:

- IDs;
- ownership;
- onboarding lifecycle internals;
- ScenarioCard ownership;
- versions;
- timestamps;
- session summaries;
- persistent progress models;
- other application-controlled fields.

`last_selected_agent` is written deterministically by the application, not proposed by the LLM.

---

## 10. Sprint 4B.7 — Context Budgeting 🟡 CURRENT

The context window must be explicitly budgeted.

### 10.1 Provider-neutral token estimation

Implemented in Sprint 4B.7A:

```text
ApproximateTokenEstimator
ContextBudgetPolicy
ContextBudgetResult
ContextBudgetService
```

Current policy model:

```text
max_context_tokens
-
reserved_output_tokens
=
input_token_budget
```

The estimator is intentionally provider-neutral so development can use Ollama and later deployment can use vLLM without redesigning `ContextService`.

### 10.2 Essential context is protected first

Current priority:

```text
current user message
ScenarioCard
SessionContext
learning/profile state
progress/adaptations
-----------------------------
recent conversation history
```

Recent history may be removed to fit the budget. Essential state must not be silently destroyed in Sprint 4B.7.

### 10.3 Recent-history selection rules

History selection must guarantee:

- configured hard message-count cap;
- token-budget cap;
- newest messages preferred;
- selected messages returned in chronological order;
- selected history forms a contiguous recent suffix;
- if a newer message cannot fit, do not skip it merely to include older messages;
- the stored archive remains untouched.

### 10.4 Candidate-window optimization

Do not load the full archive only to discard most of it.

```text
Full archive
   ↓
ConversationService fetches last N candidates
   ↓
ContextBudgetService
   ↓
selected messages that fit
```

### 10.5 `recent_message_limit`

`ContextService.build()` keeps an optional per-request `recent_message_limit`.

```text
None
→ use configured ContextBudgetPolicy.max_recent_messages

explicit integer
→ override the hard candidate-count cap for that request
```

The token budget may still select fewer messages.

### 10.6 Current ContextMetadata observability

Context metadata should expose context-build information such as:

- `trace_id`;
- `included_sections`;
- `recent_message_count`;
- `summary_included`;
- `knowledge_chunk_count`;
- `adaptation_count`;
- `estimated_input_tokens`;
- `truncation_applied`;
- `input_token_budget`;
- `estimated_essential_tokens`;
- `estimated_history_tokens`;
- `candidate_message_count`;
- `dropped_message_count`.

Example:

```text
input_token_budget           = 6144
estimated_essential_tokens   = 2200
estimated_history_tokens     = 3600
estimated_input_tokens       = 5800

candidate_message_count      = 20
recent_message_count         = 8
dropped_message_count        = 12
truncation_applied           = true
```

`recent_message_count` means messages actually included in `AgentContext`, not merely fetched as candidates.

### 10.7 Current 4B.7B integration target

`ContextService` must:

1. validate user/session ownership;
2. load/create user context;
3. load required session context;
4. fetch bounded recent-message candidates;
5. exclude `current_message_id` when the just-persisted current message is part of storage;
6. serialize essential sections for token estimation;
7. apply `ContextBudgetService`;
8. keep the selected recent suffix;
9. populate real `ContextMetadata` budget values;
10. return `AgentContext`.

The runtime/factory should construct the estimator and budget policy from settings and inject `ContextBudgetService` into `ContextService`.

Do **not** add session-summary prompt selection in 4B.7. That belongs to 4B.8.

Do **not** add RAG chunks in 4B.7. That belongs to Sprint 5.

---

## 11. Sprint 4B.8 — Session Summaries ✅

Implemented:

- `SessionSummaryService.maybe_update()` in `app/services/session_summary_service.py` with the deterministic `should_update_summary` policy (update only after `context_summary_min_new_messages` have fallen out of the recent window);
- incremental batch fetch via the existing `ConversationService.get_history(offset, limit)` — the full archive is never loaded and old messages are never re-sent raw;
- `SummaryDraft` + `build_summary_request` + `sanitize_summary_draft` in `app/context/summary_prompt.py`; the model only fills semantic fields — `user_id`, `session_id`, `summary_id`, `source_message_count`, `generated_at` are app-owned; drafts are validated and bounded (list caps, overview truncation);
- all summary failures (model error, invalid output, version conflict) are contained best-effort — the previous summary is kept and the turn never breaks;
- new `update_session_summary` LangGraph node on the specialist and fallback paths (scenario onboarding bypasses it), injected through `build_orchestration_graph`/`build_orchestration_runtime` with the existing public factory APIs preserved;
- settings: `SUMMARY_MODEL`, `CONTEXT_SUMMARY_MIN_NEW_MESSAGES`, `CONTEXT_SUMMARY_MAX_ITEMS_PER_LIST`, `CONTEXT_SUMMARY_MAX_OVERVIEW_CHARS`;
- plan: `docs/plans/sprint-4b-8-plan.md`.

Purpose: long conversations must not simply lose all semantic information when old raw messages fall outside the recent-history window.

Planned behavior:

```text
older conversation events
        ↓
summary update policy
        ↓
compact persisted Session Summary
        ↓
ContextService
        ↓
summary + recent messages + current request
```

Requirements:

- persist the summary in session-level state/storage;
- update incrementally instead of summarizing the entire archive on every request;
- summarize only information useful for future interaction;
- keep summary text separate from authoritative structured state;
- include summary only when relevant/available;
- record `summary_included` in metadata;
- preserve user/session isolation.

The summary may capture:

- important prior decisions;
- student progress/milestones;
- important student statements;
- persistent blockers;
- misconceptions;
- help requests;
- unresolved issues.

The summary is memory, not authoritative state.

---

## 12. Sprint 4B.9 — Guidance Policy ✅

Guidance must be selected centrally and consistently rather than improvised independently by every agent.

Planned policy inputs may include:

- learning profile;
- progress;
- SessionContext;
- current blocker/confusion;
- prior attempts;
- current process phase;
- agent-specific context policy.

The shared `GuidanceLevel` domain already exists from 4B.1.

Planned deterministic policy should map context to levels such as the project's current guidance-level domain, e.g. minimal/guided/directed/explicit support as defined by the codebase.

Core rule:

> Help should escalate only as needed. The system should preserve productive student reasoning.

Agents may adapt their response to the selected level, but they do not own persistent guidance/adaptation state.

Implemented in 4B.9 (response-quality driven, zero extra inference):

- `StudentResponseAssessment` (closed enum: correct / partially_correct /
  incorrect / off_track / not_applicable + `misconception_detected` +
  `reason`) in `app/agents/schemas.py` — rides on each specialist's
  **existing** structured output; a tolerant validator degrades malformed
  assessments to `None` so a broken judgment can never fail the student's
  turn;
- shared assessment rubric in `app/agents/policies.py`
  (`RESPONSE_ASSESSMENT_INSTRUCTIONS`), identical in all three specialist
  system prompts;
- `HintState.consecutive_unproductive_attempts` (new counter, `ge=0`);
- `app/domain/guidance_policy.py` — pure `select_guidance()` returning
  `GuidanceDecision(level, reason)`: manual-override pin → baseline →
  blocker floor (streak 1/2/≥3 → GUIDED/DIRECTED/EXPLICIT) → struggle
  floor (≥2 unproductive → DIRECTED, ≥3 → EXPLICIT) → productive-reasoning
  de-escalation (≥2 judged-good responses, no recent struggle, no open
  blocker → exactly one step down, MINIMAL floor);
- new `record_response_signal` LangGraph node (specialist path only,
  between `apply_state_updates` and `update_session_summary`): maps the
  assessment to bounded counter movements via `assessment_signal()` and
  persists them on `HintState` — the raw LLM assessment is never persisted;
- `ContextService.build()` selects the level centrally via
  `select_guidance()` and records `guidance_reason` in `ContextMetadata`.

The LLM never sets the guidance level; it only judges the student's
current message inside the specialist call it already makes.

---

## 13. Sprint 4B.10 — Observability ✅

Observability is local-first and useful for research/debugging. There is no
LangSmith or cloud tracing dependency, and the architecture permits later
persistent research-event storage.

Implemented (one typed event per orchestration request, aggregated from the
graph's final state — no new agent/graph/factory changes):

- `RequestObservabilityEvent` (frozen pydantic, `extra="forbid"`) + pure
  `build_request_event(state, *, duration_ms)` mapper in
  `app/observability/events.py`; missing data degrades to `None` fields and
  the mapper never raises or mutates its input;
- carries request/session identifiers, outcome + error, route/reason,
  selected agent, onboarding flag, `guidance_level`/`guidance_reason`,
  context-budget metrics, candidate/selected/dropped counts, truncation,
  summary-update fields, accepted/rejected state-update counts, response
  signal fields, model/provider name, model latency, and request
  `duration_ms`;
- emitted as a structured record by `OrchestrationService.execute()` on both
  the success and graph-error paths via the dedicated standard-library logger
  `fredi.request_events` (best-effort — an observability failure can never
  fail or mask the request);
- never carries user message content, assistant response content, or
  free-text LLM output (only `route_reason`, already public in the API
  response);
- optional local research export: `OBSERVABILITY_LOG_FILE` setting (default
  `None`/off) attaches a `FileHandler` to the request-events logger that
  writes one JSON object per line; propagation stays on so the event also
  reaches the main application log;
- plan: `docs/plans/sprint-4b-10-plan.md`.

Later RAG retrieval references/scores (Sprint 5) will be added to the same
event shape; the schema is designed to persist unchanged into a future
`research_events` table.

---

## 14. Sprint 4B.11 — Multi-User Integration / Concurrency Tests ✅

Implemented in 4B.11 (deterministic in-repo validation, no new dependency):

- `tests/concurrency/test_repositories.py` (layer A): concurrent duplicate
  creation yields exactly one winner; lost-update checks on
  UserContext/SessionContext via optimistic `VersionConflictError`
  (version advances exactly by the applied writes); concurrent appends
  lose nothing; cross-user reads never leak; reads return complete deep
  copies.
- `tests/concurrency/test_orchestration.py` (layer B): one shared
  `OrchestrationRuntime` (one service, one graph, one agent instance per
  role) serves 4 students × 2 requests; canary cross-user leakage is
  checked at response, captured-model-prompt, and conversation-archive
  level; one user / two concurrent sessions stay isolated; simultaneous
  state updates on one session stay coherent (a conflict degrades a
  state write, never the student's turn); concurrent `ContextService`
  builds carry only their own identity/session data; the agent registry
  serves single stateless instances; concurrent 4B.9 signal writes keep
  hint-state counters bounded.
- `tests/integration/test_orchestration_api.py` (layer C): concurrent
  HTTP `/orchestration/invoke` requests with wire-level per-student
  isolation asserted on the response body.
- All interleaving is forced with exact `asyncio.Event` barriers;
  assertions cover counts, versions, and canary absence — never timing
  or completion order.

Validated test categories:

- multiple concurrent users;
- multiple sessions for one user;
- strict no-cross-user state leakage;
- strict no-cross-session leakage;
- shared stateless agent instances;
- repository/service isolation;
- optimistic version/conflict handling where applicable;
- simultaneous state updates;
- concurrent ContextService builds;
- readiness for future batched vLLM inference.

Initial deployment target can remain a small pilot population. The architecture must not require one agent instance per student.

---

## 15. Sprint 5 — RAG / Knowledge Grounding ⏳

RAG begins only after the context/state/concurrency foundation is stable.

### 15.1 Knowledge separation

FREDi should maintain distinct instructional sources so agents receive role-appropriate knowledge.

#### Shared step metadata

Common facts that should not be duplicated between guides, e.g.:

- step ID;
- canonical step name;
- process phase;
- prerequisites/dependencies;
- shared business object identifiers;
- references/links between documents.

#### AP+ Execution Guide

Role: support the AP+ Navigator.

Contains execution-focused information such as:

- where the step occurs in AP+;
- navigation/menu/module information;
- required fields/inputs;
- software-specific actions;
- expected result/output;
- validation points;
- software errors relevant to execution.

#### Process Coach Guide

Role: support conceptual/process learning.

Contains learning-focused information such as:

- minimum understanding expected at the step;
- business reasoning;
- typical student misconceptions;
- common learning mistakes;
- Socratic prompts;
- what the student should be able to explain before receiving more direct help.

The two guides should not duplicate full content. Shared facts belong in shared metadata.

### 15.2 Retrieval rule

```text
Structured shared metadata controls identity/relationships
        +
role-specific documents provide knowledge
        ↓
retrieval
        ↓
only relevant chunks enter AgentContext
```

RAG enriches explanations; it must not replace authoritative workflow/session state.

### 15.3 Storage target

PostgreSQL + pgvector is a suitable planned persistence/retrieval target, but it is not the same thing as the current Sprint 4 in-memory persistence layer.

Do not rewrite current context services around PostgreSQL before the persistence/RAG sprint explicitly requires it.

---

## 16. Model Provider Strategy

### Current development

Ollama is the current local model provider.

Requirements:

- model calls remain behind provider/service abstractions;
- agents do not directly depend on Ollama APIs;
- context budgeting is provider-neutral;
- configuration controls model selection.

### Future deployment

vLLM is a planned serving option for better throughput, batching, and multi-user inference.

The architecture should support:

```text
ContextService
     ↓
AgentContext / model messages
     ↓
provider abstraction
   ↙           ↘
Ollama        vLLM
local dev     later serving/batching
```

vLLM migration should not require redesigning `UserContext`, `SessionContext`, `ContextService`, or the agents.

---

## 17. Persistence Strategy

### Current Sprint 4 implementation

Use repository contracts and in-memory persistence for deterministic development/tests.

This gives us a clean domain/service layer before database concerns are introduced.

### Planned persistent deployment

Later persistence may use PostgreSQL for:

- users/students;
- sessions;
- messages;
- user/session context;
- scenario data;
- summaries;
- research events;
- evaluation records;
- structured AP+ knowledge;
- RAG chunks and embeddings with pgvector.

Repository interfaces should allow the storage implementation to change without changing agent behavior.

---

## 18. API-First Development

Phase 1 remains backend/API-first while the core agentic behavior is stabilized.

The exact API surface should follow the repository's implemented routes rather than this context file inventing endpoints.

Typical capabilities include:

- health checks;
- session creation/retrieval;
- chat/orchestration requests;
- scenario onboarding;
- progress/state operations;
- evaluation endpoints where implemented.

Business logic belongs in services/orchestration, not FastAPI route handlers.

---

## 19. Frontend Plan — Later Phase

A custom frontend is intentionally deferred until the backend agent/context behavior is sufficiently stable.

Planned workflow:

1. Create frontend design concepts in a free/accessible design tool such as Penpot or an appropriate Figma free workflow.
2. Keep design aligned with the agent/context capabilities actually exposed by the backend.
3. Show design alternatives to students.
4. Collect student feedback/preferences.
5. Update the design.
6. Implement the frontend from the validated design.

Potential UI areas include:

- scenario onboarding;
- chat with visible agent/help context where appropriate;
- process/progress state;
- AP+ help escalation;
- status/blocker visualization.

Do not let frontend requirements drive premature changes to context/state architecture.

---

## 20. Evaluation and Research Goals

Primary Phase 1 evaluation target: **agent response quality and system behavior**, not a final claim of learning gains.

Evaluate areas such as:

- routing correctness;
- pedagogical appropriateness;
- Socratic behavior;
- AP+ grounding;
- hallucination avoidance;
- guidance-level appropriateness;
- usefulness/clarity;
- context/state correctness;
- cross-user isolation;
- stability under concurrent users.

A benchmark dataset may contain:

- example student message;
- expected agent;
- expected behavior;
- guidance constraint;
- allowed state-update behavior;
- later expected knowledge source.

---

## 21. Configuration Rules

Use `.env` and Pydantic Settings.

Never hardcode deployment-specific values such as:

- model names;
- model endpoints;
- database credentials;
- ports;
- file paths;
- API keys;
- environment names;
- context-window limits.

### Context-budget settings

Current 4B.7 configuration includes values conceptually equivalent to:

```text
CONTEXT_MAX_TOKENS
CONTEXT_RESERVED_OUTPUT_TOKENS
CONTEXT_MAX_RECENT_MESSAGES
CONTEXT_CHARACTERS_PER_TOKEN
CONTEXT_MESSAGE_OVERHEAD_TOKENS
```

Factory/runtime code should read these settings and construct the provider-neutral budget service.

---

## 22. Testing Strategy

Testing is part of feature implementation, not a final optional step. A feature is not complete merely because the happy path works.

### 22.1 Test every meaningful behavior introduced by a feature

For each new feature, identify and test the meaningful execution paths that can realistically occur. Depending on the feature, this should include:

- expected / happy-path behavior;
- boundary conditions;
- empty and missing input;
- invalid input and validation failures;
- wrong user/session ownership;
- duplicate or repeated requests;
- state-conflict behavior;
- malformed model output where relevant;
- provider/service failures where relevant;
- persistence failures where relevant;
- current-message/history edge cases;
- authorization and isolation cases;
- concurrency/race behavior when the feature is concurrency-sensitive;
- regression cases for previously fixed bugs;
- compatibility with existing public APIs and calling code.

Do not claim literal mathematical exhaustiveness where the state space is unbounded. The requirement is to cover all **meaningful known paths, edge cases, failure modes, and security-relevant cases** for the feature.

### 22.2 Test layers

Use the appropriate combination of:

- unit tests for deterministic logic;
- service tests for application behavior;
- repository/persistence tests;
- orchestration tests;
- API integration tests;
- multi-user/isolation tests;
- concurrency tests when relevant;
- Docker/container smoke tests at sprint completion;
- full-suite regression tests before a sprint is merged.

### 22.3 Current deterministic/unit test areas

Use deterministic tests for:

- schemas;
- token estimation;
- context-budget policy;
- history selection;
- state-update validation;
- ownership checks;
- guidance policy later;
- summary update logic later;
- retrieval ranking/filtering later where deterministic behavior exists.

### 22.4 ContextService tests

4B.7 tests must verify:

- estimated token values are populated;
- candidate counts are correct;
- selected recent-message counts are correct;
- old history is dropped under a constrained budget;
- newest messages survive;
- selected history remains chronological;
- the selected history is a contiguous suffix;
- `recent_message_limit` remains a hard candidate-count override;
- `current_message_id` exclusion still works;
- user/session isolation remains unchanged.

### 22.5 Regression testing after every feature

After implementing a new feature:

1. run the focused tests for that feature;
2. run directly related service/orchestration/API tests;
3. run regression tests for existing features that the new code connects to;
4. run the full test suite before considering the sprint complete.

A feature is not accepted if it passes its own unit test but breaks the existing runtime.

### 22.6 Sprint-level quality gate

Before a sprint can be merged into `main`, the following must be green:

```text
formatting / linting
type checks
feature-focused tests
service tests
orchestration tests
API/integration tests
security-relevant tests
full regression suite
Docker build
container startup
containerized application smoke/health check
```

### 22.7 Concurrency tests

Sprint 4B.11 is the dedicated multi-user/concurrency validation sprint, but concurrency-sensitive behavior introduced earlier should still receive targeted tests rather than being deferred if it can cause state leakage or corruption.

---

## 23. Development Workflow — Mandatory Sprint Governance

This workflow is mandatory for every sprint. Do not skip directly from an idea to implementation.

### 23.1 Sprint start: plan first, approval before coding

At the start of every new sprint:

1. identify the sprint goal and explicit non-goals;
2. inspect the existing code that the sprint will connect to;
3. identify affected files, public APIs, services, repositories, schemas, tests, Docker configuration, and documentation;
4. identify backwards-compatibility risks;
5. define the implementation sequence;
6. define the expected test matrix;
7. define security concerns relevant to that sprint;
8. define sprint completion criteria;
9. present the plan to the user;
10. **wait for explicit user go-ahead before implementation begins.**

Do not treat planning approval as permission for unrelated future sprint work.

### 23.2 Git branch policy

Each sprint is developed on its own feature branch.

Branch naming convention:

```text
feature/sprint-<number>
```

Examples:

```text
feature/sprint-4
feature/sprint-5
feature/sprint-6
```

For the current work, use:

```text
feature/sprint-4
```

At the beginning of the sprint:

1. start from the latest known-good `main`;
2. create or switch to `feature/sprint-<number>`;
3. push that sprint branch to the remote before substantial implementation work;
4. perform all sprint development and fixes on that branch.

Do not merge partial sprint work into `main`.

### 23.3 `main` branch rule

`main` must always represent a known-good, runnable application.

Therefore:

- do not merge a sprint while tests are failing;
- do not merge a sprint while known runtime errors remain;
- do not merge if the Dockerized application cannot build/start/run;
- do not merge if the sprint security review has unresolved high-severity issues;
- do not merge if required sprint documentation and README state are stale;
- merge only after the **whole sprint** is complete, not after individual sub-steps.

### 23.4 Existing-code coordination check before implementation

Before writing new code, trace how the feature must connect to existing behavior. For example:

```text
API / caller
    ↓
service
    ↓
repository / state / context
    ↓
orchestration
    ↓
agent/provider
```

The exact path depends on the feature. The goal is to prevent isolated implementations that are technically correct but never used by the actual runtime.

Before coding, explicitly identify:

- what calls the new code;
- what the new code calls;
- what existing schemas/contracts it must preserve;
- what state it reads or writes;
- what public interfaces must remain compatible;
- which existing tests should still pass.

### 23.5 Implementation loop for every feature/sub-step

For each feature or sprint sub-step:

1. inspect the affected existing files;
2. implement the smallest complete vertical slice;
3. wire it into the real runtime;
4. add/update tests for all meaningful behavior and edge cases;
5. run formatting/lint/type checks;
6. run focused tests;
7. run related integration/regression tests;
8. inspect failures rather than bypassing them;
9. verify the new code coordinates with existing code paths;
10. update implementation notes if the architecture changed.

Typical backend checks:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
```

Then run focused tests followed by the relevant integration/regression tests.

### 23.6 Feature test review

After a feature is implemented, provide a short review of what was tested, including:

- happy paths;
- edge cases;
- invalid/failure paths;
- user/session isolation where relevant;
- state/persistence behavior;
- integration with existing runtime code;
- regressions checked;
- any cases intentionally deferred and why.

### 23.7 Sprint security review

At the end of every sprint, perform a security-focused review appropriate to the work completed. At minimum consider:

- authentication/identity assumptions;
- user/session ownership and cross-user data leakage;
- authorization boundaries;
- input validation;
- unsafe deserialization/parsing;
- prompt-injection or untrusted model-output risks where applicable;
- LLM-proposed state changes versus application-controlled state;
- secrets/API keys/configuration leakage;
- sensitive information in logs;
- error messages that expose internals;
- dependency/security warnings relevant to newly added packages;
- database/query safety once persistent SQL storage is introduced;
- file/path handling once document ingestion is introduced;
- container exposure, ports, runtime user, and unnecessary privileges;
- concurrency/state-isolation risks;
- denial-of-service/resource-exhaustion concerns where relevant.

The review must describe:

```text
what was checked
what tests/checks were run
what issues were found
what was fixed
what remains as an accepted/deferred risk
```

A sprint should not be marked complete while known high-severity security issues introduced by that sprint remain unresolved.

### 23.8 Sprint documentation in `docs/`

After completing each sprint, create a human-readable sprint report in the repository `docs/` folder.

Recommended path:

```text
docs/sprints/sprint-<number>.md
```

Examples:

```text
docs/sprints/sprint-4.md
docs/sprints/sprint-5.md
```

Each sprint report should explain in plain human language:

- sprint goal;
- what was implemented;
- important architecture/design decisions;
- files/components materially changed;
- how the new work connects to the existing application;
- tests added;
- test commands/checks run;
- final test results;
- security review performed;
- security findings and fixes;
- Docker/container changes;
- how the application was verified to run;
- known limitations or deferred work;
- what the next sprint will address.

This file is a project record for humans, not merely a raw test log.

### 23.9 README must always show current project state

`README.md` is the current-state entry point for the repository and must remain synchronized with the project.

Update it whenever a sprint materially changes:

- current sprint/status;
- implemented capabilities;
- architecture overview if changed;
- setup/run instructions if changed;
- Docker/container instructions;
- currently supported API/runtime behavior;
- known limitations;
- pointer to sprint reports under `docs/sprints/`;
- next planned sprint.

At minimum, README must be updated before a completed sprint is merged into `main`.

### 23.10 Docker/container completion gate

The target operational model is containerized. After every completed sprint:

1. review whether Dockerfiles, Compose files, environment variables, exposed ports, volumes, health checks, dependencies, or startup commands must change;
2. rebuild affected images;
3. recreate/update the containers;
4. start the complete application stack;
5. verify container health/status;
6. run a basic application smoke test through the real containerized service;
7. run the relevant API/integration tests against the containerized application where practical;
8. confirm that the application can be used in containers before the sprint is considered done.

The sprint is not complete merely because the code runs directly on the host.

### 23.11 Sprint completion and merge gate

A sprint can be marked complete only when all applicable items below are satisfied:

```text
[ ] approved sprint plan exists
[ ] sprint work is on feature/sprint-<number>
[ ] sprint branch has been pushed
[ ] implementation is wired into existing runtime
[ ] feature tests cover meaningful paths / edge cases / failures
[ ] formatting passes
[ ] linting passes
[ ] type checking passes
[ ] focused tests pass
[ ] related integration tests pass
[ ] full regression suite passes
[ ] security review completed
[ ] security findings resolved or explicitly documented
[ ] docs/sprints/sprint-<number>.md created/updated
[ ] README.md reflects current project state
[ ] Docker/container configuration updated if needed
[ ] Docker images build successfully
[ ] application stack starts successfully in containers
[ ] containerized smoke/health checks pass
[ ] no known sprint-blocking runtime errors remain
```

Only after this gate is green should `feature/sprint-<number>` be merged into `main`.

After merge, verify `main` again as the known-good runnable branch.

### 23.12 Architecture/context maintenance

Update this project-context file whenever architecture, roadmap status, public APIs, sprint sequencing, or development governance materially changes.

---

## 24. Coding Standards

- Python 3.12+.
- Prefer clear, typed Python.
- Pydantic v2 for domain/API/context schemas where appropriate.
- Keep persistent-state logic out of agents.
- Keep business logic out of FastAPI route files.
- Keep model-provider code behind abstractions.
- Keep deterministic state separate from LLM-generated summaries.
- Prefer explicit services over hidden global state.
- Preserve user/session ownership checks.
- Use Ruff for formatting/linting.
- Use mypy where practical.
- Use pytest/pytest-asyncio for tests.
- Use `uv` for Python package/environment management.
- Do not add a large dependency unless it clearly reduces complexity or is needed by the current sprint.

---

## 25. Important Public-API Rule

Do not casually rename or remove application-facing factory/service functions that are already imported elsewhere.

Example lesson from Sprint 4B.7 integration:

```text
app/orchestration/__init__.py
        ↓
expects public factory symbols
```

When modifying factory composition for a new dependency such as `ContextBudgetService`, preserve existing public factory APIs unless the entire call chain is intentionally migrated and tested.

Prefer adding dependency construction inside the existing composition root over replacing the composition root with a new incompatible API.

---

## 26. Phase 1 Non-Goals / Deferred Work

Unless explicitly promoted into the active roadmap, do not add these to current Sprint 4 work:

- custom production frontend;
- direct AP+ backend integration;
- automatic AP+ UI observation;
- university SSO;
- Kubernetes;
- large-scale deployment infrastructure;
- cloud-only tracing dependencies;
- full learning analytics dashboard;
- teacher dashboard;
- multi-course support;
- autonomous agent self-modification;
- long-term Student Model beyond the agreed user/adaptation domains;
- RAG before Sprint 5;
- vLLM deployment before the context/concurrency foundation is ready.

---

## 27. Current Immediate Work

### Current sprint

**Sprint 4 — completion gate (all 4B substeps complete)**

### Current sprint branch

```text
feature/sprint-4
```

All remaining Sprint 4 work should stay on this branch until Sprint 4 is fully complete, fully tested, security-reviewed, documented, container-validated, and ready to merge. `main` should remain the known-good runnable branch.

### Current sub-step

**None — 4B.11 Multi-user/concurrency validation just completed; Sprint 4 completion gate is next**

### Just completed

- **4B.7 Context Budgeting** — provider-neutral token estimation,
  budget policy, token-aware history selection, real budget metadata,
  `ContextService` integration;
- **4B.8 Session Summaries** — deterministic update policy, bounded
  batch fetches, incremental persisted summaries, best-effort
  `update_session_summary` graph node;
- **4B.9 Guidance Policy** — response assessment inside the
  specialist's existing output, deterministic `record_response_signal`
  counters, central `select_guidance()` level policy with recorded
  reason in `ContextMetadata`;
- **4B.10 Observability** — one typed `RequestObservabilityEvent` per
  request aggregated from the graph's final state by a pure mapper,
  emitted best-effort from `OrchestrationService.execute()` via the
  `fredi.request_events` logger; optional `OBSERVABILITY_LOG_FILE`
  JSON-lines research export; no message/free-LLM-text content in events;
- **4B.11 Multi-user/concurrency validation** — deterministic
  barrier-forced concurrency tests across three layers (repository,
  shared-runtime orchestration, HTTP API) proving no cross-user /
  cross-session leakage, coherent optimistic version conflicts, shared
  stateless agents, and concurrent context builds; no runtime changes
  were required (the architecture was already concurrency-ready).

### Next: Sprint 4 completion gate

All 4B substeps are implemented and green. The remaining gate per
Section 23 is: sprint security review, `docs/sprints/sprint-4.md`
report, `README.md` sync to current state, Docker/container rebuild +
startup + smoke validation, and merge of `feature/sprint-4` into `main`
once the whole gate is green. Sprint 5 RAG starts only after that.

---

## 28. Future Sprint Sequence

Do not skip ahead unless explicitly requested. At the beginning of each sprint, create/push the sprint branch, present the sprint plan, and wait for explicit user approval before implementation. At sprint completion, apply the full testing, security, documentation, README, Docker, and merge gates defined in Section 23.

```text
4B.7  Context Budgeting
      ↓
4B.8  Session Summaries
      ↓
4B.9  Guidance Policy
      ↓
4B.10 Observability
      ↓
4B.11 Multi-user Integration / Concurrency Tests
      ↓
Sprint 5 RAG / Knowledge Grounding
      ↓
Persistent DB / retrieval expansion as required
      ↓
Frontend design validation with students
      ↓
Frontend implementation
      ↓
Future vLLM serving / batching deployment
```

---

## 29. Final Design Principles

- AI guides rather than solves.
- Preserve productive student thinking.
- Business-process understanding is more important than click-speed.
- Structured application state controls workflow.
- LLMs interpret, explain, route, and propose; they do not own persistence.
- AP+ Navigator is escalation support, not the default tutor.
- Full archives are not prompt stuffing.
- Context must be bounded, observable, and testable.
- Summary memory must remain separate from authoritative state.
- Knowledge retrieval must remain separate from state.
- Agents must remain stateless and reusable across users.
- User/session isolation is non-negotiable.
- Current Ollama development must not block future vLLM serving.
- Local-first/privacy-conscious operation remains a core Phase 1 requirement.
- Plan before implementation; wait for explicit approval before beginning each sprint.
- Every new feature must be integrated with and regression-tested against existing code.
- Security review is part of each sprint definition of done.
- Every completed sprint must leave a human-readable implementation/test/security record under `docs/sprints/`.
- `README.md` must always describe the current runnable project state.
- `main` must remain a known-good runnable branch.
- Each sprint is developed on `feature/sprint-<number>` and merged only after the whole sprint is green.
- A sprint is not complete until the containerized application builds, starts, and passes smoke/health validation.

---

## 30. Short Context for a New Coding Chat

If a new chat needs a compact handoff, use:

> We are building **FREDi**, a multi-agentic AI learning environment for AP+ ERP education. Sprints 1–3 are complete. Sprint 4 is Context Engineering. Identity, repository contracts, in-memory persistence, conversation/session services, UserContext, ScenarioCard/onboarding, SessionContext, ContextService, shared GuidanceLevel, agent-specific context policies, semantic onboarding confirmation, LangGraph context integration, and controlled state-update proposals are implemented. We just completed **4B.7 Context Budgeting**, **4B.8 Session Summaries**, **4B.9 Guidance Policy**, **4B.10 Observability**, and **4B.11 Multi-user/concurrency validation** (all implemented, tested, and wired into the runtime), on the Sprint 4 feature branch `feature/sprint-4`. Context budgeting protects essential context first, selects the newest contiguous history suffix that fits, retains chronological order, preserves the full archive, and records token/truncation metadata. Session summaries update incrementally in a best-effort graph node. Guidance is selected centrally and deterministically (`select_guidance`) from persisted counters that the `record_response_signal` node updates from each specialist's in-output assessment of the student's response — zero extra inference; the LLM never sets the level. Preserve `current_message_id` exclusion, `recent_message_limit`, user/session isolation, and existing public orchestration factory APIs. Observability emits one typed, best-effort event per request (no message or free-LLM-text content) with an optional `OBSERVABILITY_LOG_FILE` JSON-lines export. Multi-user/concurrency behavior is validated by deterministic in-repo tests (barrier-forced interleaving, canary leak checks at repository, captured-prompt, response, and HTTP layers); no runtime changes were needed. Sprint 4's completion gate (security review, docs, README, Docker, merge) is next, then **Sprint 5 RAG**. Agents are stateless, persistent state lives in services/repositories, and the architecture must remain compatible with current Ollama development and later vLLM multi-user serving. **Mandatory workflow:** before every sprint, inspect existing integration points, present a plan, and wait for explicit user go-ahead. Develop the sprint on `feature/sprint-<number>` and push that branch first. For every feature, add meaningful happy-path, edge-case, failure, isolation, and regression tests, and verify it is wired into the existing runtime. Before completing a sprint, run formatting/lint/type checks, focused tests, integration tests, and the full suite; perform and document a security review; create/update `docs/sprints/sprint-<number>.md` in human language; update `README.md` to the current project state; rebuild/update Docker containers; start the application in containers and verify health/smoke behavior. Merge into `main` only when the whole sprint is complete and green. `main` must always remain the known-good runnable application.
