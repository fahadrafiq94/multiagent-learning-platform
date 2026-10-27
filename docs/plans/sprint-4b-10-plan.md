# Sprint 4B.10 — Observability: Implementation Plan (v1)

**Branch:** `feature/sprint-4` (continue Sprint 4 work)
**Status:** ✅ Implemented — approved, built, and green (570 tests passing)
**Predecessor:** 4B.9 Guidance Policy (complete)
**Successor:** 4B.11 Multi-user Integration / Concurrency Tests

---

## 1. Goal

Every orchestration request produces **one structured, local-first observability
event** that records everything CLAUDE.md §13 lists — routing decision/reason,
selected agent, context-budget metrics, summary updates, accepted/rejected
state-update proposals, response-signal fields, `guidance_reason`, model/provider
name, latency, errors — as a single JSON record per request.

**Key insight (why this sprint is small):** all of that data is **already
collected** in the graph's final state (`metadata` dict + `base_context.metadata`
budget values). Nothing in the agents, services, or graph needs new
instrumentation. 4B.10 is a **pure aggregation step at the end of
`OrchestrationService.execute()`** — one typed event model, one pure mapper,
one emit.

```text
graph.ainvoke() → final OrchestrationState
        ↓
build_request_event(state)          (pure function, no I/O)
        ↓
RequestObservabilityEvent           (typed, bounded, JSON-serializable)
        ↓
emit via dedicated logger "fredi.request_events"
   ├── always → application log stream (already JSON in production envs)
   └── optional → JSON-lines file (OBSERVABILITY_LOG_FILE, research export)
```

**Non-goals:**

- No LangSmith, no OpenTelemetry, no cloud tracing (CLAUDE.md §13).
- No database/event-store table yet (Sprint 5 persistence sprint adds
  `research_events` storage; the event schema is designed to persist as-is).
- No per-node tracing — one event per request is the right granularity for
  Phase 1 evaluation (§20); node-level LangSmith-style traces are deferred.
- No RAG retrieval events (Sprint 5).
- No metrics/dashboard/anything beyond structured events.
- **No message content, no free-text LLM output** in the event
  (sensitive-data rule — see §6).

---

## 2. Current state (what 4B.10 reuses)

| Piece | Location | State |
|---|---|---|
| structlog configured (JSON renderer outside development) | `app/logging/config.py` | ✅ reuse |
| structlog dependency | `pyproject.toml` (`structlog>=26.1.0`) | ✅ reuse |
| `Orchestration_started` / `Orchestration_failed` logs | `app/orchestration/service.py` | ✅ extend |
| Graph final `metadata`: route, route_reason, selected_agent, state-update accepted/rejected, summary update fields, response signal fields, agent_metadata (model/provider/latency), context metrics | `app/orchestration/nodes.py` | ✅ read-only source |
| `ContextMetadata`: budget tokens, candidate/selected/dropped counts, truncation, `guidance_reason`, `summary_included` | `app/context/schemas.py` | ✅ read-only source (via `base_context` in final state) |
| `trace_id`, user/session IDs | `OrchestrationState` | ✅ reuse |

**What's missing today:** nothing is aggregated into a single typed, bounded
per-request record; events are scattered across the raw metadata dict and the
API response; there is no local file export for research; there is no request
wall-clock latency (only per-model latency).

---

## 3. Design

### 3.1 Event schema — `app/observability/events.py` (NEW)

```text
RequestObservabilityEvent (pydantic, extra="forbid"):
    # identity
    trace_id: UUID
    student_id: str
    session_id: str
    user_id: UUID | None

    # outcome
    outcome: Literal["completed", "error"]
    error: str | None
    route: Literal["scenario", "process_coach", "ap_plus_navigator", "fallback", None]
    route_reason: str | None
    selected_agent: AgentName | None
    onboarding_handled: bool          # onboarding turn (no route)

    # guidance (4B.9)
    guidance_level: str | None        # selected for this request
    guidance_reason: str | None       # manual_override / baseline / active_blocker /
                                      # repeated_struggle / productive_reasoning

    # context budget (4B.7) — from ContextMetadata
    estimated_input_tokens: int | None
    input_token_budget: int | None
    estimated_essential_tokens: int | None
    estimated_history_tokens: int | None
    candidate_message_count: int | None
    recent_message_count: int | None
    dropped_message_count: int | None
    truncation_applied: bool | None
    summary_included: bool | None

    # summary (4B.8)
    summary_update_attempted: bool | None
    summary_update_applied: bool | None
    summary_update_reason: str | None
    summary_message_count: int | None

    # state updates (4B.6)
    state_update_applied: bool | None
    state_update_accepted_count: int | None
    state_update_rejected_count: int | None

    # response signal (4B.9)
    response_assessed: bool | None
    response_understanding: str | None          # closed enum value only
    response_misconception_detected: bool | None
    response_signal_recorded: bool | None

    # model
    model: str | None
    model_provider: str | None
    model_latency_ms: float | None

    # timing
    duration_ms: float
```

All fields bounded (reasons free-text **excluded**, see §6); JSON-serializable
with `model_dump_json()`; the shape persists unchanged into a future
`research_events` table.

### 3.2 Pure mapper — same module

```text
build_request_event(state, *, duration_ms) -> RequestObservabilityEvent
```

- Reads `state["metadata"]` and `state["base_context"].metadata` (when present);
  every source key is optional — missing data yields `None` fields, never a
  raised error (the mapper must not be able to break a completed turn).
- `outcome` = "error" iff `state["error"]` is set, else "completed".
- `onboarding_handled` = route absent and no error (onboarding turns carry no
  route).
- Model fields come from `metadata["agent_metadata"]` when a specialist ran.
- No I/O, no mutation of `state`.

### 3.3 Emission — `OrchestrationService.execute()`

- Wall-clock the `graph.ainvoke()` call (`time.monotonic`).
- After the graph returns (success **and** the error branch), emit:
  `request_events_logger.info("request_completed", event=event.model_dump(mode="json"))`
  using the dedicated standard-library logger named `fredi.request_events`.
- The existing `Orchestration_started` / `Orchestration_failed` structlog
  calls stay as-is (application log); the new event is the structured record.
- Event emission itself is wrapped so a logging failure can never fail the
  request (best-effort, like summary/signal nodes).
- **No graph changes, no factory changes, no agent changes, no new settings
  consumed at build time** — the service already has the final state.

### 3.4 Local file export (optional, research-oriented)

- New setting: `OBSERVABILITY_LOG_FILE` (default `None` → disabled; empty
  behavior today).
- `configure_logging()`: when set, attach a `FileHandler` to the
  `fredi.request_events` logger that writes one JSON object per line
  (via `json.dumps` of the same event payload). Propagation stays on, so the
  event still appears in the main application log too.
- `.env.example` + `.env.production.example` get the new variable with a
  commented-out default.

---

## 4. Affected files

| File | Action |
|---|---|
| `backend/app/observability/__init__.py`, `events.py` | **NEW** — `RequestObservabilityEvent` + pure `build_request_event()` |
| `backend/app/orchestration/service.py` | time the graph call; emit event on success and error paths |
| `backend/app/logging/config.py` | optional JSON-lines file handler for `fredi.request_events` |
| `backend/app/config/settings.py` | `observability_log_file: str | None = None` |
| `backend/.env.example`, `backend/.env.production.example` | `OBSERVABILITY_LOG_FILE=` |
| `backend/tests/observability/test_events.py` | **NEW** — mapper matrix per outcome/branch |
| `backend/tests/orchestration/test_service.py` | extend: event emitted with correct fields on success/error |
| `backend/tests/logging/test_observability_file.py` | **NEW** — file handler on/off, valid JSON lines |
| `CLAUDE.md` | mark 4B.10 ✅ at completion |

Notably absent: no graph/node changes, no factory changes, no agent changes,
no new dependency (structlog + stdlib logging already present).

---

## 5. Test matrix

**Mapper (pure unit):**

1. Specialist turn (all fields present) → event carries route, agent, budget
   metrics, summary fields, state-update counts, response signal, guidance
   level + reason, model/provider/latency.
2. Fallback turn → route="fallback", no agent fields, no response signal.
3. Onboarding turn → `onboarding_handled=True`, route None.
4. Error state → `outcome="error"`, error populated, still well-formed.
5. Minimal/empty state (missing `base_context`, empty metadata) → all
   optional fields `None`, no exception.
6. `duration_ms` is taken from the argument; mapper does not touch the input
   state.

**Service (integration, fake graph):**

7. Successful execution emits exactly one `request_completed` event with the
   final state's fields (captured via a `caplog`/handler on
   `fredi.request_events`).
8. Graph exception → error event emitted **and** the service still raises
   `OrchestrationServiceError` (emission never masks the failure).
9. `duration_ms > 0` on the emitted event.

**Config:**

10. `OBSERVABILITY_LOG_FILE` unset → no file handler attached, no file created.
11. `OBSERVABILITY_LOG_FILE` set → handler attached; emitted event lands as one
    valid JSON line per event; event still propagates to app log.

**Regression:**

12. Full suite green (routing, onboarding, budgeting, summaries, guidance,
    state updates, isolation).

---

## 6. Security considerations (for the end-of-sprint review)

- **Sensitive data in logs:** the event contains identifiers, enums, counters,
  token counts, and latency only. **Never** user message content, assistant
  response content, or free-text LLM output (assessment reason, blocker text,
  objective text). `route_reason` is the one LLM-generated string included —
  it is already returned verbatim in the public `OrchestrationResponse`, so
  logging it adds no new exposure surface; it is bounded by the routing
  schema's `MediumText`.
- **Prompt injection via logs:** student text cannot reach the event except
  through `route_reason` (see above); everything else is app-generated
  enums/counters.
- **Failure isolation:** event emission is best-effort and cannot fail a
  request; the mapper is pure and cannot mutate state.
- **No new network/cloud endpoint, no new credential, no new port** — output
  is a local file or the existing log stream.
- **Isolation:** the event carries one request's user/session IDs only.

---

## 7. Implementation sequence

1. `RequestObservabilityEvent` + `build_request_event()` + mapper tests (1–6).
2. Service emission (success + error paths) + service tests (7–9).
3. Setting + file-handler wiring + `.env` examples + config tests (10–11).
4. `ruff format`, `ruff check`, `mypy`, focused tests, full regression suite (12).
5. CLAUDE.md status update.

---

## 8. Completion criteria

```text
One typed event per request                ✅
Route/agent/reason captured               ✅ tested
Context-budget metrics captured           ✅ tested
Summary update captured                   ✅ tested
State-update accepted/rejected captured   ✅ tested
Response signal + guidance reason         ✅ tested
Model/provider/latency captured           ✅ tested
Request duration captured                 ✅ tested
Error paths produce events                ✅ tested
No message content / free LLM text        ✅ (schema forbids)
Optional local JSON-lines file export     ✅ tested
No cloud tracing dependency               ✅ (architecture)
No graph/agent/factory changes needed     ✅ (verified)
Full regression suite green               must be green
```

---

## 9. Decisions (please confirm or edit)

| # | Question | Proposal |
|---|---|---|
| 1 | Event granularity | **One event per request** (not per-node traces) — matches §13/§20 evaluation needs; node-level tracing deferred |
| 2 | File export default | **Off** (`OBSERVABILITY_LOG_FILE=None`); event still visible in the app log (JSON in non-development envs). Enable per-deployment for research export |
| 3 | Free text in events | **Excluded** except `route_reason` (already public in the API response). Assessment/blocker/objective free text never logged |
| 4 | Event store (DB) | **Deferred to Sprint 5 persistence** — the event schema persists as-is into `research_events` later; no table now |
| 5 | Emission point | **`OrchestrationService.execute()`** (already has the final state; zero graph changes) — alternative would be a new graph node, which is more code for the same result |
