# Sprint 4B.11 — Multi-User Integration / Concurrency Tests: Implementation Plan (v1)

**Branch:** `feature/sprint-4` (continue Sprint 4 work)
**Status:** ✅ Implemented — approved, built, and green (584 tests passing)
**Predecessor:** 4B.10 Observability (complete)
**Successor:** Sprint 4 completion gate (security review, docs, README, Docker, merge to `main` (this will always be done by me))

---

## 1. Goal

Validate — with deterministic, in-repo tests — that the shared-runtime
architecture actually works under simultaneous users and sessions, per
CLAUDE.md §14:

- multiple concurrent users;
- multiple sessions for one user;
- **strict no-cross-user state leakage** (security property);
- **strict no-cross-session leakage**;
- shared stateless agent instances (one instance serves all users);
- repository/service isolation;
- optimistic version/conflict handling;
- simultaneous state updates;
- concurrent `ContextService` builds;
- readiness for future batched vLLM inference (architecture assertion).

This is a **validation sprint**: the deliverable is a deterministic
concurrency test suite plus any small root-cause fixes the tests expose.
The architecture is already designed for this — in-memory repositories
carry per-repo `asyncio.Lock`s, optimistic `VersionConflictError`
checking, and deep copies on every read/write; agents hold only
`model_service`/`model` config; the API shares one
`lru_cache`d runtime. 4B.11 proves it under real interleaving.

**Non-goals:**

- No production load/benchmark testing (pilot-scale validation only).
- No vLLM deployment — readiness is an architecture assertion
  (single shared `ModelService`, no per-user instances), not a
  deployment exercise.
- No PostgreSQL — in-memory concurrency semantics are what later map
  onto DB transaction behavior; repository contracts stay unchanged.
- No new endpoints, rate limiting, request queues, or retry policy.
- No agent/graph logic changes **unless a test exposes a defect** —
  any such fix lands in this sprint with its regression test.

---

## 2. Current state (what 4B.11 reuses)

| Piece | Location | State |
|---|---|---|
| Per-repository `asyncio.Lock`, optimistic versioning (`VersionConflictError`), deep copies in/out | `app/repositories/in_memory.py` | ✅ under test |
| `SessionContextService.update(expected_version=…)` conflict raise | `app/services/session_context_service.py:146` | ✅ under test |
| State-update read-modify-write: `apply()` reads version, saves with `expected_version` | `app/services/session_state_update_service.py:67-138` | ✅ under test |
| Best-effort degradation: `apply_state_updates` / `record_response_signal` / `update_session_summary` nodes catch `Exception` → metadata error field, turn completes | `app/orchestration/nodes.py` | ✅ under test |
| Shared process runtime: `get_orchestration_runtime()` (`lru_cache`), one agent instance per role | `app/orchestration/factory.py` | ✅ under test |
| Agents hold only `_model_service`/`_model` (no request state) | `app/agents/*.py` | ✅ asserted by tests |
| Fake model service pattern (`MagicMock(spec=ModelService)` + `AsyncMock`, task-based routing) | `tests/integration/test_orchestration_api.py:26` | ✅ reused |
| API override pattern: `app.dependency_overrides[get_orchestration_service]` | `tests/integration/test_orchestration_api.py:151` | ✅ reused |
| Manual multi-user smoke script (live Ollama, Sprint 3) | `scripts/run_concurrent_orchestration_smoke.py` | ✅ kept as-is (manual tool, not the gate) |
| pytest-asyncio (strict mode), httpx for ASGI tests | `pyproject.toml` | ✅ reused, **no new dependency** |

---

## 3. Design — three deterministic test layers

All layers run in-process with the existing fake-model pattern: no live
Ollama, no wall-clock assertions, no network. Interleaving is forced
with `asyncio.gather` and `asyncio.Event` barriers (barrier release at
the write step) where a specific race must be exercised. Determinism
rules:

- assert **counts, versions, and absence of cross-owner data** — never
  timing or completion order;
- every test uses **unique canary strings** so leakage is detectable;
- shared runtime per test module, fresh repositories per test.

### 3.1 Layer A — repository concurrency (unit)

New `tests/concurrency/test_repositories.py`:

1. **Duplicate creation:** N tasks create identities with the same
   `student_id` simultaneously (barrier at the call) → exactly one
   succeeds, N−1 get `DuplicateRecordError`, repository holds one record.
2. **Lost-update check (UserContext):** N tasks read version v, then
   concurrently `save(expected_version=v)` with different counter values
   → exactly one succeeds, N−1 get `VersionConflictError`, final version
   is `v+1`, no value silently lost.
3. **Lost-update check (SessionContext):** same shape on the
   session-context repository → same guarantees.
4. **Concurrent appends (conversation):** N tasks append distinct
   canary messages to the same session simultaneously → all succeed,
   `count_messages == N`, every canary present exactly once, order is a
   consistent chronological sequence (no interleaved corruption).
5. **Cross-user read under concurrency:** user B concurrently reads
   user A's session/messages while A writes → always `None`/empty,
   never A's data, no exception.
6. **Snapshot consistency:** concurrent `list_recent` during appends
   returns complete messages only (no half-states — deep-copy check).

### 3.2 Layer B — service + orchestration concurrency (integration)

New `tests/concurrency/test_orchestration.py`, built on
`build_orchestration_runtime(model_service=<fake>, repositories=<shared
in-memory>)` so all requests share one service, one graph, and one
agent instance per role:

7. **Multi-user, multi-request:** 4 students × 2 requests each via
   `asyncio.gather` on the shared `OrchestrationService` → all 8
   complete, each response routed to the expected agent, each
   `metadata["selected_agent"]` matches its own student.
8. **Cross-user leakage canary (the security test):** each student's
   messages contain a unique canary (e.g. `"canary-alpha-7f3a"`). After
   the gather: (a) no `final_response` mentions another student's
   canary; (b) the fake model's captured `ModelChatRequest`s show each
   request's `AgentContext` messages containing only its own student's
   canaries; (c) the conversation archive holds each student's messages
   only under their own session.
9. **One user, two concurrent sessions:** same student, two session
   UUIDs, concurrent requests → each response reflects only its own
   session state (distinct `session_context_version` streams), no
   cross-session field bleed.
10. **Simultaneous state updates on one session:** two concurrent
    specialist turns for the same session, both proposing different
    `active_blocker` values → both turns complete (no 500); exactly one
    state write is applied, the other degrades best-effort
    (`state_update_applied=False` + `state_update_error` containing the
    conflict); final `SessionContext.version` advanced exactly by the
    applied writes; final state holds one coherent value, never a
    mix.
11. **Concurrent `ContextService.build()`:** parallel builds for
    different users/sessions → all succeed, each `AgentContext`
    carries only its own identity/scenario/session data (canary
    check on serialized context).
12. **Shared stateless agents:** assert the runtime's agent registry
    serves all concurrent users from single instances
    (`registry.get(x) is registry.get(x)`), and the canary check in (8b)
    doubles as the no-cross-request-contamination proof for the shared
    instances.
13. **Guidance counters under concurrency (4B.9 interaction):**
    concurrent same-user turns both producing assessments →
    `record_response_signal` never raises out of the graph; hint-state
    counters end in a coherent state (version monotonic, counters
    within the expected bounds for N signals; a conflict may record
    `response_signal_recorded=False` — that is acceptable, corruption
    is not).

### 3.3 Layer C — API-level concurrency

Extend `tests/integration/test_orchestration_api.py` with:

14. **Concurrent HTTP orchestration:** 3+ students hitting the
    `/orchestration` endpoint simultaneously through
    `httpx.AsyncClient(transport=ASGITransport(app))` with the existing
    dependency-override pattern (shared runtime, fake model) → all
    responses 200, each body scoped to its own student/session,
    cross-user canary absence asserted on the wire.

This is the end-to-end proof that the real FastAPI + shared `lru_cache`
runtime path is concurrency-safe.

### 3.4 Defect handling

If any test exposes a real defect (leak, lost update, race), the fix
goes into this sprint at the **root cause** (repository lock scope,
copy discipline, or node error handling), with the failing test kept as
the regression case. Expected outcome based on code inspection:
zero defects — the in-memory layer was built lock-protected from the
start. If a defect is found and its fix is structural, the plan is
re-presented before the fix lands.

---

## 4. Affected files

| File | Action |
|---|---|
| `backend/tests/concurrency/__init__.py` | **NEW** — package |
| `backend/tests/concurrency/test_repositories.py` | **NEW** — layer A (tests 1–6) |
| `backend/tests/concurrency/test_orchestration.py` | **NEW** — layer B (tests 7–13), shared-runtime + fake-model fixtures, canary helpers, `asyncio.Event` barriers |
| `backend/tests/integration/test_orchestration_api.py` | extend — layer C (test 14) |
| `backend/app/**` | **none expected** — only if a test exposes a defect (§3.4) |
| `CLAUDE.md` | mark 4B.11 ✅ at completion |

Notably absent: no new dependency, no new settings, no runtime changes
by default, no changes to the Sprint 3 smoke script.

---

## 5. Test matrix (summary)

| # | Layer | Scenario | Asserts |
|---|---|---|---|
| 1 | repo | concurrent same-student identity create | exactly 1 winner, rest `DuplicateRecordError` |
| 2 | repo | N concurrent `UserContext.save(v)` | 1 winner, N−1 conflicts, version `v+1` |
| 3 | repo | N concurrent `SessionContext.save(v)` | same as 2 |
| 4 | repo | N concurrent message appends | all succeed, count exact, every canary once |
| 5 | repo | cross-user reads during writes | always empty/None, no exception |
| 6 | repo | reads during appends | complete messages only (deep-copy) |
| 7 | orch | 4 users × 2 reqs, shared service | all complete, routes correct |
| 8 | orch | canary cross-user leakage | none in responses, captured prompts, archive |
| 9 | orch | 1 user, 2 concurrent sessions | no cross-session bleed |
| 10 | orch | simultaneous state updates, 1 session | both turns complete; 1 applied, 1 degraded; version coherent |
| 11 | orch | concurrent `ContextService.build` | isolated `AgentContext`s |
| 12 | orch | shared agent instances | single instances serve all users |
| 13 | orch | concurrent guidance-signal writes | no raise; counters coherent |
| 14 | api | concurrent HTTP requests | all 200, wire-level isolation |
| — | — | full regression suite | green |

---

## 6. Security considerations (for the end-of-sprint review)

- **Cross-user leakage is the headline property** — tests 5, 8, 11, 14
  assert it at four different layers (repository, captured model
  input, response, HTTP wire). A pass here is the security evidence
  for §7/§29 ("user/session isolation is non-negotiable").
- **Deep-copy discipline** at repository boundaries is the primary
  anti-aliasing mechanism; test 6 verifies copies are true copies under
  concurrent mutation.
- **Conflict semantics:** a version conflict degrades a *state write*,
  never the student's turn and never other users' data. Test 10
  verifies the losing write is recorded, not silently dropped.
- **No new trust boundary:** no new dependency, endpoint, setting, or
  network surface; the concurrency tests only exercise existing paths.
- **Prompt isolation under concurrency:** concurrent requests must not
  share prompt material — asserted via the fake model's captured
  requests (8b).

---

## 7. Implementation sequence

1. Layer A — repository concurrency tests (1–6); fix any defect found.
2. Layer B — shared-runtime fixtures + orchestration concurrency tests
   (7–13); fix any defect found.
3. Layer C — API concurrent test (14).
4. `ruff format`, `ruff check`, `uv run mypy app tests`, focused tests,
   full regression suite.
5. CLAUDE.md status update (4B.11 ✅, Sprint 4 gate becomes the next
   item).

---

## 8. Completion criteria

```text
Concurrent users validated              ✅ tested
Concurrent sessions per user            ✅ tested
No-cross-user leakage                   ✅ tested (4 layers)
No-cross-session leakage                ✅ tested
Shared stateless agents                 ✅ tested
Repository/service isolation            ✅ tested
Optimistic version conflict handling    ✅ tested
Simultaneous state updates              ✅ tested (coherent outcome)
Concurrent ContextService builds        ✅ tested
vLLM-readiness (shared model service)   ✅ (architecture assertion, test 12 area)
Full regression suite green             must be green
No new dependency / no runtime change   ✅ by design (unless a defect demands one)
```

---

## 9. Decisions (please confirm or edit)

| # | Question | Proposal |
|---|---|---|
| 1 | Test style | **Deterministic in-process asyncio tests** (`gather` + `Event` barriers, fake model). No load generator, no real network, no new dependency. The Sprint 3 smoke script stays the optional live-model tool |
| 2 | Conflict policy | **Keep current best-effort degradation** (conflicting state write is recorded, turn completes). No retry logic unless a test exposes an actual lost update |
| 3 | Scale | **4–8 concurrent students, ≤2 requests each** — enough to force interleavings while staying fast and deterministic; scale up only if something looks suspicious |
| 4 | API level | **In-process ASGI via `httpx` + existing dependency overrides** — no separate Docker load test in this sprint (Docker gate covers the stack separately at merge) |
| 5 | vLLM readiness | **Architecture assertion only** (one shared `ModelService`, agents/services carry no per-user instance state) — actual vLLM serving remains a future deployment sprint |
| 6 | Defects | **Fix in-sprint at root cause**, failing test becomes the regression case; structural fixes get re-presented first (§3.4) |
