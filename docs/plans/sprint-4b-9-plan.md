# Sprint 4B.9 — Guidance Policy: Implementation Plan (v3)

**Branch:** `feature/sprint-4` (continue Sprint 4 work)
**Status:** DRAFT v3 — awaiting approval before implementation
**Predecessor:** 4B.8 Session Summaries (complete)
**Successor:** 4B.10 Observability

> **v2 → v3 change (per user direction):** v2 added a separate post-turn model
> call for response assessment. That adds latency to every turn and extra
> queueing load under concurrent users; Nth-turn sampling was rejected as
> noisy. v3 instead extends the specialist agents' **existing** structured
> output with an optional `response_assessment` — the judgment rides on the
> call the agent is already making. **Zero extra model calls.**

---

## 1. Goal

Guidance escalates when the student is demonstrably struggling and de-escalates when the student is demonstrably reasoning well — decided **centrally**, **consistently**, and **observably**, at **no extra inference cost**:

```text
student message
    ↓
specialist agent answers (existing call)
    └── same structured output now ALSO carries
        response_assessment: correct / partially_correct /
        incorrect / off_track / not_applicable
        + misconception_detected + reason
    ↓
deterministic signal update (application, NO model call)
    (successful_independent_steps, consecutive_unproductive_attempts)
    ↓
GuidancePolicyService (per request, pure)
    (counters + blocker + manual override → level + reason)
    ↓
AgentContext.guidance_level   (agents already adapt — no change)
```

Core rule (CLAUDE.md §12):

> Help should escalate only as needed. The system should preserve productive student reasoning.

**Non-goals:**

- The LLM never chooses or changes the level — it only judges the response, inside its existing output.
- No extra model calls (assessment reuses the specialist's call — the point of v3).
- No new API endpoint for manual overrides (flag becomes functional; a teacher API is later).
- No per-agent level clamping in v1 (decided: skip — see §9).
- No cross-session guidance memory (decided: session-scoped for now).
- No RAG-grounded assessment — the judge works from scenario/process context and model knowledge; Sprint 5 grounding improves it later.
- No observability event store (4B.10) — raw assessments ride in graph state/metadata for now.

---

## 2. Current state (what 4B.9 changes and why)

| Piece | Location | State |
|---|---|---|
| `GuidanceLevel` (MINIMAL < GUIDED < DIRECTED < EXPLICIT), `next_guidance_level()` | `app/domain/guidance.py`, `app/agents/policies.py` | ✅ reuse |
| Per-agent guidance prompt dictionaries | `app/agents/policies.py` | ✅ agents already adapt — no change |
| `SpecialistAgentOutput` (`content`, `state_update_proposal`, metadata) — shared by all three specialists | `app/agents/schemas.py` | ⚠️ extended with optional `response_assessment` |
| `HintState` (level, `consecutive_blocked_attempts`, `successful_independent_steps` — **dead**, `manually_overridden`) | `app/context/schemas.py:312` | ⚠️ counters never written meaningfully |
| Level only ratchets up via accepted blocker | `app/orchestration/nodes.py::_update_hint_state_after_blocker` | ⚠️ no de-escalation, no response-quality input |
| Per-request level = raw copy of persisted level | `app/services/context_service.py:134` | ⚠️ no central selection, no reasoning |

**Problems this sprint fixes:**
1. Nothing observes *whether the student is answering well or struggling* — the core pedagogical signal is unused.
2. Level never de-escalates.
3. Selection is invisible (no reason recorded) — blocks 4B.10 and the §20 "guidance-level appropriateness" evaluation.

---

## 3. Design

### 3.1 Response assessment inside the specialist's existing output

New schema:

```text
StudentResponseAssessment:
    understanding:  "correct" | "partially_correct" | "incorrect"
                    | "off_track" | "not_applicable"
    misconception_detected: bool
    reason: str        # one short sentence, bounded length
```

`SpecialistAgentOutput` gains one **optional** field:

```text
response_assessment: StudentResponseAssessment | None = None
```

All three specialist agents (Scenario, Process Coach, AP+ Navigator) get the same instruction block appended to their existing prompts (shared text in one place — `app/agents/policies.py` — so the rubric is identical across agents):

> Additionally, assess the student's CURRENT message (not your own answer):
> how well does it show understanding of the business process? Reply with
> `understanding` as one of correct / partially_correct / incorrect /
> off_track / not_applicable, `misconception_detected` as a boolean, and a
> one-sentence `reason`. Use `not_applicable` when the message contains no
> assessable learning content (greeting, thanks, pure navigation request).
> Judge the student's message on its own merits against the scenario, the
> current process context, and the help you are giving — do not grade it by
> how well it agrees with your reply.

**Judge's inputs (decided):** the specialist already sees the current message, the bounded agent context (scenario, session state, recent history, summary), and is composing its own response — the rubric instructs it to use all of that, including the help it is giving, while judging the student's message *on its own merits*.

**Robustness (critical — the assessment now shares the turn's single output):**

- A `field_validator(mode="before")` on `response_assessment` coerces dict → model and **returns `None` on any malformation** — a broken assessment degrades to "no assessment" and can never fail the student's turn (`content` errors still raise, as today).
- Omitted assessment → `None` → no signal change (identical behavior to `not_applicable`).

**Known bias, accepted for v1:** the agent judges while also having just answered — it could rubber-stamp a student who merely agreed with its framing. Mitigations: the rubric explicitly instructs judging against scenario/process context, not the agent's own reply; the enum stays closed; two good answers are required before any de-escalation. Documented in the sprint report; re-evaluate when Sprint 5 grounding gives the judge external facts to check against.

**Scope:** only the specialist path produces assessments. Onboarding (extract/assess/merge) and fallback responses do not — no assessable specialist answer exists there.

### 3.2 Deterministic signal update (application-owned, NO model call)

The raw assessment is LLM output — it is **not persisted as state**. Only deterministic counters are persisted on the existing `HintState` (same `UserContextService.save()` path the blocker logic already uses), by a new **pure** graph node `record_response_signal` that reads the assessment from the orchestration state:

| Assessment | `successful_independent_steps` | `consecutive_unproductive_attempts` (NEW field) |
|---|---|---|
| `correct` | **+1** | reset to 0 |
| `partially_correct` | +1 | reset to 0  *(decided: counts as progress — still reasoning forward)* |
| `incorrect` | — | **+1** |
| `off_track` | — | **+1** |
| `not_applicable` / `None` | unchanged | unchanged |

Plus the existing blocker interplay (kept from 4B.6, unchanged):

| Event | Effect |
|---|---|
| accepted blocker SET | `consecutive_blocked_attempts += 1`, persisted-level escalation as today |
| accepted blocker CLEAR | `consecutive_blocked_attempts = 0` |

`consecutive_unproductive_attempts` is added to `HintState` as an additive field (`ge=0`, default 0).

**Why counters instead of raw assessments:** CLAUDE.md §3.2 — deterministic state stays separate from LLM-generated content; counters are what the policy reasons about, and the raw assessment stays in graph state/metadata for the 4B.10 research-event store.

### 3.3 Deterministic level policy (per request, pure function)

New module `app/domain/guidance_policy.py`: `select_guidance(hint_state, session_context) -> GuidanceDecision(level, reason)` — no I/O, fully unit-testable.

```text
R0  MANUAL PIN
    if hint_state.manually_overridden:
        → level = hint_state.guidance_level, reason = "manual_override"; stop

R1  BASELINE
    level = hint_state.guidance_level
    (existing 4B.6 blocker-escalation result — backwards compatible)

R2  BLOCKER FLOOR
    if session.active_blocker:
        floor = consecutive_blocked_attempts: 1 → GUIDED, 2 → DIRECTED, ≥3 → EXPLICIT
        level = max(level, floor); reason = "active_blocker"

R3  STRUGGLE FLOOR   ← response-quality driven (decided thresholds)
    if consecutive_unproductive_attempts >= 2:
        floor = 2 → DIRECTED, ≥3 → EXPLICIT
        level = max(level, floor); reason = "repeated_struggle"

R4  PRODUCTIVE-REASONING DE-ESCALATION   ← assessment-driven (decided: ≥2, one step)
    if no active blocker and successful_independent_steps >= 2
       and consecutive_unproductive_attempts == 0:
        level = max(MINIMAL, level − 1 step); reason = "productive_reasoning"

otherwise reason = "baseline"
```

Properties:

- **Escalation is evidence-based:** repeated *judged* incorrect/off-track responses floor the level before a formal blocker exists.
- **De-escalation is evidence-based:** ≥2 judged-good responses, no recent struggle, no open blocker — one step down at most, MINIMAL floor. One lucky answer cannot drop support.
- **Backwards compatible:** fresh user (all counters 0, no blocker, no assessment) → identical to today.
- `ContextService.build()` replaces the raw level copy with the policy result; `ContextMetadata.guidance_reason` (new additive field) records the decision — the 4B.10/evaluation hook.

### 3.4 Wiring

- `OrchestrationState` gains one additive field: `response_assessment: StudentResponseAssessment | None`.
- `execute_agent_node` copies `response.response_assessment` into state (alongside the existing `agent_response` / `state_update_proposal`).
- New node `record_response_signal` (pure mapping + `UserContextService.save()` for the counters, mirroring `_update_hint_state_after_blocker`):
  ```text
  apply_state_updates → record_response_signal → update_session_summary → finalize_response
  ```
  Fallback and onboarding paths bypass it (like the summary node).
- `ContextService.build()`: use `select_guidance()` instead of the raw `hint_state.guidance_level` copy.
- Factory: the new node only needs the already-injected `UserContextService` — **no new service to construct**; graph signature gains one node, all existing public factory APIs preserved.
- **Views, orchestrator, per-agent prompt dictionaries, summary pipeline: no changes.**

---

## 4. Affected files

| File | Action |
|---|---|
| `backend/app/domain/guidance_policy.py` | **NEW** — `GuidanceDecision` + pure `select_guidance()` + pure `assessment_signal()` mapper |
| `backend/app/agents/schemas.py` | `StudentResponseAssessment` (next to `SpecialistAgentOutput`) + optional `response_assessment` field with tolerant validator |
| `backend/app/agents/policies.py` | shared assessment rubric block (single source) |
| `backend/app/agents/scenario.py`, `process_coach.py`, `ap_plus_navigator.py` | rubric block appended to each agent's system prompt |
| `backend/app/context/schemas.py` | `HintState`: add `consecutive_unproductive_attempts`; `ContextMetadata`: add `guidance_reason` |
| `backend/app/orchestration/state.py` | add `response_assessment` field |
| `backend/app/orchestration/nodes.py` | `execute_agent_node`: carry assessment; **new** `record_response_signal_node` |
| `backend/app/orchestration/graph.py` | insert node on specialist path |
| `backend/app/orchestration/factory.py` | pass node into graph (public API preserved) |
| `backend/app/services/context_service.py` | policy-selected level + reason |
| `backend/tests/domain/test_guidance_policy.py` | **NEW** — full rule matrix + signal mapper |
| `backend/tests/agents/test_scenario.py`, `test_process_coach.py`, `test_ap_plus_navigator.py` | extend: rubric in prompt; malformed/omitted assessment tolerated; valid assessment surfaced |
| `backend/tests/orchestration/test_nodes.py`, `test_graph.py`, `test_state.py` | extend: state carry, signal node behavior, graph edges, 4B.6 regression |
| `backend/tests/services/test_context_service.py` | extend: selected level + reason |
| `CLAUDE.md` | mark 4B.9 ✅ at completion |
| `README.md`, `docs/sprints/sprint-4.md` | at Sprint 4 merge gate |

Notably absent vs v2: no separate assessment prompt module, no assessment service, no `ASSESSMENT_MODEL` setting — there is no separate call to configure.

---

## 5. Test matrix

**Assessment schema & agent contract (fake model, per specialist):**

1. Valid assessment in `SpecialistAgentOutput` → parsed and surfaced (`response_assessment` non-None, enum valid).
2. Omitted assessment → `None`, output otherwise valid, turn succeeds.
3. Malformed assessment (wrong enum value / bad shape / nested junk) → coerced to `None`, `content` and `state_update_proposal` still valid, turn succeeds (tolerant validator).
4. Each specialist's system prompt contains the shared rubric block (deterministic prompt test).
5. All three specialists carry the same instruction (no agent-specific drift).

**Signal node (deterministic, no model):**

6. `correct` → `successful_independent_steps +1`, `consecutive_unproductive_attempts = 0`, persisted on UserContext.
7. `partially_correct` → same effect as correct (decided).
8. `incorrect` / `off_track` → `consecutive_unproductive_attempts +1`, success counter untouched.
9. `not_applicable` / `None` (no specialist, fallback) → no counter changes, no persistence write.
10. Cross-user isolation: signal for user A's turn cannot write user B's context.

**Guidance policy (pure unit, full matrix):**

11. R0 manual pin at every level; other signals ignored.
12. R2 blocker floor at streak 1/2/3+; baseline above floor wins while blocked.
13. R3 struggle floor at 2/3+ unproductive attempts (2 → DIRECTED, ≥3 → EXPLICIT — decided); interacts with R2 as max-of-floors.
14. R4 de-escalation (decided: ≥2 good, one step): success ≥2 + no blocker + struggle 0 → exactly one step down; MINIMAL floor; each precondition violation (blocker present, struggle >0, success 1) → no de-escalation.
15. Fresh user → baseline, reason `baseline` (backwards compatible).
16. Reason correctness per branch; pure function (no input mutation).

**ContextService:**

17. Selected level + `guidance_reason` populated per branch.
18. Budget/summary metadata unaffected; ownership/isolation regression.

**Orchestration / integration:**

19. Specialist turn: assessment carried in state → signal node runs between `apply_state_updates` and `update_session_summary`; metadata present (`response_assessed`, `response_understanding`, `response_misconception_detected`).
20. Fallback + onboarding paths do **not** run the signal node (graph edges).
21. 4B.6 regression: blocker SET still escalates the persisted level; CLEAR still resets the streak.
22. Full regression suite green (routing, onboarding, budgeting, summaries, state updates, isolation).

---

## 6. Security considerations (for the end-of-sprint review)

- **Trust boundary:** the assessment is untrusted LLM output to a deterministic mapper — it can only move two bounded counters through an explicit, closed-enum mapping; it cannot set the level, write arbitrary state, or touch another user's context (all writes via ownership-checked services).
- **Failure isolation:** a malformed assessment degrades to `None` — the student's turn and state are never harmed by the judge failing.
- **Prompt injection:** conversation content is the agent's normal input; the rubric instructs judging content as data; output is schema-validated with a closed enum.
- **Pedagogical safety:** de-escalation requires two judged-good responses with no recent struggle; escalation requires repeated judged struggle — never a single LLM whim.
- **No new persistence beyond two counters + one state field + one metadata field; no new API, no new model config, no extra inference** — the design keeps per-turn latency and concurrent-load behavior identical to today.
- **Observable:** raw assessment + `guidance_reason` in metadata → 4B.10 events, and direct support for §20 "guidance-level appropriateness" evaluation.

---

## 7. Implementation sequence

1. `StudentResponseAssessment` schema + `SpecialistAgentOutput.response_assessment` (tolerant validator) + shared rubric block + agent prompt additions + agent tests (1–5).
2. `HintState` new counter + `ContextMetadata.guidance_reason` + `guidance_policy.py` (`select_guidance` + `assessment_signal`) + full unit matrix (11–16).
3. `OrchestrationState` field + `execute_agent_node` carry + `record_response_signal_node` + node/isolation tests (6–10, 19–21).
4. `ContextService` policy integration + tests (17–18); graph wiring + factory + graph tests (20).
5. `ruff format`, `ruff check`, `mypy`, focused tests, full regression suite (22).

---

## 8. Completion criteria

```text
Assessment rides on existing specialist call        ✅ (zero extra model calls)
Closed-enum structured assessment + tolerant parse  ✅ tested
correct/partial → progress signal                   ✅ tested
incorrect/off_track → struggle signal               ✅ tested
not_applicable/None → no-op                         ✅ tested
Struggle floor escalation (2→DIRECTED, ≥3→EXPLICIT) ✅ tested
Blocker floor preserved (4B.6 regression)           ✅ tested
De-escalation on validated productive reasoning     ✅ tested
Manual-override pin                                 ✅ tested
Reason recorded in ContextMetadata                  ✅
LLM can never set the level directly                ✅ (architecture)
Malformed assessment is a contained no-op           ✅ tested
Wired into live graph (specialist path)             ✅ verified
No added latency under concurrent users             ✅ (no extra inference)
User/session isolation regression                   ✅
Full regression suite green                          must be green
```

---

## 9. Decisions (resolved with user)

| # | Question | Decision |
|---|---|---|
| 1 | Assessment cost | **(b) assessment inside the specialist's existing output** — no extra calls, no Nth-turn sampling |
| 2 | `partially_correct` | counts as progress |
| 3 | De-escalation | ≥2 good responses → one step down |
| 4 | Struggle floor | ≥2 consecutive unproductive → DIRECTED, ≥3 → EXPLICIT |
| 5 | Per-agent clamping | **skipped for v1** (prompt text already self-restrains; revisit with evaluation data) |
| 6 | Judge's inputs | specialist's full context **including its own response** (rubric: judge student's message on its merits) |
| 7 | Schema home | `StudentResponseAssessment` in `app/agents/schemas.py` next to `SpecialistAgentOutput` (agent output contract) |
