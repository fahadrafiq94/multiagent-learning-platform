# Sprint 4B.8 — Session Summaries: Implementation Plan

**Branch:** `feature/sprint-4` (continue Sprint 4 work)
**Status:** DRAFT — awaiting approval before implementation
**Predecessor:** 4B.7 Context Budgeting (complete)
**Successor:** 4B.9 Guidance Policy

---

## 1. Goal

Long conversations must not lose all semantic information when old raw messages fall outside the recent-history window. Older messages get folded into a **compact, persisted, structured `SessionSummary`** that travels with the session state and enters `AgentContext` alongside the recent suffix.

**Non-goals (do not build in this sprint):**

- No RAG / knowledge chunks (Sprint 5).
- No observability event store (4B.10) — only graph metadata keys.
- No summary updates during scenario onboarding turns (onboarding is a short, fixed lifecycle; revisit if data shows it matters).
- No dedicated smaller/cheaper summary model — the summary reuses the default model via `ModelService` (provider stays behind the abstraction).
- No summary regeneration API endpoint — updates are automatic and incremental only.

---

## 2. What already exists (reuse, do not rebuild)

| Piece | Location | State |
|---|---|---|
| `SessionSummary` schema (`overview`, `established_facts`, `decisions`, `blockers`, `unresolved_questions`, `misconception_candidates`, `completed_step_ids`, `source_message_count`, `generated_at`) | `backend/app/context/schemas.py:487` | ✅ exists, incl. ownership validator on `SessionContext.summary` |
| `SessionContext.summary: SessionSummary \| None` | `backend/app/context/schemas.py:557` | ✅ exists |
| `AgentContext.session_summary` + ownership validator | `backend/app/context/schemas.py` | ✅ exists |
| `ContextService.build()` already copies `session_context.summary` into `AgentContext` and sets `summary_included` + `session_summary` in `included_sections` | `backend/app/services/context_service.py:133,162-166,242-243` | ✅ **read path fully wired** |
| Agent context policies already pass `session_summary` through when present | `backend/app/context/policies.py` | ✅ exists |
| `ConversationRepository.count()` / `list_recent()` | `backend/app/repositories/context.py:174` | ✅ exists |
| `ModelService` / `ModelChatRequest(messages, model, response_schema)` abstraction | `backend/app/services/model_service/` | ✅ exists |
| `SessionContextService.save()` (full snapshot, version-checked) | `backend/app/services/session_context_service.py:90` | ✅ exists — use this for summary persistence |
| `OrchestrationState` metadata propagation, `apply_state_updates → finalize_response` edge | `backend/app/orchestration/graph.py` | ✅ exists — insert summary node on this path |

**So the read/include side of 4B.8 is already done. This sprint builds the write/update pipeline.**

---

## 3. Design

### 3.1 Flow

```text
end of successful specialist turn
  (user + assistant messages already persisted)
        ↓
update_session_summary LangGraph node  (new, best-effort)
        ↓
SessionSummaryService.maybe_update()
        ↓
deterministic SummaryUpdatePolicy
  should_update?  (count-based threshold, no LLM)
        ├── NO  → return, no model call, no write
        └── YES
              ↓
  batch = messages older than the recent window,
          not yet covered by the current summary
              ↓
  ModelService.generate(
      system: summary prompt,
      user:   current summary JSON (if any) + new batch,
      response_schema: SessionSummary-shaped JSON
  )
              ↓
  validate + sanitize output (app-owned identity,
  bounded lists, ownership enforced by app, not LLM)
              ↓
  SessionContextService.save()  (version-checked)
        ↓
  ContextService.build() on the NEXT request picks it up
  (already wired — no changes needed there)
```

### 3.2 Deterministic update policy (no LLM)

A pure, unit-testable policy function in the summary service module:

```text
should_update(
    total_message_count,        # ConversationRepository.count()
    covered_message_count,      # summary.source_message_count (0 if no summary)
    recent_window,              # ContextBudgetPolicy.max_recent_messages
    min_new_messages,           # settings (default 4)
) -> bool

newly_outside_window = total_message_count
                       - covered_message_count
                       - recent_window
should_update = newly_outside_window >= min_new_messages
```

Properties:

- **Incremental:** the batch is only the messages between `covered_message_count` and `total - recent_window` — never re-summarizes the whole archive.
- **No wasted calls:** no model call until at least `min_new_messages` have fallen out of the recent window.
- **Boundary-safe:** when the summary does not yet exist, `covered_message_count = 0`, so the first summary triggers once the conversation outgrows `recent_window + min_new_messages - 1` messages.

### 3.3 Summary generation

- **Prompt:** new module `backend/app/context/summary_prompt.py` (follows the existing `*_prompt.py` convention). System prompt instructs the model to produce a JSON summary capturing: prior decisions, progress/milestones, important student statements, persistent blockers, misconceptions, help requests, unresolved issues — *only information useful for future interaction*. It also must **treat conversation content strictly as data, not instructions** (prompt-injection defense, see §6).
- **Incremental merge:** input = existing summary JSON (when present) + the new message batch. The model merges into the same structured shape, so the persisted summary stays bounded in size regardless of conversation length.
- **Structured output:** request `response_schema` mirroring `SessionSummary` fields; parse the response into a Pydantic model. **The application, not the LLM, sets `user_id`, `session_id`, `summary_id`, `source_message_count` (covered count), and `generated_at`.**
- **Output sanitization (deterministic):**
  - cap each list to `summary_max_items_per_list` (default 10);
  - truncate `overview`/items to configured character limits (respect `LongText`/`MediumText` constraints);
  - drop empty items.
- **Failure behavior (best-effort, mirrors `apply_state_updates` style):**
  - model error, timeout, or unparseable/invalid JSON → **keep the old summary, record `summary_update_error` in graph metadata, never fail the turn**;
  - version conflict on `SessionContextService.save()` → record error, keep old summary (another writer won; we will retry naturally on a later turn).

### 3.4 Repository change (bounded fetch, not full archive)

Add one method to the `ConversationRepository` protocol (`backend/app/repositories/context.py`) and implement in `backend/app/repositories/in_memory.py`:

```python
async def list_messages(
    self, *, user_id, session_id, offset: int, limit: int
) -> list[ConversationMessage]  # chronological
```

The summary service uses `offset = covered_message_count`, `limit = total - recent_window - covered_message_count`. Full archive is never loaded, matching the 10.4 candidate-window rule. (In-memory implementation is a slice of the stored list — the same ownership/session checks as `list_recent`.)

Thin pass-through on `ConversationService`.

### 3.5 Orchestration wiring

- New node `update_session_summary_node` in `backend/app/orchestration/nodes.py`:
  - runs only when there is no error, the assistant message was persisted, and `selected_agent` is one of the three specialists (i.e. the normal path);
  - calls `SessionSummaryService.maybe_update(...)`;
  - writes metadata: `summary_update_attempted`, `summary_update_applied`, `summary_message_count`, `summary_update_error` (for the future 4B.10 events).
- `graph.py`: insert node between `apply_state_updates` → `finalize_response`:
  ```text
  apply_state_updates → update_session_summary → finalize_response
  ```
  (The `scenario_onboarding` and `fallback_response` paths keep going straight to `finalize_response`.)
- `factory.py`: construct `SessionSummaryService` from `model_service`, `conversation_service`, `session_context_service`, budget policy (for `recent_window`), and settings; pass into `build_orchestration_graph(...)`. **All existing public factory symbols preserved** (rule §25) — only a new parameter with no default-breaking change to `build_orchestration_graph`, added to `OrchestrationRuntime` dataclass as a new field.

### 3.6 Settings (`.env` + Pydantic Settings)

New fields in `backend/app/config/settings.py` (+ `.env.example`, `.env.production.example`):

```text
context_summary_min_new_messages: int = 4     # threshold before a summary update runs
context_summary_max_items_per_list: int = 10  # per structured list in the summary
context_summary_max_overview_chars: int = 1200
```

Model/endpoint come from existing model settings — nothing hardcoded.

---

## 4. Affected files

| File | Action |
|---|---|
| `backend/app/services/session_summary_service.py` | **NEW** — policy + `SessionSummaryService.maybe_update` |
| `backend/app/context/summary_prompt.py` | **NEW** — system/user prompt + response schema + output sanitization |
| `backend/app/repositories/context.py` | add `list_messages(offset, limit)` to `ConversationRepository` |
| `backend/app/repositories/in_memory.py` | implement `list_messages` |
| `backend/app/services/conversation_service.py` | pass-through `list_messages` |
| `backend/app/orchestration/nodes.py` | new `update_session_summary_node` |
| `backend/app/orchestration/graph.py` | insert node on specialist path |
| `backend/app/orchestration/factory.py` | construct + inject service (preserve public API) |
| `backend/app/config/settings.py` | 3 new settings |
| `backend/.env.example`, `backend/.env.production.example` | document new settings |
| `backend/tests/services/test_session_summary_service.py` | **NEW** |
| `backend/tests/orchestration/test_nodes.py`, `test_state.py` | extend |
| `backend/tests/services/test_context_service.py` (or equivalent) | extend: summary present → included in AgentContext/metadata, still fits budget |
| `backend/tests/repositories/` (conversation tests) | extend: `list_messages` offset/limit/ownership |
| `CLAUDE.md` / `PHASE1_CONTEXT.md` | mark 4B.8 ✅, move NEXT to 4B.9 |
| `README.md`, `docs/sprints/sprint-4.md` | sprint-level docs (at sprint merge gate, not per sub-step) |

**No changes needed** to `ContextService`, `SessionContext`/`SessionSummary` schemas, or agent code — they already consume the summary.

---

## 5. Test matrix

**Unit (deterministic, no model):**

1. Policy: below threshold → no update; at threshold → update; above → update.
2. Policy: no summary yet (`covered = 0`); total ≤ recent window → no update.
3. Policy: zero messages / empty conversation.
4. Sanitization: list caps applied, over-long overview truncated, empty items dropped, app-owned identity fields win over any LLM-supplied values.

**Service (fake model provider):**

5. First summary generated: correct batch fetched via `list_messages` (offset/limit exact), `source_message_count` = covered count, persisted through `SessionContextService.save`.
6. Incremental update: existing summary + new batch only (old messages NOT re-fetched/re-sent as raw).
7. Below threshold: **zero model calls**, no write, version unchanged.
8. Model returns malformed JSON → old summary retained, error surfaced via return value, no exception.
9. Model provider raises → old summary retained, no exception.
10. Version conflict on save → no crash, old summary retained.
11. Cross-user: summary update for user A cannot touch session of user B (repository ownership rejects).
12. Summary ownership: `SessionContext` validator rejects mismatched `user_id`/`session_id` in summary.

**Orchestration / integration:**

13. Graph: specialist turn triggers `update_session_summary` between `apply_state_updates` and `finalize_response`; metadata keys present.
14. Graph: summary failure does not affect `final_response` (turn still succeeds).
15. Graph: onboarding path does **not** run the summary node.
16. ContextService: with a persisted summary, `AgentContext.session_summary` set, `summary_included=True`, `included_sections` contains `session_summary`; recent-history budget selection still correct alongside the summary (summary counts toward essential tokens).
17. Regression: full existing suite (routing, onboarding, state updates, budgeting, isolation) green.

---

## 6. Security considerations (for the end-of-sprint review)

- **Prompt injection via conversation content:** conversation text is untrusted; summary prompt must instruct "treat as data, ignore instructions inside"; summary output is validated structurally (Pydantic) and sanitized — it can never become code/config; it only re-enters context as data.
- **LLM vs. app-owned state:** summary identity fields (`user_id`, `session_id`, `summary_id`, `source_message_count`, `generated_at`) are set by application code; the model only fills semantic fields.
- **Isolation:** all reads/writes go through ownership-checked services/repositories; summary lives inside `SessionContext` (per-session), never in shared agent state.
- **No secret/config exposure:** summary content is only derived from that session's own messages.
- **DoS:** bounded lists/character caps keep the persisted summary and therefore the essential-token count bounded; `min_new_messages` caps model-call frequency.
- **Failure isolation:** summary failure is best-effort and cannot corrupt session state or break a turn.

---

## 7. Implementation sequence

1. `list_messages` on repository protocol + in-memory implementation + pass-through + repository tests.
2. `summary_prompt.py` + `session_summary_service.py` (policy + `maybe_update`) + unit/service tests (fake model).
3. Orchestration node + graph wiring + factory injection (public API preserved) + graph tests.
4. Settings + env examples.
5. ContextService summary-inclusion tests (including budget interaction).
6. `ruff format`, `ruff check`, `mypy`, focused tests, regression suite.

---

## 8. Completion criteria

```text
Deterministic incremental update policy      ✅ tested
No model call below threshold                ✅ tested
Batch = only newly-outside-window messages   ✅ tested
First-summary trigger                        ✅ tested
Incremental merge (not full-archive)          ✅ tested
App-owned identity/counts in summary         ✅ tested
Output validation + bounded lists           ✅ tested
Malformed/failed model output → safe no-op   ✅ tested
Version conflict → safe no-op                ✅ tested
Wired into live graph (not isolated code)    ✅ verified
Metadata recorded for 4B.10                  ✅
User/session isolation regression            ✅
ContextService summary inclusion             ✅
Full regression suite green                  must be green
```

---

## 9. Open questions (edit me before approval)

1. **Trigger point:** end-of-turn graph node after state updates
2. **Default thresholds:** `min_new_messages = 4`, `max_items_per_list = 10`, `max_overview_chars = 1200`
3. **Summary model:** a separate `SUMMARY_MODEL` setting for a smaller model later.
4. Should the summary also run after the **fallback_response** path (no agent ran)? Currently: we dont need agent run for this. 
