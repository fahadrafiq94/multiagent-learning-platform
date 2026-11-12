# Hybrid retrieval decisions

Checkpoint 7 adds HybridRetrievalService above the existing uncached
KnowledgeRetrievalService. The decision executor and reader-only operator are usable
now. Connecting this service after supervisor routing and projecting its evidence into
specialist prompts/history are Checkpoint 8. HTTP chat currently behaves as before.
No migrations, document edits, publication, classifier, instructor workflow or Redis
integration are introduced here. vLLM remains outside Sprint 5.

## Application and model responsibilities

The trusted caller supplies a collection, optional admitted revision, AccessScope,
internal student/session IDs, selected specialist and bounded context. Neither student
metadata nor model output can establish course membership, ownership or publication.
An omitted revision uses only the active pointer, never the latest staging revision.
The pilot therefore still needs its explicit ready revision for operator evaluation.

Before inference, application code checks enabled/onboarding/budget gates and admits
an authorized ready revision. It pins the snapshot across the decision and every
lookup/search/reuse action. The chosen specialist supplies only a relevance purpose;
there is no exclusive source-to-agent assignment. Model inference holds no SQL
transaction open. Scope/readiness are rechecked after inference and before returning
evidence. Permission revocation discards the evidence.

The model reads bounded question, specialist, step/phase, objective, blocker,
conversation, summary and fresh available evidence. It proposes one of:

| Action | Appropriate use |
|---|---|
| skip | Conversation management or a question already answered by supplied runtime state |
| reuse_evidence | Supplied cited passages cover the actual requested facts |
| metadata_lookup | Exact source-authored step fields such as prerequisites or next step |
| document_search | Conceptual, procedural or relationship knowledge not already covered |
| clarify | A genuinely unresolved target prevents a useful lookup/search |

Up to three lookup/search/reuse actions may be combined. Skip and clarification stand
alone. A typed proposal contains the need, short reason and action arguments, with
no confidence field. Application code rejects malformed/inconsistent plans, unseen
evidence IDs, unsupported fields and invented canonical lookup IDs. Generation uses a
compact action-specific schema; strict Pydantic length/count/UUID validation still
applies afterward. This avoids decoder grammar expansion from nested string bounds.
The same schema is included in the system prompt to ground the output structure.
Planner calls request reasoning_enabled=false through ModelService; Ollama receives
think=false. Other chat requests default to None and retain their previous provider
behavior. A prompt hint alone is not the serving control. See
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
and [thinking controls](https://docs.ollama.com/capabilities/thinking).

Exact lookup IDs must occur in the question, current step or available cited evidence.
A bare number is not converted into a course-specific ID by string formatting. Semantic
questions without canonical IDs can use document search. Full alias catalogs are not
invented here. Explicit references from the original question survive a model search
rewrite; current step remains a soft preference and never advances automatically.
References already supplied by earlier successful operations do not receive an extra
identifier boost in a search for a different gap. Missing lookup results retain their
references for the later search; this does not certify full semantic coverage.

## Reuse and budgets

Available evidence is a trusted server-side EvidenceReceipt, not an arbitrary earlier
assistant answer or ContextChunk. Receipts bind student, session, access scope,
collection, revision and a timezone-aware creation time. Default lifetime is five
minutes; expired/future or mismatched receipts are excluded before loading sources.
Within-request revision pinning survives active-pointer changes; a later request on a
newly active revision rejects an old receipt.

Even valid receipts are only handles: text and provenance are reloaded through the
scoped repository before the planner sees them and again when selected for reuse.
Stored receipt content cannot substitute for current PostgreSQL source text. The
model must choose reuse for semantic coverage, with a reason tied to the supplied facts.
A summary and a high confidence claim do not establish evidence sufficiency. There
is no Redis cache yet, and a receipt does not grant future access.

Default limits are three operations, 650 output tokens, a 60-second overall planner
timeout, 8,000 response characters and 18,000 planner message characters, including
instructions and the schema rendered in the system message. The response-format
copy adds bounded provider overhead; these are not exact token counts. Whole evidence
previews fit 5,000
characters. Optional older conversation/summary text is dropped before essential
question/state. No question is silently shortened to make fallback work.

All executed operations share one serialized-evidence budget (default 8,000
characters) and item cap (default 12). Duplicate quotes merge citations and whole
items are omitted when they cannot fit. This bounds evidence items, not the final
specialist prompt or the result envelope. Final rendered prompt budgeting remains
Checkpoint 8. A found result means selected evidence exists; it does not certify that
every requested fact has been answered. Inspect per-action outcomes and omissions.

## Failure and trace contract

Malformed output, timeout or model failure triggers a deterministic fallback.
A trusted clear_course_query assertion permits a direct search of the complete
original question; a pure explicit reference also permits that search. Otherwise
the result requests a course/topic clarification. This flag is a caller assertion
used only for failure handling, not an ingestion classifier or student permission.
Checkpoint 8 must establish it from trusted request/routing context rather than
accepting a client value or marking every greeting as a course question.

Outcomes are skipped, clarify, found, no_evidence, unavailable and error. A database,
embedding or source-integrity failure is never reported as no evidence. A later
operational failure suppresses partial evidence and the proposed plan, whose reason
or query could quote previously read source text. Cancellation
propagates rather than retrying the workflow. Decisions expose origin
(gate/model/fallback), failure category, planner latency/call count, per-action
outcomes/embedding-call counts, receipt rejections and budget omissions. Counts
represent logical calls, not all HTTP requests or provider retries.
Planner model/version and a bounded failure detail distinguish invalid arguments,
unseen IDs, omitted references, input-budget failures and provider errors without
including failed model output, database internals or credentials.

No decision executes a student turn, writes messages/context, changes guidance,
increments struggle counters or marks the student incorrect. Clarification and
operational fallbacks still need careful student-facing projection in Checkpoint 8.
The context projection helper preserves existing authoritative state and removes a
duplicate current user message from its bounded recent conversation.

## Operator usage

Use the existing knowledge-retrieve one-off container with its entry point overridden:

```powershell
docker compose --env-file backend/.env.knowledge `
  -f docker/compose/docker-compose.dev.yml `
  -f docker/compose/docker-compose.knowledge.yml `
  -f docker/compose/docker-compose.knowledge.dev.yml `
  --profile knowledge-tools run --rm --no-deps --build --entrypoint python `
  knowledge-retrieve -m scripts.plan_knowledge `
  --collection 89c7a7b4-70d8-58d4-913b-7cde21bf3a40 `
  --revision ec469aea-bef2-5b57-b631-c30340092055 `
  --scope-domain course:ap40 --planner-model qwen3.5:4b `
  --model-url http://host.docker.internal:11434 `
  --specialist process_coach --current-step AP40-02 `
  --clear-course-query `
  --question "Why must goods issue be posted before creating an invoice?"
```

This reproduces the final evaluated host-model candidate; its action-quality target
is not yet met. See the [Checkpoint 7 report](../checkpoints/sprint-5-checkpoint-7.md).
The host endpoint is reachable from this workstation's Docker containers and already
has the named model and matching pinned embedding digest. --model-url changes the
operator's model endpoint for both planning and embedding; compatible source-index
embedding identity is still verified. Omitting it uses the configured Docker endpoint.

The explicit planner model changes only the operator's planner configuration, not
the application's chat model. No download is performed. Default operator timeout is
60 seconds; --timeout adjusts it up to 180. --seed-query fetches fresh evidence first
for reuse testing. Replace --question with --cases /config/planner-smoke.json to run
the frozen development cases. Their expectations/targets were defined before model
evaluation; prompt changes against these cases are development tuning, not held-out
quality certification. Wider held-out/per-specialist evaluation remains Checkpoint 11.

Default JSON omits evidence content/context_prefix. --include-evidence explicitly
includes quotes. The command exits zero only when a model-origin decision passes
the case expectations (or produces a valid model-origin decision in single-query
mode); fallback/gate outcomes fail this semantic smoke even when safely handled.
Use result.status to distinguish evidence absence from an operational error. Batch
operator cases are trusted bounded local input, not a student upload format.

The tool retains the existing reader-only credential, 512 MiB/one-CPU limit, no
source-volume mount and no application-schema privileges. It creates synthetic
request identities in memory and never creates student rows. From backend, the
equivalent host command is python -m scripts.plan_knowledge with an explicitly
reachable --model-url and existing protected reader credentials. Windows uses
SelectorEventLoop. Never put credentials in CLI arguments or regular logs.

## Focused regression

From backend, with the existing PostgreSQL container and protected local credentials:

```powershell
$env:FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION = "1"
$plannerRun = [guid]::NewGuid().ToString("N")
.venv/Scripts/python.exe -m pytest `
  tests/knowledge/test_planner.py tests/knowledge/test_planner_postgres.py `
  tests/services/model_service/test_chat_controls.py `
  -q -p no:cacheprovider --basetemp "reports/pytest-planner-$plannerRun"
```

These 55 cases cover proposals, all actions/combinations, scoped real SQL execution,
receipt freshness, full-question fallback, cancellation, global budgets, default chat
compatibility and revocation of source-derived plan text. The PostgreSQL fixture
creates/removes uniquely named disposable databases. Without the integration flag,
six database cases are skipped. Real-model behavior is evaluated separately through
the frozen smoke file; doubles and passing regression do not certify semantic quality.
