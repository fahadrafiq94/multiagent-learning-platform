# RAG evaluation and recovery

Checkpoint 11 provides repeatable evaluation and a private, coordinated backup.
These tools do not enable student RAG or approve a release. Use the existing local
credential-file owner account, running Docker/PostgreSQL and the installed native
Ollama models. Commands below run from `backend` with the existing virtual environment.

## Frozen evaluation

The committed `docs/knowledge/rag-evaluation-v1.json` fixes 12 development and 15
held-out questions, gold reference/document identities and 0.90 targets before
execution. It pins the currently published course revision. Development and held-out
results must be reported separately; never change held-out labels to improve a score.
Record a new suite version and an untouched holdout after changing the evaluation.

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_rag --split development --stages retrieval planner --output reports/rag-development-new
.venv/Scripts/python.exe -m scripts.evaluate_rag --split heldout --stages retrieval planner teacher --output reports/rag-heldout-new
```

Choose a new output directory for every invocation. Windows permissions restrict it
to the executing account, SYSTEM and Administrators; POSIX directories use mode 0700.
Results contain source passages and generated answers: keep them private and out of
Git. Routine console output contains identifiers/timings, not source text or credentials.

The default candidate is the installed `qwen3.5:4b` at `127.0.0.1:11434`, with an
8192-token request window, reasoning disabled and a 60-second model timeout.
Specialist output reserves 1024 tokens. `--model`, `--model-url` and `--timeout` select
an explicit evaluation profile; they do not alter the running chat configuration.
Record serving changes and compare equivalent profiles, rather than mixing their results.

For a focused development diagnostic:

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_rag --split development --case-id dev-scenario-start --stages teacher --probe-teaching --guidance minimal --diagnostics --output reports/diagnostic-new
```

Repeat `--case-id` to select questions. Extra teaching probes are development-only;
`--guidance` selects the existing guidance levels. `--diagnostics` retains bounded raw
model output only inside the protected result directory. `model_calls` records chat
generation attempts, token counts, provider loading/prompt/generation durations and
safe failure/cancellation codes; it does not include embedding calls or copy provider
error bodies/hidden thinking. New artifacts include grounded/planner code hashes.
Subset completeness gates remain false when the full split has not been exercised.
See [acceptance follow-up 1](../checkpoints/sprint-5-rag-diagnostics-1.md) for the
confirmed example-ID copying bug, its fix and laptop comparisons. Consumed v1 failures
copied into separate development regressions do not become new held-out evidence.

The evaluator reads the course using the knowledge reader. It creates its own
`fredi_eval_*` database, imports a synthetic general PDF through the existing intake
and real embedding pipeline, evaluates it for all specialists, then closes pools
and drops only that generated database. Corpus setup requires local admin credentials;
the course is never written to. Do not kill the process during fixture creation or
cleanup. A hard crash may leave a generated scratch database for operator inspection.

Gold labels are used only by scoring, never appended to retrieval/model requests.
The JSON records the suite hash, source revision, actual embedding/model identities,
effective retrieval configuration and serving profile. Each case is saved incrementally.
Missing observations cannot pass complete-stage gates. Inspect summary completeness
before interpreting a partial artifact. Exit code 0 means the experiment ran; inspect
the gate booleans for quality. It never means release authorization.

Scoring reports reference/document recall among the first four returned items,
strict no-evidence rejection, planner validity/expected behavior, missed evidence,
unnecessary searches on context-only cases and planner p50/p95. It accepts an
appropriate metadata lookup or document search for course facts; this is a different
rubric from the earlier strict action-selection experiment. It does not supersede
that experiment or measure optimal search cost. Three specialists at four guidance
levels produce twelve separate teaching probes. Valid JSON and source labels do not
prove factual support, teaching quality or appropriate disclosure: inspect answers
against their supplied evidence and the existing guidance policies.

This is a small pilot suite. Explicit step references and document-level gold for a
single general PDF are useful wiring checks, not broad semantic-retrieval certification.
The generic negative includes a document statement that keys are unspecified; retrieving
that statement could correctly support a refusal even though strict no-evidence scoring
counts it as a rejection failure. Keep that outcome visible and revise the rubric in
a separately versioned evaluation, rather than silently changing this frozen suite.

## Backup and scratch restore

```powershell
.venv/Scripts/python.exe -m scripts.knowledge_backup create --output reports/backup-new
.venv/Scripts/python.exe -m scripts.knowledge_backup verify --backup reports/backup-new --recover
```

`create` exports a read-only repeatable-read PostgreSQL snapshot and passes that exact
snapshot to `pg_dump`. Its manifest fingerprints all application/knowledge tables,
including chat, contexts, turns, migrations, revisions, vectors and intake jobs.
It copies every referenced original source hash, including queued jobs and revision
manifests, from the native `.knowledge-sources` store or the existing Docker source
volume. Multiple `--source-store` arguments add other trusted native stores.
Blobs remain immutable while the snapshot is held. Redis is disposable and is excluded.

The result is `database.dump`, `manifest.json` and content-addressed `sources/`.
An incomplete copy remains marked incomplete; an existing output is never overwritten.
Before restoring, validation checks the dump hash and every source identity, size and
hash. Verification restores only into a newly generated `fredi_restore_*` database;
there is no caller-provided restore target and no live-database overwrite option.
It compares every table's count/checksum, verifies the source-reference set, opens
connections as the real reader/application roles and checks schema isolation.
`--recover` additionally starts the application ownership/recovery mechanism there.
The generated restore database is removed afterward; the backup remains available.

Only verify archives created and retained under trusted operator control. Checksums
detect accidental damage; they do not authenticate an archive against an attacker.
`pg_restore` processes SQL under the local admin role. Backups contain student data
and are private, unencrypted local archives; arrange protected off-machine retention
and protect the credential file separately. Credentials/cluster role definitions are
not part of `pg_dump`. Keep compatible PostgreSQL/pgvector and bootstrap the expected
roles before a recovery on another machine.

This is a same-cluster disposable restore drill, not an automatic disaster failover.
Live replacement, a new host restore, source-volume installation, endpoint changes
and starting workers against recovered data require a separately reviewed deployment
procedure. The tool never performs those actions. Preserve the old database/volumes
until a recovered deployment is verified. Use the normal intake worker to resume
queued indexing with its recovered source store and trusted configuration.

## Recovery and independent rollback

Keep one HTTP worker per application database. A restart acquires exclusive ownership,
fences the old worker and interrupts abandoned processing turns. A student can then
make a new request; completed request IDs still replay the stored response.
Do not use elapsed time alone to declare an active request abandoned. The existing
`scripts.application_admin recover` refuses recovery while a backend owns the database.

| Change in the existing deployment configuration | Expected behavior after backend restart |
|---|---|
| `KNOWLEDGE_CACHE_ENABLED=false`, RAG still enabled | Direct authorized PostgreSQL retrieval; chat/history and citations continue |
| `RAG_ENABLED=false`, cache setting retained | Existing chat/history; no knowledge reader/cache runtime is constructed |
| `KNOWLEDGE_INTAKE_ENABLED=false` | HTTP management intake unavailable; existing published knowledge/history remain |

Use the same deployment overlays and trusted binding paths that were used to enable
the feature. Update flags and recreate the backend with that deployment's command;
changing a terminal variable without recreating the container is insufficient.
Verify `/ready`, signed chat/history and the container's effective flag values.
Do not add an overlay with an absent configuration file merely to disable a feature.
Never remove data volumes, reset migrations or flush shared Redis as a rollback.
Technical/model failures are not student-learning signals.

## Verification commands

### Quality Step 5: cited-passage support

Grounded Scenario, Process Coach and Navigator responses now include internal
claim-to-label links. Their schema is `specialist-agent-v4-claim-support`; the
student still receives ordinary text and source references. Unknown labels are
rejected. Inconsistent labels or missing required links abstain without a judge.
If a claim copied source wording instead of matching the answer, a bounded fallback
checks the entire unchanged answer against the same labels; it never substitutes
citations, guesses an equivalent span or rewrites the student's answer.

One `citation_support` call checks full-answer coverage and assigned passages after
generation, before the graph's final source re-authorization. It uses the existing
specialist model, 30-second outer timeout, 512 output tokens, disabled reasoning,
temperature 0 and at most min(6144, window minus 512) estimated input tokens including
the schema. Only already-admitted, actually assigned passages enter the check.
Oversized inputs abstain rather than cropping passages. The provider's existing retry
policy stays inside the deadline; the feature adds no retry/repair loop. Timing out
the client does not ensure immediate server-side inference cancellation.

Every grounded found/skipped specialist answer is checked, including one declaring
no claims; software instructions and question presuppositions can still assert
external facts. No-evidence/error/outage answers use a fixed abstention and zero
support calls. Step 3 direct acknowledgements/recall and non-RAG agents do not run
this check. The judge sees filtered owned facts for mixed replies, not full history,
student profiles or arbitrary repository access.

`metadata.rag.status` describes retrieval; `support_status` describes answer checking.
For example, `found` plus `unsupported` suppresses the proposed source answer,
references, assessment and specialist state updates, while preserving the normal
saved user/assistant turn. Valid earlier student progress declarations are separate.
Replay returns the saved result without another check. An outage/uncertainty is
neutral about student understanding. `support_link_mode` identifies declared-claim
versus whole-answer checking. Exact quotes permit removal of one matching display
quotation pair only; quotes and model judgments do not guarantee entailment.

Safe request events add `citation_support_status`, `_calls`, `_latency_ms`,
`_claim_count`, `_abstained` and `_link_mode`. Raw claims, excerpts and judgments
remain out of routine events. A slow or timed-out check is visible separately from
retrieval and specialist generation. Do not interpret a clean HTTP completion or a
neutral abstention as successful factual-answer acceptance.

From `backend`, use a fresh protected output directory:

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_citation_support --output reports/citation-support-new
.venv/Scripts/python.exe -m scripts.evaluate_citation_support --output reports/citation-targeted-new --case-id ui-correct-passage --skip-generation
```

These are real-model **synthetic development component probes**, not HTTP requests,
PDF search, reserved acceptance cases or natural guidance adaptation. Eight fixed
checks and three separate specialist probes record code/fixture hashes and protected
per-call input/output/timing. Explicit guidance is a controlled probe input only.
Expectations do not enter model prompts. Uncertainty, timeout and malformed results
cannot count as successful semantic detection. Keep failed runs and choose new paths;
monitor the log while executing and read atomic JSON archives only after exit on
Windows. See the [Step 5 report](../checkpoints/sprint-5-quality-step-5.md) for
validation and limitations. Step 4 and integrated/latency acceptance remain open.

### Quality Step 3: routing and context-only replies

The existing supervisor call returns an `interaction` decision as well as the
specialist route and current-step proposal. It receives previews of confirmed
company/role/problem/goals and the last four available messages (450 characters
each, with source references removed). The full scenario, learning profile,
progress, source chunks and conversation archive remain outside this routing view.
The previous agent is a weak hint; recent topic and current need determine routing.
The joint generation grammar keeps purpose, route and progress consistent:
help/mixed requires a specialist with set/unchanged progress; pure conversation
requires fallback with unchanged progress; genuinely unclear intent can clarify.
Legacy non-writing progress ambiguity is deferred when a known conversation/help
need can proceed using saved state. It does not infer or save a different step.

When a valid primary decision says `unchanged` on a `help`/`mixed` message, one
progress-only verification call checks for a declaration missed while routing the
question. It sees only the current message, uses the same configured model, limits
output to 250 tokens and has a 30-second outer timeout. Only a `set` proposal is
forwarded to the existing current-message/ownership/version guard. No-proposal,
invalid, unavailable or timed-out checks keep the resolved route and stored progress;
the focused pass cannot introduce its own clarification. There is no orchestration
retry loop; provider retry settings remain subject to the outer timeout. Existing
proposals, social/recall and omitted legacy purposes avoid this additional call.
`student_step_verification_status`, `_calls` and `_latency_ms` report its cost;
the separate `student_step_status` remains authoritative about persistence.

Pure acknowledgements/greetings and explicit personal-fact recall use an application
reply after admission of the owned context. They still use the supervisor call,
but skip the specialist, retrieval planner, embeddings, learning-signal recording
and summary-generation call. Messages and the completed turn are persisted normally;
replay/restart behavior is unchanged. Summary maintenance is deferred to the next
substantive turn. A `fallback` route with `context_reply_handled=true` is an intentional
conversation reply, not a routing error.

Recall answers use the actual saved step or confirmed scenario value, rather than
the model's wording or truncated fact preview. An optional history quote must match
a student-authored recent message; assistant text and policy examples cannot supply
it. One enclosing quotation-mark pair may be removed before the exact normalized
student-text match; this is formatting normalization, not fuzzy matching.
Missing facts/history produce a clarification, with no document search. Quote
matching establishes provenance, not semantic entailment or complete history recall.
Questions about new external facts or the meaning of a concept still use normal
specialists and hybrid planning. Mixed thanks/recall plus a substantive question
must preserve that question. The planner also receives bounded confirmed facts so
that it can recognize state/history sufficiency when it runs. Existing permission,
revision, freshness, reuse and evidence-budget checks remain authoritative.

Routine request events add only allowed `interaction_kind` values and the
`context_reply_handled` flag. Raw quotes, saved facts and replies are not new event
fields. A malformed/omitted interaction decision retains the ordinary workflow;
an actual `set` proposal retains current-message/ownership/version validation.
Genuinely unclear intent retains clarification; a known recall/help need does not
require a new progress declaration. Model mistakes can still cause poor routing or unnecessary work;
the model's semantic accuracy is evaluated separately from these application guards.
For mixed recall and external questions, the planner places the current question
last and treats an explicitly requested reference/topic as the lookup subject.
Saved current_step_id and prior evidence remain background. The shared specialist
policy likewise separates saved progress from the requested topic; looking up a
different task must not switch student progress or answer about the old task.

Run bounded **development-only** supervisor/reply-node probes, then a separate real
signed-HTTP journey from `backend`. Use new output directories on every invocation:

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_student_interactions --output reports/interactions-new
.venv/Scripts/python.exe -m scripts.evaluate_student_interactions --suite ../docs/knowledge/student-interactions-state-development-v1.json --output reports/interactions-state-new
.venv/Scripts/python.exe -m scripts.run_student_simulation --fixture ../docs/knowledge/student-interactions-http-smoke-v1.json --output reports/interactions-http-new --presentation ../docs/demos/interactions-http-new
```

The component probes use frozen development labels and historical synthetic
pre-turn context. They do **not** execute specialists, HTTP or source searches;
unobserved retrieval stays unobserved. The journey uses real supervisor/planner/
specialist calls, signed ASGI HTTP, a disposable application database and read-only
pinned course evidence. The frozen smoke binding preserves the baseline PDF revision
for comparison; latest-published-version acceptance is separate. It saves presentation
files plus protected raw diagnostics. On Windows, monitor the log/journal during a
run and open the atomic JSON archive after it exits to avoid file-sharing conflicts.
These commands do not read reserved quality cases, promote a serving profile,
enable live RAG/cache/intake, or close assessment/citation/latency release gates.
The additional six-case development pack captures combined progress/help and
recall-source contrasts discovered during this step. Preserve it once consumed;
create a new version for changed expectations. The original 42-case pack stays
unchanged. The adapter observes saved state through the actual service and has
controlled regression tests; its model calls remain separate from HTTP coverage.

### Full regression

```powershell
$env:FREDI_APPLICATION_REPOSITORY_INTEGRATION='1'
$env:FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION='1'
$env:FREDI_KNOWLEDGE_REDIS_INTEGRATION='1'
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=reports/regression-new
```

For the separate actual supervisor/planner/specialist signed-HTTP probe:

```powershell
$env:FREDI_RAG_REAL_MODEL='1'
$env:FREDI_C11_REAL_REPORT='reports/real-http-new.json'
.venv/Scripts/python.exe -m pytest tests/persistence/test_sprint5_operations.py -k real_router -q -p no:cacheprovider --basetemp=reports/real-http-temp-new
```

This uses a disposable student database, actual semantic routing, real course/planner/
specialist inference, stored history and retry replay. Its failure must be reported;
do not replace its router with a double to claim an end-to-end model pass.
Keep real-model experiments serial to avoid competing for local model capacity.
The four-student concurrency/rollback regression uses real PostgreSQL and model
doubles: it tests isolation and persistence, not GPU throughput or a student latency SLA.

See [Checkpoint 11 results](../checkpoints/sprint-5-checkpoint-11.md) and the
[release checklist](../releases/sprint-5-release-checklist.md) before any pilot enablement.
