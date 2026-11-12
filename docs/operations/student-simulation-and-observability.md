# Student simulation and retained observability

The simulation exercises real scenario collection, correction and confirmation,
then normal supervisor routing, adaptive guidance and hybrid RAG. Nothing forces
a specialist, confirms a scenario directly, or sets a student's guidance level.
The script provisions a synthetic pilot identity because the application currently
has signed pilot authentication, not a public registration/login flow.

## Run and present

Start the existing local PostgreSQL/pgvector stack and native Ollama. The script
uses the protected `backend/.env.knowledge` credentials under their owner account.
From `backend`, with the existing virtual environment:

```powershell
.venv/Scripts/python.exe -m scripts.run_student_simulation
```

Defaults: `qwen3.5:4b` for onboarding/router/specialists/summary;
`qwen3:1.7b` for the retrieval planner; 8192-token model window, reasoning off,
at most 1024 generated tokens per call, 150-second provider timeout and no retries.
These are development candidates, not promoted serving defaults. A student turn
can contain several model calls, so its duration can exceed the per-call timeout.
Use `--model`, `--planner-model`, `--model-url`, `--timeout`, `--fixture`,
`--output` and `--presentation` for an explicit new run. Output directories must
be new; previous runs are not overwritten. The controlled fixture is
[`student-journey-v2.json`](../knowledge/student-journey-v2.json) (same case
content as the historical `student-journey-v1.json`, which remains frozen with
its recorded sha256).

The runner uses actual signed HTTP handlers through native ASGI transport. This
tests application HTTP behavior without TCP or the Docker chat model. A generated
`fredi_app_test_*` PostgreSQL database stores the synthetic conversation; the
published course is read through the restricted knowledge reader. RAG is enabled
only inside this script's isolated process, and Redis is disabled for this baseline.
Live server flags and records are unchanged. All main application/knowledge table
fingerprints are compared before/after. No PDFs, published revision or frozen
evaluation labels are changed. Cleanup closes runtime resources and removes only
the generated scratch database after archiving. Hard termination may leave that
database; inspect the private manifest before any operator cleanup.

Outputs:

- `docs/demos/student-journey-<run>/report.html`: self-contained presentation with
  escaped transcript, per-turn timing, route, observed guidance, RAG actions,
  citation titles/pages and technical checks. Open locally in a browser.
- The same directory's `report.md` and `metrics.csv`: readable notes and chart data.
- `backend/reports/student-journey-<run>/run.json`: full private request/response,
  state and history snapshots, model identities, fixture hash and checks.
- `model-calls.jsonl`: private synthetic prompts, bounded generated outputs and
  per-call timing/token diagnostics; appended after each chat generation attempt.
- `request-events.jsonl`: retained versioned routine request telemetry, correlated
  by server trace ID and request ID. Final restored history/context are saved too.

The private directory uses verified Windows ACLs or POSIX mode 0700 and is ignored
by Git. Presentation exports require explicitly synthetic data; never pass a real
student record to this exporter. Responses and snapshots are saved after every
turn, so an interrupted run remains inspectable. Model output capture is private
and confined to this operator script; it is not enabled in deployed request logs.

Transport/storage checks cover completed durable turns, one request event per
graph execution, request correlation, history pairs, context estimates/budgets,
corrected scenario persistence, exact-request replay without new model calls or
duplicate messages, and a runtime rebuild followed by history/context reload.
Replays return stored responses and do not create another graph event.
HTTP success alone cannot pass these checks. Missing or failed observations remain
visible; a failed run stops without bypassing onboarding. Expected pedagogical
behavior is printed for human comparison, not automatically declared correct.
Guidance levels are observed rather than manufactured; this journey does not
guarantee all four levels, evidence reuse, cache coverage or a particular route.
Keep the separate deterministic, four-level and held-out evaluations.

## Deployment retention

Messages, confirmed scenario, learning signals and session state already persist
in PostgreSQL. Request telemetry is a separate operational stream. Set:

```dotenv
OBSERVABILITY_LOG_FILE=/var/log/fredi/request-events.jsonl
OBSERVABILITY_LOG_MAX_BYTES=10485760
OBSERVABILITY_LOG_BACKUP_COUNT=5
```

The optional [`docker-compose.observability.yml`](../../docker/compose/docker-compose.observability.yml)
overlay mounts a named volume at `/var/log/fredi` and supplies these settings.
Append it after the selected base/application overlays when deploying. It has
not been applied to the running server. Configure ownership for the actual backend
user; the local development image currently runs as root. The volume survives
container recreation; deleting volumes removes its retained logs.

One process retains the current file plus five rotated files, about 60 MiB maximum
in total. Rotation is size based, not a promise to retain a fixed number of days.
Request export remains opt-in. Disabling the setting closes its file handler;
reconfiguration does not duplicate records. A configured unwritable destination
fails logging setup rather than silently pretending to retain events.

Version 2 events include UTC timestamp, request/trace and pseudonymous user/session
identifiers, outcome/error presence, routing, guidance, model and context metrics,
retrieval status/actions, planner/embedding counts, planner latency and provided
evidence counts. They exclude message/answer text, document passages, raw prompts,
route explanations, free-text summary/guidance reasons and raw exception bodies.
Readers of the older export must migrate from `error`/`route_reason` to the
structured outcome fields. These exclusions concern this request-event stream;
other application/provider loggers have separate logging behavior.

Restrict access to the log volume: identifiers are pseudonymous, not anonymous.
For multiple backend workers/replicas, ship each instance's events to a central
collector with separate local paths/stdout streams; multiple processes must not
rotate the same file. A centralized dashboard, alerting, time-based retention and
backup/export policy are deployment work, not features provided by this overlay.
Collect these versioned events before local rotation if longer retention is needed.
The stream currently covers graph requests, not all HTTP/auth failures or per-request
Redis hit/miss attribution. Trace request events to stored messages and restricted
operator diagnostics when investigating an incident.

## Explicit student facts and current-step diagnostics

Quality Step 2 adds optional `scenario_role_status` and `student_step_status` fields
to version 2 events. Current-step statuses distinguish `omitted`, `rejected`,
`unchanged`, `clarification_required`, `persisted`, `unavailable` and `failed`.
The quote used to anchor a proposed step is not retained in routine events.
Completed HTTP responses also identify the admitted source message for a saved step.

The supervisor interprets the current message in its existing call. Application code
validates and commits an accepted step before planning/specialist context; a current
step never marks progress complete or grants knowledge-management rights. A committed
declaration can remain saved if a later context/model operation fails; inspect the
status rather than assuming the entire turn rolled back. Existing failed/completed
turn replay rules still apply.

For bounded development probes from `backend`, choose new output directories:

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_student_state --output reports/student-state-new
.venv/Scripts/python.exe -m scripts.evaluate_student_state --finding extraction --output reports/student-role-new
```

The tool uses real models with isolated in-memory repositories and protected raw
diagnostics. It never uses held-out cases or writes the main database. See the
[Step 2 evidence](../checkpoints/sprint-5-quality-step-2.md) for the separate real
PostgreSQL/signed-HTTP run, negative cases, source/identifier checks and limitations.
