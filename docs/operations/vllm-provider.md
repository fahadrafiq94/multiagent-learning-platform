# vLLM and independent model profiles (Sprint 6 Checkpoints 1–2)

The chat adapter is implemented behind the existing model-service facade. Ollama
remains the default. Provider selection is explicit; there is no automatic fallback
or model download. Independent planner, embedding and per-role profiles are wired
through runtime and operator workflows in Checkpoint 2.

## Configuration

For native backend execution, configure its private `backend/.env`:

```dotenv
MODEL_PROVIDER=vllm
VLLM_BASE_URL=http://192.168.40.19:8000/v1
VLLM_CHAT_MODEL=nvidia/Qwen3.8-27B-NVFP4
VLLM_REASONING_CONTROL=enable_thinking
VLLM_REASONING_ENABLED=false

# Optional, only when required by the inference endpoint:
# VLLM_API_KEY=your-private-key

ORCHESTRATOR_MODEL=
SCENARIO_AGENT_MODEL=
PROCESS_COACH_MODEL=
AP_PLUS_NAVIGATOR_MODEL=
SUMMARY_MODEL=

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://192.168.40.19:11434
EMBEDDING_MODEL=nomic-embed-text-v2-moe:latest

RAG_PLANNER_PROVIDER=vllm
RAG_PLANNER_BASE_URL=http://192.168.40.19:8000/v1
RAG_PLANNER_MODEL=nvidia/Qwen3.8-27B-NVFP4
```

Empty role values inherit the selected provider's default. `PROCESS_COACH_AGENT_MODEL`
is also accepted for compatibility; prefer `PROCESS_COACH_MODEL` and do not set both.
The model names must be aliases actually served at this endpoint. The known Spark
server currently advertises only `nvidia/Qwen3.8-27B-NVFP4`. A missing alias is rejected,
not replaced by the default. Setting a different name does not start another model.
Existing role names are wired through the actual orchestration factory, including
the shared Scenario agent used for onboarding and the configured summary model.

`VLLM_REASONING_CONTROL=enable_thinking` declares a verified chat-template capability.
The configured default is used when a request has no explicit reasoning preference;
an explicit request takes precedence. Unset `VLLM_REASONING_ENABLED` preserves the
server default. A request asking for reasoning control without a declared supported
mapping fails clearly. Do not assume this Qwen setting applies to every other model.

Settings load on process startup. Restart the backend through its existing workflow
after deliberate configuration changes. This does not call `vllm-start`/`vllm-stop`
or modify the Spark server. Checkpoint 1 left private configuration unchanged.
During Checkpoint 2, six missing embedding/planner entries shown above were added
to the local `.env`, preserving existing values, secrets and serving opt-ins.
The full Checkpoint 2 validation subsequently cleared four old `qwen3:4b` agent
overrides so those roles inherit the new primary vLLM model. If switching providers,
review all explicit role aliases: an old Ollama alias is not automatically translated
to a vLLM model. Missing aliases fail rather than silently falling back.

Development Compose reads the provider from `backend/.env`; production Compose reads
it from `backend/.env.production`. Their previous hardcoded `MODEL_PROVIDER=ollama`
entries were removed; the settings default and supplied templates remain Ollama.
The existing stacks still start their Ollama service. A dedicated Spark application
deployment without unnecessary local inference containers belongs to Checkpoint 6.

Keep RAG/intake/cache opt-ins unchanged until deliberate application admission.
The vLLM chat adapter deliberately rejects embedding
generation/inspection instead of directing the current vectors to a chat model.
The separate Spark embedding setup is documented in
[the embedding runbook](spark-embedding-setup.md).

## Profile inheritance and overrides

The supported role prefixes are `ORCHESTRATOR`, `SCENARIO_AGENT`, `PROCESS_COACH`,
`AP_PLUS_NAVIGATOR`, `SUMMARY` and `RAG_PLANNER`. Each supports `_MODEL`, `_PROVIDER`,
`_BASE_URL`, `_API_KEY`, `_REASONING_CONTROL` and `_REASONING_ENABLED`.
These are trusted server settings; request metadata cannot change routing.

For example, use a different served Process Coach model without editing an agent:

```dotenv
PROCESS_COACH_PROVIDER=vllm
PROCESS_COACH_BASE_URL=http://another-private-host:8000/v1
PROCESS_COACH_MODEL=the-alias-served-there
PROCESS_COACH_REASONING_CONTROL=enable_thinking
PROCESS_COACH_REASONING_ENABLED=false
# PROCESS_COACH_API_KEY=inject-this-endpoints-key-privately-if-required
```

The selected provider supplies its default URL/model for unspecified fields.
The five chat roles inherit the primary profile by default. A model-only override
uses the same endpoint. A different provider uses that provider's configured defaults.
Keys and reasoning defaults inherit only for the same normalized endpoint/provider;
changing endpoints requires deliberate endpoint-specific configuration. An explicit
empty role API key clears an inherited key. Ollama API-key configuration is rejected
because the existing adapter does not implement authenticated Ollama serving.
An explicit request reasoning setting takes precedence over the profile default.

Planner settings are compatible with existing trusted RAG JSON: absent new overrides
retain its Ollama model/URL and timeout. Explicit `RAG_PLANNER_*` values take precedence;
selecting vLLM does not reuse an old Ollama alias. Legacy planner URL/model may still
supply unspecified Ollama fields. The effective planner model is also passed into
the actual planner request, rather than merely changing its HTTP endpoint.

Embedding settings always resolve independently, including model inspection, query
retrieval, index construction, intake worker and the management wizard. Ollama is
the supported embedding provider; vLLM embedding selection is rejected until an
inspection-compatible implementation exists. The admitted revision still determines
model digest, dimensions, normalization and input prefixes. A conflicting explicit
`EMBEDDING_MODEL` fails before generation; digest/dimension checks still run against
the selected server. Moving the same model to Spark does not rebuild the index.

Operator precedence is an explicit CLI endpoint, then new embedding/planner settings,
then legacy trusted JSON URL, then provider default. CLI endpoint changes never forward
a key from the configured endpoint to a different endpoint. `plan_knowledge.py`
uses `--model-url` for the planner, `--planner-provider` for its provider and
`--embedding-url` independently. Index/retrieve scripts use `--model-url` for embeddings.
The management wizard no longer falls back to localhost when an explicit embedding
endpoint is unavailable; it reports the configured service failure.

Identical role profiles share their client within the facade. Composed health reports
chat, configured role and embedding readiness separately, with one ten-second total
deadline. This is availability/model-presence inspection, not evidence that the active
index digest is admitted or that answer quality is acceptable. Lifespan and operator
exit paths close all owned model clients, including error/partial-startup paths.

## Adapter behavior

- Accepts a server-root or `/v1` API base; normalizes the API suffix once. Rejects
  credentials/query strings embedded in URLs. Optional API keys use protected headers;
  errors never echo provider bodies, keys or source/student text.
- Discovers advertised aliases/context limits through `/v1/models`. Before generation,
  uses `/tokenize` with the same messages and reasoning-template options to check
  the request's actual input/output reservation. No prompt/evidence truncation.
- Translates structured requests to `response_format.json_schema`; validation of
  domain output remains with application Pydantic models and source/teaching guards.
- Applies an explicit output ceiling: request override, configured `MODEL_MAX_TOKENS`,
  then a 2048-token vLLM default. Retains the application's context budgets; the
  server advertising 262144 tokens does not enlarge an 8192-token request.
- Rejects truncated/unsupported finishes, unexpected models/tools, malformed usage,
  measured budget overflow and reasoning output when reasoning is explicitly disabled.
- Bounded retries apply to connection setup and selected transient HTTP statuses.
  Authentication/client errors, read timeouts, cancellations and partially delivered
  streams are not automatically replayed. A wall deadline includes metadata/tokenization.
- Parses provider SSE through `[DONE]`, validates final usage and reports malformed,
  incomplete or over-budget streams as failures. This is provider-level support, not
  permission to expose unverified specialist output to students.
- Reuses bounded HTTP connection pools. The app owns/cleans the service through lifespan,
  and `/model/*` reuses it rather than leaking per-request providers. Health checks
  refresh the default model's availability, use a short deadline and do not retry.

The `/model/*` routes are technical model APIs. The student frontend must use the
authorized orchestration/conversation APIs and display completed verified replies.
Provider streaming does not bypass RAG, teaching, assessment or persistence checks.

## Validation and scope

Checkpoint 1 includes mocked HTTP/factory/ASGI tests, actual per-agent request selection,
existing real PostgreSQL/Redis runtime regression, and read-only Linux validation.
Small opt-in technical smokes on Spark confirmed simple structured JSON, the real
routing schema, streaming and unavailable-alias rejection. These are not teaching,
citation, concurrency or final student-release acceptance.

The original and failed/corrected technical probe artifacts are retained privately
under `backend/reports/`; see the local Checkpoint 1 report for exact outcomes.
The 30-second support-verifier limit remains unchanged.
