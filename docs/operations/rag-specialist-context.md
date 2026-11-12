# Optional specialist RAG runtime

Checkpoint 8 connects the existing hybrid planner to the three specialist agents.
It is opt-in and requires authenticated PostgreSQL application storage. The running
development stack keeps RAG disabled. The Checkpoint 7 decision-quality result
(10/13 expected actions) and measured latency still prevent routine student enablement.

## Request flow

1. Admit the signed student identity and durable turn through the existing runtime.
2. Load application-owned context and let the existing supervisor select a specialist.
   Onboarding and fallback responses bypass retrieval.
3. Resolve a trusted server-side course binding using the internal user UUID. Admit
   the collection/revision with that identity and access domain. Request metadata,
   student text and model output cannot grant course access.
4. The Checkpoint 7 hybrid planner chooses skip, reuse, metadata lookup, search or
   clarification. Reuse freshly hydrates previously cited sources under current scope;
   archived citations are hints, never authorization or stored source bodies.
5. Filter the specialist context using the existing policies, then budget the complete
   rendered model request. Supply selected evidence as inert reference data with
   temporary labels K1, K2, etc. Preserve all four adaptive guidance levels.
6. Validate returned labels against evidence actually supplied, recheck authorization
   after generation, and atomically save the assistant response and typed provenance
   using the existing durable turn completion. Stored replay performs no model calls.

The specialist receives relevant material regardless of which PDF supplied it.
There is no classifier, exclusive document-to-agent assignment or instructor review.
Planning and search create no additional student turns. Clarification is one normal
assistant reply. Missing evidence and technical failures do not generate learning
assessments or state proposals that could increase struggle/blocker counters.

## Trusted configuration

Start with [the example](../knowledge/rag-runtime.example.json), copying it to a
server-controlled file such as the ignored repository-root `.rag-runtime.json`.
Replace its dummy IDs/domain with actual authorized internal user and collection IDs.
The pilot supports one unambiguous course binding per user. This file is read at
startup; changing its bindings requires a restart. It is deployment configuration,
not an upload or client-controlled request field.

With revision_id omitted, retrieval follows the active collection pointer. The
current retained collection has no active revision. An explicitly configured,
technically ready revision supports controlled pilot testing before using the
Checkpoint 9 publication flow; adding such a binding does not activate or rewrite that collection.
Neither a filename nor a claimed step ID establishes permission.

Required environment when opting in:

| Setting | Purpose |
|---|---|
| APPLICATION_STORAGE=postgres | Existing authenticated durable runtime |
| RAG_ENABLED=true | Explicit enablement; default is false |
| RAG_CONFIG_FILE | Trusted JSON file path inside the backend |
| RAG_DATABASE_URL | Protected reader DSN for direct process deployment |
| RAG_DATABASE_HOST / RAG_READER_PASSWORD | Compose alternative using the fixed knowledge reader role |
| RAG_DATABASE_POOL_SIZE | Separate reader pool; default 4, maximum 16 |

Keep credentials out of binding JSON and command output. Startup rejects writable
knowledge roles, including roles writable only on another knowledge table or able
to create objects in the knowledge schema. The application writer and knowledge
reader retain separate permissions even when both schemas share local PostgreSQL.

The optional [Compose overlay](../../docker/compose/docker-compose.rag.yml) extends
the existing dev/knowledge/knowledge.dev/application overlays. It binds the file
read-only at `/rag/config.json`; set RAG_CONFIG_PATH to its absolute host path and
use the existing protected backend/.env.knowledge for interpolation. It defaults
RAG_ENABLED to false. Do not enable it merely because regression tests pass.

planner.model/planner_url select the planner only. embedding_url may select the
embedding endpoint, but retrieval still verifies the revision's exact stored model
profile. The specialist continues using the existing application chat ModelService.
The example host endpoint/model is an experimental profile, not a certified release
recommendation. No model installation or main application model change is required
by this implementation.

## Prompt and source contract

The default window is 8192 tokens with 2048 reserved for generation. Budgeting counts
all rendered messages, the response JSON schema and message overhead using the
existing approximate estimator. Essential state/instructions/current question are
preserved; whole evidence items are admitted in rank order, followed by a recent
chronological history suffix. Qualifications are not cut out of passages. The
current message is deduplicated by its persisted ID and rendered once.

If essential content exceeds the budget, the request stops before generation.
The provider receives the configured context window as Ollama num_ctx and the
output reservation as its generation limit. Reported actual prompt usage is checked
against the input allowance. This is approximate admission, not an exact tokenizer
or a guarantee against every provider-specific tokenization difference. See
[Ollama's parameter documentation](https://docs.ollama.com/modelfile).

Prompt reasoning_enabled is optional and preserves the existing teacher default
unless explicitly configured. Planner-only reasoning control remains separate.
RAG-disabled requests retain their previous schema, prompts and provider defaults.

Evidence admission is private server-side context, excluded from serialized agent
input schemas. Public agent JSON cannot submit verified grounding. Text is still
treated as untrusted data; instructions in passages/history do not override teaching
policy, source permissions or application-owned state.

Returned source_references and archived assistant messages record document/revision,
section/pages/hash and a validated current label. Chunk references retain their real
chunk ID. Metadata references record fact/evidence ID and field name with chunk_id
null. Legacy chunk citations remain readable; no SQL migration was needed for these
existing JSONB records. Citation validation proves supplied-source identity, not
that every claim is supported or the answer is complete. No source download route
or frontend citation interface is introduced here.

The API accepts longer student messages than the planner's current 2000-character
question limit. Such a question produces a controlled RAG error rather than silently
truncating the query; the full question remains available to the specialist. Runtime
does not set an untrusted clear-course-query flag to force broad search on planner
failure; the existing safe clarification fallback remains available.

## Verification and rollback

From backend, run the focused suites with the existing virtual environment:

```powershell
.venv/Scripts/python.exe -m pytest tests/agents/test_grounding.py tests/orchestration/test_grounding_graph.py tests/unit/test_rag_lifespan.py -q -p no:cacheprovider --basetemp=reports/pytest-rag-context
```

Real PostgreSQL coverage additionally requires the two existing integration flags:
FREDI_APPLICATION_REPOSITORY_INTEGRATION=1 and
FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION=1. It uses disposable databases. On Windows,
run as the credential-file owner so the existing Windows ACL tests can execute.

The separate real-model test is explicitly opt-in:

```powershell
$env:FREDI_APPLICATION_REPOSITORY_INTEGRATION='1'
$env:FREDI_RAG_REAL_MODEL='1'
$env:FREDI_RAG_MODEL_URL='http://127.0.0.1:11434'
$env:FREDI_RAG_MODEL='qwen3.5:4b'
$env:FREDI_RAG_REAL_REPORT='reports/checkpoint-8-real-model.json'
.venv/Scripts/python.exe -m pytest tests/persistence/test_grounded_real_model.py -q -p no:cacheprovider --basetemp=reports/pytest-rag-real
```

It reads the retained course through the reader role and writes synthetic chat only
to a disposable application database. It fixes supervisor routing to isolate this
integration test, while planner and specialist inference are real. It does not
replace multi-question model evaluation or latency/capacity measurement.

Disable RAG_ENABLED and restart to return to existing chat behavior; stored messages
and citations remain readable. Remove the optional overlay when no longer needed.
Do not delete PostgreSQL volumes, PDFs, indexed revisions or applied migrations.
Automatic intake/publication is provided by [Checkpoint 9](knowledge-intake.md),
optional Redis caching by [Checkpoint 10](knowledge-cache.md),
and broader [Checkpoint 11 evaluation/recovery](rag-evaluation-and-recovery.md).
The Checkpoint 11 actual-model HTTP probe also exercises the real supervisor.
vLLM is outside this sprint.
