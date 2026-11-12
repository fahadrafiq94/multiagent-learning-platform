# Student-quality review rubric v1

Prepared: 2026-11-09, before application quality fixes.
Scope: development regressions and later acceptance review. This rubric does not
change student-facing policy, award learning signals or authorize a release.

## What each result means

| Result | Meaning |
|---|---|
| `pass` | Every declared observed contract passes and no semantic review remains. |
| `fail` | At least one observed contract contradicts its declared expectation. |
| `not_observed` | The case was not executed, failed before observation, or a required observable is absent. |
| `review_pending` | Observed contracts pass, but the semantic criteria still require review. |

Explicit null is different from a missing field. No response, absent telemetry,
HTTP/model errors, unknown support and unexecuted variants cannot be counted as
successes. Boolean metrics are not interchangeable with integers. A case with a
known failure remains failed even if other observations are missing.

Automated checks compare saved/extracted fields, routes, assessment categories,
signal-recording flags, retrieval actions, selected guidance and citation identity.
They do not infer understanding from question marks, word counts, exact phrasing,
the existence of a citation or a successful HTTP response.

## Semantic review

Record a verdict (`pass`, `fail` or `uncertain`), a short reason and the case/run ID.
For support checks, include the actual answer claim, its cited label and the
supporting passage/location. Store raw excerpts in the protected diagnostics.
Uncertain or unreviewed criteria stay pending. Any clear failure prevents case
acceptance. An automated judge may assist but cannot be the sole authority.

| Rubric | Pass condition | Failure examples |
|---|---|---|
| `fact_fidelity` | Preserve explicitly supplied facts and corrections without inventing missing facts. Assess actual reasoning when the student supplies it. | Ignore a supplied role; copy a role from a quoted example; invent a company goal. |
| `state_intent` | Distinguish an explicit current-step change from a hypothetical, quotation, denial or source instruction. Current does not mean completed. | Update progress because a question mentions a different step; answer correctly from history while the required saved-state update is missing. |
| `routing` | Select the specialist matching the immediate need: scenario facts, conceptual reasoning or UI execution. Resolve short follow-ups from context. | Route prerequisite reasoning to Scenario solely because the company scenario appears in history; discard a substantive question after “thanks.” |
| `teaching_style` | Follow the selected guidance level for learning help, staying focused on the current need. | Give the prerequisite answer before a minimal-guidance question; turn a hint into a future-step procedure. |
| `claim_support` | Each substantive source claim is supported by the cited passage or explicit combination of cited passages. Distinguish prerequisites, business explanation and UI details. | Cite a step-name fact for an Address/Picklist claim; add an unsupported extra instruction to a supported answer; attach irrelevant evidence to an invented configuration value. |
| `acknowledgement` | Acknowledge a clear social turn naturally without needless routing clarification, retrieval or grading. Preserve substantive content in mixed turns. | Ask the student to choose a type of help after an unambiguous thank-you. |
| `retrieval_need` | Use student state/history for recall; retrieve for genuinely new external facts; handle mixed requests by their separate needs. | Search PDFs for the supplier the student chose; skip a new handbook question because the message also asks for history recall. |

A correct citation label proves identity only. A source quotation alone does not
prove that a broader explanation follows from it. If the answer is supported by
an uncited supplied passage, its citation support is still incomplete. Evaluate
the claims actually made: omission of an unnecessary UI detail does not require
citing that UI passage. Unsupported factual claims must be omitted, qualified or
repaired with admitted supporting evidence.

## Four-level teaching expectations

| Level | Expected learning-help behavior |
|---|---|
| Minimal | One focused reasoning/context question; no corrective answer before it. |
| Guided | One conceptual clue and one focused question; no procedural sequence. |
| Directed | One targeted hint with a brief explanation, limited to the immediate decision/action. |
| Explicit | Direct verified help for the immediate question, with a brief explanation and no unrelated steps. |

The 12 controlled teaching probes cover three specialists x four levels. Their
specified specialist/level are isolated test inputs, not production routing or
guidance overrides. Natural journeys must observe the policy's real choices.
Only minimal guidance in a natural journey is a coverage limitation, not proof
that deterministic adaptation is broken. Do not force escalation to improve a chart.
Step 4 will explicitly review guidance scope for acknowledgements and state recall;
this rubric does not impose minimal teaching questions on those interactions.

## Case fixtures and application wiring

The [42-case development pack](student-quality-development-v1.json) contains 11
case mappings to historical turns, 19 other conversation/extraction contrasts and
12 controlled teaching probes. One turn can test more than one finding; case counts
are not independent student counts. The new contrast cases use different roles,
companies, references and general source material without mandatory step IDs.

`context.evidence` describes synthetic source fixtures. For a conversation/RAG
execution adapter, ingest/serve these through the authorized fixture knowledge
repository; do not put their contents directly into student memory and then grade
retrieval as necessary. In controlled specialist-only probes, evidence is explicitly
supplied as admitted current evidence, and retrieval is not being scored.

Conversation cases start from a confirmed synthetic scenario. Original mapped
cases use the actual archived pre-turn context when reproduced live; the Step 1
tool only replays recorded observations. Context overrides describe each contrast.
Focused live/component execution adapters are added with the corresponding fixes;
the historical replay does not claim to execute all 42 cases against the agents.

Only inputs, trusted setup and admitted evidence enter agent requests. `checks`,
`reviews`, finding labels and gold expectations are evaluator data and must never
be appended to model prompts. Tests of the scorer use controlled observations;
they verify detection of failures, not real-model quality or a fixed application.

The original AP40-02/AP40-10 navigation/recall wording remains unchanged. Its new
replay case checks retrieval necessity, not a disputed exact progress label.
New contrasts distinguish explicit switches from hypothetical references.

## Baseline replay

From `backend`, as the protected archive's owner:

```powershell
.venv/Scripts/python.exe -m scripts.student_quality --archive reports/student-journey-20261106-presentation-2/run.json --output reports/student-quality-baseline-new
```

Choose a new output directory. The tool reads the declared synthetic baseline,
records its SHA-256 and the suite/analyzer hashes, and writes protected JSON and
Markdown. It performs no model calls, database operations or feature fixes.
Semantic verdicts remain pending rather than being guessed from strings.
Exit code 0 means the replay/export completed, not that quality passed. Inspect
the case statuses and completeness field before interpreting any result.

The baseline includes the actual serving profile and per-turn/per-chat-call timing
from the historical archive. Missing historical source-code hashes remain unknown;
the analyzer's present-day hash is not misrepresented as the code used in November 6
model execution. Current application hashes can be recorded separately as review
provenance before Step 2.

## Reserved acceptance cases

The [reserved-pack manifest](student-quality-reserved-v1-manifest.json) records the
hash and coverage of 16 separate, unexecuted cases. Inputs/expectations are in the
protected, Git-ignored path in that manifest, not in development prompts or tests.
The development replay command rejects held-out packs. Retain the private pack in
the operator's backup; a Git clone contains only its public manifest.

These are assistant-authored cases, not an independent human-authored benchmark.
Review educational expectations before execution; any revision requires a new
version/hash rather than overwriting the reserved pack. Enable acceptance execution
explicitly in Step 6, verify its hash, freeze labels before use and report all cases,
errors and missing results. Once consumed for debugging/tuning, cases become
development regressions and cannot be described as fresh held-out evidence again.

Per-case criteria are fixed here. Broader pilot quality/capacity thresholds and a
latency target must be agreed before final candidate acceptance, as Step 6 requires.
No global quality target or passing latency is invented from the observed baseline.
