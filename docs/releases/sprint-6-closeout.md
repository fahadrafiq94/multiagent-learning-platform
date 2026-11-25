# Sprint 6 closeout

Closed on 25 November 2026. FREDi now connects the student browser workspace to
the agent backend and can run inference on DGX Spark. The local pilot uses
operator-issued credentials. The full application deployment and institutional
student login are scheduled for Sprint 7.

| Delivered | What students or operators can do |
| --- | --- |
| Spark model integration | Select vLLM chat models and endpoints per agent role, configure the retrieval planner separately, and use the existing course index with compatible Ollama embeddings on Spark. One Qwen serving configuration was exercised. |
| Student workspace | Sign in, describe and confirm a company scenario, switch between saved chats, see their headings and archived sources, edit messages, and recover a turn after a lost connection. Earlier untitled chats with a student message received headings. |
| Onboarding assistance | Ask the Scenario Agent for suggestions while building or reviewing the scenario and choose an option before confirming the draft. |
| Changes after onboarding | Propose a new learner role in chat. The original role stays active until the student confirms the change in that conversation. Students can cancel the proposal. |
| Performance groundwork | Measure reuse of common prompt prefixes on Spark. A serial synthetic run observed reuse on supervisor routing calls. Concurrent pilot capacity still needs measurement. |

Recorded workspace validation included 13 Chromium tests and 17 targeted PostgreSQL
tests. Focused onboarding tests passed, and a regression test verified that another
chat cannot confirm a proposed role change. The local backend passed readiness checks.
Live-model recognition of role changes, the quality of onboarding suggestions, and
teaching and citation quality remain to be evaluated.

For local operation, see [student browser access](../operations/student-browser-access.md)
and the [vLLM provider runbook](../operations/vllm-provider.md).
The [Sprint 7 roadmap](../ROADMAP.md#sprint-7) owns Prometheus and Grafana, Spark
application deployment, student login integration and the deferred Sprint 5 evaluation.
