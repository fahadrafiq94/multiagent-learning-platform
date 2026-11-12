# Sprint 5 release gate

Checkpoint 11 implements evaluation and operational verification. That does not
declare Sprint 5 released or permit student RAG enablement. vLLM remains outside scope.

**2026-11-12 decision:** The user deferred the remaining evaluation to the final
evaluation stage. Its scope, known failures and resumption rules are recorded in the
[deferred evaluation plan](../plans/sprint-5-deferred-evaluation-plan.md). Proceed
with implementation handover; keep unmet acceptance checkboxes open. The verifier
remains at 30 seconds. This is not a release or enablement decision.

- [x] Frozen development/held-out suite, labels, thresholds and serving identity recorded.
- [x] Separate application regression and actual model assessment.
- [x] Real PostgreSQL/pgvector/Redis coverage and Linux regression.
- [x] Coordinated database/source backup and disposable restore verification.
- [x] Restored roles, abandoned-turn recovery and preserved original data checked.
- [x] Independent Redis/RAG rollback and multi-student SQL/HTTP isolation checks.
- [x] Held-out reference/document recall@4 >= 0.90 for each specialist on the narrow v1 suite.
- [ ] Held-out strict unanswerable rejection >= 0.90, with rubric limitations reviewed.
- [ ] Held-out valid and expected hybrid planner behavior >= 0.90.
- [ ] Successful, source-supported teaching probes for all three specialists and four levels.
- [x] Actual supervisor/planner/specialist signed HTTP, history and replay wiring (single exchange).
- [ ] Model latency/capacity accepted for the intended student pilot.
- [ ] Decide trusted pilot bindings, deployment flags and rollback procedure.
- [ ] Explicit release decision; commit/push/merge only when authorized.

Check boxes describe measured gates, not promises. See
[Checkpoint 11](../checkpoints/sprint-5-checkpoint-11.md) for the final outcome and
[the evaluation/recovery runbook](../operations/rag-evaluation-and-recovery.md) for
commands, private artifacts and limitations. New host disaster recovery and
multi-replica serving have not been certified by a same-cluster scratch restore.

Latest follow-up (2026-11-12): [D1/r4 validation](../checkpoints/sprint-5-quality-step6-d1-r4.md)
records 1,254 real-service and 1,086 Linux passes. All twelve r4 requests, replay,
restart and preservation checks pass, but zero citations were archived; source-supported
teaching remains open. Guidance now escalates on app-owned unproductive outcomes.
D2 is deferred under its r4 gate. This does not change any release authorization.
