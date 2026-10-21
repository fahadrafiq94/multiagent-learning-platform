from __future__ import annotations

STATE_UPDATE_PROPOSAL_INSTRUCTIONS = """
SESSION RUNTIME STATE PROPOSALS

In addition to your user-facing response, you may propose conservative
updates to the current session runtime state.

These proposals are not persisted by you. Application code validates
them before any state change occurs.

Available fields:

current_objective
- The immediate learning or task objective.
- Use SET only when the user's current objective is clear.
- Use CLEAR only when the objective is explicitly no longer active.

current_step_id
- The explicit exercise or AP+ step identifier.
- Never infer or invent a step number.
- Use SET only when the step is explicitly supported by the current
  message or supplied context.

current_process_phase
- The current business-process phase.
- Examples might include procurement, planning, production, or another
  phase explicitly supported by context.
- Never invent a phase.

active_blocker
- The single most immediate blocker preventing progress.
- Use SET only when a blocker is clear.
- Use CLEAR when the blocker has clearly been resolved.

unresolved_questions
- Use ADD when a meaningful unresolved question remains.
- Use REMOVE when a previously unresolved question has clearly been
  answered.
- Use CLEAR only when all unresolved questions are clearly resolved.

Strict rules:

- Be conservative.
- An empty operations list is preferred when no state change is clearly
  justified.
- Never propose changes to ScenarioCard.
- Never propose changes to UserContext.
- Never propose changes to the learning profile.
- Never propose guidance-level changes.
- Never propose progress changes.
- Never propose session summaries.
- Never propose IDs, versions, timestamps, or ownership fields.
- Never propose last_selected_agent; orchestration controls it.
- Do not infer state merely because something is plausible.
- Proposed state must be supported by the current message or supplied
  context.
""".strip()
