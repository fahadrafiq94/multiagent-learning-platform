from __future__ import annotations

from app.agents.schemas import (
    AgentName,
)
from app.context.state_updates import (
    SessionStateField,
)

_FIELD_ORDER: tuple[
    SessionStateField,
    ...,
] = (
    SessionStateField.CURRENT_OBJECTIVE,
    SessionStateField.CURRENT_STEP_ID,
    SessionStateField.CURRENT_PROCESS_PHASE,
    SessionStateField.ACTIVE_BLOCKER,
    SessionStateField.UNRESOLVED_QUESTIONS,
)

_FIELD_DESCRIPTIONS: dict[
    SessionStateField,
    str,
] = {
    SessionStateField.CURRENT_OBJECTIVE: (
        "current_objective\n"
        "- The immediate learning or task objective.\n"
        "- Use SET only when the user's current objective is clear.\n"
        "- Use CLEAR only when the objective is explicitly no longer active."
    ),
    SessionStateField.CURRENT_STEP_ID: (
        "current_step_id\n"
        "- The explicit exercise or AP+ step identifier.\n"
        "- Never infer or invent a step number.\n"
        "- Use SET only when the step is explicitly supported by the current\n"
        "  message or supplied context."
    ),
    SessionStateField.CURRENT_PROCESS_PHASE: (
        "current_process_phase\n"
        "- The current business-process phase.\n"
        "- Examples might include procurement, planning, production, or another\n"
        "  phase explicitly supported by context.\n"
        "- Never invent a phase."
    ),
    SessionStateField.ACTIVE_BLOCKER: (
        "active_blocker\n"
        "- The single most immediate blocker preventing progress.\n"
        "- Use SET only when a blocker is clear.\n"
        "- Use CLEAR when the blocker has clearly been resolved."
    ),
    SessionStateField.UNRESOLVED_QUESTIONS: (
        "unresolved_questions\n"
        "- Use ADD when a meaningful unresolved question remains.\n"
        "- Use REMOVE when a previously unresolved question has clearly been\n"
        "  answered.\n"
        "- Use CLEAR only when all unresolved questions are clearly resolved."
    ),
}

# Single source of truth for which runtime state fields each agent may
# propose. SessionStateUpdateService enforces the same mapping.
AGENT_STATE_FIELD_POLICY: dict[
    AgentName,
    frozenset[SessionStateField],
] = {
    "scenario": frozenset(
        {
            SessionStateField.CURRENT_OBJECTIVE,
            SessionStateField.ACTIVE_BLOCKER,
            SessionStateField.UNRESOLVED_QUESTIONS,
        },
    ),
    "process_coach": frozenset(
        {
            SessionStateField.CURRENT_OBJECTIVE,
            SessionStateField.CURRENT_PROCESS_PHASE,
            SessionStateField.ACTIVE_BLOCKER,
            SessionStateField.UNRESOLVED_QUESTIONS,
        },
    ),
    "ap_plus_navigator": frozenset(
        {
            SessionStateField.CURRENT_STEP_ID,
            SessionStateField.CURRENT_PROCESS_PHASE,
            SessionStateField.ACTIVE_BLOCKER,
            SessionStateField.UNRESOLVED_QUESTIONS,
        },
    ),
}


def state_update_proposal_instructions(
    agent_name: AgentName,
) -> str:
    """
    Build the state-update proposal instructions for one agent.

    Only the fields the application actually allows this agent to
    propose are listed, so the model does not waste operations on
    proposals that are always rejected.
    """

    allowed = AGENT_STATE_FIELD_POLICY[agent_name]

    field_sections = "\n\n".join(
        _FIELD_DESCRIPTIONS[field] for field in _FIELD_ORDER if field in allowed
    )

    return f"""
SESSION RUNTIME STATE PROPOSALS

In addition to your user-facing response, you may propose conservative
updates to the current session runtime state.

These proposals are not persisted by you. Application code validates
them before any state change occurs.

You may ONLY propose changes to these fields:

{field_sections}

Strict rules:

- Be conservative.
- An empty operations list is preferred when no state change is clearly
  justified.
- Never propose fields that are not listed above.
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
