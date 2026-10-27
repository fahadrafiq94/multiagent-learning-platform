from __future__ import annotations

from typing import (
    Any,
    Literal,
    TypedDict,
)
from uuid import UUID

from app.agents.schemas import (
    AgentName,
    StudentResponseAssessment,
)
from app.context.schemas import (
    AgentContext,
)
from app.context.state_updates import (
    SessionStateUpdateProposal,
)

RouteName = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
    "fallback",
]


class OrchestrationState(
    TypedDict,
    total=False,
):
    """
    Request-scoped LangGraph state.

    No agent owns this state.
    """

    # External request identity
    session_id: str
    student_id: str

    # Internal identity
    user_id: UUID
    session_uuid: UUID

    # Request trace
    trace_id: UUID

    # Input
    user_message: str
    current_message_id: UUID

    # Context
    base_context: AgentContext
    onboarding_required: bool

    # Routing
    route: RouteName
    route_reason: str

    # Agent output
    selected_agent: AgentName

    agent_response: str

    state_update_proposal: SessionStateUpdateProposal | None

    # Specialist's judgment of the student's current message (4B.9).
    # LLM-generated; only deterministic counters derived from it are
    # persisted.
    response_assessment: StudentResponseAssessment | None

    # Final
    final_response: str

    # Errors
    error: str | None

    # Request-scoped observability
    metadata: dict[str, Any]
