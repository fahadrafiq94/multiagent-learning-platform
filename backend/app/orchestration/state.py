from __future__ import annotations

from typing import (
    Any,
    Literal,
    TypedDict,
)
from uuid import UUID

from app.context.schemas import (
    AgentContext,
)
from app.domain.guidance import (
    GuidanceLevel,
)
from app.services.scenario_onboarding_interaction_schemas import (
    ScenarioOnboardingTurnResult,
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

    # Onboarding
    onboarding_result: ScenarioOnboardingTurnResult

    # Routing
    route: RouteName
    route_reason: str

    # Guidance fallback
    guidance_level: GuidanceLevel

    # Agent output
    agent_response: str

    # Compatibility
    model_response: str

    # Final
    final_response: str

    # Errors
    error: str | None

    # Request-scoped observability
    metadata: dict[str, Any]
