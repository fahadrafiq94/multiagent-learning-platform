from __future__ import annotations

from typing import Any, Literal, TypedDict

RouteName = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
    "fallback",
]


class OrchestrationState(TypedDict, total=False):
    """Shared state passed through the LangGraph workflow.

    This is intentionally minimal in Sprint 2.
    Later sprints will extend this with context, memory, checklist state,
    retrieved knowledge, and agent-specific outputs.
    """

    # Request identity
    session_id: str
    student_id: str | None

    # Input
    user_message: str

    # Routing
    route: RouteName
    route_reason: str

    # Model interaction
    model_response: str

    # Final output
    final_response: str

    # Error handling
    error: str | None

    # Debug / observability metadata
    metadata: dict[str, Any]
