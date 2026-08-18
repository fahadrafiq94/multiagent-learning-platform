from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.agents.policies import GuidanceLevel

RouteName = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
    "fallback",
]


class OrchestrationState(TypedDict, total=False):
    """Shared state passed through the LangGraph workflow."""

    # Request identity
    session_id: str
    student_id: str | None

    # Input
    user_message: str

    # Routing
    route: RouteName
    route_reason: str

    # Guidance Level
    guidance_level: GuidanceLevel

    # Agent execution
    agent_response: str

    # Retained temporarily for compatibility with earlier Sprint 2 code.
    model_response: str

    # Final output
    final_response: str

    # Error handling
    error: str | None

    # Debug / observability metadata
    metadata: dict[str, Any]
