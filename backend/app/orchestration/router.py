from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.orchestration.state import (
    OrchestrationState,
    RouteName,
)

GraphRouteTarget = Literal[
    "scenario_agent",
    "process_coach_agent",
    "ap_plus_navigator_agent",
    "fallback_response",
]


PrepareTarget = Literal[
    "route_request",
    "finalize_response",
]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Deterministic fallback-routing decision.

    The primary Sprint 3 routing strategy is the semantic OrchestratorAgent.

    This decision type is used only by the deterministic keyword router when
    semantic routing cannot confidently select a specialized agent or when
    semantic routing fails.
    """

    route: RouteName
    reason: str


AP_PLUS_NAVIGATOR_KEYWORDS: tuple[str, ...] = (
    "ap+",
    "ap plus",
    "screen",
    "menu",
    "field",
    "button",
    "click",
    "interface",
    "navigation",
    "navigate",
    "where do i find",
    "where can i find",
    "where should i enter",
    "where do i enter",
    "where should i put",
    "where do i put",
    "error message",
    "error code",
)


SCENARIO_KEYWORDS: tuple[str, ...] = (
    "scenario",
    "company name",
    "company address",
    "company context",
    "business problem",
    "business case",
    "my company",
    "our company",
    "student role",
    "my role",
    "our role",
)


PROCESS_COACH_KEYWORDS: tuple[str, ...] = (
    "business process",
    "process",
    "why",
    "what should happen next",
    "what happens next",
    "next business step",
    "brainstorm",
    "understand",
    "reason",
    "reasoning",
    "procurement",
    "production",
    "sales",
    "logistics",
    "concept",
)


def classify_route(
    user_message: str,
) -> RouteDecision:
    """Classify a message using deterministic keyword fallback rules.

    This router is intentionally simple, deterministic, and observable.

    It is NOT the primary Sprint 3 routing mechanism.

    The semantic OrchestratorAgent should run first. This function is used
    only when:

    1. the semantic orchestrator explicitly selects ``fallback``; or
    2. semantic routing fails and deterministic recovery is required.

    If no deterministic rule matches, ``fallback`` is returned and the graph
    should ask the student to clarify what type of help they need.
    """

    normalized_message = _normalize_message(user_message)

    if not normalized_message:
        return RouteDecision(
            route="fallback",
            reason=("The deterministic fallback router received no usable message content."),
        )

    if _contains_keyword(
        normalized_message,
        AP_PLUS_NAVIGATOR_KEYWORDS,
    ):
        return RouteDecision(
            route="ap_plus_navigator",
            reason=(
                "The deterministic fallback router detected "
                "AP+-specific interface or navigation intent."
            ),
        )

    if _contains_keyword(
        normalized_message,
        SCENARIO_KEYWORDS,
    ):
        return RouteDecision(
            route="scenario",
            reason=(
                "The deterministic fallback router detected company or business-scenario context."
            ),
        )

    if _contains_keyword(
        normalized_message,
        PROCESS_COACH_KEYWORDS,
    ):
        return RouteDecision(
            route="process_coach",
            reason=(
                "The deterministic fallback router detected business-process reasoning intent."
            ),
        )

    return RouteDecision(
        route="fallback",
        reason=("No deterministic fallback routing rule matched the message."),
    )


def route_to_graph_target(
    state: OrchestrationState,
) -> GraphRouteTarget:
    """Map the selected routing target to its LangGraph node.

    Semantic classification has already happened before this function runs.

    This function performs only deterministic graph control-flow mapping.
    """

    route = state.get(
        "route",
        "fallback",
    )

    route_targets: dict[
        RouteName,
        GraphRouteTarget,
    ] = {
        "scenario": "scenario_agent",
        "process_coach": "process_coach_agent",
        "ap_plus_navigator": ("ap_plus_navigator_agent"),
        "fallback": "fallback_response",
    }

    return route_targets[route]


def route_after_input_preparation(
    state: OrchestrationState,
) -> PrepareTarget:
    """Determine whether routing should run after input preparation.

    Invalid input skips semantic and deterministic routing and proceeds
    directly to finalization where the existing error is returned.
    """

    if state.get("error"):
        return "finalize_response"

    return "route_request"


def _normalize_message(
    user_message: str,
) -> str:
    """Normalize user input for deterministic keyword matching."""

    return " ".join(user_message.strip().lower().split())


def _contains_keyword(
    normalized_message: str,
    keywords: tuple[str, ...],
) -> bool:
    """Return whether any fallback-routing keyword is present."""

    return any(keyword in normalized_message for keyword in keywords)
