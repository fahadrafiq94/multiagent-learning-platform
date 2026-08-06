from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.orchestration.state import OrchestrationState, RouteName

GraphRouteTarget = Literal[
    "scenario_path",
    "process_coach_path",
    "ap_plus_navigator_path",
    "fallback_path",
]


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic routing result for the Sprint 2 graph."""

    route: RouteName
    reason: str


AP_PLUS_NAVIGATOR_KEYWORDS = (
    "ap+",
    "ap plus",
    "screen",
    "menu",
    "field",
    "button",
    "click",
    "interface",
    "navigation",
    "where do i find",
    "where can i find",
    "error message",
    "error code",
)

SCENARIO_KEYWORDS = (
    "scenario",
    "company name",
    "company address",
    "company context",
    "business problem",
    "business case",
    "my company",
    "our company",
    "student role",
)

PROCESS_COACH_KEYWORDS = (
    "business process",
    "process",
    "why",
    "what should happen next",
    "next business step",
    "brainstorm",
    "understand",
    "reason",
    "procurement",
    "concept",
)


def classify_route(user_message: str) -> RouteDecision:
    """Classify a message using deterministic Sprint 2 routing rules.

    This is intentionally simple and observable. Sprint 3 may introduce
    richer agent selection, but deterministic fallback behavior should remain.
    """

    normalized_message = " ".join(user_message.lower().split())

    if _contains_keyword(
        normalized_message,
        AP_PLUS_NAVIGATOR_KEYWORDS,
    ):
        return RouteDecision(
            route="ap_plus_navigator",
            reason="The message requests AP+-specific interface or navigation help.",
        )

    if _contains_keyword(
        normalized_message,
        SCENARIO_KEYWORDS,
    ):
        return RouteDecision(
            route="scenario",
            reason="The message concerns company or business scenario context.",
        )

    if _contains_keyword(
        normalized_message,
        PROCESS_COACH_KEYWORDS,
    ):
        return RouteDecision(
            route="process_coach",
            reason="The message concerns business-process reasoning or understanding.",
        )

    return RouteDecision(
        route="fallback",
        reason="No specialized Sprint 2 routing rule matched the message.",
    )


def route_to_graph_target(state: OrchestrationState) -> GraphRouteTarget:
    """Map the selected application route to a LangGraph node name."""

    route = state.get("route", "fallback")

    route_targets: dict[RouteName, GraphRouteTarget] = {
        "scenario": "scenario_path",
        "process_coach": "process_coach_path",
        "ap_plus_navigator": "ap_plus_navigator_path",
        "fallback": "fallback_path",
    }

    return route_targets[route]


PrepareTarget = Literal[
    "route_request",
    "finalize_response",
]


def route_after_input_preparation(
    state: OrchestrationState,
) -> PrepareTarget:
    """Skip routing and model execution when input preparation failed."""

    if state.get("error"):
        return "finalize_response"

    return "route_request"


def _contains_keyword(
    normalized_message: str,
    keywords: tuple[str, ...],
) -> bool:
    return any(keyword in normalized_message for keyword in keywords)
