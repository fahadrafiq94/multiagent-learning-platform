from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.orchestration.state import (
    OrchestrationState,
    RouteName,
)


@dataclass(frozen=True)
class RouteDecision:
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
    "my role",
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


def classify_route(
    user_message: str,
) -> RouteDecision:
    """Deterministic fallback router."""

    normalized = " ".join(user_message.casefold().split())

    if _contains_keyword(
        normalized,
        AP_PLUS_NAVIGATOR_KEYWORDS,
    ):
        return RouteDecision(
            route="ap_plus_navigator",
            reason=("The message requests AP+-specific system help."),
        )

    if _contains_keyword(
        normalized,
        SCENARIO_KEYWORDS,
    ):
        return RouteDecision(
            route="scenario",
            reason=("The message concerns the business scenario."),
        )

    if _contains_keyword(
        normalized,
        PROCESS_COACH_KEYWORDS,
    ):
        return RouteDecision(
            route="process_coach",
            reason=("The message concerns business-process reasoning."),
        )

    return RouteDecision(
        route="fallback",
        reason=("No deterministic fallback route matched."),
    )


InputPreparationTarget = Literal[
    "load_context",
    "finalize_response",
]


def route_after_input_preparation(
    state: OrchestrationState,
) -> InputPreparationTarget:
    if state.get("error"):
        return "finalize_response"

    return "load_context"


ContextLoadTarget = Literal[
    "scenario_onboarding",
    "route_request",
    "finalize_response",
]


def route_after_context_load(
    state: OrchestrationState,
) -> ContextLoadTarget:
    if state.get("error"):
        return "finalize_response"

    if state.get(
        "onboarding_required",
        False,
    ):
        return "scenario_onboarding"

    return "route_request"


GraphRouteTarget = Literal[
    "scenario_agent",
    "process_coach_agent",
    "ap_plus_navigator_agent",
    "fallback_response",
]


def route_to_graph_target(
    state: OrchestrationState,
) -> GraphRouteTarget:
    route = state.get(
        "route",
        "fallback",
    )

    mapping: dict[
        RouteName,
        GraphRouteTarget,
    ] = {
        "scenario": "scenario_agent",
        "process_coach": ("process_coach_agent"),
        "ap_plus_navigator": ("ap_plus_navigator_agent"),
        "fallback": ("fallback_response"),
    }

    return mapping[route]


def _contains_keyword(
    normalized_message: str,
    keywords: tuple[str, ...],
) -> bool:
    return any(keyword in normalized_message for keyword in keywords)
