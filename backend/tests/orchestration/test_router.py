from __future__ import annotations

from app.orchestration.router import (
    classify_route,
    route_after_context_load,
    route_after_input_preparation,
    route_to_graph_target,
)


def test_classify_route_detects_ap_plus() -> None:
    decision = classify_route("Where can I find this field in AP+?")

    assert decision.route == "ap_plus_navigator"


def test_classify_route_detects_scenario() -> None:
    decision = classify_route("I need help understanding our company scenario.")

    assert decision.route == "scenario"


def test_classify_route_detects_process_reasoning() -> None:
    decision = classify_route("Why is this purchase order needed?")

    assert decision.route == "process_coach"


def test_classify_route_returns_fallback() -> None:
    decision = classify_route("Hello there.")

    assert decision.route == "fallback"


def test_ap_plus_has_priority_over_process_keyword() -> None:
    decision = classify_route("Why do I need this field in AP+?")

    assert decision.route == "ap_plus_navigator"


def test_scenario_has_priority_over_process_keyword() -> None:
    decision = classify_route("Why is our company scenario defined this way?")

    assert decision.route == "scenario"


def test_classify_route_is_case_insensitive() -> None:
    decision = classify_route("WHERE CAN I FIND THIS IN AP+?")

    assert decision.route == "ap_plus_navigator"


def test_route_after_input_preparation_loads_context() -> None:
    result = route_after_input_preparation(
        {
            "error": None,
        }
    )

    assert result == "load_context"


def test_route_after_input_preparation_finalizes_on_error() -> None:
    result = route_after_input_preparation(
        {
            "error": ("Invalid request"),
        }
    )

    assert result == "finalize_response"


def test_route_after_context_load_uses_onboarding_when_required() -> None:
    result = route_after_context_load(
        {
            "error": None,
            "onboarding_required": (True),
        }
    )

    assert result == "scenario_onboarding"


def test_route_after_context_load_routes_normally_after_onboarding() -> None:
    result = route_after_context_load(
        {
            "error": None,
            "onboarding_required": (False),
        }
    )

    assert result == "route_request"


def test_route_after_context_load_finalizes_on_error() -> None:
    result = route_after_context_load(
        {
            "error": ("Context failed"),
        }
    )

    assert result == "finalize_response"


def test_route_to_graph_target_scenario() -> None:
    assert (
        route_to_graph_target(
            {
                "route": "scenario",
            }
        )
        == "scenario_agent"
    )


def test_route_to_graph_target_process_coach() -> None:
    assert (
        route_to_graph_target(
            {
                "route": ("process_coach"),
            }
        )
        == "process_coach_agent"
    )


def test_route_to_graph_target_ap_plus() -> None:
    assert route_to_graph_target(
        {
            "route": ("ap_plus_navigator"),
        }
    ) == ("ap_plus_navigator_agent")


def test_route_to_graph_target_defaults_to_fallback() -> None:
    assert route_to_graph_target({}) == "fallback_response"
