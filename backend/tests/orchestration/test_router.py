from __future__ import annotations

from app.orchestration.router import (
    classify_route,
    route_after_input_preparation,
    route_to_graph_target,
)


def test_classify_route_selects_scenario() -> None:
    decision = classify_route("I want to define my company name and business problem.")

    assert decision.route == "scenario"
    assert "scenario context" in decision.reason


def test_classify_route_selects_process_coach() -> None:
    decision = classify_route("Why is procurement important in this business process?")

    assert decision.route == "process_coach"
    assert "business-process reasoning" in decision.reason


def test_classify_route_selects_ap_plus_navigator() -> None:
    decision = classify_route("Where can I find the purchase-order screen in AP+?")

    assert decision.route == "ap_plus_navigator"
    assert "AP+-specific" in decision.reason


def test_classify_route_uses_fallback() -> None:
    decision = classify_route("Hello there.")

    assert decision.route == "fallback"
    assert "no specialized" in decision.reason.lower()


def test_ap_plus_route_has_priority_over_process_route() -> None:
    decision = classify_route("Why can I not find this process field in AP+?")

    assert decision.route == "ap_plus_navigator"


def test_scenario_route_has_priority_over_process_route() -> None:
    decision = classify_route("I need help defining our company business problem.")

    assert decision.route == "scenario"


def test_classify_route_is_case_insensitive() -> None:
    decision = classify_route("WHERE CAN I FIND THIS SCREEN IN AP+?")

    assert decision.route == "ap_plus_navigator"


def test_classify_route_normalizes_whitespace() -> None:
    decision = classify_route("Where   can   I   find   this   field   in   AP+?")

    assert decision.route == "ap_plus_navigator"


def test_route_to_graph_target_maps_scenario() -> None:
    target = route_to_graph_target(
        {
            "route": "scenario",
        }
    )

    assert target == "scenario_agent"


def test_route_to_graph_target_maps_process_coach() -> None:
    target = route_to_graph_target(
        {
            "route": "process_coach",
        }
    )

    assert target == "process_coach_agent"


def test_route_to_graph_target_maps_ap_plus_navigator() -> None:
    target = route_to_graph_target(
        {
            "route": "ap_plus_navigator",
        }
    )

    assert target == "ap_plus_navigator_agent"


def test_route_to_graph_target_maps_fallback() -> None:
    target = route_to_graph_target(
        {
            "route": "fallback",
        }
    )

    assert target == "fallback_response"


def test_route_to_graph_target_defaults_to_fallback() -> None:
    target = route_to_graph_target({})

    assert target == "fallback_response"


def test_route_after_input_preparation_routes_valid_input() -> None:
    target = route_after_input_preparation(
        {
            "user_message": "Hello",
            "error": None,
        }
    )

    assert target == "route_request"


def test_route_after_input_preparation_routes_when_error_missing() -> None:
    target = route_after_input_preparation(
        {
            "user_message": "Hello",
        }
    )

    assert target == "route_request"


def test_route_after_input_preparation_skips_invalid_input() -> None:
    target = route_after_input_preparation(
        {
            "user_message": "",
            "error": "User message cannot be empty.",
        }
    )

    assert target == "finalize_response"
