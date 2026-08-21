from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
)


def test_orchestrator_request_accepts_valid_input() -> None:
    request = OrchestratorRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=("I know what needs to happen, but I do not know where to do it."),
        metadata={
            "source": "test",
        },
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"

    assert request.user_message == (
        "I know what needs to happen, but I do not know where to do it."
    )

    assert request.metadata == {
        "source": "test",
    }


def test_orchestrator_request_accepts_missing_student_id() -> None:
    request = OrchestratorRequest(
        session_id="session-1",
        user_message="Help me understand this.",
    )

    assert request.student_id is None
    assert request.metadata == {}


def test_orchestrator_request_strips_string_values() -> None:
    request = OrchestratorRequest(
        session_id="  session-1  ",
        student_id="  student-1  ",
        user_message="  Help me understand this.  ",
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"
    assert request.user_message == "Help me understand this."


def test_orchestrator_request_rejects_blank_session_id() -> None:
    with pytest.raises(ValidationError):
        OrchestratorRequest(
            session_id="   ",
            user_message="Hello",
        )


def test_orchestrator_request_rejects_blank_student_id() -> None:
    with pytest.raises(ValidationError):
        OrchestratorRequest(
            session_id="session-1",
            student_id="   ",
            user_message="Hello",
        )


def test_orchestrator_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        OrchestratorRequest(
            session_id="session-1",
            user_message="   ",
        )


@pytest.mark.parametrize(
    "route",
    [
        "scenario",
        "process_coach",
        "ap_plus_navigator",
        "fallback",
    ],
)
def test_routing_decision_accepts_supported_routes(
    route: str,
) -> None:
    decision = RoutingDecision.model_validate(
        {
            "route": route,
            "reason": "Valid routing reason.",
        }
    )

    assert decision.route == route
    assert decision.reason == "Valid routing reason."


def test_routing_decision_strips_reason() -> None:
    decision = RoutingDecision(
        route="scenario",
        reason="  The student needs scenario clarification.  ",
    )

    assert decision.reason == ("The student needs scenario clarification.")


def test_routing_decision_rejects_unknown_route() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision.model_validate(
            {
                "route": "unknown_agent",
                "reason": "Unknown route.",
            }
        )


def test_routing_decision_rejects_blank_reason() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            route="process_coach",
            reason="   ",
        )


def test_routing_decision_rejects_reason_over_limit() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            route="scenario",
            reason="x" * 301,
        )


def test_orchestrator_result_accepts_decision() -> None:
    decision = RoutingDecision(
        route="process_coach",
        reason=("The student needs business-process reasoning."),
    )

    result = OrchestratorResult(
        decision=decision,
        metadata={
            "routing_strategy": "llm_semantic_router_v1",
        },
    )

    assert result.decision is decision

    assert result.metadata["routing_strategy"] == "llm_semantic_router_v1"


def test_orchestrator_result_defaults_metadata() -> None:
    result = OrchestratorResult(
        decision=RoutingDecision(
            route="fallback",
            reason=("The student's immediate need cannot yet be determined."),
        )
    )

    assert result.metadata == {}
