from __future__ import annotations

from typing import cast

import pytest

from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
    RoutingTarget,
)
from app.orchestration.nodes import (
    prepare_input_node,
    route_request_node,
)
from app.orchestration.state import (
    OrchestrationState,
)


class FakeOrchestrator:
    """Configurable semantic orchestrator for isolated node tests."""

    def __init__(
        self,
        *,
        route: RoutingTarget,
        reason: str = "Fake semantic routing decision.",
    ) -> None:
        self._route = route
        self._reason = reason

        self.call_count = 0
        self.last_request: OrchestratorRequest | None = None

    async def route(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        self.call_count += 1
        self.last_request = request

        return OrchestratorResult(
            decision=RoutingDecision(
                route=self._route,
                reason=self._reason,
            ),
            metadata={
                "model": "fake-model",
                "model_provider": "test",
                "model_latency_ms": 1.0,
                "routing_strategy": ("llm_semantic_router_v1"),
            },
        )


class FailingOrchestrator:
    """Semantic orchestrator that always fails."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_request: OrchestratorRequest | None = None

    async def route(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        self.call_count += 1
        self.last_request = request

        raise OrchestratorRoutingError(
            "Fake routing failure.",
            details={
                "reason": "test failure",
            },
        )


def create_state(
    *,
    user_message: str,
) -> OrchestrationState:
    """Create a minimal prepared orchestration state."""

    return {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": user_message,
        "metadata": {
            "source": "node-test",
        },
        "error": None,
    }


def as_orchestrator(
    orchestrator: FakeOrchestrator | FailingOrchestrator,
) -> OrchestratorAgent:
    """Cast a structurally compatible test double."""

    return cast(
        OrchestratorAgent,
        orchestrator,
    )


def test_prepare_input_trims_message() -> None:
    state: OrchestrationState = {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": "   Hello student   ",
        "metadata": {},
    }

    result = prepare_input_node(state)

    assert result["user_message"] == "Hello student"
    assert result["error"] is None

    assert result["metadata"]["prepared"] is True


def test_prepare_input_rejects_blank_message() -> None:
    state: OrchestrationState = {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": "   ",
        "metadata": {},
    }

    result = prepare_input_node(state)

    assert result["user_message"] == ""

    assert result["error"] == "User message cannot be empty."

    assert result["metadata"]["prepared"] is False


@pytest.mark.asyncio
async def test_route_node_uses_semantic_scenario_route() -> None:
    orchestrator = FakeOrchestrator(
        route="scenario",
        reason=("The student needs scenario clarification."),
    )

    state = create_state(
        user_message=("I am unclear about the situation our company is in."),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "scenario"

    assert result["route_reason"] == ("The student needs scenario clarification.")

    assert result["metadata"]["routing_strategy"] == "llm_semantic_router_v1"

    assert result["metadata"]["semantic_route"] == "scenario"

    assert result["metadata"]["keyword_fallback_used"] is False

    assert orchestrator.call_count == 1


@pytest.mark.asyncio
async def test_route_node_uses_semantic_process_route() -> None:
    orchestrator = FakeOrchestrator(
        route="process_coach",
        reason=("The student needs business-process reasoning."),
    )

    state = create_state(
        user_message=("I know where I am but I do not understand why this action is necessary."),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["semantic_route"] == "process_coach"

    assert result["metadata"]["keyword_fallback_used"] is False


@pytest.mark.asyncio
async def test_route_node_uses_semantic_ap_plus_route() -> None:
    orchestrator = FakeOrchestrator(
        route="ap_plus_navigator",
        reason=("The student needs AP+ system guidance."),
    )

    state = create_state(
        user_message=("I know what needs to be recorded, but I cannot work out how to do it."),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["semantic_route"] == "ap_plus_navigator"

    assert result["metadata"]["keyword_fallback_used"] is False


@pytest.mark.asyncio
async def test_semantic_fallback_uses_keyword_router() -> None:
    orchestrator = FakeOrchestrator(
        route="fallback",
        reason=("The semantic router needs more information."),
    )

    state = create_state(
        user_message=("Where can I find this field in AP+?"),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["semantic_route"] == "fallback"

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_fallback_trigger"] == "semantic_fallback"

    assert result["metadata"]["keyword_route"] == "ap_plus_navigator"

    assert result["metadata"]["routing_strategy"] == (
        "llm_semantic_router_v1->deterministic_keywords_v1"
    )


@pytest.mark.asyncio
async def test_semantic_fallback_can_recover_process_route() -> None:
    orchestrator = FakeOrchestrator(
        route="fallback",
    )

    state = create_state(
        user_message=("Why is procurement important in this business process?"),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["keyword_route"] == "process_coach"


@pytest.mark.asyncio
async def test_semantic_fallback_can_recover_scenario_route() -> None:
    orchestrator = FakeOrchestrator(
        route="fallback",
    )

    state = create_state(
        user_message=("I need help defining our company business problem."),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "scenario"

    assert result["metadata"]["keyword_route"] == "scenario"


@pytest.mark.asyncio
async def test_both_routers_can_return_fallback() -> None:
    orchestrator = FakeOrchestrator(
        route="fallback",
        reason=("The student's need is unclear."),
    )

    state = create_state(
        user_message="Hello there.",
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "fallback"

    assert result["metadata"]["semantic_route"] == "fallback"

    assert result["metadata"]["keyword_route"] == "fallback"

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_fallback_trigger"] == "semantic_fallback"


@pytest.mark.asyncio
async def test_orchestrator_failure_uses_keyword_router() -> None:
    orchestrator = FailingOrchestrator()

    state = create_state(
        user_message=("Where should I enter the supplier?"),
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["semantic_routing_attempted"] is True

    assert result["metadata"]["semantic_routing_failed"] is True

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_fallback_trigger"] == "orchestrator_error"

    assert result["metadata"]["keyword_route"] == "ap_plus_navigator"

    assert result["metadata"]["orchestrator_error"]["error"] == "orchestrator_routing_error"

    assert orchestrator.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_failure_can_end_in_fallback() -> None:
    orchestrator = FailingOrchestrator()

    state = create_state(
        user_message="Hello there.",
    )

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result["route"] == "fallback"

    assert result["metadata"]["semantic_routing_failed"] is True

    assert result["metadata"]["keyword_route"] == "fallback"

    assert result["metadata"]["keyword_fallback_trigger"] == "orchestrator_error"


@pytest.mark.asyncio
async def test_route_node_passes_identity_to_orchestrator() -> None:
    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    state: OrchestrationState = {
        "session_id": "session-123",
        "student_id": "student-456",
        "user_message": ("Help me understand our company situation."),
        "metadata": {
            "source": "identity-test",
        },
        "error": None,
    }

    await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert orchestrator.last_request is not None

    assert orchestrator.last_request.session_id == "session-123"

    assert orchestrator.last_request.student_id == "student-456"

    assert orchestrator.last_request.user_message == ("Help me understand our company situation.")

    assert orchestrator.last_request.metadata["source"] == "identity-test"


@pytest.mark.asyncio
async def test_route_node_does_nothing_when_state_has_error() -> None:
    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    state: OrchestrationState = {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": "",
        "error": "User message cannot be empty.",
        "metadata": {
            "prepared": False,
        },
    }

    result = await route_request_node(
        state=state,
        orchestrator=as_orchestrator(orchestrator),
    )

    assert result == state

    assert orchestrator.call_count == 0
