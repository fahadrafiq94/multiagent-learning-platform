from __future__ import annotations

from typing import cast

import pytest

from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.registry import AgentRegistry
from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
    RoutingTarget,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)
from app.orchestration.graph import (
    build_orchestration_graph,
)


class FakeAgent:
    def __init__(
        self,
        name: AgentName,
    ) -> None:
        self._name = name
        self.call_count = 0
        self.last_request: AgentRequest | None = None

    @property
    def name(self) -> AgentName:
        return self._name

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        self.call_count += 1
        self.last_request = request

        return AgentResponse(
            agent=self.name,
            content=(f"{self.name} response to: {request.user_message}"),
            metadata={
                "fake_agent": True,
            },
        )


class FailingAgent(FakeAgent):
    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        self.call_count += 1
        self.last_request = request

        raise AgentExecutionError(
            "Fake agent failed.",
            agent=self.name,
            details={
                "reason": "test failure",
            },
        )


class FakeOrchestrator:
    """Configurable semantic orchestrator used by graph tests."""

    def __init__(
        self,
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
                "routing_strategy": ("llm_semantic_router_v1"),
                "fake_orchestrator": True,
            },
        )


class FailingOrchestrator:
    """Semantic orchestrator that simulates model routing failure."""

    def __init__(self) -> None:
        self.call_count = 0

    async def route(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        self.call_count += 1

        raise OrchestratorRoutingError(
            "Fake semantic routing failure.",
            details={
                "reason": "test failure",
            },
        )


def create_registry() -> tuple[
    AgentRegistry,
    FakeAgent,
    FakeAgent,
    FakeAgent,
]:
    scenario = FakeAgent("scenario")

    process_coach = FakeAgent("process_coach")

    navigator = FakeAgent("ap_plus_navigator")

    registry = AgentRegistry(
        agents=[
            scenario,
            process_coach,
            navigator,
        ]
    )

    return (
        registry,
        scenario,
        process_coach,
        navigator,
    )


def build_test_graph(
    *,
    registry: AgentRegistry,
    orchestrator: FakeOrchestrator | FailingOrchestrator,
):
    return build_orchestration_graph(
        agent_registry=registry,
        orchestrator=cast(
            OrchestratorAgent,
            orchestrator,
        ),
    )


@pytest.mark.asyncio
async def test_graph_routes_to_scenario_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="scenario",
        reason=("The student needs scenario clarification."),
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Tell me more about the situation I am working with."),
            "metadata": {},
        }
    )

    assert result["route"] == "scenario"

    assert result["metadata"]["selected_agent"] == "scenario"

    assert result["metadata"]["agent_called"] is True

    assert result["metadata"]["routing_strategy"] == "llm_semantic_router_v1"

    assert result["metadata"]["keyword_fallback_used"] is False

    assert scenario.call_count == 1
    assert process_coach.call_count == 0
    assert navigator.call_count == 0

    assert orchestrator.call_count == 1


@pytest.mark.asyncio
async def test_graph_routes_to_process_coach_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="process_coach",
        reason=("The student needs business-process reasoning."),
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": (
                "I understand the software action, but I do not understand its purpose."
            ),
            "metadata": {},
        }
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["selected_agent"] == "process_coach"

    assert scenario.call_count == 0
    assert process_coach.call_count == 1
    assert navigator.call_count == 0


@pytest.mark.asyncio
async def test_graph_routes_to_ap_plus_navigator_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="ap_plus_navigator",
        reason=("The student needs help executing an action in AP+."),
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": (
                "I know the supplier must be recorded, but I cannot figure out how to do it."
            ),
            "metadata": {},
        }
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["selected_agent"] == "ap_plus_navigator"

    assert scenario.call_count == 0
    assert process_coach.call_count == 0
    assert navigator.call_count == 1


@pytest.mark.asyncio
async def test_graph_uses_keyword_router_after_semantic_fallback() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="fallback",
        reason=("The semantic router cannot determine the student's need."),
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Where can I find this field in AP+?"),
            "metadata": {},
        }
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_fallback_trigger"] == "semantic_fallback"

    assert result["metadata"]["semantic_route"] == "fallback"

    assert result["metadata"]["keyword_route"] == "ap_plus_navigator"

    assert navigator.call_count == 1
    assert scenario.call_count == 0
    assert process_coach.call_count == 0


@pytest.mark.asyncio
async def test_graph_uses_keyword_router_after_orchestrator_error() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FailingOrchestrator()

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Why is procurement important in this business process?"),
            "metadata": {},
        }
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_fallback_trigger"] == "orchestrator_error"

    assert result["metadata"]["semantic_routing_failed"] is True

    assert result["metadata"]["keyword_route"] == "process_coach"

    assert process_coach.call_count == 1

    assert orchestrator.call_count == 1


@pytest.mark.asyncio
async def test_graph_uses_clarification_when_both_routers_fallback() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="fallback",
        reason=("The semantic router cannot determine the student's need."),
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "Hello there.",
            "metadata": {},
        }
    )

    assert result["route"] == "fallback"

    assert result["metadata"]["keyword_fallback_used"] is True

    assert result["metadata"]["keyword_route"] == "fallback"

    assert result["metadata"]["fallback_used"] is True

    assert result["metadata"]["selected_agent"] is None

    assert scenario.call_count == 0
    assert process_coach.call_count == 0
    assert navigator.call_count == 0

    assert "business scenario" in result["final_response"]

    assert "business-process reasoning" in result["final_response"]

    assert "AP+" in result["final_response"]


@pytest.mark.asyncio
async def test_graph_does_not_route_empty_input() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "   ",
            "metadata": {},
        }
    )

    assert result["error"] == "User message cannot be empty."

    assert result["metadata"]["prepared"] is False

    assert result["metadata"]["finalized"] is True

    assert orchestrator.call_count == 0

    assert scenario.call_count == 0
    assert process_coach.call_count == 0
    assert navigator.call_count == 0


@pytest.mark.asyncio
async def test_graph_converts_agent_failure_into_state_error() -> None:
    scenario = FailingAgent("scenario")

    registry = AgentRegistry(
        agents=[
            scenario,
            FakeAgent("process_coach"),
            FakeAgent("ap_plus_navigator"),
        ]
    )

    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("I need help understanding the scenario."),
            "metadata": {},
        }
    )

    assert result["error"] == "Fake agent failed."

    assert result["metadata"]["selected_agent"] == "scenario"

    assert result["metadata"]["agent_error"]["error"] == "agent_execution_error"

    assert result["metadata"]["finalized"] is True

    assert result["final_response"] == (
        "I could not process the message because: Fake agent failed."
    )


@pytest.mark.asyncio
async def test_graph_passes_identity_to_orchestrator_and_agent() -> None:
    (
        registry,
        scenario,
        _,
        _,
    ) = create_registry()

    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    await graph.ainvoke(
        {
            "session_id": "session-123",
            "student_id": "student-456",
            "user_message": ("Help me understand my situation."),
            "metadata": {
                "source": "graph-test",
            },
        }
    )

    assert orchestrator.last_request is not None

    assert orchestrator.last_request.session_id == "session-123"

    assert orchestrator.last_request.student_id == "student-456"

    assert scenario.last_request is not None

    assert scenario.last_request.session_id == "session-123"

    assert scenario.last_request.student_id == "student-456"


@pytest.mark.asyncio
async def test_graph_handles_missing_registered_agent() -> None:
    registry = AgentRegistry(
        agents=[
            FakeAgent("process_coach"),
            FakeAgent("ap_plus_navigator"),
        ]
    )

    orchestrator = FakeOrchestrator(
        route="scenario",
    )

    graph = build_test_graph(
        registry=registry,
        orchestrator=orchestrator,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("I need help understanding the scenario."),
            "metadata": {},
        }
    )

    assert result["metadata"]["agent_error"]["error"] == "agent_configuration_error"

    assert result["metadata"]["selected_agent"] == "scenario"

    assert "not registered" in result["error"]
