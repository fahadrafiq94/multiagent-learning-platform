from __future__ import annotations

import pytest

from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.registry import AgentRegistry
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


@pytest.mark.asyncio
async def test_graph_routes_to_scenario_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("I need to define my company name and business problem."),
            "metadata": {},
        }
    )

    assert result["route"] == "scenario"

    assert result["metadata"]["selected_agent"] == "scenario"

    assert result["metadata"]["agent_called"] is True

    assert scenario.call_count == 1
    assert process_coach.call_count == 0
    assert navigator.call_count == 0

    assert result["final_response"] == (
        "scenario response to: I need to define my company name and business problem."
    )


@pytest.mark.asyncio
async def test_graph_routes_to_process_coach_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
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

    assert result["metadata"]["selected_agent"] == "process_coach"

    assert scenario.call_count == 0
    assert process_coach.call_count == 1
    assert navigator.call_count == 0

    assert result["final_response"].startswith("process_coach response")


@pytest.mark.asyncio
async def test_graph_routes_to_ap_plus_navigator_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Where can I find the purchase order screen in AP+?"),
            "metadata": {},
        }
    )

    assert result["route"] == "ap_plus_navigator"

    assert result["metadata"]["selected_agent"] == "ap_plus_navigator"

    assert scenario.call_count == 0
    assert process_coach.call_count == 0
    assert navigator.call_count == 1

    assert result["final_response"].startswith("ap_plus_navigator response")


@pytest.mark.asyncio
async def test_graph_uses_fallback_without_agent() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
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

    assert result["metadata"]["fallback_used"] is True

    assert result["metadata"]["selected_agent"] is None

    assert scenario.call_count == 0
    assert process_coach.call_count == 0
    assert navigator.call_count == 0

    assert "business scenario" in result["final_response"]

    assert "business-process reasoning" in result["final_response"]

    assert "AP+" in result["final_response"]


@pytest.mark.asyncio
async def test_graph_does_not_invoke_agent_for_empty_input() -> None:
    (
        registry,
        scenario,
        process_coach,
        navigator,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
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

    graph = build_orchestration_graph(
        agent_registry=registry,
    )

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("I need help defining my business scenario."),
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
async def test_graph_passes_identity_to_agent() -> None:
    (
        registry,
        scenario,
        _,
        _,
    ) = create_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
    )

    await graph.ainvoke(
        {
            "session_id": "session-123",
            "student_id": "student-456",
            "user_message": ("Help me define my company scenario."),
            "metadata": {
                "source": "graph-test",
            },
        }
    )

    assert scenario.last_request is not None

    assert scenario.last_request.session_id == "session-123"

    assert scenario.last_request.student_id == "student-456"

    assert scenario.last_request.user_message == "Help me define my company scenario."

    @pytest.mark.asyncio
    async def test_graph_handles_missing_registered_agent() -> None:
        registry = AgentRegistry(
            agents=[
                FakeAgent("process_coach"),
                FakeAgent("ap_plus_navigator"),
            ]
        )

        graph = build_orchestration_graph(
            agent_registry=registry,
        )

        result = await graph.ainvoke(
            {
                "session_id": "session-1",
                "student_id": "student-1",
                "user_message": ("I need help with my company scenario."),
                "metadata": {},
            }
        )

        assert result["metadata"]["agent_error"]["error"] == "agent_configuration_error"

        assert result["metadata"]["selected_agent"] == "scenario"

        assert "not registered" in result["error"]
