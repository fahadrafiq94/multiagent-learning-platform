from __future__ import annotations

from typing import cast

import pytest

from app.agents.base import Agent
from app.agents.exceptions import AgentConfigurationError
from app.agents.registry import AgentRegistry
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)


class FakeAgent:
    def __init__(
        self,
        name: AgentName,
    ) -> None:
        self._name = name

    @property
    def name(self) -> AgentName:
        return self._name

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        return AgentResponse(
            agent=self.name,
            content=f"{self.name}: {request.user_message}",
        )


def test_registry_can_be_created_empty() -> None:
    registry = AgentRegistry()

    assert registry.names == ()


def test_registry_registers_agent() -> None:
    registry = AgentRegistry()

    agent = FakeAgent("scenario")

    registry.register(agent)

    assert registry.contains("scenario") is True
    assert registry.get("scenario") is agent


def test_registry_initializes_with_agents() -> None:
    scenario = FakeAgent("scenario")
    process_coach = FakeAgent("process_coach")

    registry = AgentRegistry(
        agents=[
            scenario,
            process_coach,
        ]
    )

    assert registry.get("scenario") is scenario
    assert registry.get("process_coach") is process_coach


def test_registry_returns_registered_names() -> None:
    registry = AgentRegistry(
        agents=[
            FakeAgent("scenario"),
            FakeAgent("process_coach"),
            FakeAgent("ap_plus_navigator"),
        ]
    )

    assert registry.names == (
        "scenario",
        "process_coach",
        "ap_plus_navigator",
    )


def test_registry_rejects_duplicate_agent_name() -> None:
    registry = AgentRegistry()

    first_agent = FakeAgent("scenario")
    second_agent = FakeAgent("scenario")

    registry.register(first_agent)

    with pytest.raises(
        AgentConfigurationError,
        match="already registered",
    ) as exc_info:
        registry.register(second_agent)

    assert exc_info.value.agent == "scenario"


def test_registry_raises_for_unregistered_agent() -> None:
    registry = AgentRegistry(
        agents=[
            FakeAgent("scenario"),
        ]
    )

    # Intentionally bypass static AgentName validation to test the
    # registry's runtime failure behavior.
    unknown_name = cast(
        AgentName,
        "process_coach",
    )

    with pytest.raises(
        AgentConfigurationError,
        match="not registered",
    ) as exc_info:
        registry.get(unknown_name)

    assert exc_info.value.agent == "process_coach"


def test_fake_agent_satisfies_agent_protocol() -> None:
    agent: Agent = FakeAgent("scenario")

    assert agent.name == "scenario"


@pytest.mark.asyncio
async def test_registered_agent_can_be_executed() -> None:
    registry = AgentRegistry(
        agents=[
            FakeAgent("scenario"),
        ]
    )

    agent = registry.get("scenario")

    response = await agent.execute(
        AgentRequest(
            session_id="session-1",
            student_id="student-1",
            user_message="Describe my company scenario.",
        )
    )

    assert response.agent == "scenario"
    assert response.content == ("scenario: Describe my company scenario.")
