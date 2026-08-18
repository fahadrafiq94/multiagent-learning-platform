from __future__ import annotations

from app.agents.base import Agent
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)


class FakeAgent:
    @property
    def name(self) -> AgentName:
        return "scenario"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        return AgentResponse(
            agent=self.name,
            content=f"Processed: {request.user_message}",
        )


def accepts_agent(agent: Agent) -> Agent:
    """Used only to verify structural compatibility."""
    return agent


def test_fake_agent_satisfies_agent_protocol() -> None:
    agent = FakeAgent()

    result = accepts_agent(agent)

    assert result.name == "scenario"
