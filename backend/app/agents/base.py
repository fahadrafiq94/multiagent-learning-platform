from __future__ import annotations

from typing import Protocol

from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)


class Agent(Protocol):
    """Common contract implemented by every specialized FREDi agent."""

    @property
    def name(self) -> AgentName:
        """Return the stable identifier for this agent."""
        ...

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute the agent for one request."""
        ...
