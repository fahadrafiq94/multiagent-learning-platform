from __future__ import annotations

from collections.abc import Iterable

from app.agents.base import Agent
from app.agents.exceptions import AgentConfigurationError
from app.agents.schemas import AgentName


class AgentRegistry:
    """Registry of specialized FREDi agents.

    The registry provides a single lookup mechanism for orchestration code
    and prevents agent construction logic from spreading across the graph.
    """

    def __init__(
        self,
        agents: Iterable[Agent] | None = None,
    ) -> None:
        self._agents: dict[AgentName, Agent] = {}

        if agents is not None:
            for agent in agents:
                self.register(agent)

    def register(
        self,
        agent: Agent,
    ) -> None:
        """Register one specialized agent.

        Duplicate names are rejected because each AgentName must resolve to
        exactly one active implementation.
        """

        if agent.name in self._agents:
            raise AgentConfigurationError(
                f"Agent '{agent.name}' is already registered.",
                agent=agent.name,
            )

        self._agents[agent.name] = agent

    def get(
        self,
        name: AgentName,
    ) -> Agent:
        """Return the registered agent for a stable agent name."""

        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentConfigurationError(
                f"Agent '{name}' is not registered.",
                agent=name,
            ) from exc

    def contains(
        self,
        name: AgentName,
    ) -> bool:
        """Return whether an agent is currently registered."""

        return name in self._agents

    @property
    def names(self) -> tuple[AgentName, ...]:
        """Return registered agent names."""

        return tuple(self._agents)
