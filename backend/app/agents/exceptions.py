from __future__ import annotations

from typing import Any

from app.agents.schemas import AgentName


class AgentError(Exception):
    """Base exception for specialized agent failures."""

    error_code = "agent_error"

    def __init__(
        self,
        message: str,
        *,
        agent: AgentName,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.agent = agent
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "agent": self.agent,
            "details": self.details,
        }


class AgentExecutionError(AgentError):
    """Raised when a specialized agent cannot complete its execution."""

    error_code = "agent_execution_error"


class AgentConfigurationError(AgentError):
    """Raised when an agent is configured incorrectly."""

    error_code = "agent_configuration_error"
