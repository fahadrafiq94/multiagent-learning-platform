from __future__ import annotations

from app.agents.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
)


def test_agent_error_contains_agent_information() -> None:
    error = AgentError(
        "Agent failed.",
        agent="scenario",
        details={
            "reason": "test",
        },
    )

    assert str(error) == "Agent failed."
    assert error.message == "Agent failed."
    assert error.agent == "scenario"
    assert error.details == {
        "reason": "test",
    }


def test_agent_error_to_dict() -> None:
    error = AgentError(
        "Agent failed.",
        agent="process_coach",
        details={
            "reason": "test",
        },
    )

    assert error.to_dict() == {
        "error": "agent_error",
        "message": "Agent failed.",
        "agent": "process_coach",
        "details": {
            "reason": "test",
        },
    }


def test_agent_execution_error_uses_specific_error_code() -> None:
    error = AgentExecutionError(
        "Execution failed.",
        agent="ap_plus_navigator",
    )

    assert error.to_dict()["error"] == "agent_execution_error"


def test_agent_configuration_error_uses_specific_error_code() -> None:
    error = AgentConfigurationError(
        "Configuration failed.",
        agent="scenario",
    )

    assert error.to_dict()["error"] == "agent_configuration_error"
