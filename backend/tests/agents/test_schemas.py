from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.policies import GuidanceLevel
from app.agents.schemas import (
    AgentRequest,
    AgentResponse,
)


def test_agent_request_accepts_valid_input() -> None:
    request = AgentRequest(
        session_id="session-1",
        student_id="student-1",
        user_message="I need help understanding procurement.",
        metadata={
            "source": "test",
        },
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"

    assert request.user_message == ("I need help understanding procurement.")

    assert request.guidance_level == GuidanceLevel.MINIMAL

    assert request.metadata == {
        "source": "test",
    }


def test_agent_request_accepts_missing_student_id() -> None:
    request = AgentRequest(
        session_id="session-1",
        user_message="Hello",
    )

    assert request.student_id is None
    assert request.guidance_level == GuidanceLevel.MINIMAL
    assert request.metadata == {}


def test_agent_request_strips_string_values() -> None:
    request = AgentRequest(
        session_id="  session-1  ",
        student_id="  student-1  ",
        user_message="  Hello  ",
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"
    assert request.user_message == "Hello"


def test_agent_request_defaults_to_minimal_guidance() -> None:
    request = AgentRequest(
        session_id="session-1",
        user_message="Hello",
    )

    assert request.guidance_level == GuidanceLevel.MINIMAL


@pytest.mark.parametrize(
    ("guidance_level", "expected"),
    [
        ("minimal", GuidanceLevel.MINIMAL),
        ("guided", GuidanceLevel.GUIDED),
        ("directed", GuidanceLevel.DIRECTED),
        ("explicit", GuidanceLevel.EXPLICIT),
    ],
)
def test_agent_request_parses_guidance_levels(
    guidance_level: str,
    expected: GuidanceLevel,
) -> None:
    request = AgentRequest.model_validate(
        {
            "session_id": "session-1",
            "user_message": "Hello",
            "guidance_level": guidance_level,
        }
    )

    assert request.guidance_level == expected


def test_agent_request_accepts_guidance_enum() -> None:
    request = AgentRequest(
        session_id="session-1",
        user_message="I am stuck.",
        guidance_level=GuidanceLevel.EXPLICIT,
    )

    assert request.guidance_level == GuidanceLevel.EXPLICIT


def test_agent_request_rejects_invalid_guidance_level() -> None:
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(
            {
                "session_id": "session-1",
                "user_message": "Hello",
                "guidance_level": "maximum",
            }
        )


def test_agent_request_rejects_blank_session_id() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(
            session_id="   ",
            user_message="Hello",
        )


def test_agent_request_rejects_blank_student_id() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(
            session_id="session-1",
            student_id="   ",
            user_message="Hello",
        )


def test_agent_request_rejects_blank_user_message() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(
            session_id="session-1",
            user_message="   ",
        )


def test_agent_response_accepts_valid_response() -> None:
    response = AgentResponse(
        agent="process_coach",
        content=("What do you think should happen before ordering?"),
        metadata={
            "model": "fake-model",
            "guidance_level": "minimal",
        },
    )

    assert response.agent == "process_coach"

    assert response.content == ("What do you think should happen before ordering?")

    assert response.metadata["model"] == "fake-model"

    assert response.metadata["guidance_level"] == "minimal"


def test_agent_response_strips_content() -> None:
    response = AgentResponse(
        agent="scenario",
        content="  Tell me about your company.  ",
    )

    assert response.content == "Tell me about your company."


def test_agent_response_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            agent="scenario",
            content="   ",
        )


def test_agent_response_rejects_unknown_agent() -> None:
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(
            {
                "agent": "unknown",
                "content": "Hello",
            }
        )
