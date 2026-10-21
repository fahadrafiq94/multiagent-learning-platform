from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.policies import GuidanceLevel
from app.agents.schemas import (
    AgentRequest,
    AgentResponse,
    SpecialistAgentOutput,
)
from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
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


def test_specialist_output_accepts_empty_state_proposal() -> None:
    output = SpecialistAgentOutput(content="Useful response.")

    assert output.content == "Useful response."

    assert output.state_update_proposal.operations == []


def test_specialist_output_accepts_state_proposal() -> None:
    output = SpecialistAgentOutput.model_validate(
        {
            "content": ("Let's focus on step 8."),
            "state_update_proposal": {
                "operations": [
                    {
                        "field": ("current_step_id"),
                        "action": "set",
                        "value": "8",
                        "reason": ("The user explicitly stated step 8."),
                    }
                ]
            },
        }
    )

    operation = output.state_update_proposal.operations[0]

    assert operation.field == (SessionStateField.CURRENT_STEP_ID)

    assert operation.action == SessionStateAction.SET


def test_specialist_output_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SpecialistAgentOutput.model_validate(
            {
                "content": "Response.",
                "state_update_proposal": {
                    "operations": [],
                },
                "unexpected": True,
            }
        )


def test_specialist_output_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        SpecialistAgentOutput(content="   ")


def test_agent_response_accepts_state_update_proposal() -> None:
    response = AgentResponse.model_validate(
        {
            "agent": ("ap_plus_navigator"),
            "content": "Continue here.",
            "state_update_proposal": {
                "operations": [
                    {
                        "field": ("active_blocker"),
                        "action": "set",
                        "value": ("Supplier field cannot be found."),
                        "reason": ("The user explicitly reported this blocker."),
                    }
                ]
            },
        }
    )

    assert response.state_update_proposal is not None

    assert len(response.state_update_proposal.operations) == 1
