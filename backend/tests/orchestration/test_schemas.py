import pytest
from pydantic import ValidationError

from app.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)


def test_orchestration_request_accepts_valid_input() -> None:
    request = OrchestrationRequest(
        session_id="session-1",
        student_id="student-1",
        user_message="Hello",
        metadata={"source": "test"},
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"
    assert request.user_message == "Hello"
    assert request.metadata == {"source": "test"}


def test_orchestration_request_strips_values() -> None:
    request = OrchestrationRequest(
        session_id="  session-1  ",
        student_id="  student-1  ",
        user_message="  Hello  ",
    )

    assert request.session_id == "session-1"
    assert request.student_id == "student-1"
    assert request.user_message == "Hello"


def test_orchestration_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        OrchestrationRequest(
            session_id="session-1",
            user_message="   ",
        )


def test_orchestration_request_rejects_blank_session_id() -> None:
    with pytest.raises(ValidationError):
        OrchestrationRequest(
            session_id="   ",
            user_message="Hello",
        )


def test_orchestration_response_accepts_graph_result() -> None:
    response = OrchestrationResponse(
        session_id="session-1",
        student_id="student-1",
        final_response="Hello student",
        route=None,
        error=None,
        metadata={"model_called": True},
    )

    assert response.final_response == "Hello student"
    assert response.metadata["model_called"] is True
