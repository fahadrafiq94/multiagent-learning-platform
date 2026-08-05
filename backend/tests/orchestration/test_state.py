from app.orchestration.state import OrchestrationState


def test_orchestration_state_accepts_minimal_values() -> None:
    state: OrchestrationState = {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": "Hello",
        "metadata": {},
    }

    assert state["session_id"] == "session-1"
    assert state["user_message"] == "Hello"


def test_orchestration_state_accepts_route_and_response() -> None:
    state: OrchestrationState = {
        "session_id": "session-1",
        "student_id": "student-1",
        "user_message": "I need help with the business problem.",
        "route": "process_coach",
        "route_reason": "Student asks for process reasoning support.",
        "model_response": "Let us reason about the process first.",
        "final_response": "Let us reason about the process first.",
        "error": None,
        "metadata": {"test": True},
    }

    assert state["route"] == "process_coach"
    assert state["final_response"] == "Let us reason about the process first."
