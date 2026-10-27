from __future__ import annotations

import copy
from uuid import UUID

from app.context.schemas import (
    AgentContext,
    ContextMetadata,
)
from app.domain.guidance import (
    GuidanceLevel,
)
from app.observability.events import (
    RequestObservabilityEvent,
    build_request_event,
)

USER_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
TRACE_ID = UUID(int=3)


def base_context() -> AgentContext:
    return AgentContext(
        user_id=USER_ID,
        session_id=SESSION_ID,
        current_message=("Why do we need a purchase order?"),
        guidance_level=(GuidanceLevel.DIRECTED),
        metadata=(
            ContextMetadata(
                recent_message_count=4,
                candidate_message_count=20,
                dropped_message_count=16,
                truncation_applied=True,
                summary_included=True,
                guidance_reason="repeated_struggle",
                estimated_input_tokens=1500,
                input_token_budget=2000,
                estimated_essential_tokens=600,
                estimated_history_tokens=900,
            )
        ),
    )


def specialist_state() -> dict:
    return {
        "trace_id": (TRACE_ID),
        "student_id": ("K12345"),
        "session_id": (str(SESSION_ID)),
        "user_id": (USER_ID),
        "route": ("process_coach"),
        "route_reason": ("The student asks about process reasoning."),
        "error": None,
        "onboarding_required": False,
        "base_context": (base_context()),
        "metadata": {
            "selected_agent": ("process_coach"),
            "state_update_applied": True,
            "state_update_accepted_count": 2,
            "state_update_rejected_count": 1,
            "summary_update_attempted": True,
            "summary_update_applied": True,
            "summary_update_reason": ("updated"),
            "summary_message_count": 4,
            "response_assessed": True,
            "response_understanding": ("partially_correct"),
            "response_misconception_detected": (True),
            "response_signal_recorded": (True),
            "agent_metadata": {
                "model": ("qwen3:4b"),
                "model_provider": ("ollama"),
                "model_latency_ms": (42.5),
            },
        },
    }


def test_specialist_turn_event_carries_all_fields() -> None:
    event = build_request_event(
        specialist_state(),
        duration_ms=123.4,
    )

    assert event.trace_id == TRACE_ID

    assert event.student_id == "K12345"

    assert event.session_id == str(SESSION_ID)

    assert event.user_id == USER_ID

    assert event.outcome == "completed"

    assert event.error is None

    assert event.route == "process_coach"

    assert event.route_reason == ("The student asks about process reasoning.")

    assert event.selected_agent == "process_coach"

    assert event.onboarding_handled is False

    assert event.guidance_level == "directed"

    assert event.guidance_reason == "repeated_struggle"

    assert event.estimated_input_tokens == 1500

    assert event.input_token_budget == 2000

    assert event.estimated_essential_tokens == 600

    assert event.estimated_history_tokens == 900

    assert event.candidate_message_count == 20

    assert event.recent_message_count == 4

    assert event.dropped_message_count == 16

    assert event.truncation_applied is True

    assert event.summary_included is True

    assert event.summary_update_attempted is True

    assert event.summary_update_applied is True

    assert event.summary_update_reason == "updated"

    assert event.summary_message_count == 4

    assert event.state_update_applied is True

    assert event.state_update_accepted_count == 2

    assert event.state_update_rejected_count == 1

    assert event.response_assessed is True

    assert event.response_understanding == "partially_correct"

    assert event.response_misconception_detected is True

    assert event.response_signal_recorded is True

    assert event.model == "qwen3:4b"

    assert event.model_provider == "ollama"

    assert event.model_latency_ms == 42.5

    assert event.duration_ms == 123.4


def test_fallback_turn_event_has_no_agent_fields() -> None:
    state = specialist_state()

    state["route"] = "fallback"

    state["metadata"]["selected_agent"] = None

    state["metadata"].pop("agent_metadata")

    state["metadata"].pop("response_assessed")

    state["metadata"].pop("response_understanding")

    state["metadata"].pop("response_misconception_detected")

    state["metadata"].pop("response_signal_recorded")

    event = build_request_event(state, duration_ms=10.0)

    assert event.route == "fallback"

    assert event.selected_agent is None

    assert event.model is None

    assert event.model_provider is None

    assert event.model_latency_ms is None

    assert event.response_assessed is None

    assert event.response_understanding is None

    assert event.response_signal_recorded is None


def test_onboarding_turn_event_is_marked() -> None:
    state = specialist_state()

    state["route"] = None

    state["onboarding_required"] = True

    state["route_reason"] = None

    state["metadata"]["selected_agent"] = None

    event = build_request_event(state, duration_ms=10.0)

    assert event.onboarding_handled is True

    assert event.route is None

    assert event.route_reason is None

    assert event.outcome == "completed"


def test_error_state_produces_error_event() -> None:
    state = specialist_state()

    state["error"] = "The specialist agent failed."

    event = build_request_event(state, duration_ms=10.0)

    assert event.outcome == "error"

    assert event.error == "The specialist agent failed."

    assert event.onboarding_handled is False


def test_minimal_state_degrades_to_optional_fields() -> None:
    event = build_request_event({}, duration_ms=5.0)

    assert isinstance(event, RequestObservabilityEvent)

    assert event.trace_id is None

    assert event.student_id is None

    assert event.session_id is None

    assert event.user_id is None

    assert event.outcome == "completed"

    assert event.error is None

    assert event.route is None

    assert event.selected_agent is None

    assert event.onboarding_handled is False

    assert event.guidance_level is None

    assert event.guidance_reason is None

    assert event.estimated_input_tokens is None

    assert event.state_update_applied is None

    assert event.response_assessed is None

    assert event.model is None

    assert event.duration_ms == 5.0


def test_malformed_metadata_values_never_raise() -> None:
    state = {
        "trace_id": TRACE_ID,
        "error": None,
        "route_reason": ["not", "a", "string"],
        "base_context": (base_context()),
        "metadata": {
            "selected_agent": 42,
            "state_update_accepted_count": "two",
            "summary_update_applied": "yes",
            "response_understanding": 7,
            "agent_metadata": "junk",
        },
    }

    event = build_request_event(state, duration_ms=5.0)

    assert event.selected_agent is None

    assert event.route_reason is None

    assert event.state_update_accepted_count is None

    assert event.summary_update_applied is None

    assert event.response_understanding is None

    assert event.model is None


def test_mapper_does_not_mutate_input_and_uses_argument_duration() -> None:
    state = specialist_state()

    snapshot = copy.deepcopy(state)

    build_request_event(state, duration_ms=99.9)

    assert state == snapshot
