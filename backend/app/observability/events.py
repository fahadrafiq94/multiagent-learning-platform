from __future__ import annotations

from typing import (
    Any,
    Literal,
    cast,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.agents.schemas import (
    AgentName,
)
from app.orchestration.state import (
    RouteName,
)

REQUEST_EVENT_LOGGER_NAME = "fredi.request_events"

REQUEST_COMPLETED_EVENT = "request_completed"


class RequestObservabilityEvent(BaseModel):
    """
    One structured observability record per orchestration request (4B.10).

    Carries identifiers, enums, counters, token counts, and latency only.
    It never carries user message content, assistant response content, or
    free-text LLM output, so it is safe to log and to persist as-is into a
    future research-event store.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Identity
    trace_id: UUID | None = None

    student_id: str | None = None

    session_id: str | None = None

    user_id: UUID | None = None

    # Outcome
    outcome: Literal[
        "completed",
        "error",
    ]

    error: str | None = None

    route: RouteName | None = None

    route_reason: str | None = None

    selected_agent: AgentName | None = None

    onboarding_handled: bool = False

    # Guidance (4B.9)
    guidance_level: str | None = None

    guidance_reason: str | None = None

    # Context budget (4B.7)
    estimated_input_tokens: int | None = None

    input_token_budget: int | None = None

    estimated_essential_tokens: int | None = None

    estimated_history_tokens: int | None = None

    candidate_message_count: int | None = None

    recent_message_count: int | None = None

    dropped_message_count: int | None = None

    truncation_applied: bool | None = None

    summary_included: bool | None = None

    # Session summary maintenance (4B.8)
    summary_update_attempted: bool | None = None

    summary_update_applied: bool | None = None

    summary_update_reason: str | None = None

    summary_message_count: int | None = None

    # State-update proposals (4B.6)
    state_update_applied: bool | None = None

    state_update_accepted_count: int | None = None

    state_update_rejected_count: int | None = None

    # Response signal (4B.9)
    response_assessed: bool | None = None

    response_understanding: str | None = None

    response_misconception_detected: bool | None = None

    response_signal_recorded: bool | None = None

    # Model
    model: str | None = None

    model_provider: str | None = None

    model_latency_ms: float | None = None

    # Timing
    duration_ms: float


def _optional_int(
    value: object,
) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    return None


def _optional_bool(
    value: object,
) -> bool | None:
    if isinstance(value, bool):
        return value

    return None


def _optional_str(
    value: object,
) -> str | None:
    if isinstance(value, str):
        return value

    return None


def _optional_float(
    value: object,
) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    return None


def build_request_event(
    state: dict[str, Any],
    *,
    duration_ms: float,
) -> RequestObservabilityEvent:
    """
    Aggregate one finished orchestration request into a structured event.

    Pure function: reads the final state (state metadata + base context
    metadata) and returns the event. Missing data degrades to `None`
    fields; the mapper never raises for incomplete states and never
    mutates its input.
    """

    metadata = state.get("metadata")

    metadata = metadata if isinstance(metadata, dict) else {}

    error = state.get("error")

    route = state.get("route")

    base_context = state.get("base_context")

    context_metadata = getattr(base_context, "metadata", None) if base_context is not None else None

    context_metadata = (
        context_metadata if hasattr(context_metadata, "estimated_input_tokens") else None
    )

    agent_metadata = metadata.get("agent_metadata")

    agent_metadata = agent_metadata if isinstance(agent_metadata, dict) else {}

    onboarding_handled = (
        state.get("onboarding_required") is True and route is None and error is None
    )

    guidance_level = getattr(base_context, "guidance_level", None)

    guidance_level = str(guidance_level.value) if guidance_level is not None else None

    event = RequestObservabilityEvent(
        trace_id=(state.get("trace_id")),
        student_id=_optional_str(state.get("student_id")),
        session_id=_optional_str(state.get("session_id")),
        user_id=(state.get("user_id")),
        outcome=("error" if error else ("completed")),
        error=(str(error) if error else None),
        route=(
            cast(
                RouteName,
                route,
            )
            if isinstance(route, str)
            else None
        ),
        route_reason=_optional_str(state.get("route_reason")),
        selected_agent=(
            cast(
                AgentName,
                metadata.get("selected_agent"),
            )
            if isinstance(metadata.get("selected_agent"), str)
            else None
        ),
        onboarding_handled=(onboarding_handled),
        guidance_level=(guidance_level),
        guidance_reason=_optional_str(getattr(context_metadata, "guidance_reason", None)),
        estimated_input_tokens=_optional_int(
            getattr(context_metadata, "estimated_input_tokens", None)
        ),
        input_token_budget=_optional_int(getattr(context_metadata, "input_token_budget", None)),
        estimated_essential_tokens=_optional_int(
            getattr(context_metadata, "estimated_essential_tokens", None)
        ),
        estimated_history_tokens=_optional_int(
            getattr(context_metadata, "estimated_history_tokens", None)
        ),
        candidate_message_count=_optional_int(
            getattr(context_metadata, "candidate_message_count", None)
        ),
        recent_message_count=_optional_int(getattr(context_metadata, "recent_message_count", None)),
        dropped_message_count=_optional_int(
            getattr(context_metadata, "dropped_message_count", None)
        ),
        truncation_applied=_optional_bool(getattr(context_metadata, "truncation_applied", None)),
        summary_included=_optional_bool(getattr(context_metadata, "summary_included", None)),
        summary_update_attempted=_optional_bool(
            metadata.get("summary_update_attempted"),
        ),
        summary_update_applied=_optional_bool(
            metadata.get("summary_update_applied"),
        ),
        summary_update_reason=_optional_str(
            metadata.get("summary_update_reason"),
        ),
        summary_message_count=_optional_int(
            metadata.get("summary_message_count"),
        ),
        state_update_applied=_optional_bool(
            metadata.get("state_update_applied"),
        ),
        state_update_accepted_count=_optional_int(
            metadata.get("state_update_accepted_count"),
        ),
        state_update_rejected_count=_optional_int(
            metadata.get("state_update_rejected_count"),
        ),
        response_assessed=_optional_bool(
            metadata.get("response_assessed"),
        ),
        response_understanding=_optional_str(
            metadata.get("response_understanding"),
        ),
        response_misconception_detected=_optional_bool(
            metadata.get("response_misconception_detected"),
        ),
        response_signal_recorded=_optional_bool(
            metadata.get("response_signal_recorded"),
        ),
        model=_optional_str(agent_metadata.get("model")),
        model_provider=_optional_str(
            agent_metadata.get("model_provider"),
        ),
        model_latency_ms=_optional_float(
            agent_metadata.get("model_latency_ms"),
        ),
        duration_ms=(duration_ms),
    )

    return event
