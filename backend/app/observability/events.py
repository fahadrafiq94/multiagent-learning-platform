from __future__ import annotations

from datetime import UTC, datetime
from typing import (
    Any,
    Literal,
    cast,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
    Use log_payload() for retention: legacy free-text diagnostic fields
    remain available in memory but are excluded from the event export.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Identity
    schema_version: Literal[2] = 2
    student_step_status: str | None = None
    scenario_role_status: str | None = None
    interaction_kind: str | None = None
    context_reply_handled: bool | None = None
    student_step_verification_status: str | None = None
    student_step_verification_calls: int | None = None
    student_step_verification_latency_ms: float | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: UUID | None = None
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
    guidance_level_used: str | None = None
    guidance_level_next: str | None = None
    assessment_status: str | None = None
    teaching_status: str | None = None
    teaching_abstained: bool | None = None

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

    # Bounded retrieval telemetry, never queries, passages or planner reasons.
    rag_status: str | None = None
    rag_origin: str | None = None
    rag_planner_failure: str | None = None
    rag_learning_signal_suppressed: bool | None = None
    rag_actions: list[str] = Field(default_factory=list)
    rag_planner_calls: int | None = None
    rag_planner_latency_ms: float | None = None
    rag_embedding_calls: int | None = None
    rag_selected_count: int | None = None
    rag_provided_count: int | None = None
    rag_budget_omissions: int | None = None
    citation_support_status: str | None = None
    citation_support_calls: int | None = None
    citation_support_latency_ms: float | None = None
    citation_support_claim_count: int | None = None
    citation_support_abstained: bool | None = None
    citation_support_link_mode: str | None = None

    def log_payload(self) -> dict[str, Any]:
        """Only structured metrics/identifiers cross the routine logging boundary."""
        payload = self.model_dump(mode="json", exclude={"error", "route_reason"})
        # These older fields can contain model-generated prose or exception text.
        for name in ("guidance_reason", "summary_update_reason"):
            payload.pop(name, None)
        payload["error_present"] = self.error is not None
        return payload


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


def _choice(value: object, choices: set[str]) -> str | None:
    return value if isinstance(value, str) and value in choices else None


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
    rag = metadata.get("rag")
    rag = rag if isinstance(rag, dict) else {}

    error = state.get("error")

    route = state.get("route")

    base_context = state.get("base_context")

    context_metadata = getattr(base_context, "metadata", None) if base_context is not None else None

    context_metadata = (
        context_metadata if hasattr(context_metadata, "estimated_input_tokens") else None
    )

    agent_metadata = metadata.get("agent_metadata")

    agent_metadata = agent_metadata if isinstance(agent_metadata, dict) else {}

    onboarding_handled = state.get("onboarding_required") is True and error is None

    def context_metric(name: str):
        # Grounding can re-budget the rendered specialist prompt after the base
        # context was assembled. Retain the final request's values when present.
        return metadata.get(name, getattr(context_metadata, name, None))

    guidance_level = getattr(base_context, "guidance_level", None)

    guidance_level = str(guidance_level.value) if guidance_level is not None else None

    event = RequestObservabilityEvent(
        guidance_level_used=_choice(
            metadata.get("guidance_level_used"), {"minimal", "guided", "directed", "explicit"}
        ),
        guidance_level_next=_choice(
            metadata.get("guidance_level_next"), {"minimal", "guided", "directed", "explicit"}
        ),
        assessment_status=_choice(
            metadata.get("assessment_status"),
            {
                "omitted",
                "accepted",
                "not_assessable",
                "uncertain",
                "unanchored",
                "contradictory",
                "suppressed",
            },
        ),
        teaching_status=_choice(
            rag.get("teaching_status"),
            {
                "followed",
                "violated",
                "uncertain",
                "not_checked",
                "timeout",
                "unavailable",
                "invalid_verdict",
                "budget_exceeded",
                "unsupported",
            },
        ),
        teaching_abstained=_optional_bool(rag.get("teaching_abstained")),
        interaction_kind=_choice(
            metadata.get("interaction_kind"),
            {
                "help",
                "acknowledgement",
                "greeting",
                "recall",
                "mixed",
                "unclear",
            },
        ),
        student_step_verification_status=_choice(
            metadata.get("student_step_verification_status"),
            {
                "proposal",
                "no_proposal",
                "timeout",
                "unavailable",
                "invalid",
                "conversation_normalized",
                "clarification_deferred",
            },
        ),
        student_step_verification_calls=_optional_int(
            metadata.get("student_step_verification_calls")
        ),
        student_step_verification_latency_ms=_optional_float(
            metadata.get("student_step_verification_latency_ms")
        ),
        context_reply_handled=_optional_bool(metadata.get("context_reply_handled")),
        student_step_status=_choice(
            metadata.get("student_step_status"),
            {
                "persisted",
                "unchanged",
                "clarification_required",
                "rejected",
                "omitted",
                "unavailable",
                "failed",
            },
        ),
        scenario_role_status=_choice(
            metadata.get("scenario_role_status"), {"persisted", "missing"}
        ),
        request_id=metadata.get("request_id"),
        rag_status=_choice(
            rag.get("status"),
            {
                "skipped",
                "clarify",
                "unavailable",
                "error",
                "found",
                "no_evidence",
            },
        ),
        rag_origin=_choice(rag.get("origin"), {"gate", "model", "fallback"}),
        rag_planner_failure=_choice(
            rag.get("planner_failure"),
            {
                "invalid_plan",
                "planner_timeout",
                "planner_unavailable",
            },
        ),
        rag_learning_signal_suppressed=_optional_bool(rag.get("learning_signal_suppressed")),
        rag_actions=[
            a
            for a in rag.get("actions", [])
            if _choice(
                a,
                {
                    "skip",
                    "clarify",
                    "reuse_evidence",
                    "metadata_lookup",
                    "document_search",
                },
            )
            is not None
        ]
        if isinstance(rag.get("actions"), list)
        else [],
        rag_planner_calls=_optional_int(rag.get("planner_calls")),
        rag_planner_latency_ms=_optional_float(rag.get("planner_latency_ms")),
        rag_embedding_calls=_optional_int(rag.get("embedding_calls")),
        rag_selected_count=_optional_int(rag.get("selected_count")),
        rag_provided_count=_optional_int(rag.get("provided_count")),
        rag_budget_omissions=_optional_int(rag.get("budget_omissions")),
        citation_support_status=_choice(
            rag.get("support_status"),
            {
                "passed",
                "unsupported",
                "uncertain",
                "invalid_links",
                "invalid_verdict",
                "timeout",
                "unavailable",
                "budget_exceeded",
                "no_evidence",
            },
        ),
        citation_support_calls=_optional_int(rag.get("support_calls")),
        citation_support_latency_ms=_optional_float(rag.get("support_latency_ms")),
        citation_support_claim_count=_optional_int(rag.get("support_claim_count")),
        citation_support_abstained=_optional_bool(rag.get("support_abstained")),
        citation_support_link_mode=_choice(
            rag.get("support_link_mode"), {"declared_claims", "whole_answer"}
        ),
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
        estimated_input_tokens=_optional_int(context_metric("estimated_input_tokens")),
        input_token_budget=_optional_int(context_metric("input_token_budget")),
        estimated_essential_tokens=_optional_int(context_metric("estimated_essential_tokens")),
        estimated_history_tokens=_optional_int(context_metric("estimated_history_tokens")),
        candidate_message_count=_optional_int(
            getattr(context_metadata, "candidate_message_count", None)
        ),
        recent_message_count=_optional_int(context_metric("recent_message_count")),
        dropped_message_count=_optional_int(context_metric("dropped_message_count")),
        truncation_applied=_optional_bool(context_metric("truncation_applied")),
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
