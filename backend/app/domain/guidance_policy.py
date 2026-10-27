from __future__ import annotations

from dataclasses import dataclass

from app.agents.policies import (
    GUIDANCE_LEVEL_ORDER,
)
from app.agents.schemas import (
    StudentResponseAssessment,
    UnderstandingLevel,
)
from app.context.schemas import (
    HintState,
    SessionContext,
)
from app.domain.guidance import (
    GuidanceLevel,
)


def _order(level: GuidanceLevel) -> int:
    return GUIDANCE_LEVEL_ORDER.index(level)


def _one_step_down(level: GuidanceLevel) -> GuidanceLevel:
    index = _order(level)

    if index == 0:
        return GuidanceLevel.MINIMAL

    return GUIDANCE_LEVEL_ORDER[index - 1]


@dataclass(frozen=True)
class GuidanceDecision:
    """
    Result of the deterministic guidance policy.

    `reason` is one of: manual_override, baseline, active_blocker,
    repeated_struggle, productive_reasoning.
    """

    level: GuidanceLevel

    reason: str


def _blocker_floor(
    consecutive_blocked_attempts: int,
) -> GuidanceLevel:
    if consecutive_blocked_attempts >= 3:
        return GuidanceLevel.EXPLICIT

    if consecutive_blocked_attempts == 2:
        return GuidanceLevel.DIRECTED

    return GuidanceLevel.GUIDED


def _struggle_floor(
    consecutive_unproductive_attempts: int,
) -> GuidanceLevel:
    if consecutive_unproductive_attempts >= 3:
        return GuidanceLevel.EXPLICIT

    return GuidanceLevel.DIRECTED


def select_guidance(
    *,
    hint_state: HintState,
    session_context: SessionContext,
) -> GuidanceDecision:
    """
    Central deterministic guidance-level selection (Sprint 4B.9).

    Pure function: reads persisted signals (hint state) and session
    state, returns the level one request should use. No I/O, no
    mutation, no LLM involvement — the level is always the output of
    this policy, never of a model.
    """

    if hint_state.manually_overridden:
        return GuidanceDecision(
            level=hint_state.guidance_level,
            reason="manual_override",
        )

    level = hint_state.guidance_level

    reason = "baseline"

    if session_context.active_blocker:
        floor = _blocker_floor(
            hint_state.consecutive_blocked_attempts,
        )

        if _order(floor) > _order(level):
            level = floor
            reason = "active_blocker"

    if hint_state.consecutive_unproductive_attempts >= 2:
        floor = _struggle_floor(
            hint_state.consecutive_unproductive_attempts,
        )

        if _order(floor) > _order(level):
            level = floor
            reason = "repeated_struggle"

    if (
        session_context.active_blocker is None
        and hint_state.consecutive_unproductive_attempts == 0
        and hint_state.successful_independent_steps >= 2
    ):
        level = _one_step_down(level)
        reason = "productive_reasoning"

    return GuidanceDecision(
        level=level,
        reason=reason,
    )


@dataclass(frozen=True)
class ResponseSignalUpdate:
    """
    Deterministic counter movement for one judged student response.
    """

    success_delta: int = 0

    struggle_delta: int = 0

    reset_struggle: bool = False


def assessment_signal(
    assessment: StudentResponseAssessment | None,
) -> ResponseSignalUpdate:
    """
    Map one student-response assessment to counter movements.

    The raw assessment is LLM output and is never persisted; only
    these bounded, deterministic movements are.
    """

    if assessment is None:
        return ResponseSignalUpdate()

    understanding = assessment.understanding

    if understanding in (
        UnderstandingLevel.CORRECT,
        UnderstandingLevel.PARTIALLY_CORRECT,
    ):
        return ResponseSignalUpdate(
            success_delta=1,
            reset_struggle=True,
        )

    if understanding in (
        UnderstandingLevel.INCORRECT,
        UnderstandingLevel.OFF_TRACK,
    ):
        return ResponseSignalUpdate(
            struggle_delta=1,
        )

    return ResponseSignalUpdate()
