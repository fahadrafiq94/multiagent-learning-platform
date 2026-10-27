from __future__ import annotations

from uuid import UUID

import pytest

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
from app.domain.guidance_policy import (
    assessment_signal,
    select_guidance,
)

USER_ID = UUID(int=1)
SESSION_ID = UUID(int=2)


def hint_state(**kwargs: object) -> HintState:
    return HintState(**kwargs)


def session_context(
    *,
    active_blocker: str | None = None,
) -> SessionContext:
    return SessionContext(
        user_id=USER_ID,
        session_id=SESSION_ID,
        active_blocker=active_blocker,
    )


def assessment(
    understanding: UnderstandingLevel,
    *,
    misconception: bool = False,
) -> StudentResponseAssessment:
    return StudentResponseAssessment(
        understanding=understanding,
        misconception_detected=misconception,
        reason="Test judgment.",
    )


# ---------------------------------------------------------------------------
# R0: manual override pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pinned",
    [level for level in GuidanceLevel],
)
def test_manual_override_pins_level_and_ignores_other_signals(
    pinned: GuidanceLevel,
) -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=pinned,
                manually_overridden=True,
                consecutive_blocked_attempts=3,
                consecutive_unproductive_attempts=3,
            )
        ),
        session_context=(session_context(active_blocker="Blocked.")),
    )

    assert decision.level is pinned

    assert decision.reason == "manual_override"


# ---------------------------------------------------------------------------
# R1: baseline (backwards compatible)
# ---------------------------------------------------------------------------


def test_fresh_user_stays_at_baseline() -> None:
    decision = select_guidance(
        hint_state=(hint_state()),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.MINIMAL

    assert decision.reason == "baseline"


def test_persisted_level_is_used_as_baseline() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.DIRECTED),
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.DIRECTED

    assert decision.reason == "baseline"


# ---------------------------------------------------------------------------
# R2: blocker floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("streak", "expected"),
    [
        (1, GuidanceLevel.GUIDED),
        (2, GuidanceLevel.DIRECTED),
        (3, GuidanceLevel.EXPLICIT),
        (5, GuidanceLevel.EXPLICIT),
    ],
)
def test_active_blocker_floors_level_by_blocked_streak(
    streak: int,
    expected: GuidanceLevel,
) -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                consecutive_blocked_attempts=streak,
            )
        ),
        session_context=(session_context(active_blocker="Blocked.")),
    )

    assert decision.level is expected

    assert decision.reason == "active_blocker"


def test_baseline_above_blocker_floor_is_kept() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.EXPLICIT),
                consecutive_blocked_attempts=1,
            )
        ),
        session_context=(session_context(active_blocker="Blocked.")),
    )

    assert decision.level is GuidanceLevel.EXPLICIT

    assert decision.reason == "baseline"


# ---------------------------------------------------------------------------
# R3: struggle floor (response-quality driven)
# ---------------------------------------------------------------------------


def test_single_unproductive_attempt_does_not_escalate() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                consecutive_unproductive_attempts=1,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.MINIMAL

    assert decision.reason == "baseline"


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [
        (2, GuidanceLevel.DIRECTED),
        (3, GuidanceLevel.EXPLICIT),
        (4, GuidanceLevel.EXPLICIT),
    ],
)
def test_repeated_struggle_floors_level(
    attempts: int,
    expected: GuidanceLevel,
) -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                consecutive_unproductive_attempts=attempts,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is expected

    assert decision.reason == "repeated_struggle"


def test_struggle_and_blocker_floors_interact_as_max() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                consecutive_blocked_attempts=1,
                consecutive_unproductive_attempts=3,
            )
        ),
        session_context=(session_context(active_blocker="Blocked.")),
    )

    assert decision.level is GuidanceLevel.EXPLICIT

    assert decision.reason == "repeated_struggle"


# ---------------------------------------------------------------------------
# R4: productive-reasoning de-escalation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        (GuidanceLevel.MINIMAL, GuidanceLevel.MINIMAL),
        (GuidanceLevel.GUIDED, GuidanceLevel.MINIMAL),
        (GuidanceLevel.DIRECTED, GuidanceLevel.GUIDED),
        (GuidanceLevel.EXPLICIT, GuidanceLevel.DIRECTED),
    ],
)
def test_productive_reasoning_de_escalates_exactly_one_step(
    base: GuidanceLevel,
    expected: GuidanceLevel,
) -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=base,
                successful_independent_steps=2,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is expected

    assert decision.reason == "productive_reasoning"


def test_de_escalation_requires_two_successes() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.GUIDED),
                successful_independent_steps=1,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.GUIDED

    assert decision.reason == "baseline"


def test_de_escalation_blocked_by_recent_struggle() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.GUIDED),
                successful_independent_steps=3,
                consecutive_unproductive_attempts=1,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.GUIDED

    assert decision.reason == "baseline"


def test_de_escalation_blocked_by_active_blocker() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.MINIMAL),
                successful_independent_steps=3,
                consecutive_blocked_attempts=1,
            )
        ),
        session_context=(session_context(active_blocker="Blocked.")),
    )

    # The blocker floor raised the level, and the open blocker kept
    # the three successful steps from de-escalating.
    assert decision.level is GuidanceLevel.GUIDED

    assert decision.reason == "active_blocker"


def test_de_escalation_is_at_most_one_step_even_after_long_streak() -> None:
    decision = select_guidance(
        hint_state=(
            hint_state(
                guidance_level=(GuidanceLevel.EXPLICIT),
                successful_independent_steps=20,
            )
        ),
        session_context=(session_context()),
    )

    assert decision.level is GuidanceLevel.DIRECTED


# ---------------------------------------------------------------------------
# Purity / no input mutation
# ---------------------------------------------------------------------------


def test_select_guidance_does_not_mutate_inputs() -> None:
    original_hint = hint_state(
        guidance_level=(GuidanceLevel.EXPLICIT),
        successful_independent_steps=2,
    )

    original_session = session_context()

    original_hint_dump = original_hint.model_dump()

    original_session_dump = original_session.model_dump()

    select_guidance(
        hint_state=original_hint,
        session_context=original_session,
    )

    assert original_hint.model_dump() == original_hint_dump

    assert original_session.model_dump() == original_session_dump


# ---------------------------------------------------------------------------
# assessment_signal mapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "understanding",
    [
        UnderstandingLevel.CORRECT,
        UnderstandingLevel.PARTIALLY_CORRECT,
    ],
)
def test_correct_and_partial_assessments_record_progress(
    understanding: UnderstandingLevel,
) -> None:
    update = assessment_signal(assessment(understanding))

    assert update.success_delta == 1

    assert update.struggle_delta == 0

    assert update.reset_struggle is True


@pytest.mark.parametrize(
    "understanding",
    [
        UnderstandingLevel.INCORRECT,
        UnderstandingLevel.OFF_TRACK,
    ],
)
def test_unproductive_assessments_record_struggle(
    understanding: UnderstandingLevel,
) -> None:
    update = assessment_signal(assessment(understanding))

    assert update.success_delta == 0

    assert update.struggle_delta == 1

    assert update.reset_struggle is False


def test_misconception_does_not_change_signal() -> None:
    update = assessment_signal(
        assessment(
            UnderstandingLevel.CORRECT,
            misconception=True,
        ),
    )

    assert update.success_delta == 1

    assert update.struggle_delta == 0


@pytest.mark.parametrize(
    "signal_input",
    [
        pytest.param(None, id=("none")),
        pytest.param(
            assessment(UnderstandingLevel.NOT_APPLICABLE),
            id=("not_applicable"),
        ),
    ],
)
def test_no_signal_inputs_are_no_ops(signal_input: object) -> None:
    update = assessment_signal(signal_input)  # type: ignore[arg-type]

    assert update.success_delta == 0

    assert update.struggle_delta == 0

    assert update.reset_struggle is False
