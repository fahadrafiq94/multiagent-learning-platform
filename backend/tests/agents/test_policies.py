from __future__ import annotations

from app.agents.policies import (
    AP_PLUS_NAVIGATOR_GUIDANCE,
    GUIDANCE_LEVEL_ORDER,
    PROCESS_COACH_GUIDANCE,
    SCENARIO_GUIDANCE,
    GuidanceLevel,
    next_guidance_level,
)


def test_guidance_levels_have_expected_order() -> None:
    assert GUIDANCE_LEVEL_ORDER == (
        GuidanceLevel.MINIMAL,
        GuidanceLevel.GUIDED,
        GuidanceLevel.DIRECTED,
        GuidanceLevel.EXPLICIT,
    )


def test_next_guidance_level_escalates_minimal() -> None:
    assert next_guidance_level(GuidanceLevel.MINIMAL) == GuidanceLevel.GUIDED


def test_next_guidance_level_escalates_guided() -> None:
    assert next_guidance_level(GuidanceLevel.GUIDED) == GuidanceLevel.DIRECTED


def test_next_guidance_level_escalates_directed() -> None:
    assert next_guidance_level(GuidanceLevel.DIRECTED) == GuidanceLevel.EXPLICIT


def test_explicit_guidance_is_maximum() -> None:
    assert next_guidance_level(GuidanceLevel.EXPLICIT) == GuidanceLevel.EXPLICIT


def test_process_coach_has_policy_for_every_level() -> None:
    assert set(PROCESS_COACH_GUIDANCE) == set(GuidanceLevel)


def test_scenario_agent_has_policy_for_every_level() -> None:
    assert set(SCENARIO_GUIDANCE) == set(GuidanceLevel)


def test_ap_plus_navigator_has_policy_for_every_level() -> None:
    assert set(AP_PLUS_NAVIGATOR_GUIDANCE) == set(GuidanceLevel)


def test_ap_plus_explicit_policy_requires_verified_information() -> None:
    policy = AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.EXPLICIT].lower()

    assert "verified" in policy
    assert "cannot be verified" in policy
    assert "invent" in policy


def test_process_coach_minimal_policy_is_socratic() -> None:
    policy = PROCESS_COACH_GUIDANCE[GuidanceLevel.MINIMAL].lower()

    assert "socratic" in policy
    assert "question" in policy


def test_process_coach_explicit_policy_still_limits_solution_scope() -> None:
    policy = PROCESS_COACH_GUIDANCE[GuidanceLevel.EXPLICIT].lower()

    assert "direct" in policy
    assert "rest of the exercise" in policy
