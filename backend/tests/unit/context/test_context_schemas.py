from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.policies import GuidanceLevel
from app.context import (
    AdaptationSource,
    AgentContext,
    ConfirmedScenarioFact,
    ConversationMessage,
    ConversationRole,
    ConversationSession,
    HintState,
    ProgressStatus,
    ScenarioCard,
    ScenarioFactSource,
    SessionContext,
    SessionSummary,
    StepProgress,
    UserContext,
    UserSpecificAdaptation,
)


def test_new_user_context_starts_from_common_baseline() -> None:
    user_id = uuid4()

    context = UserContext(
        user_id=user_id,
    )

    assert context.user_id == user_id
    assert context.scenario_card is None

    assert context.learning_profile.demonstrated_competencies == []

    assert context.learning_profile.recurring_misconceptions == []

    assert context.progress.steps == []

    assert context.hint_state.guidance_level == GuidanceLevel.MINIMAL

    assert context.adaptations == []
    assert context.version == 1


def test_scenario_card_contains_only_core_business_context() -> None:
    user_id = uuid4()

    card = ScenarioCard(
        user_id=user_id,
        company_context=(
            "UrbanCycle Manufacturing GmbH is a bicycle "
            "manufacturer producing city bicycles, spare "
            "parts, and maintenance services."
        ),
        learner_role="Production Planner",
        business_problem=(
            "Customer orders are delayed because sales, "
            "material planning, purchasing, and production "
            "planning are not sufficiently coordinated."
        ),
        business_goals=[
            "Provide reliable delivery dates.",
            "Improve material availability.",
        ],
        constraints=[
            "One production site.",
            "Limited assembly capacity.",
        ],
    )

    assert card.user_id == user_id

    assert card.learner_role == "Production Planner"

    assert len(card.business_goals) == 2
    assert len(card.constraints) == 2


def test_scenario_card_rejects_company_size_as_core_field() -> None:
    user_id = uuid4()

    with pytest.raises(ValidationError):
        ScenarioCard.model_validate(
            {
                "user_id": user_id,
                "company_context": ("UrbanCycle is a bicycle manufacturer."),
                "company_size": "180 employees",
            }
        )


def test_company_size_can_be_saved_as_confirmed_fact_when_relevant() -> None:
    user_id = uuid4()

    fact = ConfirmedScenarioFact(
        fact=("The company has approximately 180 employees."),
        source_type=ScenarioFactSource.USER,
    )

    card = ScenarioCard(
        user_id=user_id,
        confirmed_facts=[fact],
    )

    assert len(card.confirmed_facts) == 1

    assert card.confirmed_facts[0].fact == "The company has approximately 180 employees."


def test_confirmed_fact_has_provenance() -> None:
    fact = ConfirmedScenarioFact(
        fact=("RollingParts supplies the Wheel Set."),
        source_type=(ScenarioFactSource.ASSIGNMENT_DOCUMENT),
        source_reference="AP+ pilot assignment",
    )

    assert fact.source_type == ScenarioFactSource.ASSIGNMENT_DOCUMENT

    assert fact.source_reference == "AP+ pilot assignment"


def test_user_context_can_store_scenario_card() -> None:
    user_id = uuid4()

    card = ScenarioCard(
        user_id=user_id,
        learner_role="Production Planner",
    )

    context = UserContext(
        user_id=user_id,
        scenario_card=card,
    )

    assert context.scenario_card is not None

    assert context.scenario_card.learner_role == "Production Planner"


def test_user_specific_adaptation_is_user_scoped_through_context() -> None:
    user_a_id = uuid4()
    user_b_id = uuid4()

    adaptation = UserSpecificAdaptation(
        category="hint_delivery",
        instruction="Give one hint at a time.",
        source=(AdaptationSource.EXPLICIT_USER_PREFERENCE),
    )

    context_a = UserContext(
        user_id=user_a_id,
        adaptations=[adaptation],
    )

    context_b = UserContext(
        user_id=user_b_id,
    )

    assert len(context_a.adaptations) == 1
    assert context_b.adaptations == []


def test_default_hint_state_is_minimal() -> None:
    hint_state = HintState()

    assert hint_state.guidance_level == GuidanceLevel.MINIMAL

    assert hint_state.consecutive_blocked_attempts == 0

    assert hint_state.successful_independent_steps == 0

    assert hint_state.manually_overridden is False


def test_user_progress_supports_step_status() -> None:
    progress = StepProgress(
        step_id="8",
        status=ProgressStatus.BLOCKED,
    )

    assert progress.step_id == "8"

    assert progress.status == ProgressStatus.BLOCKED


def test_one_user_can_have_multiple_sessions() -> None:
    user_id = uuid4()

    first = ConversationSession(
        user_id=user_id,
        title="Procurement Discussion",
    )

    second = ConversationSession(
        user_id=user_id,
        title="AP+ Navigation",
    )

    assert first.user_id == second.user_id

    assert first.session_id != second.session_id


def test_messages_keep_user_and_session_identity() -> None:
    user_id = uuid4()
    session_id = uuid4()

    message = ConversationMessage(
        user_id=user_id,
        session_id=session_id,
        role=ConversationRole.USER,
        content="Why do we need this business action?",
    )

    assert message.user_id == user_id
    assert message.session_id == session_id

    assert message.role == ConversationRole.USER


def test_session_context_is_separate_from_user_context() -> None:
    user_id = uuid4()
    session_id = uuid4()

    session_context = SessionContext(
        user_id=user_id,
        session_id=session_id,
        current_objective=("Understand supplier selection."),
        current_step_id="8",
        active_blocker=("The user does not understand why supplier selection is required."),
    )

    assert session_context.current_step_id == "8"

    assert session_context.version == 1


def test_agent_context_is_request_scoped() -> None:
    user_id = uuid4()
    session_id = uuid4()

    context = AgentContext(
        user_id=user_id,
        session_id=session_id,
        current_message=("Why do we need to choose a supplier?"),
    )

    assert context.user_id == user_id
    assert context.session_id == session_id

    assert context.guidance_level == GuidanceLevel.MINIMAL

    assert context.recent_messages == []
    assert context.knowledge_chunks == []


def test_unknown_context_fields_are_rejected() -> None:
    user_id = uuid4()

    with pytest.raises(ValidationError):
        UserContext.model_validate(
            {
                "user_id": user_id,
                "unexpected_field": "should fail",
            }
        )


def test_user_context_rejects_scenario_card_from_another_user() -> None:
    user_a_id = uuid4()
    user_b_id = uuid4()

    scenario_card = ScenarioCard(
        user_id=user_b_id,
        company_context=("A bicycle manufacturing company."),
    )

    with pytest.raises(
        ValidationError,
        match=("scenario_card.user_id must match UserContext.user_id"),
    ):
        UserContext(
            user_id=user_a_id,
            scenario_card=scenario_card,
        )


def test_session_context_rejects_summary_from_another_user() -> None:
    user_a_id = uuid4()
    user_b_id = uuid4()
    session_id = uuid4()

    summary = SessionSummary(
        user_id=user_b_id,
        session_id=session_id,
        overview=("The user is working on procurement."),
    )

    with pytest.raises(
        ValidationError,
        match=("summary.user_id must match SessionContext.user_id"),
    ):
        SessionContext(
            user_id=user_a_id,
            session_id=session_id,
            summary=summary,
        )


def test_session_context_rejects_summary_from_another_session() -> None:
    user_id = uuid4()

    current_session_id = uuid4()
    other_session_id = uuid4()

    summary = SessionSummary(
        user_id=user_id,
        session_id=other_session_id,
        overview=("The user is working on procurement."),
    )

    with pytest.raises(
        ValidationError,
        match=("summary.session_id must match SessionContext.session_id"),
    ):
        SessionContext(
            user_id=user_id,
            session_id=current_session_id,
            summary=summary,
        )


def test_agent_context_rejects_scenario_card_from_another_user() -> None:
    user_a_id = uuid4()
    user_b_id = uuid4()

    scenario_card = ScenarioCard(
        user_id=user_b_id,
        learner_role="Production Planner",
    )

    with pytest.raises(
        ValidationError,
        match=("scenario_card.user_id must match AgentContext.user_id"),
    ):
        AgentContext(
            user_id=user_a_id,
            session_id=uuid4(),
            current_message=("What should I do next?"),
            scenario_card=scenario_card,
        )


def test_agent_context_rejects_session_context_from_another_session() -> None:
    user_id = uuid4()

    current_session_id = uuid4()
    other_session_id = uuid4()

    session_context = SessionContext(
        user_id=user_id,
        session_id=other_session_id,
    )

    with pytest.raises(
        ValidationError,
        match=("session_context.session_id must match AgentContext.session_id"),
    ):
        AgentContext(
            user_id=user_id,
            session_id=current_session_id,
            current_message=("Why is this step required?"),
            session_context=session_context,
        )


def test_agent_context_rejects_message_from_another_user() -> None:
    user_a_id = uuid4()
    user_b_id = uuid4()

    session_id = uuid4()

    foreign_message = ConversationMessage(
        user_id=user_b_id,
        session_id=session_id,
        role=ConversationRole.USER,
        content=("This message belongs to another user."),
    )

    with pytest.raises(
        ValidationError,
        match=("recent message user_id must match AgentContext.user_id"),
    ):
        AgentContext(
            user_id=user_a_id,
            session_id=session_id,
            current_message=("What should I do?"),
            recent_messages=[foreign_message],
        )


def test_agent_context_rejects_message_from_another_session() -> None:
    user_id = uuid4()

    current_session_id = uuid4()
    other_session_id = uuid4()

    foreign_message = ConversationMessage(
        user_id=user_id,
        session_id=other_session_id,
        role=ConversationRole.USER,
        content=("This belongs to another conversation."),
    )

    with pytest.raises(
        ValidationError,
        match=("recent message session_id must match AgentContext.session_id"),
    ):
        AgentContext(
            user_id=user_id,
            session_id=current_session_id,
            current_message=("What should I do?"),
            recent_messages=[foreign_message],
        )


def test_agent_context_accepts_consistent_nested_context() -> None:
    user_id = uuid4()
    session_id = uuid4()

    scenario_card = ScenarioCard(
        user_id=user_id,
        learner_role="Production Planner",
    )

    summary = SessionSummary(
        user_id=user_id,
        session_id=session_id,
        overview=("The user is currently reasoning about supplier selection."),
    )

    session_context = SessionContext(
        user_id=user_id,
        session_id=session_id,
        current_step_id="8",
        summary=summary,
    )

    message = ConversationMessage(
        user_id=user_id,
        session_id=session_id,
        role=ConversationRole.USER,
        content=("Why do we need to select the supplier now?"),
    )

    context = AgentContext(
        user_id=user_id,
        session_id=session_id,
        current_message=("What business information should I consider next?"),
        scenario_card=scenario_card,
        session_context=session_context,
        session_summary=summary,
        recent_messages=[message],
    )

    assert context.user_id == user_id

    assert context.session_id == session_id

    assert context.scenario_card is not None

    assert context.session_context is not None

    assert len(context.recent_messages) == 1
