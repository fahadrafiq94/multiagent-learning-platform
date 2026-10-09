from __future__ import annotations

from uuid import uuid4

import pytest

from app.context import (
    ScenarioDraft,
    ScenarioRequiredField,
    ScenarioSetupStatus,
)
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.scenario_card_service import (
    ScenarioCardService,
)
from app.services.scenario_onboarding_service import (
    ScenarioOnboardingService,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.user_context_service import (
    UserContextService,
)


def create_services() -> tuple[
    InMemoryRepositories,
    ConversationService,
    UserContextService,
    SessionContextService,
    ScenarioCardService,
    ScenarioOnboardingService,
]:
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
    )

    user_context_service = UserContextService(
        repository=(repositories.user_contexts),
    )

    session_context_service = SessionContextService(
        repository=(repositories.session_contexts),
    )

    scenario_card_service = ScenarioCardService(
        user_context_service=(user_context_service),
    )

    onboarding_service = ScenarioOnboardingService(
        session_context_service=(session_context_service),
        scenario_card_service=(scenario_card_service),
    )

    return (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        scenario_card_service,
        onboarding_service,
    )


async def create_user(
    *,
    repositories: InMemoryRepositories,
    student_id: str,
) -> UserIdentity:
    identity = UserIdentity(
        student_id=student_id,
    )

    return await repositories.identities.create(identity)


@pytest.mark.asyncio
async def test_new_session_starts_with_not_started_status() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        _,
        _,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    context = await repositories.session_contexts.get(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context is not None

    assert context.scenario_setup_status == ScenarioSetupStatus.NOT_STARTED

    assert context.scenario_draft is None


@pytest.mark.asyncio
async def test_start_enters_collecting_when_no_card_exists() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        _,
        onboarding_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    status = await onboarding_service.start(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert status == ScenarioSetupStatus.COLLECTING


@pytest.mark.asyncio
async def test_needs_onboarding_returns_true_without_card() -> None:
    (
        repositories,
        _,
        _,
        _,
        _,
        onboarding_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    assert (
        await onboarding_service.needs_onboarding(
            user_id=user.user_id,
        )
        is True
    )


@pytest.mark.asyncio
async def test_assessment_detects_all_missing_required_fields() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        onboarding_service,
    ) = create_services()

    draft = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
    )

    assessment = onboarding_service.assess(draft)

    assert assessment.ready_for_review is False

    assert set(assessment.missing_required_fields) == {
        ScenarioRequiredField.COMPANY_CONTEXT,
        ScenarioRequiredField.LEARNER_ROLE,
        ScenarioRequiredField.BUSINESS_PROBLEM,
        ScenarioRequiredField.BUSINESS_GOALS,
    }


@pytest.mark.asyncio
async def test_assessment_detects_missing_fields_together() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        onboarding_service,
    ) = create_services()

    draft = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        company_context=("UrbanCycle is a bicycle manufacturer."),
        business_problem=("Customer orders are delayed."),
    )

    assessment = onboarding_service.assess(draft)

    assert assessment.ready_for_review is False

    assert set(assessment.missing_required_fields) == {
        ScenarioRequiredField.LEARNER_ROLE,
        ScenarioRequiredField.BUSINESS_GOALS,
    }
