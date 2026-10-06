from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

import pytest

from app.agents.policies import GuidanceLevel
from app.context import (
    ProgressStatus,
    ScenarioCard,
    StepProgress,
)
from app.identity import UserIdentity
from app.repositories import (
    RepositoryNotFoundError,
    VersionConflictError,
    create_in_memory_repositories,
)
from app.services.user_context_service import (
    UserContextService,
)


def create_service():
    repositories = create_in_memory_repositories()

    service = UserContextService(
        repository=(repositories.user_contexts),
    )

    return repositories, service


async def create_user(
    *,
    repositories,
    student_id: str,
) -> UserIdentity:
    identity = UserIdentity(
        student_id=student_id,
    )

    created = await repositories.identities.create(identity)
    return cast(UserIdentity, created)


@pytest.mark.asyncio
async def test_get_or_create_returns_common_baseline() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    context = await service.get_or_create(
        user_id=user.user_id,
    )

    assert context.user_id == user.user_id

    assert context.scenario_card is None

    assert context.learning_profile.demonstrated_competencies == []

    assert context.learning_profile.recurring_misconceptions == []

    assert context.learning_profile.support_preferences == []

    assert context.progress.steps == []

    assert context.hint_state.guidance_level == GuidanceLevel.MINIMAL

    assert context.hint_state.consecutive_blocked_attempts == 0

    assert context.hint_state.successful_independent_steps == 0

    assert context.adaptations == []

    assert context.version == 1


@pytest.mark.asyncio
async def test_get_or_create_returns_existing_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    initial = await service.get_or_create(
        user_id=user.user_id,
    )

    initial.progress.steps.append(
        StepProgress(
            step_id="8",
            status=(ProgressStatus.COMPLETED),
        )
    )

    saved = await service.save(
        user_id=user.user_id,
        context=initial,
        expected_version=1,
    )

    loaded = await service.get_or_create(
        user_id=user.user_id,
    )

    assert loaded.version == 2

    assert len(loaded.progress.steps) == 1

    assert loaded.progress.steps[0].step_id == "8"

    assert loaded.progress.steps[0].status == ProgressStatus.COMPLETED

    assert saved.version == loaded.version


@pytest.mark.asyncio
async def test_different_users_receive_independent_baselines() -> None:
    repositories, service = create_service()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    context_a = await service.get_or_create(
        user_id=user_a.user_id,
    )

    context_b = await service.get_or_create(
        user_id=user_b.user_id,
    )

    context_a.progress.steps.append(
        StepProgress(
            step_id="5",
            status=(ProgressStatus.COMPLETED),
        )
    )

    assert len(context_a.progress.steps) == 1

    assert context_b.progress.steps == []

    assert context_a.user_id != context_b.user_id


@pytest.mark.asyncio
async def test_scenario_card_survives_cross_session_user_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    context = await service.get_or_create(
        user_id=user.user_id,
    )

    context.scenario_card = ScenarioCard(
        user_id=user.user_id,
        company_context=("UrbanCycle Manufacturing GmbH is a bicycle manufacturer."),
        learner_role="Production Planner",
        business_problem=(
            "Customer orders are delayed because "
            "planning activities are not sufficiently "
            "coordinated."
        ),
    )

    saved = await service.save(
        user_id=user.user_id,
        context=context,
        expected_version=1,
    )

    assert saved.version == 2

    later_context = await service.get_or_create(
        user_id=user.user_id,
    )

    assert later_context.scenario_card is not None

    assert later_context.scenario_card.learner_role == "Production Planner"

    assert later_context.scenario_card.company_context == (
        "UrbanCycle Manufacturing GmbH is a bicycle manufacturer."
    )


@pytest.mark.asyncio
async def test_get_or_create_requires_existing_identity() -> None:
    _, service = create_service()

    unknown_user_id = uuid4()

    with pytest.raises(
        RepositoryNotFoundError,
        match="UserIdentity not found",
    ):
        await service.get_or_create(
            user_id=unknown_user_id,
        )


@pytest.mark.asyncio
async def test_get_does_not_create_missing_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    context = await service.get(
        user_id=user.user_id,
    )

    assert context is None

    repository_context = await repositories.user_contexts.get(user.user_id)

    assert repository_context is None


@pytest.mark.asyncio
async def test_get_required_fails_when_context_is_missing() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    with pytest.raises(
        RepositoryNotFoundError,
        match="UserContext not found",
    ):
        await service.get_required(
            user_id=user.user_id,
        )


@pytest.mark.asyncio
async def test_get_required_returns_existing_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    created = await service.get_or_create(
        user_id=user.user_id,
    )

    loaded = await service.get_required(
        user_id=user.user_id,
    )

    assert loaded.user_id == created.user_id

    assert loaded.version == 1


@pytest.mark.asyncio
async def test_save_rejects_context_from_another_user() -> None:
    repositories, service = create_service()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    context_b = await service.get_or_create(
        user_id=user_b.user_id,
    )

    with pytest.raises(
        ValueError,
        match=("context.user_id must match UserContextService user_id"),
    ):
        await service.save(
            user_id=user_a.user_id,
            context=context_b,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_save_increments_context_version() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    context = await service.get_or_create(
        user_id=user.user_id,
    )

    context.progress.steps.append(
        StepProgress(
            step_id="3",
            status=(ProgressStatus.IN_PROGRESS),
        )
    )

    saved = await service.save(
        user_id=user.user_id,
        context=context,
        expected_version=1,
    )

    assert saved.version == 2


@pytest.mark.asyncio
async def test_stale_context_save_is_rejected() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    initial = await service.get_or_create(
        user_id=user.user_id,
    )

    request_a = initial.model_copy(deep=True)

    request_b = initial.model_copy(deep=True)

    request_a.progress.steps.append(
        StepProgress(
            step_id="4",
            status=(ProgressStatus.COMPLETED),
        )
    )

    await service.save(
        user_id=user.user_id,
        context=request_a,
        expected_version=1,
    )

    request_b.progress.steps.append(
        StepProgress(
            step_id="5",
            status=(ProgressStatus.BLOCKED),
        )
    )

    with pytest.raises(VersionConflictError):
        await service.save(
            user_id=user.user_id,
            context=request_b,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_concurrent_first_access_returns_single_user_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    results = await asyncio.gather(
        *[
            service.get_or_create(
                user_id=user.user_id,
            )
            for _ in range(10)
        ]
    )

    assert len(results) == 10

    assert all(context.user_id == user.user_id for context in results)

    assert all(context.version == 1 for context in results)

    persisted = await repositories.user_contexts.get(user.user_id)

    assert persisted is not None

    assert persisted.version == 1
