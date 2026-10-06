from __future__ import annotations

from uuid import uuid4

import pytest

from app.context import (
    ConversationSession,
    ScenarioDraft,
    ScenarioSetupStatus,
    SessionContextUpdate,
)
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    RepositoryNotFoundError,
    VersionConflictError,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.session_context_service import (
    SessionContextService,
)


def create_services() -> tuple[
    InMemoryRepositories,
    ConversationService,
    SessionContextService,
]:
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
    )

    session_context_service = SessionContextService(
        repository=(repositories.session_contexts),
    )

    return (
        repositories,
        conversation_service,
        session_context_service,
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


async def create_session(
    *,
    repositories: InMemoryRepositories,
    conversation_service: ConversationService,
    student_id: str = "K12345",
) -> tuple[
    UserIdentity,
    ConversationSession,
]:
    user = await create_user(
        repositories=repositories,
        student_id=student_id,
    )

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    return user, session


@pytest.mark.asyncio
async def test_get_required_returns_initial_session_context() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.user_id == user.user_id

    assert context.session_id == session.session_id

    assert context.version == 1

    assert context.current_objective is None
    assert context.current_step_id is None
    assert context.current_process_phase is None
    assert context.active_blocker is None
    assert context.last_selected_agent is None

    assert context.unresolved_questions == []


@pytest.mark.asyncio
async def test_get_required_rejects_missing_session_context() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, _ = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    with pytest.raises(RepositoryNotFoundError):
        await service.get_required(
            user_id=user.user_id,
            session_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_save_rejects_context_from_another_user() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user_a, session_a = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    context = await service.get_required(
        user_id=user_a.user_id,
        session_id=session_a.session_id,
    )

    with pytest.raises(
        ValueError,
        match=("context.user_id must match SessionContextService user_id"),
    ):
        await service.save(
            user_id=user_b.user_id,
            session_id=session_a.session_id,
            context=context,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_save_rejects_context_from_another_session() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session_a = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    session_b = await conversation_service.create_session(
        user_id=user.user_id,
    )

    context = await service.get_required(
        user_id=user.user_id,
        session_id=session_a.session_id,
    )

    with pytest.raises(
        ValueError,
        match=("context.session_id must match SessionContextService session_id"),
    ):
        await service.save(
            user_id=user.user_id,
            session_id=session_b.session_id,
            context=context,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_update_can_change_multiple_fields_in_one_version() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    updated = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_objective=("Understand supplier selection."),
            current_step_id="8",
            current_process_phase=("supplier_selection"),
            active_blocker=("Supplier evaluation criteria are unclear."),
            last_selected_agent=("process_coach"),
        ),
    )

    assert updated.version == 2

    assert updated.current_objective == "Understand supplier selection."

    assert updated.current_step_id == "8"

    assert updated.current_process_phase == "supplier_selection"

    assert updated.active_blocker == ("Supplier evaluation criteria are unclear.")

    assert updated.last_selected_agent == "process_coach"


@pytest.mark.asyncio
async def test_partial_update_does_not_overwrite_other_fields() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_objective=("Understand procurement."),
            current_step_id="8",
            active_blocker=("Supplier evaluation is unclear."),
        ),
    )

    second = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_step_id="9",
        ),
        expected_version=first.version,
    )

    assert second.version == 3

    assert second.current_objective == "Understand procurement."

    assert second.current_step_id == "9"

    assert second.active_blocker == ("Supplier evaluation is unclear.")


@pytest.mark.asyncio
async def test_explicit_none_clears_nullable_field() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.set_active_blocker(
        user_id=user.user_id,
        session_id=session.session_id,
        blocker=("The user is unsure about supplier criteria."),
    )

    assert first.active_blocker is not None

    cleared = await service.clear_active_blocker(
        user_id=user.user_id,
        session_id=session.session_id,
        expected_version=first.version,
    )

    assert cleared.active_blocker is None

    assert cleared.version == first.version + 1


@pytest.mark.asyncio
async def test_empty_update_does_not_increment_version() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    updated = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(),
    )

    assert updated.version == 1


@pytest.mark.asyncio
async def test_set_current_step_updates_step() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.set_current_step(
        user_id=user.user_id,
        session_id=session.session_id,
        step_id="12",
    )

    assert context.current_step_id == "12"
    assert context.version == 2


@pytest.mark.asyncio
async def test_set_process_phase_updates_phase() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.set_process_phase(
        user_id=user.user_id,
        session_id=session.session_id,
        process_phase="production_planning",
    )

    assert context.current_process_phase == "production_planning"


@pytest.mark.asyncio
async def test_last_selected_agent_can_be_recorded() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.set_last_selected_agent(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="ap_plus_navigator",
    )

    assert context.last_selected_agent == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_unresolved_question_can_be_added() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question=("Which supplier should be selected?"),
    )

    assert context.unresolved_questions == ["Which supplier should be selected?"]

    assert context.version == 2


@pytest.mark.asyncio
async def test_duplicate_unresolved_question_is_not_added() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question=("Which supplier should be selected?"),
    )

    second = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question=("  which supplier should be selected?  "),
    )

    assert len(second.unresolved_questions) == 1

    assert second.version == first.version


@pytest.mark.asyncio
async def test_unresolved_question_can_be_removed() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question=("Which supplier should be selected?"),
    )

    second = await service.remove_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question=("which supplier should be selected?"),
        expected_version=first.version,
    )

    assert second.unresolved_questions == []

    assert second.version == first.version + 1


@pytest.mark.asyncio
async def test_clear_unresolved_questions_removes_all_questions() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question="Question one?",
    )

    second = await service.add_unresolved_question(
        user_id=user.user_id,
        session_id=session.session_id,
        question="Question two?",
        expected_version=first.version,
    )

    cleared = await service.clear_unresolved_questions(
        user_id=user.user_id,
        session_id=session.session_id,
        expected_version=second.version,
    )

    assert cleared.unresolved_questions == []


@pytest.mark.asyncio
async def test_update_deduplicates_unresolved_questions() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=[
                "Which supplier?",
                "  which supplier? ",
                "What quantity is required?",
            ],
        ),
    )

    assert context.unresolved_questions == [
        "Which supplier?",
        "What quantity is required?",
    ]


@pytest.mark.asyncio
async def test_stale_expected_version_is_rejected() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    initial = await service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    first = await service.set_current_step(
        user_id=user.user_id,
        session_id=session.session_id,
        step_id="8",
        expected_version=initial.version,
    )

    assert first.version == 2

    with pytest.raises(VersionConflictError):
        await service.set_process_phase(
            user_id=user.user_id,
            session_id=session.session_id,
            process_phase="procurement",
            expected_version=initial.version,
        )


@pytest.mark.asyncio
async def test_sessions_keep_independent_state() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session_a = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    session_b = await conversation_service.create_session(
        user_id=user.user_id,
    )

    await service.set_current_step(
        user_id=user.user_id,
        session_id=session_a.session_id,
        step_id="8",
    )

    await service.set_current_step(
        user_id=user.user_id,
        session_id=session_b.session_id,
        step_id="17",
    )

    context_a = await service.get_required(
        user_id=user.user_id,
        session_id=session_a.session_id,
    )

    context_b = await service.get_required(
        user_id=user.user_id,
        session_id=session_b.session_id,
    )

    assert context_a.current_step_id == "8"

    assert context_b.current_step_id == "17"


@pytest.mark.asyncio
async def test_user_cannot_read_another_users_session_context() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user_a, session_a = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    with pytest.raises(RepositoryNotFoundError):
        await service.get_required(
            user_id=user_b.user_id,
            session_id=session_a.session_id,
        )


@pytest.mark.asyncio
async def test_runtime_update_preserves_scenario_onboarding_state() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    context.scenario_setup_status = ScenarioSetupStatus.COLLECTING

    context.scenario_draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=session.session_id,
        company_context=("UrbanCycle is a bicycle manufacturer."),
    )

    saved = await service.save(
        user_id=user.user_id,
        session_id=session.session_id,
        context=context,
        expected_version=context.version,
    )

    updated = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_step_id="8",
            last_selected_agent=("scenario"),
        ),
        expected_version=saved.version,
    )

    assert updated.scenario_setup_status == ScenarioSetupStatus.COLLECTING

    assert updated.scenario_draft is not None

    assert updated.scenario_draft.company_context == ("UrbanCycle is a bicycle manufacturer.")

    assert updated.current_step_id == "8"

    assert updated.last_selected_agent == "scenario"


@pytest.mark.asyncio
async def test_unresolved_questions_none_explicitly_clears_list() -> None:
    (
        repositories,
        conversation_service,
        service,
    ) = create_services()

    user, session = await create_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=[
                "Question one?",
                "Question two?",
            ],
        ),
    )

    cleared = await service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=None,
        ),
        expected_version=first.version,
    )

    assert cleared.unresolved_questions == []
