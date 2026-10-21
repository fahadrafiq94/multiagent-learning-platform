from __future__ import annotations

import pytest

from app.context.schemas import (
    ScenarioDraft,
    ScenarioSetupStatus,
    SessionContextUpdate,
)
from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
    SessionStateUpdateOperation,
    SessionStateUpdateProposal,
)
from app.identity import UserIdentity
from app.repositories import (
    RepositoryNotFoundError,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.session_state_update_service import (
    SessionStateUpdateService,
)


def create_services():
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
    )

    session_context_service = SessionContextService(
        repository=(repositories.session_contexts),
    )

    state_update_service = SessionStateUpdateService(
        session_context_service=(session_context_service),
    )

    return (
        repositories,
        conversation_service,
        session_context_service,
        state_update_service,
    )


async def create_user_and_session(
    *,
    repositories,
    conversation_service,
    student_id: str = "K12345",
):
    user = await repositories.identities.create(UserIdentity(student_id=student_id))

    session = await conversation_service.create_session(user_id=user.user_id)

    return user, session


def proposal(
    *operations: (SessionStateUpdateOperation),
) -> SessionStateUpdateProposal:
    return SessionStateUpdateProposal(operations=list(operations))


def operation(
    *,
    field: SessionStateField,
    action: SessionStateAction,
    value: str | None = None,
    reason: str = "Test reason.",
) -> SessionStateUpdateOperation:
    return SessionStateUpdateOperation(
        field=field,
        action=action,
        value=value,
        reason=reason,
    )


@pytest.mark.asyncio
async def test_ap_plus_navigator_can_set_current_step() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name=("ap_plus_navigator"),
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_STEP_ID),
                action=(SessionStateAction.SET),
                value="8",
                reason=("The user explicitly said they are at step 8."),
            )
        ),
    )

    assert result.applied is True

    assert len(result.accepted_operations) == 1

    assert result.rejected_operations == []

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id == "8"

    assert context.last_selected_agent == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_process_coach_cannot_set_current_step() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_STEP_ID),
                action=(SessionStateAction.SET),
                value="8",
            )
        ),
    )

    assert result.accepted_operations == []

    assert len(result.rejected_operations) == 1

    assert result.rejected_operations[0].reason == ("Field is not allowed for this agent.")

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id is None

    # Deterministic application-owned state.
    assert context.last_selected_agent == "process_coach"


@pytest.mark.asyncio
async def test_scenario_agent_cannot_set_current_step() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="scenario",
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_STEP_ID),
                action=(SessionStateAction.SET),
                value="17",
            )
        ),
    )

    assert len(result.rejected_operations) == 1

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id is None


@pytest.mark.asyncio
async def test_process_coach_can_set_current_objective() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_OBJECTIVE),
                action=(SessionStateAction.SET),
                value=("Understand supplier selection."),
            )
        ),
    )

    assert result.applied is True

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_objective == ("Understand supplier selection.")


@pytest.mark.asyncio
async def test_process_coach_can_set_process_phase() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_PROCESS_PHASE),
                action=(SessionStateAction.SET),
                value="procurement",
            )
        ),
    )

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_process_phase == "procurement"


@pytest.mark.asyncio
async def test_active_blocker_can_be_set() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.ACTIVE_BLOCKER),
                action=(SessionStateAction.SET),
                value=("Supplier criteria are unclear."),
            )
        ),
    )

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.active_blocker == ("Supplier criteria are unclear.")


@pytest.mark.asyncio
async def test_active_blocker_can_be_cleared() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await session_context_service.update(
        user_id=user.user_id,
        session_id=(session.session_id),
        changes=(
            SessionContextUpdate(
                active_blocker=("Supplier criteria are unclear."),
            )
        ),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.ACTIVE_BLOCKER),
                action=(SessionStateAction.CLEAR),
                reason=("The blocker was resolved."),
            )
        ),
    )

    assert result.applied is True

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.active_blocker is None

    assert context.version > first.version


@pytest.mark.asyncio
async def test_unresolved_question_can_be_added() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.ADD),
                value=("Which supplier should be selected?"),
            )
        ),
    )

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.unresolved_questions == [("Which supplier should be selected?")]


@pytest.mark.asyncio
async def test_duplicate_unresolved_question_is_not_added() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=[("Which supplier should be selected?")],
            last_selected_agent=("process_coach"),
        ),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.ADD),
                value=(" which SUPPLIER should be selected? "),
            )
        ),
    )

    assert result.applied is False

    assert len(result.rejected_operations) == 1

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert len(context.unresolved_questions) == 1


@pytest.mark.asyncio
async def test_unresolved_question_can_be_removed() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=[
                "Question one?",
                "Question two?",
            ],
        ),
    )

    await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="scenario",
        proposal=proposal(
            operation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.REMOVE),
                value="Question one?",
            )
        ),
    )

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.unresolved_questions == ["Question two?"]


@pytest.mark.asyncio
async def test_unresolved_questions_can_be_cleared() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            unresolved_questions=[
                "Question one?",
                "Question two?",
            ],
        ),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.CLEAR),
            )
        ),
    )

    assert result.applied is True

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.unresolved_questions == []


@pytest.mark.asyncio
async def test_noop_scalar_update_is_rejected() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_process_phase=("procurement"),
            last_selected_agent=("process_coach"),
        ),
    )

    before = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.CURRENT_PROCESS_PHASE),
                action=(SessionStateAction.SET),
                value="procurement",
            )
        ),
    )

    assert result.applied is False

    assert len(result.rejected_operations) == 1

    after = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert after.version == before.version


@pytest.mark.asyncio
async def test_last_selected_agent_is_application_controlled() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="scenario",
        proposal=None,
    )

    assert result.applied is True

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.last_selected_agent == "scenario"


@pytest.mark.asyncio
async def test_empty_proposal_is_noop_when_agent_is_already_selected() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    first = await session_context_service.update(
        user_id=user.user_id,
        session_id=(session.session_id),
        changes=SessionContextUpdate(
            last_selected_agent=("scenario"),
        ),
    )

    result = await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="scenario",
        proposal=(SessionStateUpdateProposal()),
    )

    assert result.applied is False

    assert result.session_context_version == first.version


@pytest.mark.asyncio
async def test_runtime_update_preserves_scenario_onboarding_state() -> None:
    (
        repositories,
        conversation_service,
        session_context_service,
        service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    context.scenario_setup_status = ScenarioSetupStatus.REVIEWING

    context.scenario_draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=(session.session_id),
        company_context=("UrbanCycle is a bicycle manufacturer."),
        learner_role=("Production Planner"),
        business_problem=("Orders are delayed."),
        business_goals=[("Improve delivery reliability.")],
    )

    await session_context_service.save(
        user_id=user.user_id,
        session_id=session.session_id,
        context=context,
        expected_version=(context.version),
    )

    await service.apply(
        user_id=user.user_id,
        session_id=session.session_id,
        agent_name="process_coach",
        proposal=proposal(
            operation(
                field=(SessionStateField.ACTIVE_BLOCKER),
                action=(SessionStateAction.SET),
                value=("Supplier selection is unclear."),
            )
        ),
    )

    updated = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert updated.scenario_setup_status == (ScenarioSetupStatus.REVIEWING)

    assert updated.scenario_draft is not None

    assert updated.scenario_draft.company_context == ("UrbanCycle is a bicycle manufacturer.")

    assert updated.active_blocker == ("Supplier selection is unclear.")


@pytest.mark.asyncio
async def test_one_user_cannot_update_another_users_session() -> None:
    (
        repositories,
        conversation_service,
        _,
        service,
    ) = create_services()

    user_a, session_a = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
        student_id="K11111",
    )

    user_b, _ = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
        student_id="K22222",
    )

    with pytest.raises(RepositoryNotFoundError):
        await service.apply(
            user_id=user_b.user_id,
            session_id=(session_a.session_id),
            agent_name=("process_coach"),
            proposal=proposal(
                operation(
                    field=(SessionStateField.ACTIVE_BLOCKER),
                    action=(SessionStateAction.SET),
                    value="Illegal update.",
                )
            ),
        )
