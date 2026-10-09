from __future__ import annotations

from uuid import uuid4

import pytest

from app.context import (
    ConversationRole,
    ConversationSession,
    ScenarioCard,
    ScenarioDraft,
    ScenarioSetupStatus,
    SessionContextUpdate,
)
from app.domain import GuidanceLevel
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    RepositoryNotFoundError,
    create_in_memory_repositories,
)
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
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
    ContextService,
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

    context_service = ContextService(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    return (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        context_service,
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


async def create_user_and_session(
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
async def test_build_creates_common_user_context_baseline() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=("What should I do next?"),
    )

    assert context.user_id == user.user_id

    assert context.session_id == session.session_id

    assert context.current_message == "What should I do next?"

    assert context.scenario_card is None

    assert context.learning_profile is not None

    assert context.progress is not None

    assert context.guidance_level == GuidanceLevel.MINIMAL

    assert context.user_adaptations == []

    assert context.knowledge_chunks == []


@pytest.mark.asyncio
async def test_build_initializes_persistent_user_context() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    before = await user_context_service.get(
        user_id=user.user_id,
    )

    assert before is None

    await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Hello",
    )

    after = await user_context_service.get(
        user_id=user.user_id,
    )

    assert after is not None

    assert after.user_id == user.user_id


@pytest.mark.asyncio
async def test_build_includes_current_session_state() -> None:
    (
        repositories,
        conversation_service,
        _,
        session_context_service,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_objective=("Understand supplier selection."),
            current_step_id="8",
            current_process_phase=("supplier_selection"),
            active_blocker=("Supplier criteria are unclear."),
            last_selected_agent=("process_coach"),
        ),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=("Why do I need these criteria?"),
    )

    assert context.session_context is not None

    assert context.session_context.current_objective == "Understand supplier selection."

    assert context.session_context.current_step_id == "8"

    assert context.session_context.current_process_phase == "supplier_selection"

    assert context.session_context.active_blocker == "Supplier criteria are unclear."

    assert context.session_context.last_selected_agent == "process_coach"


@pytest.mark.asyncio
async def test_build_includes_recent_messages() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Message 1",
    )

    await conversation_service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Message 2",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Message 3",
    )

    assert len(context.recent_messages) == 2

    assert context.recent_messages[0].content == "Message 1"

    assert context.recent_messages[1].content == "Message 2"


@pytest.mark.asyncio
async def test_recent_message_history_is_bounded() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(8):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=f"Message {index}",
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
        recent_message_limit=3,
    )

    assert len(context.recent_messages) == 3

    assert [message.content for message in (context.recent_messages)] == [
        "Message 5",
        "Message 6",
        "Message 7",
    ]


@pytest.mark.asyncio
async def test_zero_recent_message_limit_returns_no_history() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Previous message",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current message",
        recent_message_limit=0,
    )

    assert context.recent_messages == []

    assert context.metadata.recent_message_count == 0


@pytest.mark.asyncio
async def test_negative_recent_message_limit_is_rejected() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    with pytest.raises(
        ValueError,
        match=("recent_message_limit must not be negative"),
    ):
        await context_service.build(
            user_id=user.user_id,
            session_id=session.session_id,
            current_message="Hello",
            recent_message_limit=-1,
        )


@pytest.mark.asyncio
async def test_current_persisted_message_can_be_excluded_from_history() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Earlier message",
    )

    current = await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content=("Why do we select a supplier?"),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=current.content,
        current_message_id=(current.message_id),
    )

    assert context.current_message == "Why do we select a supplier?"

    assert len(context.recent_messages) == 1

    assert context.recent_messages[0].content == "Earlier message"

    assert all(message.message_id != current.message_id for message in (context.recent_messages))


@pytest.mark.asyncio
async def test_excluding_current_message_preserves_previous_history_limit() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(4):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=f"Previous {index}",
        )

    current = await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Current",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
        current_message_id=(current.message_id),
        recent_message_limit=3,
    )

    assert [message.content for message in (context.recent_messages)] == [
        "Previous 1",
        "Previous 2",
        "Previous 3",
    ]


@pytest.mark.asyncio
async def test_recent_messages_preserve_conversation_roles() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="User message",
    )

    await conversation_service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Assistant response",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Next message",
    )

    assert context.recent_messages[0].role == ConversationRole.USER

    assert context.recent_messages[1].role == ConversationRole.ASSISTANT


@pytest.mark.asyncio
async def test_unconfirmed_scenario_draft_stays_in_session_context_only() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await user_context_service.get_or_create(
        user_id=user.user_id,
    )

    session_context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    session_context.scenario_setup_status = ScenarioSetupStatus.COLLECTING

    session_context.scenario_draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=session.session_id,
        company_context=("UrbanCycle is a bicycle manufacturer."),
    )

    await session_context_service.save(
        user_id=user.user_id,
        session_id=session.session_id,
        context=session_context,
        expected_version=(session_context.version),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=("I work as a production planner."),
    )

    assert context.scenario_card is None

    assert context.session_context is not None

    assert context.session_context.scenario_setup_status == ScenarioSetupStatus.COLLECTING

    assert context.session_context.scenario_draft is not None

    persistent_user_context = await user_context_service.get_required(
        user_id=user.user_id,
    )

    assert persistent_user_context.scenario_card is None


@pytest.mark.asyncio
async def test_confirmed_scenario_card_is_included() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    user_context = await user_context_service.get_or_create(
        user_id=user.user_id,
    )

    user_context.scenario_card = ScenarioCard(
        user_id=user.user_id,
        company_context=("UrbanCycle is a bicycle manufacturer."),
        learner_role="Production Planner",
        business_problem=("Customer orders are delayed."),
        business_goals=[
            "Improve delivery reliability.",
        ],
    )

    await user_context_service.save(
        user_id=user.user_id,
        context=user_context,
        expected_version=(user_context.version),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="What should I do?",
    )

    assert context.scenario_card is not None

    assert context.scenario_card.learner_role == "Production Planner"

    assert "scenario_card" in context.metadata.included_sections


@pytest.mark.asyncio
async def test_context_metadata_describes_loaded_context() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    trace_id = uuid4()

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Previous message",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
        trace_id=trace_id,
    )

    assert context.metadata.trace_id == trace_id

    assert context.metadata.recent_message_count == 1

    assert context.metadata.summary_included is False

    assert context.metadata.knowledge_chunk_count == 0

    assert context.metadata.adaptation_count == 0

    assert context.metadata.estimated_input_tokens == 0

    assert context.metadata.truncation_applied is False

    assert "session_context" in context.metadata.included_sections

    assert "recent_messages" in context.metadata.included_sections


@pytest.mark.asyncio
async def test_context_service_rejects_cross_user_session_access() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user_a, session_a = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    with pytest.raises(RepositoryNotFoundError):
        await context_service.build(
            user_id=user_b.user_id,
            session_id=session_a.session_id,
            current_message=("This must not access User A."),
        )


@pytest.mark.asyncio
async def test_context_service_keeps_sessions_isolated() -> None:
    (
        repositories,
        conversation_service,
        _,
        session_context_service,
        context_service,
    ) = create_services()

    user, session_a = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    session_b = await conversation_service.create_session(
        user_id=user.user_id,
    )

    await session_context_service.set_current_step(
        user_id=user.user_id,
        session_id=session_a.session_id,
        step_id="8",
    )

    await session_context_service.set_current_step(
        user_id=user.user_id,
        session_id=session_b.session_id,
        step_id="17",
    )

    context_a = await context_service.build(
        user_id=user.user_id,
        session_id=session_a.session_id,
        current_message="Session A",
    )

    context_b = await context_service.build(
        user_id=user.user_id,
        session_id=session_b.session_id,
        current_message="Session B",
    )

    assert context_a.session_context is not None
    assert context_b.session_context is not None

    assert context_a.session_context.current_step_id == "8"

    assert context_b.session_context.current_step_id == "17"
