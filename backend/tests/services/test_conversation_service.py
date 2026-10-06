from __future__ import annotations

from typing import cast
from uuid import UUID, uuid4

import pytest

from app.context import (
    ConversationRole,
)
from app.identity import (
    UserIdentity,
)
from app.repositories import (
    RepositoryNotFoundError,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)


def create_service():
    repositories = create_in_memory_repositories()

    service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
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
async def test_create_session_creates_session_and_context() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
        title="Procurement Discussion",
    )

    assert isinstance(
        session.session_id,
        UUID,
    )

    assert session.user_id == user.user_id

    assert session.title == "Procurement Discussion"

    context = await repositories.session_contexts.get(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context is not None

    assert context.user_id == user.user_id

    assert context.session_id == session.session_id

    assert context.version == 1


@pytest.mark.asyncio
async def test_create_session_requires_existing_user() -> None:
    _, service = create_service()

    with pytest.raises(
        RepositoryNotFoundError,
        match="UserIdentity not found",
    ):
        await service.create_session(
            user_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_one_user_can_create_multiple_sessions() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    first = await service.create_session(
        user_id=user.user_id,
        title="Scenario",
    )

    second = await service.create_session(
        user_id=user.user_id,
        title="AP+ Help",
    )

    assert first.session_id != second.session_id

    sessions = await service.list_sessions(
        user_id=user.user_id,
    )

    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_users_only_see_their_own_sessions() -> None:
    repositories, service = create_service()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    await service.create_session(
        user_id=user_a.user_id,
        title="User A Chat",
    )

    await service.create_session(
        user_id=user_b.user_id,
        title="User B Chat",
    )

    sessions_a = await service.list_sessions(
        user_id=user_a.user_id,
    )

    sessions_b = await service.list_sessions(
        user_id=user_b.user_id,
    )

    assert len(sessions_a) == 1
    assert len(sessions_b) == 1

    assert sessions_a[0].user_id == user_a.user_id

    assert sessions_b[0].user_id == user_b.user_id


@pytest.mark.asyncio
async def test_user_cannot_get_another_users_session() -> None:
    repositories, service = create_service()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    session = await service.create_session(
        user_id=user_a.user_id,
    )

    with pytest.raises(
        RepositoryNotFoundError,
        match="ConversationSession not found",
    ):
        await service.get_session(
            user_id=user_b.user_id,
            session_id=session.session_id,
        )


@pytest.mark.asyncio
async def test_user_message_is_persisted() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    trace_id = uuid4()

    message = await service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content=("Why do we need to select a supplier?"),
        trace_id=trace_id,
    )

    assert message.role == ConversationRole.USER

    assert message.content == "Why do we need to select a supplier?"

    assert message.trace_id == trace_id

    history = await service.get_history(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert len(history) == 1

    assert history[0].message_id == message.message_id


@pytest.mark.asyncio
async def test_assistant_message_preserves_agent_metadata() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    trace_id = uuid4()

    message = await service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content=("What information would you need before choosing one?"),
        agent_name="process_coach",
        route="process_coach",
        trace_id=trace_id,
    )

    assert message.role == ConversationRole.ASSISTANT

    assert message.agent_name == "process_coach"

    assert message.route == "process_coach"
    assert message.trace_id == trace_id


@pytest.mark.asyncio
async def test_conversation_history_contains_both_sides() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    await service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Why is this step needed?",
    )

    await service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content=("What business event happened before this step?"),
        agent_name="process_coach",
        route="process_coach",
    )

    history = await service.get_history(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert len(history) == 2

    assert history[0].role == ConversationRole.USER

    assert history[1].role == ConversationRole.ASSISTANT


@pytest.mark.asyncio
async def test_messages_from_different_sessions_remain_isolated() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session_a = await service.create_session(
        user_id=user.user_id,
        title="Chat A",
    )

    session_b = await service.create_session(
        user_id=user.user_id,
        title="Chat B",
    )

    await service.append_user_message(
        user_id=user.user_id,
        session_id=session_a.session_id,
        content="Message for A.",
    )

    await service.append_user_message(
        user_id=user.user_id,
        session_id=session_b.session_id,
        content="Message for B.",
    )

    history_a = await service.get_history(
        user_id=user.user_id,
        session_id=session_a.session_id,
    )

    history_b = await service.get_history(
        user_id=user.user_id,
        session_id=session_b.session_id,
    )

    assert len(history_a) == 1
    assert len(history_b) == 1

    assert history_a[0].content == "Message for A."

    assert history_b[0].content == "Message for B."


@pytest.mark.asyncio
async def test_another_user_cannot_append_to_session() -> None:
    repositories, service = create_service()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    session = await service.create_session(
        user_id=user_a.user_id,
    )

    with pytest.raises(
        RepositoryNotFoundError,
        match="ConversationSession not found",
    ):
        await service.append_user_message(
            user_id=user_b.user_id,
            session_id=session.session_id,
            content=("This must not be persisted."),
        )

    count = await repositories.conversations.count(
        user_id=user_a.user_id,
        session_id=session.session_id,
    )

    assert count == 0


@pytest.mark.asyncio
async def test_get_recent_messages_returns_bounded_history() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    for index in range(5):
        await service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=f"Message {index}",
        )

    recent = await service.get_recent_messages(
        user_id=user.user_id,
        session_id=session.session_id,
        limit=2,
    )

    assert len(recent) == 2

    assert recent[0].content == "Message 3"

    assert recent[1].content == "Message 4"


@pytest.mark.asyncio
async def test_message_count_is_exposed_through_service() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    await service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="First",
    )

    await service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Second",
    )

    count = await service.count_messages(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert count == 2


@pytest.mark.asyncio
async def test_appending_message_updates_session_activity() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    original_updated_at = session.updated_at

    await service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="New activity.",
    )

    updated_session = await service.get_session(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert updated_session.updated_at >= original_updated_at


@pytest.mark.asyncio
async def test_get_session_context_returns_initial_state() -> None:
    repositories, service = create_service()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    session = await service.create_session(
        user_id=user.user_id,
    )

    context = await service.get_session_context(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.user_id == user.user_id

    assert context.session_id == session.session_id

    assert context.version == 1
    assert context.current_objective is None
    assert context.current_step_id is None
