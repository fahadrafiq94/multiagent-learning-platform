from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from app.context import (
    ConversationMessage,
    ConversationRole,
    ConversationSession,
    ProgressStatus,
    ScenarioCard,
    SessionContext,
    StepProgress,
    UserContext,
)
from app.identity import (
    ExternalIdentity,
    IdentityProvider,
    UserIdentity,
)
from app.repositories import (
    ConversationRepository,
    ConversationSessionRepository,
    DuplicateRecordError,
    RepositoryNotFoundError,
    SessionContextRepository,
    UserContextRepository,
    UserIdentityRepository,
    VersionConflictError,
    create_in_memory_repositories,
)


@pytest.mark.asyncio
async def test_repository_bundle_implements_contracts() -> None:
    repositories = create_in_memory_repositories()

    assert isinstance(
        repositories.identities,
        UserIdentityRepository,
    )

    assert isinstance(
        repositories.user_contexts,
        UserContextRepository,
    )

    assert isinstance(
        repositories.sessions,
        ConversationSessionRepository,
    )

    assert isinstance(
        repositories.session_contexts,
        SessionContextRepository,
    )

    assert isinstance(
        repositories.conversations,
        ConversationRepository,
    )


@pytest.mark.asyncio
async def test_identity_can_be_created_and_resolved_by_user_id() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    created = await repositories.identities.create(identity)

    resolved = await repositories.identities.get_by_user_id(identity.user_id)

    assert created.user_id == identity.user_id

    assert resolved is not None

    assert resolved.student_id == "K12345"


@pytest.mark.asyncio
async def test_identity_can_be_resolved_by_student_id() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    resolved = await repositories.identities.get_by_student_id("  K12345  ")

    assert resolved is not None

    assert resolved.user_id == identity.user_id


@pytest.mark.asyncio
async def test_identity_can_be_resolved_from_moodle_mapping() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
        external_identities=[
            ExternalIdentity(
                provider=(IdentityProvider.MOODLE),
                provider_instance=("university-moodle"),
                external_user_id="837",
                external_student_id=("K12345"),
            )
        ],
    )

    await repositories.identities.create(identity)

    resolved = await repositories.identities.get_by_external_identity(
        provider=(IdentityProvider.MOODLE),
        provider_instance=("university-moodle"),
        external_user_id="837",
    )

    assert resolved is not None

    assert resolved.user_id == identity.user_id


@pytest.mark.asyncio
async def test_duplicate_student_id_is_rejected() -> None:
    repositories = create_in_memory_repositories()

    first = UserIdentity(
        student_id="K12345",
    )

    second = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(first)

    with pytest.raises(DuplicateRecordError):
        await repositories.identities.create(second)


@pytest.mark.asyncio
async def test_external_identity_cannot_belong_to_two_users() -> None:
    repositories = create_in_memory_repositories()

    external_identity = ExternalIdentity(
        provider=IdentityProvider.MOODLE,
        provider_instance=("university-moodle"),
        external_user_id="837",
    )

    first = UserIdentity(
        student_id="K12345",
        external_identities=[external_identity],
    )

    second = UserIdentity(
        student_id="K67890",
        external_identities=[external_identity.model_copy(deep=True)],
    )

    await repositories.identities.create(first)

    with pytest.raises(DuplicateRecordError):
        await repositories.identities.create(second)


@pytest.mark.asyncio
async def test_identity_update_can_add_external_mapping() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    created = await repositories.identities.create(identity)

    created.external_identities.append(
        ExternalIdentity(
            provider=(IdentityProvider.MAHARA),
            provider_instance=("university-mahara"),
            external_user_id="1198",
        )
    )

    updated = await repositories.identities.update(created)

    assert len(updated.external_identities) == 1

    resolved = await repositories.identities.get_by_external_identity(
        provider=(IdentityProvider.MAHARA),
        provider_instance=("university-mahara"),
        external_user_id="1198",
    )

    assert resolved is not None

    assert resolved.user_id == identity.user_id


@pytest.mark.asyncio
async def test_repository_reads_are_deep_copies() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    first_read = await repositories.identities.get_by_user_id(identity.user_id)

    assert first_read is not None

    first_read.external_identities.append(
        ExternalIdentity(
            provider=(IdentityProvider.MOODLE),
            provider_instance="test",
            external_user_id="999",
        )
    )

    second_read = await repositories.identities.get_by_user_id(identity.user_id)

    assert second_read is not None

    assert second_read.external_identities == []


@pytest.mark.asyncio
async def test_user_context_requires_existing_identity() -> None:
    repositories = create_in_memory_repositories()

    context = UserContext(
        user_id=uuid4(),
    )

    with pytest.raises(
        RepositoryNotFoundError,
        match="UserIdentity not found",
    ):
        await repositories.user_contexts.create(context)


@pytest.mark.asyncio
async def test_user_context_can_be_created_for_identity() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    context = UserContext(
        user_id=identity.user_id,
    )

    created = await repositories.user_contexts.create(context)

    assert created.user_id == identity.user_id

    assert created.version == 1


@pytest.mark.asyncio
async def test_user_context_deep_copy_prevents_untracked_mutation() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    await repositories.user_contexts.create(
        UserContext(
            user_id=identity.user_id,
        )
    )

    first_read = await repositories.user_contexts.get(identity.user_id)

    assert first_read is not None

    first_read.progress.steps.append(
        StepProgress(
            step_id="8",
            status=(ProgressStatus.COMPLETED),
        )
    )

    second_read = await repositories.user_contexts.get(identity.user_id)

    assert second_read is not None

    assert second_read.progress.steps == []


@pytest.mark.asyncio
async def test_user_context_save_increments_version() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    initial = UserContext(
        user_id=identity.user_id,
    )

    created = await repositories.user_contexts.create(initial)

    created.scenario_card = ScenarioCard(
        user_id=identity.user_id,
        company_context=("UrbanCycle is a bicycle manufacturer."),
    )

    saved = await repositories.user_contexts.save(
        created,
        expected_version=1,
    )

    assert saved.version == 2

    assert saved.scenario_card is not None


@pytest.mark.asyncio
async def test_stale_user_context_save_is_rejected() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    initial = await repositories.user_contexts.create(
        UserContext(
            user_id=identity.user_id,
        )
    )

    request_a = initial.model_copy(deep=True)

    request_b = initial.model_copy(deep=True)

    await repositories.user_contexts.save(
        request_a,
        expected_version=1,
    )

    with pytest.raises(VersionConflictError):
        await repositories.user_contexts.save(
            request_b,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_one_user_can_have_multiple_sessions() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    first = ConversationSession(
        user_id=identity.user_id,
        title="Procurement Discussion",
    )

    second = ConversationSession(
        user_id=identity.user_id,
        title="AP+ Navigation",
    )

    await repositories.sessions.create(first)

    await repositories.sessions.create(second)

    sessions = await repositories.sessions.list_for_user(identity.user_id)

    assert len(sessions) == 2

    assert {session.session_id for session in sessions} == {
        first.session_id,
        second.session_id,
    }


@pytest.mark.asyncio
async def test_session_is_not_visible_to_another_user() -> None:
    repositories = create_in_memory_repositories()

    user_a = UserIdentity(
        student_id="K12345",
    )

    user_b = UserIdentity(
        student_id="K67890",
    )

    await repositories.identities.create(user_a)

    await repositories.identities.create(user_b)

    session = ConversationSession(
        user_id=user_a.user_id,
        title="Private conversation",
    )

    await repositories.sessions.create(session)

    result = await repositories.sessions.get(
        user_id=user_b.user_id,
        session_id=session.session_id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_duplicate_session_id_is_rejected_globally() -> None:
    repositories = create_in_memory_repositories()

    user_a = UserIdentity(
        student_id="K12345",
    )

    user_b = UserIdentity(
        student_id="K67890",
    )

    await repositories.identities.create(user_a)

    await repositories.identities.create(user_b)

    session_id = uuid4()

    await repositories.sessions.create(
        ConversationSession(
            session_id=session_id,
            user_id=user_a.user_id,
        )
    )

    with pytest.raises(DuplicateRecordError):
        await repositories.sessions.create(
            ConversationSession(
                session_id=session_id,
                user_id=user_b.user_id,
            )
        )


@pytest.mark.asyncio
async def test_sessions_are_listed_most_recently_updated_first() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    older = ConversationSession(
        user_id=identity.user_id,
        title="Older",
    )

    newer = ConversationSession(
        user_id=identity.user_id,
        title="Newer",
        updated_at=(older.updated_at + timedelta(seconds=10)),
    )

    await repositories.sessions.create(older)

    await repositories.sessions.create(newer)

    sessions = await repositories.sessions.list_for_user(identity.user_id)

    assert sessions[0].session_id == newer.session_id


@pytest.mark.asyncio
async def test_session_context_requires_valid_session() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    with pytest.raises(RepositoryNotFoundError):
        await repositories.session_contexts.create(
            SessionContext(
                user_id=identity.user_id,
                session_id=uuid4(),
            )
        )


@pytest.mark.asyncio
async def test_session_context_save_increments_version() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session)

    context = await repositories.session_contexts.create(
        SessionContext(
            user_id=identity.user_id,
            session_id=(session.session_id),
            current_step_id="8",
        )
    )

    context.current_step_id = "9"

    saved = await repositories.session_contexts.save(
        context,
        expected_version=1,
    )

    assert saved.version == 2

    assert saved.current_step_id == "9"


@pytest.mark.asyncio
async def test_stale_session_context_write_is_rejected() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session)

    initial = await repositories.session_contexts.create(
        SessionContext(
            user_id=identity.user_id,
            session_id=(session.session_id),
        )
    )

    request_a = initial.model_copy(deep=True)

    request_b = initial.model_copy(deep=True)

    await repositories.session_contexts.save(
        request_a,
        expected_version=1,
    )

    with pytest.raises(VersionConflictError):
        await repositories.session_contexts.save(
            request_b,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_messages_can_be_appended_and_read_chronologically() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session)

    first = ConversationMessage(
        user_id=identity.user_id,
        session_id=session.session_id,
        role=ConversationRole.USER,
        content="Why do we need this action?",
    )

    second = ConversationMessage(
        user_id=identity.user_id,
        session_id=session.session_id,
        role=ConversationRole.ASSISTANT,
        content=("What business event happened immediately before it?"),
        created_at=(first.created_at + timedelta(seconds=1)),
    )

    await repositories.conversations.append(second)

    await repositories.conversations.append(first)

    messages = await repositories.conversations.list_for_session(
        user_id=identity.user_id,
        session_id=session.session_id,
    )

    assert len(messages) == 2

    assert messages[0].message_id == first.message_id

    assert messages[1].message_id == second.message_id


@pytest.mark.asyncio
async def test_recent_messages_returns_latest_messages_in_chronological_order() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session)

    messages: list[ConversationMessage] = []

    for index in range(5):
        message = ConversationMessage(
            user_id=identity.user_id,
            session_id=session.session_id,
            role=ConversationRole.USER,
            content=f"Message {index}",
        )

        messages.append(message)

        await repositories.conversations.append(message)

    recent = await repositories.conversations.list_recent(
        user_id=identity.user_id,
        session_id=session.session_id,
        limit=2,
    )

    assert len(recent) == 2

    assert recent[0].content == "Message 3"

    assert recent[1].content == "Message 4"


@pytest.mark.asyncio
async def test_user_cannot_read_another_users_messages() -> None:
    repositories = create_in_memory_repositories()

    user_a = UserIdentity(
        student_id="K12345",
    )

    user_b = UserIdentity(
        student_id="K67890",
    )

    await repositories.identities.create(user_a)

    await repositories.identities.create(user_b)

    session = ConversationSession(
        user_id=user_a.user_id,
    )

    await repositories.sessions.create(session)

    await repositories.conversations.append(
        ConversationMessage(
            user_id=user_a.user_id,
            session_id=session.session_id,
            role=ConversationRole.USER,
            content=("Private user A message."),
        )
    )

    with pytest.raises(RepositoryNotFoundError):
        await repositories.conversations.list_for_session(
            user_id=user_b.user_id,
            session_id=session.session_id,
        )


@pytest.mark.asyncio
async def test_duplicate_message_id_is_rejected() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session)

    message = ConversationMessage(
        user_id=identity.user_id,
        session_id=session.session_id,
        role=ConversationRole.USER,
        content="First message.",
    )

    await repositories.conversations.append(message)

    duplicate = ConversationMessage(
        message_id=message.message_id,
        user_id=identity.user_id,
        session_id=session.session_id,
        role=ConversationRole.USER,
        content="Duplicate message.",
    )

    with pytest.raises(DuplicateRecordError):
        await repositories.conversations.append(duplicate)


@pytest.mark.asyncio
async def test_message_count_is_session_specific() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    session_a = ConversationSession(
        user_id=identity.user_id,
    )

    session_b = ConversationSession(
        user_id=identity.user_id,
    )

    await repositories.sessions.create(session_a)

    await repositories.sessions.create(session_b)

    await repositories.conversations.append(
        ConversationMessage(
            user_id=identity.user_id,
            session_id=session_a.session_id,
            role=ConversationRole.USER,
            content="Session A.",
        )
    )

    count_a = await repositories.conversations.count(
        user_id=identity.user_id,
        session_id=session_a.session_id,
    )

    count_b = await repositories.conversations.count(
        user_id=identity.user_id,
        session_id=session_b.session_id,
    )

    assert count_a == 1
    assert count_b == 0
