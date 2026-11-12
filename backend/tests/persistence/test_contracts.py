from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.context import ConversationMessage, ConversationSession, SessionContext, UserContext
from app.identity import ExternalIdentity, IdentityProvider, UserIdentity
from app.repositories.bundle import RepositoryBundle
from app.repositories.exceptions import (
    DuplicateRecordError,
    RepositoryNotFoundError,
    VersionConflictError,
)

pytestmark = pytest.mark.asyncio


async def seed(repositories):
    user = await repositories.identities.create(UserIdentity(student_id="student-" + uuid4().hex))
    other = await repositories.identities.create(UserIdentity(student_id="student-" + uuid4().hex))
    session = await repositories.sessions.create(ConversationSession(user_id=user.user_id))
    return user, other, session


async def test_bundle_normalization_and_external_identity_uniqueness(repositories):
    assert isinstance(repositories, RepositoryBundle)
    link = ExternalIdentity(
        provider=IdentityProvider.MOODLE, provider_instance=" campus ", external_user_id=" 7 "
    )
    original = UserIdentity(student_id=" Learner ", external_identities=[link])
    await repositories.identities.create(original)
    assert (
        await repositories.identities.get_by_student_id(" Learner ")
    ).user_id == original.user_id
    assert (
        await repositories.identities.get_by_external_identity(
            provider=IdentityProvider.MOODLE, provider_instance=" campus ", external_user_id="7"
        )
    ).user_id == original.user_id
    for duplicate in [
        original,
        UserIdentity(student_id="Learner"),
        UserIdentity(student_id="Other", external_identities=[link]),
    ]:
        with pytest.raises(DuplicateRecordError):
            await repositories.identities.create(duplicate)
    assert await repositories.identities.get_by_student_id("Other") is None
    changed = original.model_copy(update={"student_id": "Renamed", "external_identities": []})
    await repositories.identities.update(changed)
    assert await repositories.identities.get_by_student_id("Learner") is None
    assert (
        await repositories.identities.get_by_external_identity(
            provider=IdentityProvider.MOODLE, provider_instance="campus", external_user_id="7"
        )
        is None
    )


async def test_rejected_identity_update_preserves_old_indexes(repositories):
    link = ExternalIdentity(provider="local", provider_instance="campus", external_user_id="7")
    first = await repositories.identities.create(
        UserIdentity(student_id="A", external_identities=[link])
    )
    second = await repositories.identities.create(UserIdentity(student_id="B"))
    with pytest.raises(DuplicateRecordError):
        await repositories.identities.update(
            second.model_copy(update={"external_identities": [link]})
        )
    assert (await repositories.identities.get_by_student_id("B")).external_identities == []
    assert (
        await repositories.identities.get_by_external_identity(
            provider=IdentityProvider.LOCAL, provider_instance="campus", external_user_id="7"
        )
    ).user_id == first.user_id


async def test_sessions_and_contexts_are_owner_scoped(repositories):
    user, other, session = await seed(repositories)
    assert (
        await repositories.sessions.get(user_id=other.user_id, session_id=session.session_id)
        is None
    )
    assert await repositories.sessions.list_for_user(other.user_id) == []
    with pytest.raises(RepositoryNotFoundError):
        await repositories.sessions.update(session.model_copy(update={"user_id": other.user_id}))
    context = SessionContext(
        user_id=user.user_id, session_id=session.session_id, current_step_id="AP40-04"
    )
    await repositories.session_contexts.create(context)
    for operation in [
        repositories.session_contexts.get(user_id=other.user_id, session_id=session.session_id),
        repositories.session_contexts.create(context.model_copy(update={"user_id": other.user_id})),
        repositories.session_contexts.save(
            context.model_copy(update={"user_id": other.user_id}), expected_version=1
        ),
        repositories.conversations.list_for_session(
            user_id=other.user_id, session_id=session.session_id
        ),
        repositories.conversations.count(user_id=other.user_id, session_id=session.session_id),
    ]:
        with pytest.raises(RepositoryNotFoundError):
            await operation
    assert (
        await repositories.session_contexts.get(user_id=user.user_id, session_id=session.session_id)
    ).current_step_id == "AP40-04"


@pytest.mark.parametrize("kind", ["user", "session"])
async def test_context_versions_and_returned_mutation_are_isolated(repositories, kind):
    user, _, session = await seed(repositories)
    repository = repositories.user_contexts if kind == "user" else repositories.session_contexts
    context = (
        UserContext(user_id=user.user_id)
        if kind == "user"
        else SessionContext(user_id=user.user_id, session_id=session.session_id)
    )
    original = await repository.create(context)
    with pytest.raises(DuplicateRecordError):
        await repository.create(context)
    first, second = original.model_copy(deep=True), original.model_copy(deep=True)
    if kind == "user":
        first.learning_profile.support_preferences.append("Short explanations")
    else:
        first.unresolved_questions.append("Why this dependency?")
    saved = await repository.save(first, expected_version=1)
    assert saved.version == 2
    with pytest.raises(VersionConflictError) as error:
        await repository.save(second, expected_version=1)
    assert error.value.actual_version == 2
    saved.version = 99
    current = (
        await repository.get(user.user_id)
        if kind == "user"
        else await repository.get(user_id=user.user_id, session_id=session.session_id)
    )
    assert current.version == 2


@pytest.mark.parametrize("kind", ["user", "session"])
async def test_concurrent_context_writers_have_one_winner(repositories, kind):
    user, _, session = await seed(repositories)
    repository = repositories.user_contexts if kind == "user" else repositories.session_contexts
    context = (
        UserContext(user_id=user.user_id)
        if kind == "user"
        else SessionContext(user_id=user.user_id, session_id=session.session_id)
    )
    await repository.create(context)
    results = await asyncio.gather(
        *(repository.save(context.model_copy(deep=True), expected_version=1) for _ in range(2)),
        return_exceptions=True,
    )
    assert sum(isinstance(r, VersionConflictError) for r in results) == 1
    assert sum(getattr(r, "version", None) == 2 for r in results) == 1


async def test_message_order_pagination_replay_and_private_history(repositories):
    user, other, session = await seed(repositories)
    now = datetime.now(UTC)
    messages = [
        ConversationMessage(
            user_id=user.user_id,
            session_id=session.session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            created_at=now,
        )
        for i in range(6)
    ]
    for message in messages:
        assert await repositories.conversations.append(message) == message
    assert (
        await repositories.conversations.count(user_id=user.user_id, session_id=session.session_id)
        == 6
    )
    assert (
        await repositories.conversations.list_for_session(
            user_id=user.user_id, session_id=session.session_id, limit=2, offset=2
        )
        == messages[2:4]
    )
    assert (
        await repositories.conversations.list_recent(
            user_id=user.user_id, session_id=session.session_id, limit=2
        )
        == messages[-2:]
    )
    with pytest.raises(DuplicateRecordError):
        await repositories.conversations.append(messages[0])
    with pytest.raises(RepositoryNotFoundError):
        await repositories.conversations.append(
            messages[0].model_copy(update={"user_id": other.user_id})
        )
    for limit, offset in [(0, 0), (1, -1)]:
        with pytest.raises(ValueError):
            await repositories.conversations.list_for_session(
                user_id=user.user_id, session_id=session.session_id, limit=limit, offset=offset
            )


async def test_concurrent_duplicate_identity_has_one_winner(repositories):
    candidates = [UserIdentity(student_id="same-student") for _ in range(4)]
    results = await asyncio.gather(
        *(repositories.identities.create(u) for u in candidates), return_exceptions=True
    )
    assert sum(isinstance(r, UserIdentity) for r in results) == 1
    assert sum(isinstance(r, DuplicateRecordError) for r in results) == 3
