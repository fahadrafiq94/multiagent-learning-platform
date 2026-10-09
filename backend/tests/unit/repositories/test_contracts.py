from __future__ import annotations

import pytest

from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    UserIdentityRepository,
    create_in_memory_repositories,
)


def test_factory_returns_typed_repository_bundle() -> None:
    repositories: InMemoryRepositories = create_in_memory_repositories()

    assert repositories.identities is not None
    assert repositories.user_contexts is not None
    assert repositories.sessions is not None
    assert repositories.session_contexts is not None
    assert repositories.conversations is not None


def test_identity_repository_satisfies_repository_protocol() -> None:
    repositories = create_in_memory_repositories()

    repository: UserIdentityRepository = repositories.identities

    assert repository is not None


@pytest.mark.asyncio
async def test_identity_create_returns_user_identity() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    created: UserIdentity = await repositories.identities.create(identity)

    assert isinstance(
        created,
        UserIdentity,
    )

    assert created.user_id == identity.user_id

    assert created.student_id == "K12345"


@pytest.mark.asyncio
async def test_identity_get_by_user_id_returns_typed_identity() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    created = await repositories.identities.create(identity)

    loaded: UserIdentity | None = await repositories.identities.get_by_user_id(
        user_id=created.user_id,
    )

    assert loaded is not None

    assert loaded.user_id == created.user_id


@pytest.mark.asyncio
async def test_identity_get_by_student_id_returns_typed_identity() -> None:
    repositories = create_in_memory_repositories()

    identity = UserIdentity(
        student_id="K12345",
    )

    await repositories.identities.create(identity)

    loaded: UserIdentity | None = await repositories.identities.get_by_student_id(
        student_id="K12345",
    )

    assert loaded is not None

    assert loaded.student_id == "K12345"
