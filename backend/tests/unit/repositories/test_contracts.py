from __future__ import annotations

from app.repositories import (
    ConversationRepository,
    ConversationSessionRepository,
    DuplicateRecordError,
    RepositoryNotFoundError,
    SessionContextRepository,
    UserContextRepository,
    UserIdentityRepository,
    VersionConflictError,
)


def test_repository_protocols_are_runtime_checkable() -> None:
    incomplete_repository = object()

    assert not isinstance(
        incomplete_repository,
        UserIdentityRepository,
    )

    assert not isinstance(
        incomplete_repository,
        UserContextRepository,
    )

    assert not isinstance(
        incomplete_repository,
        ConversationSessionRepository,
    )

    assert not isinstance(
        incomplete_repository,
        SessionContextRepository,
    )

    assert not isinstance(
        incomplete_repository,
        ConversationRepository,
    )


def test_repository_not_found_error_contains_entity_details() -> None:
    error = RepositoryNotFoundError(
        entity_name="SessionContext",
        entity_id="session-123",
    )

    assert error.entity_name == "SessionContext"
    assert error.entity_id == "session-123"

    assert str(error) == "SessionContext not found: session-123"


def test_duplicate_record_error_contains_unique_key() -> None:
    error = DuplicateRecordError(
        entity_name="UserIdentity",
        key="student_id=K12345",
    )

    assert error.entity_name == "UserIdentity"
    assert error.key == "student_id=K12345"

    assert str(error) == ("Duplicate UserIdentity: student_id=K12345")


def test_version_conflict_error_contains_versions() -> None:
    error = VersionConflictError(
        entity_name="SessionContext",
        entity_id="session-123",
        expected_version=4,
        actual_version=5,
    )

    assert error.entity_name == "SessionContext"
    assert error.entity_id == "session-123"
    assert error.expected_version == 4
    assert error.actual_version == 5

    assert str(error) == (
        "Version conflict for SessionContext session-123: expected version 4, actual version 5"
    )
