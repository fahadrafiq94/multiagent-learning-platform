from __future__ import annotations


class RepositoryError(RuntimeError):
    """Base exception for repository operations."""


class RepositoryNotFoundError(RepositoryError):
    """Raised when an expected persistent entity does not exist."""

    def __init__(
        self,
        *,
        entity_name: str,
        entity_id: str,
    ) -> None:
        self.entity_name = entity_name
        self.entity_id = entity_id

        super().__init__(f"{entity_name} not found: {entity_id}")


class DuplicateRecordError(RepositoryError):
    """
    Raised when a create operation violates a uniqueness rule.

    Examples:
    - duplicate student_id
    - duplicate external identity
    - duplicate session_id
    """

    def __init__(
        self,
        *,
        entity_name: str,
        key: str,
    ) -> None:
        self.entity_name = entity_name
        self.key = key

        super().__init__(f"Duplicate {entity_name}: {key}")


class VersionConflictError(RepositoryError):
    """
    Raised when optimistic concurrency detects a stale write.

    The caller attempted to save a version that is no longer the
    current persisted version.
    """

    def __init__(
        self,
        *,
        entity_name: str,
        entity_id: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.entity_name = entity_name
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version

        super().__init__(
            f"Version conflict for {entity_name} "
            f"{entity_id}: expected version "
            f"{expected_version}, actual version "
            f"{actual_version}"
        )
