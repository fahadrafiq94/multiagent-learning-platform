from __future__ import annotations

from uuid import UUID

from app.context import UserContext
from app.repositories import (
    DuplicateRecordError,
    RepositoryNotFoundError,
    UserContextRepository,
)


class UserContextService:
    """
    Application service for persistent cross-session user context.

    Responsibilities:

    - return an existing UserContext
    - create the common baseline for a new user
    - safely handle concurrent first access
    - persist validated UserContext changes
    - preserve optimistic concurrency semantics

    This service does not decide:

    - scenario facts
    - hint escalation
    - learning-profile updates
    - user-specific adaptations

    Those behaviors will be implemented by dedicated services and
    policies later in Sprint 4.
    """

    def __init__(
        self,
        *,
        repository: UserContextRepository,
    ) -> None:
        self._repository = repository

    async def get_or_create(
        self,
        *,
        user_id: UUID,
    ) -> UserContext:
        """
        Return the user's persistent context.

        If this is the user's first interaction, create the common
        baseline.

        The duplicate-create recovery handles concurrent first
        requests for the same user.
        """

        existing = await self._repository.get(user_id)

        if existing is not None:
            return existing

        baseline = self._create_baseline(user_id=user_id)

        try:
            return await self._repository.create(baseline)

        except DuplicateRecordError:
            # Another concurrent request may have created the
            # baseline between our get() and create().
            existing_after_race = await self._repository.get(user_id)

            if existing_after_race is None:
                raise

            return existing_after_race

    async def get(
        self,
        *,
        user_id: UUID,
    ) -> UserContext | None:
        """
        Return the current persistent context when it exists.

        Unlike get_or_create(), this method never creates context.
        """

        return await self._repository.get(user_id)

    async def get_required(
        self,
        *,
        user_id: UUID,
    ) -> UserContext:
        """
        Return existing UserContext or fail explicitly.

        Useful when a workflow requires that user context has
        already been initialized.
        """

        context = await self._repository.get(user_id)

        if context is None:
            raise RepositoryNotFoundError(
                entity_name="UserContext",
                entity_id=str(user_id),
            )

        return context

    async def save(
        self,
        *,
        user_id: UUID,
        context: UserContext,
        expected_version: int,
    ) -> UserContext:
        """
        Persist a changed UserContext.

        user_id is supplied separately from the context so the
        caller's execution identity can be checked against the
        object being saved.

        The repository performs optimistic concurrency checking.
        """

        if context.user_id != user_id:
            raise ValueError("context.user_id must match UserContextService user_id")

        return await self._repository.save(
            context,
            expected_version=expected_version,
        )

    @staticmethod
    def _create_baseline(
        *,
        user_id: UUID,
    ) -> UserContext:
        """
        Construct a fresh common baseline.

        Pydantic default factories ensure every user receives
        independent nested objects and lists.
        """

        return UserContext(
            user_id=user_id,
        )
