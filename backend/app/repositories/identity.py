from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from app.identity import (
    IdentityProvider,
    UserIdentity,
)


@runtime_checkable
class UserIdentityRepository(Protocol):
    """
    Persistence contract for FREDi user identities.

    user_id is the stable internal FREDi identifier.

    student_id is an institutional/login identifier and is used
    only through the identity layer.

    Moodle, Mahara, and future authentication providers are
    resolved through external identity mappings.
    """

    async def get_by_user_id(
        self,
        user_id: UUID,
    ) -> UserIdentity | None:
        """Return an identity by internal FREDi user_id."""

        ...

    async def get_by_student_id(
        self,
        student_id: str,
    ) -> UserIdentity | None:
        """
        Resolve an identity using its institutional student_id.

        This is currently useful for future local login.
        """

        ...

    async def get_by_external_identity(
        self,
        *,
        provider: IdentityProvider,
        provider_instance: str,
        external_user_id: str,
    ) -> UserIdentity | None:
        """
        Resolve a FREDi identity from an external platform.

        Example:
            Moodle installation + Moodle user ID
            -> FREDi UserIdentity
        """

        ...

    async def create(
        self,
        identity: UserIdentity,
    ) -> UserIdentity:
        """
        Persist a new identity.

        Implementations should reject duplicate unique identities
        using DuplicateRecordError.
        """

        ...

    async def update(
        self,
        identity: UserIdentity,
    ) -> UserIdentity:
        """
        Persist changes to an existing identity.

        Typical changes include:
        - status
        - newly linked external identity
        """

        ...
