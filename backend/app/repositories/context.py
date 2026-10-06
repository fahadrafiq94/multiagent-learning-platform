from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from app.context import (
    ConversationMessage,
    ConversationSession,
    SessionContext,
    UserContext,
)


@runtime_checkable
class UserContextRepository(Protocol):
    """
    Persistence contract for cross-session user context.

    UserContext is always addressed through FREDi's internal
    immutable user_id.
    """

    async def get(
        self,
        user_id: UUID,
    ) -> UserContext | None:
        """Return the persistent context for one user."""

        ...

    async def create(
        self,
        context: UserContext,
    ) -> UserContext:
        """
        Persist the initial UserContext.

        Implementations should reject duplicate user contexts.
        """

        ...

    async def save(
        self,
        context: UserContext,
        *,
        expected_version: int,
    ) -> UserContext:
        """
        Persist a changed UserContext using optimistic concurrency.

        expected_version is the version originally read by the
        caller.

        If another request has already modified the same context,
        implementations should raise VersionConflictError.

        A successful save should increment the persisted version.
        """

        ...


@runtime_checkable
class ConversationSessionRepository(Protocol):
    """
    Persistence contract for user-visible chatbot sessions.

    These records allow the frontend to show and reopen previous
    conversations.
    """

    async def create(
        self,
        session: ConversationSession,
    ) -> ConversationSession:
        """Create a new conversation session."""

        ...

    async def get(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> ConversationSession | None:
        """
        Return one session only when it belongs to the supplied user.

        Implementations must never resolve a session using only
        session_id when handling user-owned data.
        """

        ...

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSession]:
        """
        Return conversation sessions belonging to one user.

        Results should be ordered with the most recently updated
        session first.
        """

        ...

    async def update(
        self,
        session: ConversationSession,
    ) -> ConversationSession:
        """
        Update conversation metadata.

        Examples:
        - title
        - status
        - updated_at
        """

        ...


@runtime_checkable
class SessionContextRepository(Protocol):
    """
    Persistence contract for state belonging to one conversation.

    SessionContext describes what is currently happening in the
    conversation and is separate from message history.
    """

    async def get(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionContext | None:
        """Return state for one specific user/session pair."""

        ...

    async def create(
        self,
        context: SessionContext,
    ) -> SessionContext:
        """Persist initial state for a conversation."""

        ...

    async def save(
        self,
        context: SessionContext,
        *,
        expected_version: int,
    ) -> SessionContext:
        """
        Save session state using optimistic concurrency.

        Implementations should raise VersionConflictError when the
        persisted version differs from expected_version.

        A successful save should increment version.
        """

        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """
    Persistence contract for complete chatbot message history.

    The full archive can be stored here while ContextService later
    chooses only a bounded subset for model context.
    """

    async def append(
        self,
        message: ConversationMessage,
    ) -> ConversationMessage:
        """
        Persist one new conversation message.

        Implementations must verify that the user/session ownership
        relationship is valid.
        """

        ...

    async def list_for_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        """
        Return persisted messages for one user/session.

        Results should be chronological:
        oldest -> newest.

        This method is primarily intended for frontend history.
        """

        ...

    async def list_recent(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        """
        Return the most recent messages for context construction.

        Although the newest messages are selected, the returned
        list should remain chronological:
        oldest -> newest.
        """

        ...

    async def count(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> int:
        """Return the number of messages in one conversation."""

        ...
