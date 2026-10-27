from __future__ import annotations

from uuid import UUID

from app.context import (
    ConversationMessage,
    ConversationRole,
    ConversationSession,
    SessionContext,
)
from app.logging.logger import (
    logger,
)
from app.repositories import (
    ConversationRepository,
    ConversationSessionRepository,
    RepositoryError,
    RepositoryNotFoundError,
    SessionContextRepository,
)


class ConversationService:
    """
    Application service for chatbot conversations and sessions.

    This service coordinates:

    - conversation session creation
    - session retrieval/listing
    - initial SessionContext creation
    - user/assistant message persistence
    - conversation history retrieval
    - session activity timestamps

    It intentionally does not contain:

    - authentication
    - LLM generation
    - agent routing
    - context building
    - personalization logic

    Those responsibilities belong to other services.
    """

    def __init__(
        self,
        *,
        session_repository: ConversationSessionRepository,
        session_context_repository: SessionContextRepository,
        conversation_repository: ConversationRepository,
    ) -> None:
        self._session_repository = session_repository

        self._session_context_repository = session_context_repository

        self._conversation_repository = conversation_repository

    async def create_session(
        self,
        *,
        user_id: UUID,
        title: str | None = None,
    ) -> ConversationSession:
        """
        Create one new conversation for a user.

        Every new conversation receives both:

        - ConversationSession
        - initial SessionContext

        The user must already exist in the identity repository.
        """

        session = ConversationSession(
            user_id=user_id,
            title=title,
        )

        created_session = await self._session_repository.create(session)

        session_context = SessionContext(
            user_id=user_id,
            session_id=(created_session.session_id),
        )

        await self._session_context_repository.create(session_context)

        return created_session

    async def list_sessions(
        self,
        *,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSession]:
        """
        Return the user's previous conversations.

        The repository returns sessions ordered with the most
        recently updated conversation first.
        """

        return await self._session_repository.list_for_user(
            user_id,
            limit=limit,
            offset=offset,
        )

    async def get_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> ConversationSession:
        """
        Return one conversation owned by the user.

        A session belonging to another user is treated as not
        found rather than exposing ownership information.
        """

        session = await self._session_repository.get(
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            raise RepositoryNotFoundError(
                entity_name=("ConversationSession"),
                entity_id=str(session_id),
            )

        return session

    async def get_session_context(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionContext:
        """
        Return the structured state belonging to one conversation.
        """

        await self.get_session(
            user_id=user_id,
            session_id=session_id,
        )

        context = await self._session_context_repository.get(
            user_id=user_id,
            session_id=session_id,
        )

        if context is None:
            raise RepositoryNotFoundError(
                entity_name="SessionContext",
                entity_id=str(session_id),
            )

        return context

    async def append_user_message(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        trace_id: UUID | None = None,
    ) -> ConversationMessage:
        """
        Persist one user message.

        Session ownership is checked before the message enters the
        conversation archive.
        """

        return await self._append_message(
            user_id=user_id,
            session_id=session_id,
            role=ConversationRole.USER,
            content=content,
            trace_id=trace_id,
        )

    async def append_assistant_message(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_name: str | None = None,
        route: str | None = None,
        trace_id: UUID | None = None,
    ) -> ConversationMessage:
        """
        Persist one assistant response.

        agent_name and route are optional because clarification or
        fallback responses may not be produced by a specialist.
        """

        return await self._append_message(
            user_id=user_id,
            session_id=session_id,
            role=ConversationRole.ASSISTANT,
            content=content,
            agent_name=agent_name,
            route=route,
            trace_id=trace_id,
        )

    async def get_history(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        """
        Return persisted conversation history.

        This is primarily intended for:

        - frontend chat restoration
        - research/debugging
        - history APIs

        It is not automatically the context sent to an LLM.
        """

        await self.get_session(
            user_id=user_id,
            session_id=session_id,
        )

        return await self._conversation_repository.list_for_session(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    async def get_recent_messages(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        """
        Return recent messages for future ContextService use.

        This is deliberately separate from get_history().
        """

        await self.get_session(
            user_id=user_id,
            session_id=session_id,
        )

        return await self._conversation_repository.list_recent(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

    async def count_messages(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> int:
        """Return the number of persisted messages."""

        await self.get_session(
            user_id=user_id,
            session_id=session_id,
        )

        return await self._conversation_repository.count(
            user_id=user_id,
            session_id=session_id,
        )

    async def _append_message(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        role: ConversationRole,
        content: str,
        agent_name: str | None = None,
        route: str | None = None,
        trace_id: UUID | None = None,
    ) -> ConversationMessage:
        """
        Shared message-persistence implementation.

        Explicitly verifies ownership at the service boundary even
        though the repository performs its own ownership check.
        """

        session = await self.get_session(
            user_id=user_id,
            session_id=session_id,
        )

        message = ConversationMessage(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            agent_name=agent_name,
            route=route,
            trace_id=trace_id,
        )

        stored_message = await self._conversation_repository.append(message)

        try:
            await self._touch_session(session)
        except RepositoryError as exc:
            # The activity touch is best-effort: the message is
            # already persisted, and re-raising would make a caller
            # retry duplicate the message.
            logger.warning(
                "Session_activity_touch_failed",
                session_id=str(session_id),
                message_id=str(stored_message.message_id),
                error=str(exc),
            )

        return stored_message

    async def _touch_session(
        self,
        session: ConversationSession,
    ) -> None:
        """
        Mark a conversation as recently active.

        The repository owns the actual updated_at timestamp.
        """

        await self._session_repository.update(session)
