from __future__ import annotations

from uuid import UUID

from app.context import (
    AgentContext,
    ContextMetadata,
    ConversationMessage,
    ScenarioCard,
    SessionSummary,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.user_context_service import (
    UserContextService,
)


class ContextService:
    """
    Assemble the base request-scoped context for FREDi.

    This service combines persistent user context, temporary session
    state, recent conversation history, and the current user message.

    Agent-specific filtering, token budgeting, summarization, and
    knowledge retrieval are intentionally handled elsewhere/later.
    """

    def __init__(
        self,
        *,
        user_context_service: UserContextService,
        session_context_service: SessionContextService,
        conversation_service: ConversationService,
    ) -> None:
        self._user_context_service = user_context_service
        self._session_context_service = session_context_service
        self._conversation_service = conversation_service

    async def build(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
        trace_id: UUID | None = None,
        current_message_id: UUID | None = None,
        recent_message_limit: int = 10,
    ) -> AgentContext:
        """
        Build the common AgentContext for one request.

        If current_message_id is supplied, that message is excluded
        from recent_messages so the current user message is not sent
        twice to the model.
        """

        if recent_message_limit < 0:
            raise ValueError("recent_message_limit must not be negative")

        user_context = await self._user_context_service.get_or_create(
            user_id=user_id,
        )

        session_context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        recent_messages = await self._load_recent_messages(
            user_id=user_id,
            session_id=session_id,
            current_message_id=(current_message_id),
            limit=recent_message_limit,
        )

        included_sections = self._build_included_sections(
            scenario_card=(user_context.scenario_card),
            recent_messages=(recent_messages),
            session_summary=(session_context.summary),
            adaptation_count=len(user_context.adaptations),
        )

        metadata = ContextMetadata(
            trace_id=trace_id,
            included_sections=(included_sections),
            recent_message_count=len(recent_messages),
            summary_included=(session_context.summary is not None),
            knowledge_chunk_count=0,
            adaptation_count=len(user_context.adaptations),
            estimated_input_tokens=0,
            truncation_applied=False,
        )

        return AgentContext(
            user_id=user_id,
            session_id=session_id,
            current_message=current_message,
            scenario_card=(user_context.scenario_card),
            learning_profile=(user_context.learning_profile),
            progress=(user_context.progress),
            session_context=(session_context),
            recent_messages=(recent_messages),
            session_summary=(session_context.summary),
            guidance_level=(user_context.hint_state.guidance_level),
            user_adaptations=list(user_context.adaptations),
            knowledge_chunks=[],
            metadata=metadata,
        )

    async def _load_recent_messages(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        current_message_id: UUID | None,
        limit: int,
    ) -> list[ConversationMessage]:
        """
        Load bounded recent conversation history.
        """

        if limit == 0:
            return []

        repository_limit = limit

        if current_message_id is not None:
            repository_limit += 1

        messages = await self._conversation_service.get_recent_messages(
            user_id=user_id,
            session_id=session_id,
            limit=repository_limit,
        )

        if current_message_id is not None:
            messages = [
                message for message in messages if (message.message_id != current_message_id)
            ]

        if len(messages) > limit:
            messages = messages[-limit:]

        return messages

    @staticmethod
    def _build_included_sections(
        *,
        scenario_card: ScenarioCard | None,
        recent_messages: list[ConversationMessage],
        session_summary: SessionSummary | None,
        adaptation_count: int,
    ) -> list[str]:
        """
        Describe the context sections included in this request.

        This metadata is intended for future observability.
        """

        sections = [
            "learning_profile",
            "progress",
            "guidance_level",
            "session_context",
        ]

        if scenario_card is not None:
            sections.append("scenario_card")

        if recent_messages:
            sections.append("recent_messages")

        if session_summary is not None:
            sections.append("session_summary")

        if adaptation_count > 0:
            sections.append("user_adaptations")

        return sections
