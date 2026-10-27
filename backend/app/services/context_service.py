from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from app.context import (
    AgentContext,
    ContextMetadata,
    ConversationMessage,
    ScenarioCard,
    SessionSummary,
)
from app.context.budget import ContextBudgetService
from app.domain.guidance_policy import (
    select_guidance,
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


class ContextBudgetExceededError(ValueError):
    """
    Raised when the essential context alone exceeds the available input budget.

    History cannot be removed to solve this situation because no history has yet
    been included. The caller should reduce retrieved knowledge, adaptations,
    summary size, system instructions, or increase the configured budget.
    """

    def __init__(
        self,
        *,
        essential_tokens: int,
        input_token_budget: int,
    ) -> None:
        self.essential_tokens = essential_tokens
        self.input_token_budget = input_token_budget

        super().__init__(
            "Essential context exceeds the available input token budget: "
            f"{essential_tokens} > {input_token_budget}."
        )


class ContextService:
    """
    Assemble the base request-scoped context for FREDi.

    This service combines persistent user context, temporary session
    state, recent conversation history, and the current user message.

    Token budgeting bounds recent history while preserving essential context.
    Agent-specific filtering is handled by the context policy service.
    """

    def __init__(
        self,
        *,
        user_context_service: UserContextService,
        session_context_service: SessionContextService,
        conversation_service: ConversationService,
        budget_service: ContextBudgetService | None = None,
        agent_prompt_reserve_tokens: int = 1500,
    ) -> None:
        self._user_context_service = user_context_service
        self._session_context_service = session_context_service
        self._conversation_service = conversation_service
        self._budget_service = budget_service or ContextBudgetService()

        # Each selected agent prepends its own system prompt,
        # guidance policy, and state-update instructions. Reserve
        # budget for that prompt content so recent-history
        # selection does not assume it costs zero tokens.
        self._agent_prompt_reserve_tokens = agent_prompt_reserve_tokens

    async def build(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
        trace_id: UUID | None = None,
        current_message_id: UUID | None = None,
        recent_message_limit: int | None = None,
    ) -> AgentContext:
        """
        Build the common AgentContext for one request.

        If current_message_id is supplied, that message is excluded
        from recent_messages so the current user message is not sent
        twice to the model.
        """

        if recent_message_limit is not None and recent_message_limit < 0:
            raise ValueError("recent_message_limit must not be negative")

        user_context = await self._user_context_service.get_or_create(
            user_id=user_id,
        )

        session_context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        policy = self._budget_service.policy
        if recent_message_limit is not None:
            policy = replace(
                policy,
                max_recent_messages=min(recent_message_limit, policy.max_recent_messages),
            )

        recent_messages = await self._load_recent_messages(
            user_id=user_id,
            session_id=session_id,
            current_message_id=(current_message_id),
            limit=policy.max_recent_messages,
        )

        # Guidance is selected centrally by deterministic policy
        # (Sprint 4B.9) from persisted signals + session state —
        # never copied raw and never chosen by an LLM.
        guidance = select_guidance(
            hint_state=user_context.hint_state,
            session_context=session_context,
        )

        context = AgentContext(
            user_id=user_id,
            session_id=session_id,
            current_message=current_message,
            scenario_card=(user_context.scenario_card),
            learning_profile=(user_context.learning_profile),
            progress=(user_context.progress),
            session_context=(session_context),
            recent_messages=(recent_messages),
            session_summary=(session_context.summary),
            guidance_level=(guidance.level),
            user_adaptations=list(user_context.adaptations),
            knowledge_chunks=[],
        )

        # Estimate structured essentials without duplicating history or metadata.
        essential = context.model_dump_json(
            exclude={"recent_messages", "metadata"},
            exclude_none=True,
        )
        budget = self._budget_service.select_recent_messages(
            messages=recent_messages,
            essential_texts=[essential],
            policy=policy,
            extra_essential_tokens=(self._agent_prompt_reserve_tokens),
        )
        if budget.estimated_essential_tokens > budget.input_token_budget:
            raise ContextBudgetExceededError(
                essential_tokens=budget.estimated_essential_tokens,
                input_token_budget=budget.input_token_budget,
            )

        context.recent_messages = budget.selected_messages
        context.metadata = ContextMetadata(
            trace_id=trace_id,
            included_sections=self._build_included_sections(
                scenario_card=context.scenario_card,
                recent_messages=context.recent_messages,
                session_summary=context.session_summary,
                adaptation_count=len(context.user_adaptations),
            ),
            recent_message_count=budget.selected_message_count,
            summary_included=(context.session_summary is not None),
            guidance_reason=(guidance.reason),
            knowledge_chunk_count=0,
            adaptation_count=len(context.user_adaptations),
            estimated_input_tokens=budget.estimated_input_tokens,
            input_token_budget=budget.input_token_budget,
            estimated_essential_tokens=budget.estimated_essential_tokens,
            estimated_history_tokens=budget.estimated_history_tokens,
            candidate_message_count=budget.candidate_message_count,
            dropped_message_count=budget.dropped_message_count,
            truncation_applied=budget.truncation_applied,
        )
        return context

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
