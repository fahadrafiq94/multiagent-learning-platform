from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.context.schemas import (
    SessionSummary,
)
from app.context.summary_prompt import (
    SummaryDraft,
    build_summary_request,
    sanitize_summary_draft,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.model_service import (
    ModelService,
)
from app.services.session_context_service import (
    SessionContextService,
)


def should_update_summary(
    *,
    total_message_count: int,
    covered_message_count: int,
    recent_window: int,
    min_new_messages: int,
) -> bool:
    """
    Deterministic summary-update policy.

    A summary update runs only when at least `min_new_messages`
    messages have fallen out of the recent-history window since the
    last update. Below the threshold there is no model call at all.
    """

    newly_outside_window = total_message_count - covered_message_count - recent_window

    return newly_outside_window >= min_new_messages


class SessionSummaryUpdateResult(BaseModel):
    """
    Outcome of one best-effort summary-update attempt.

    `error` is set when the attempt could not persist a new summary;
    the previous summary (if any) is kept unchanged in that case.
    """

    updated: bool = False

    reason: str = "skipped"

    summary: SessionSummary | None = None

    covered_message_count: int = 0

    error: str | None = None


class SessionSummaryService:
    """
    Incrementally maintain the persisted session summary.

    The service owns the summary lifecycle: it decides when an update
    is needed (deterministically), fetches only the newly summarized
    message batch, asks the model for a merged draft, and persists the
    result through SessionContextService.

    All failures are contained in the returned result so the
    orchestration turn can never break because of a summary update.
    """

    def __init__(
        self,
        *,
        conversation_service: ConversationService,
        session_context_service: SessionContextService,
        model_service: ModelService,
        recent_window: int,
        min_new_messages: int = 4,
        max_items_per_list: int = 10,
        max_overview_chars: int = 1200,
        model: str | None = None,
    ) -> None:
        self._conversation_service = conversation_service
        self._session_context_service = session_context_service
        self._model_service = model_service

        self._recent_window = recent_window
        self._min_new_messages = min_new_messages
        self._max_items_per_list = max_items_per_list
        self._max_overview_chars = max_overview_chars

        # None means: use the provider's default chat model.
        self._model = model

    async def maybe_update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionSummaryUpdateResult:
        """
        Update the session summary when the deterministic policy says
        it is due.

        Ownership is enforced by the underlying services; a cross-user
        caller gets the usual repository rejection.
        """

        context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        total = await self._conversation_service.count_messages(
            user_id=user_id,
            session_id=session_id,
        )

        existing = context.summary

        covered = existing.source_message_count if existing is not None else 0

        if not should_update_summary(
            total_message_count=total,
            covered_message_count=covered,
            recent_window=self._recent_window,
            min_new_messages=self._min_new_messages,
        ):
            return SessionSummaryUpdateResult(
                reason="below_threshold",
                covered_message_count=covered,
            )

        batch_limit = total - self._recent_window - covered

        if batch_limit < 1:
            return SessionSummaryUpdateResult(
                reason="no_new_messages",
                covered_message_count=covered,
            )

        messages = await self._conversation_service.get_history(
            user_id=user_id,
            session_id=session_id,
            limit=batch_limit,
            offset=covered,
        )

        if not messages:
            return SessionSummaryUpdateResult(
                reason="no_new_messages",
                covered_message_count=covered,
            )

        try:
            response = await self._model_service.generate(
                build_summary_request(
                    existing=existing,
                    messages=messages,
                    model=self._model,
                )
            )
        except Exception as exc:
            return SessionSummaryUpdateResult(
                reason="model_failed",
                covered_message_count=covered,
                error=f"Session summary model call failed: {exc}",
            )

        try:
            draft = SummaryDraft.model_validate_json(response.content)
        except ValidationError as exc:
            return SessionSummaryUpdateResult(
                reason="invalid_model_output",
                covered_message_count=covered,
                error=(f"Session summary model returned an invalid draft: {exc}"),
            )

        draft = sanitize_summary_draft(
            draft,
            max_items_per_list=self._max_items_per_list,
            max_overview_chars=self._max_overview_chars,
        )

        summary = SessionSummary(
            summary_id=(existing.summary_id if existing is not None else uuid4()),
            user_id=user_id,
            session_id=session_id,
            overview=draft.overview,
            established_facts=draft.established_facts,
            decisions=draft.decisions,
            blockers=draft.blockers,
            unresolved_questions=draft.unresolved_questions,
            misconception_candidates=draft.misconception_candidates,
            completed_step_ids=draft.completed_step_ids,
            source_message_count=covered + len(messages),
            generated_at=datetime.now(UTC),
        )

        context.summary = summary

        try:
            await self._session_context_service.save(
                user_id=user_id,
                session_id=session_id,
                context=context,
                expected_version=context.version,
            )
        except Exception as exc:
            return SessionSummaryUpdateResult(
                reason="persistence_failed",
                covered_message_count=covered,
                error=f"Session summary persistence failed: {exc}",
            )

        return SessionSummaryUpdateResult(
            updated=True,
            reason="updated",
            summary=summary,
            covered_message_count=summary.source_message_count,
        )
