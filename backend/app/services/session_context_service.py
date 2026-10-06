from __future__ import annotations

from uuid import UUID

from app.context import (
    SessionContext,
    SessionContextUpdate,
)
from app.repositories import (
    RepositoryNotFoundError,
    SessionContextRepository,
    VersionConflictError,
)


def _normalized_text(
    value: str,
) -> str:
    """
    Normalize text for duplicate comparison.

    Stored text preserves the original user-facing formatting.
    """

    return " ".join(value.split()).casefold()


class SessionContextService:
    """
    Application service for mutable per-conversation state.

    SessionContext represents what is happening in one specific
    conversation.

    Examples:

    - current learning objective
    - current AP+ step
    - current process phase
    - active blocker
    - selected agent
    - unresolved questions

    Cross-session durable information belongs in UserContext,
    not SessionContext.

    Scenario onboarding state is also stored in SessionContext,
    but its lifecycle is controlled by ScenarioOnboardingService.
    """

    def __init__(
        self,
        *,
        repository: SessionContextRepository,
    ) -> None:
        self._repository = repository

    async def get(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionContext | None:
        """
        Return SessionContext when it exists.

        Repository ownership checks ensure another user's session
        cannot be loaded.
        """

        return await self._repository.get(
            user_id=user_id,
            session_id=session_id,
        )

    async def get_required(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionContext:
        """
        Return SessionContext or fail explicitly.
        """

        context = await self._repository.get(
            user_id=user_id,
            session_id=session_id,
        )

        if context is None:
            raise RepositoryNotFoundError(
                entity_name="SessionContext",
                entity_id=str(session_id),
            )

        return context

    async def save(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        context: SessionContext,
        expected_version: int,
    ) -> SessionContext:
        """
        Persist a complete SessionContext snapshot.

        This method exists for services with specialized state
        ownership, such as ScenarioOnboardingService.

        Normal runtime updates should generally use update().
        """

        self._validate_identity(
            user_id=user_id,
            session_id=session_id,
            context=context,
        )

        return await self._repository.save(
            context,
            expected_version=expected_version,
        )

    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        changes: SessionContextUpdate,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Atomically apply several session-state changes.

        Only fields explicitly supplied in SessionContextUpdate are
        modified.

        If expected_version is supplied, stale callers are rejected.

        A successful non-empty update increments SessionContext
        version exactly once.
        """

        context = await self.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if not changes.model_fields_set:
            return context

        if expected_version is not None and context.version != expected_version:
            raise VersionConflictError(
                entity_name="SessionContext",
                entity_id=str(session_id),
                expected_version=(expected_version),
                actual_version=context.version,
            )

        original_version = context.version

        self._apply_changes(
            context=context,
            changes=changes,
        )

        return await self._repository.save(
            context,
            expected_version=original_version,
        )

    async def set_current_objective(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        objective: str | None,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Set or clear the conversation's current objective.
        """

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                current_objective=objective,
            ),
            expected_version=expected_version,
        )

    async def set_current_step(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        step_id: str | None,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Set or clear the currently active AP+/learning step.
        """

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                current_step_id=step_id,
            ),
            expected_version=expected_version,
        )

    async def set_process_phase(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        process_phase: str | None,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Set or clear the current process phase.
        """

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                current_process_phase=(process_phase),
            ),
            expected_version=expected_version,
        )

    async def set_active_blocker(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        blocker: str | None,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Set or clear the user's current session blocker.
        """

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                active_blocker=blocker,
            ),
            expected_version=expected_version,
        )

    async def clear_active_blocker(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Explicit convenience method for clearing the blocker.
        """

        return await self.set_active_blocker(
            user_id=user_id,
            session_id=session_id,
            blocker=None,
            expected_version=expected_version,
        )

    async def set_last_selected_agent(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        agent_name: str | None,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Record the agent selected most recently in this session.
        """

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                last_selected_agent=agent_name,
            ),
            expected_version=expected_version,
        )

    async def add_unresolved_question(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        question: str,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Add one unresolved question.

        Duplicate questions are ignored using normalized
        case-insensitive comparison.
        """

        context = await self.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if expected_version is not None and context.version != expected_version:
            raise VersionConflictError(
                entity_name="SessionContext",
                entity_id=str(session_id),
                expected_version=(expected_version),
                actual_version=context.version,
            )

        if self._contains_question(
            questions=context.unresolved_questions,
            question=question,
        ):
            return context

        updated_questions = [
            *context.unresolved_questions,
            question,
        ]

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                unresolved_questions=(updated_questions),
            ),
            expected_version=context.version,
        )

    async def remove_unresolved_question(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        question: str,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Remove one unresolved question using normalized matching.

        If the question is absent, no persistence write occurs.
        """

        context = await self.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if expected_version is not None and context.version != expected_version:
            raise VersionConflictError(
                entity_name="SessionContext",
                entity_id=str(session_id),
                expected_version=(expected_version),
                actual_version=context.version,
            )

        target = _normalized_text(question)

        updated_questions = [
            existing
            for existing in (context.unresolved_questions)
            if (_normalized_text(existing) != target)
        ]

        if updated_questions == context.unresolved_questions:
            return context

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                unresolved_questions=(updated_questions),
            ),
            expected_version=context.version,
        )

    async def clear_unresolved_questions(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        expected_version: int | None = None,
    ) -> SessionContext:
        """
        Clear all unresolved questions.

        No write occurs if the list is already empty.
        """

        context = await self.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if expected_version is not None and context.version != expected_version:
            raise VersionConflictError(
                entity_name="SessionContext",
                entity_id=str(session_id),
                expected_version=(expected_version),
                actual_version=context.version,
            )

        if not context.unresolved_questions:
            return context

        return await self.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(
                unresolved_questions=[],
            ),
            expected_version=context.version,
        )

    @staticmethod
    def _validate_identity(
        *,
        user_id: UUID,
        session_id: UUID,
        context: SessionContext,
    ) -> None:
        if context.user_id != user_id:
            raise ValueError("context.user_id must match SessionContextService user_id")

        if context.session_id != session_id:
            raise ValueError("context.session_id must match SessionContextService session_id")

    @staticmethod
    def _apply_changes(
        *,
        context: SessionContext,
        changes: SessionContextUpdate,
    ) -> None:
        """
        Apply only explicitly supplied fields.

        Pydantic's model_fields_set allows us to distinguish:

            field omitted
                -> do not change it

            field explicitly set to None
                -> clear it
        """

        supplied = changes.model_fields_set

        if "current_objective" in supplied:
            context.current_objective = changes.current_objective

        if "current_step_id" in supplied:
            context.current_step_id = changes.current_step_id

        if "current_process_phase" in supplied:
            context.current_process_phase = changes.current_process_phase

        if "active_blocker" in supplied:
            context.active_blocker = changes.active_blocker

        if "last_selected_agent" in supplied:
            context.last_selected_agent = changes.last_selected_agent

        if "unresolved_questions" in supplied:
            context.unresolved_questions = SessionContextService._normalize_question_list(
                changes.unresolved_questions
            )

    @staticmethod
    def _contains_question(
        *,
        questions: list[str],
        question: str,
    ) -> bool:
        target = _normalized_text(question)

        return any(_normalized_text(existing) == target for existing in questions)

    @staticmethod
    def _normalize_question_list(
        questions: list[str] | None,
    ) -> list[str]:
        """
        Remove duplicate unresolved questions while preserving their
        original order and formatting.

        Explicit None is treated as clearing the list.
        """

        if questions is None:
            return []

        unique: list[str] = []
        seen: set[str] = set()

        for question in questions:
            normalized = _normalized_text(question)

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(question)

        return unique
