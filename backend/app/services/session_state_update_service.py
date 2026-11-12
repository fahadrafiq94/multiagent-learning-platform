from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.routing import StudentStepDecision
from app.agents.schemas import (
    AgentName,
)
from app.agents.state_update_prompt import (
    AGENT_STATE_FIELD_POLICY,
)
from app.context.schemas import (
    SessionContext,
    SessionContextUpdate,
)
from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
    SessionStateUpdateOperation,
    SessionStateUpdateProposal,
)
from app.repositories.exceptions import VersionConflictError
from app.services.session_context_service import (
    SessionContextService,
)
from app.utils.text import (
    text_comparison_key,
)


class RejectedStateUpdate(BaseModel):
    operation: SessionStateUpdateOperation

    reason: str


class SessionStateUpdateResult(BaseModel):
    applied: bool = False

    accepted_operations: list[SessionStateUpdateOperation] = Field(default_factory=list)

    rejected_operations: list[RejectedStateUpdate] = Field(default_factory=list)

    session_context_version: int | None = None


class SessionStateUpdateService:
    """
    Validate and persist agent-proposed runtime state.

    Agents never receive SessionContextService directly.
    """

    def __init__(
        self,
        *,
        session_context_service: (SessionContextService),
    ) -> None:
        self._session_context_service = session_context_service

    async def apply_student_step(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        message_id: UUID | None,
        current_message: str,
        decision: StudentStepDecision,
        expected_version: int,
    ) -> tuple[str, SessionContext]:
        """Persist only an anchored current-step interpretation, before specialist work.

        Literal quote/identifier checks establish provenance, not semantic truth.
        The supervisor interprets affirmation versus quotation/negation; regression
        probes must continue checking that semantic distinction.
        """
        current = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )
        if decision.action == "unchanged":
            return "unchanged", current
        if decision.action == "clarify":
            return "clarification_required", current

        def normalize(value: str) -> str:
            return " ".join(value.split())

        quote = normalize(decision.source_quote or "")
        step = normalize(decision.step_id or "")
        if (
            message_id is None
            or not quote
            or quote not in normalize(current_message)
            or not re.search(r"(?<![\w-])" + re.escape(step) + r"(?![\w-])", quote)
        ):
            return "rejected", current
        if current.version != expected_version:
            raise VersionConflictError(
                entity_name="SessionContext",
                entity_id=str(session_id),
                expected_version=expected_version,
                actual_version=current.version,
            )
        if current.current_step_id == step:
            return "unchanged", current
        updated = await self._session_context_service.update(
            user_id=user_id,
            session_id=session_id,
            changes=SessionContextUpdate(current_step_id=step),
            expected_version=expected_version,
        )
        return "persisted", updated

    async def apply(
        self,
        *,
        user_id,
        session_id,
        agent_name: AgentName,
        proposal: (SessionStateUpdateProposal | None),
        allow_specialist_step_updates: bool = True,
    ) -> SessionStateUpdateResult:
        current = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        accepted: list[SessionStateUpdateOperation] = []

        rejected: list[RejectedStateUpdate] = []

        changes: dict[
            str,
            object,
        ] = {}

        unresolved_questions = list(current.unresolved_questions)

        allowed_fields = AGENT_STATE_FIELD_POLICY[agent_name]

        operations = proposal.operations if proposal is not None else []

        for operation in operations:
            if (
                operation.field == SessionStateField.CURRENT_STEP_ID
                and not allow_specialist_step_updates
            ):
                # A specialist may echo the already accepted current step. It
                # performs no write and is not a conflicting proposal.
                if (
                    operation.action == SessionStateAction.SET
                    and operation.value == current.current_step_id
                ):
                    continue
                rejected.append(
                    RejectedStateUpdate(
                        operation=operation,
                        reason="Current step requires the current-message student decision.",
                    )
                )
                continue
            if operation.field not in allowed_fields:
                rejected.append(
                    RejectedStateUpdate(
                        operation=operation,
                        reason=("Field is not allowed for this agent."),
                    )
                )

                continue

            changed = self._apply_operation(
                operation=operation,
                current=current,
                changes=changes,
                unresolved_questions=(unresolved_questions),
            )

            if not changed:
                rejected.append(
                    RejectedStateUpdate(
                        operation=operation,
                        reason=("Operation would not change the current state."),
                    )
                )

                continue

            accepted.append(operation)

        if unresolved_questions != current.unresolved_questions:
            changes["unresolved_questions"] = unresolved_questions

        if current.last_selected_agent != agent_name:
            changes["last_selected_agent"] = agent_name

        if not changes:
            return SessionStateUpdateResult(
                applied=False,
                accepted_operations=(accepted),
                rejected_operations=(rejected),
                session_context_version=(current.version),
            )

        update = SessionContextUpdate(**changes)

        updated = await self._session_context_service.update(
            user_id=user_id,
            session_id=session_id,
            changes=update,
            expected_version=(current.version),
        )

        return SessionStateUpdateResult(
            applied=True,
            accepted_operations=accepted,
            rejected_operations=rejected,
            session_context_version=(updated.version),
        )

    @staticmethod
    def _apply_operation(
        *,
        operation: (SessionStateUpdateOperation),
        current: SessionContext,
        changes: dict[
            str,
            object,
        ],
        unresolved_questions: (list[str]),
    ) -> bool:
        if operation.field == SessionStateField.UNRESOLVED_QUESTIONS:
            return SessionStateUpdateService._apply_question_operation(
                operation=operation,
                unresolved_questions=(unresolved_questions),
            )

        field_name = operation.field.value

        current_value = getattr(
            current,
            field_name,
        )

        if operation.action == SessionStateAction.CLEAR:
            if current_value is None:
                return False

            changes[field_name] = None

            return True

        new_value = operation.value

        if new_value == current_value:
            return False

        changes[field_name] = new_value

        return True

    @staticmethod
    def _apply_question_operation(
        *,
        operation: (SessionStateUpdateOperation),
        unresolved_questions: (list[str]),
    ) -> bool:
        if operation.action == SessionStateAction.CLEAR:
            if not unresolved_questions:
                return False

            unresolved_questions.clear()

            return True

        value = operation.value

        if value is None:
            return False

        key = text_comparison_key(value)

        existing_keys = {text_comparison_key(item) for item in unresolved_questions}

        if operation.action == SessionStateAction.ADD:
            if key in existing_keys:
                return False

            unresolved_questions.append(value)

            return True

        if operation.action == SessionStateAction.REMOVE:
            for index, item in enumerate(unresolved_questions):
                if text_comparison_key(item) == key:
                    unresolved_questions.pop(index)

                    return True

            return False

        return False
