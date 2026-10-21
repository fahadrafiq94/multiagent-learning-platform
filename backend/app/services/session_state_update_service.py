from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.schemas import (
    AgentName,
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
from app.services.session_context_service import (
    SessionContextService,
)

_ALLOWED_FIELDS: dict[
    AgentName,
    set[SessionStateField],
] = {
    "scenario": {
        SessionStateField.CURRENT_OBJECTIVE,
        SessionStateField.ACTIVE_BLOCKER,
        SessionStateField.UNRESOLVED_QUESTIONS,
    },
    "process_coach": {
        SessionStateField.CURRENT_OBJECTIVE,
        SessionStateField.CURRENT_PROCESS_PHASE,
        SessionStateField.ACTIVE_BLOCKER,
        SessionStateField.UNRESOLVED_QUESTIONS,
    },
    "ap_plus_navigator": {
        SessionStateField.CURRENT_STEP_ID,
        SessionStateField.CURRENT_PROCESS_PHASE,
        SessionStateField.ACTIVE_BLOCKER,
        SessionStateField.UNRESOLVED_QUESTIONS,
    },
}


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

    async def apply(
        self,
        *,
        user_id,
        session_id,
        agent_name: AgentName,
        proposal: (SessionStateUpdateProposal | None),
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

        allowed_fields = _ALLOWED_FIELDS[agent_name]

        operations = proposal.operations if proposal is not None else []

        for operation in operations:
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

        key = SessionStateUpdateService._comparison_key(value)

        existing_keys = {
            SessionStateUpdateService._comparison_key(item) for item in unresolved_questions
        }

        if operation.action == SessionStateAction.ADD:
            if key in existing_keys:
                return False

            unresolved_questions.append(value)

            return True

        if operation.action == SessionStateAction.REMOVE:
            for index, item in enumerate(unresolved_questions):
                if SessionStateUpdateService._comparison_key(item) == key:
                    unresolved_questions.pop(index)

                    return True

            return False

        return False

    @staticmethod
    def _comparison_key(
        value: str,
    ) -> str:
        return " ".join(value.casefold().split())
