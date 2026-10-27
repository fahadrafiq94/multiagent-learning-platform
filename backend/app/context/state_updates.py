from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class SessionStateField(StrEnum):
    CURRENT_OBJECTIVE = "current_objective"

    CURRENT_STEP_ID = "current_step_id"

    CURRENT_PROCESS_PHASE = "current_process_phase"

    ACTIVE_BLOCKER = "active_blocker"

    UNRESOLVED_QUESTIONS = "unresolved_questions"


class SessionStateAction(StrEnum):
    SET = "set"

    CLEAR = "clear"

    ADD = "add"

    REMOVE = "remove"


class SessionStateUpdateOperation(BaseModel):
    """
    One proposed runtime SessionContext mutation.

    This is only a proposal.

    It does not mutate persistent state.
    """

    # Upper bound per target field. These mirror the persistent
    # SessionContext field constraints (ShortText/MediumText/LongText)
    # so an oversized proposal is rejected per-operation instead of
    # failing the whole batch at persistence time.
    _FIELD_VALUE_MAX_LENGTH: ClassVar[dict[SessionStateField, int]] = {
        SessionStateField.CURRENT_OBJECTIVE: 4_000,
        SessionStateField.CURRENT_STEP_ID: 255,
        SessionStateField.CURRENT_PROCESS_PHASE: 255,
        SessionStateField.ACTIVE_BLOCKER: 4_000,
        SessionStateField.UNRESOLVED_QUESTIONS: 1_000,
    }

    model_config = ConfigDict(extra="forbid")

    field: SessionStateField

    action: SessionStateAction

    value: str | None = None

    reason: str = Field(
        min_length=1,
        max_length=300,
    )

    @field_validator("value")
    @classmethod
    def normalize_value(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())

        if not normalized:
            return None

        field = info.data.get("field")

        if isinstance(field, SessionStateField):
            max_length = cls._FIELD_VALUE_MAX_LENGTH[field]

            if len(normalized) > max_length:
                raise ValueError(
                    f"Value for {field.value} exceeds the "
                    f"target field limit of {max_length} characters."
                )

        return normalized

    @field_validator("reason")
    @classmethod
    def normalize_reason(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Reason must not be blank.")

        return normalized

    @model_validator(mode="after")
    def validate_operation(
        self,
    ) -> SessionStateUpdateOperation:
        scalar_fields = {
            SessionStateField.CURRENT_OBJECTIVE,
            SessionStateField.CURRENT_STEP_ID,
            SessionStateField.CURRENT_PROCESS_PHASE,
            SessionStateField.ACTIVE_BLOCKER,
        }

        if self.field in scalar_fields:
            if self.action not in {
                SessionStateAction.SET,
                SessionStateAction.CLEAR,
            }:
                raise ValueError("Scalar session state fields only support SET or CLEAR.")

        if self.field == SessionStateField.UNRESOLVED_QUESTIONS:
            if self.action not in {
                SessionStateAction.ADD,
                SessionStateAction.REMOVE,
                SessionStateAction.CLEAR,
            }:
                raise ValueError("unresolved_questions supports ADD, REMOVE, or CLEAR.")

        if (
            self.action
            in {
                SessionStateAction.SET,
                SessionStateAction.ADD,
                SessionStateAction.REMOVE,
            }
            and self.value is None
        ):
            raise ValueError(f"{self.action.value} requires a value.")

        if self.action == SessionStateAction.CLEAR and self.value is not None:
            raise ValueError("CLEAR must not include a value.")

        return self


class SessionStateUpdateProposal(BaseModel):
    """
    Agent-proposed runtime state changes.

    Application code validates these
    operations before persistence.
    """

    model_config = ConfigDict(extra="forbid")

    operations: list[SessionStateUpdateOperation] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def reject_ambiguous_operations(
        self,
    ) -> SessionStateUpdateProposal:
        scalar_fields_seen: set[SessionStateField] = set()

        unresolved_clear_seen = False

        unresolved_other_seen = False

        for operation in self.operations:
            if operation.field == SessionStateField.UNRESOLVED_QUESTIONS:
                if operation.action == SessionStateAction.CLEAR:
                    unresolved_clear_seen = True
                else:
                    unresolved_other_seen = True

                continue

            if operation.field in scalar_fields_seen:
                raise ValueError("A scalar session field may only be updated once per proposal.")

            scalar_fields_seen.add(operation.field)

        if unresolved_clear_seen and unresolved_other_seen:
            raise ValueError(
                "CLEAR cannot be combined with ADD or REMOVE for unresolved_questions."
            )

        return self
