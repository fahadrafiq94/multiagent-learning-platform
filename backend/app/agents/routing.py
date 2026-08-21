from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RoutingTarget = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
    "fallback",
]


class OrchestratorRequest(BaseModel):
    """Input required by the semantic routing agent."""

    session_id: str = Field(min_length=1)
    student_id: str | None = None
    user_message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "session_id",
        "user_message",
    )
    @classmethod
    def required_value_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("Value must not be blank.")

        return stripped

    @field_validator("student_id")
    @classmethod
    def student_id_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        stripped = value.strip()

        if not stripped:
            raise ValueError("Student ID must not be blank.")

        return stripped


class RoutingDecision(BaseModel):
    """Machine-readable routing decision produced by the orchestrator."""

    route: RoutingTarget

    reason: str = Field(
        min_length=1,
        max_length=300,
    )

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("Routing reason must not be blank.")

        return stripped


class OrchestratorResult(BaseModel):
    """Validated semantic routing result plus observability metadata."""

    decision: RoutingDecision

    metadata: dict[str, Any] = Field(default_factory=dict)
