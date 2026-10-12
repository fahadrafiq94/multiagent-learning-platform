from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from app.context.agent_views import (
    AgentContextView,
    ContextTarget,
)

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

    context: AgentContextView | None = None

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

    @model_validator(mode="after")
    def context_must_be_orchestrator_view(
        self,
    ) -> OrchestratorRequest:
        if self.context is not None and self.context.target is not ContextTarget.ORCHESTRATOR:
            raise ValueError("OrchestratorRequest context must target ORCHESTRATOR.")

        return self


class RoutingDecision(BaseModel):
    """Machine-readable routing result."""

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
    """Validated routing result plus metadata."""

    decision: RoutingDecision

    metadata: dict[str, Any] = Field(default_factory=dict)
