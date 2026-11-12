from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
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


class StudentStepDecision(BaseModel):
    """Semantic interpretation of the current student's statement; no permissions."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["set", "unchanged", "clarify"]
    step_id: str | None = Field(min_length=1, max_length=255)
    source_quote: str | None = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def consistent(self):
        if self.action == "set":
            if not self.step_id or not self.step_id.strip() or not self.source_quote:
                raise ValueError("A step change needs its identifier and current-message quote.")
        elif self.step_id is not None or self.source_quote is not None:
            raise ValueError("A non-change decision cannot propose a step or quote.")
        return self


class InteractionDecision(BaseModel):
    """Semantic purpose only. Application code supplies any context-only answer."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["help", "acknowledgement", "greeting", "recall", "mixed", "unclear"]
    target: (
        Literal[
            "current_step_id",
            "company_context",
            "learner_role",
            "business_problem",
            "business_goals",
            "user_history",
        ]
        | None
    )
    source_quote: str | None = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "recall":
            if self.target is None:
                raise ValueError("Recall needs a stored-fact target.")
            if self.target != "user_history" and self.source_quote is not None:
                raise ValueError("Only student-history recall may carry a source quote.")
        elif self.target is not None or self.source_quote is not None:
            raise ValueError("Other purposes cannot propose a recall target or quotation.")
        return self


class RoutingDecision(BaseModel):
    """Machine-readable routing result."""

    route: RoutingTarget

    # Historical responses remain readable. The generation schema requires an
    # explicit decision; an omitted/malformed decision never grants a state write.
    current_step: StudentStepDecision | None = None
    interaction: InteractionDecision | None = None

    reason: str = Field(
        min_length=1,
        max_length=500,
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

    @field_validator("current_step", mode="before")
    @classmethod
    def step_is_best_effort(cls, value):
        if value is None or isinstance(value, StudentStepDecision):
            return value
        try:
            return StudentStepDecision.model_validate(value)
        except (ValueError, TypeError):
            return None

    @field_validator("interaction", mode="before")
    @classmethod
    def interaction_is_best_effort(cls, value):
        if value is None or isinstance(value, InteractionDecision):
            return value
        try:
            return InteractionDecision.model_validate(value)
        except (ValueError, TypeError):
            return None


class OrchestratorResult(BaseModel):
    """Validated routing result plus metadata."""

    decision: RoutingDecision

    metadata: dict[str, Any] = Field(default_factory=dict)
