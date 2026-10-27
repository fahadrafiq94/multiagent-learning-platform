from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.context.agent_views import (
    AgentContextView,
)
from app.context.schemas import (
    MediumText,
)
from app.context.state_updates import (
    SessionStateUpdateProposal,
)
from app.domain.guidance import (
    GuidanceLevel,
)

AgentName = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
]


class AgentRequest(BaseModel):
    """
    Common request contract for FREDi specialist agents.
    """

    session_id: str = Field(min_length=1)

    student_id: str | None = None

    user_message: str = Field(min_length=1)

    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL

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


class UnderstandingLevel(StrEnum):
    """
    How well the student's current message shows process understanding.

    Closed enum: the deterministic signal mapper (4B.9) only accepts
    these values.
    """

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    OFF_TRACK = "off_track"
    NOT_APPLICABLE = "not_applicable"


class StudentResponseAssessment(BaseModel):
    """
    The specialist's judgment of the STUDENT's current message —
    not of the agent's own answer.

    This is LLM-generated content: it is never persisted as state.
    Only the deterministic counters derived from it are persisted
    (Sprint 4B.9 guidance policy).
    """

    understanding: UnderstandingLevel

    misconception_detected: bool = False

    reason: MediumText | None = None


class SpecialistAgentOutput(BaseModel):
    """
    Structured LLM output from one specialist.

    content:
        user-facing answer

    state_update_proposal:
        optional runtime state proposal

    response_assessment:
        optional judgment of the student's current message;
        a malformed assessment degrades to None and can never
        fail the student's turn
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)

    state_update_proposal: SessionStateUpdateProposal = Field(
        default_factory=(SessionStateUpdateProposal)
    )

    response_assessment: StudentResponseAssessment | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("Specialist response must not be blank.")

        return stripped

    @field_validator("response_assessment", mode="before")
    @classmethod
    def response_assessment_is_best_effort(
        cls,
        value: Any,
    ) -> Any:
        """
        The assessment is a best-effort addition to the output.

        A valid dict is coerced into the schema; anything malformed
        (bad enum value, wrong shape, nested junk) degrades to None
        so the student's turn and the rest of the output are never
        harmed by a failed judgment.
        """

        if value is None:
            return None

        if isinstance(value, StudentResponseAssessment):
            return value

        try:
            return StudentResponseAssessment.model_validate(value)

        except (ValidationError, ValueError, TypeError):
            return None


class AgentResponse(BaseModel):
    """
    Validated specialist-agent output.
    """

    agent: AgentName

    content: str = Field(min_length=1)

    state_update_proposal: SessionStateUpdateProposal | None = None

    response_assessment: StudentResponseAssessment | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("Agent response must not be blank.")

        return stripped
