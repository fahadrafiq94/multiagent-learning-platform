from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.context.agent_views import (
    AgentContextView,
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


class SpecialistAgentOutput(BaseModel):
    """
    Structured LLM output from one specialist.

    content:
        user-facing answer

    state_update_proposal:
        optional runtime state proposal
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)

    state_update_proposal: SessionStateUpdateProposal = Field(
        default_factory=(SessionStateUpdateProposal)
    )

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


class AgentResponse(BaseModel):
    """
    Validated specialist-agent output.
    """

    agent: AgentName

    content: str = Field(min_length=1)

    state_update_proposal: SessionStateUpdateProposal | None = None

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
