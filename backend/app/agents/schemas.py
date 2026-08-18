from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.agents.policies import GuidanceLevel

AgentName = Literal[
    "scenario",
    "process_coach",
    "ap_plus_navigator",
]


class AgentRequest(BaseModel):
    """Common input contract for specialized FREDi agents."""

    session_id: str = Field(min_length=1)
    student_id: str | None = None
    user_message: str = Field(min_length=1)

    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL

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
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("Value must not be blank.")

        return stripped_value

    @field_validator("student_id")
    @classmethod
    def student_id_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("Student ID must not be blank.")

        return stripped_value


class AgentResponse(BaseModel):
    """Common output contract returned by a specialized agent."""

    agent: AgentName
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("Agent response must not be blank.")

        return stripped_value
