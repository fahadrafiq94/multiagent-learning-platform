from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.orchestration.state import RouteName


class OrchestrationRequest(BaseModel):
    """Application-level request for an orchestration workflow."""

    session_id: str = Field(min_length=1)
    student_id: str | None = None
    user_message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "user_message")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
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


class OrchestrationResponse(BaseModel):
    """Application-level result returned by the orchestration workflow."""

    session_id: str
    student_id: str | None = None
    final_response: str
    route: RouteName | None = None
    route_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
