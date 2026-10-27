from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from app.orchestration.state import (
    RouteName,
)


class OrchestrationRequest(BaseModel):
    session_id: str = Field(min_length=1)

    student_id: str = Field(min_length=1)

    # The bound matches ConversationMessage.content (LongText, 4000)
    # so an over-long message is rejected at the API boundary with a
    # 422 instead of failing deep inside message persistence.
    user_message: str = Field(min_length=1, max_length=4_000)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "session_id",
        "student_id",
        "user_message",
    )
    @classmethod
    def value_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()

        if not stripped:
            raise ValueError("Value must not be blank.")

        return stripped


class OrchestrationResponse(BaseModel):
    session_id: str

    student_id: str

    final_response: str

    route: RouteName | None = None

    route_reason: str | None = None

    error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
