"""Turn values shared by storage and runtime; no database I/O."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TurnStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TurnRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn_id: UUID
    user_id: UUID
    session_id: UUID
    request_id: UUID
    status: TurnStatus
    user_message_id: UUID
    assistant_message_id: UUID | None = None
    error_code: str | None = None
    response_payload: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class TurnAdmission(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn: TurnRecord
    created: bool
