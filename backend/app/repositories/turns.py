"""Storage contract for durable request admission, replay and terminal outcomes."""

from typing import Any, Protocol
from uuid import UUID

from app.context import ConversationMessage
from app.persistence.turn_models import TurnAdmission, TurnRecord, TurnStatus


class TurnStore(Protocol):
    async def get(
        self, *, user_id: UUID, session_id: UUID, request_id: UUID
    ) -> TurnRecord | None: ...

    async def begin(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        content: str,
        trace_id: UUID | None = None,
        edit_message_id: UUID | None = None,
    ) -> TurnAdmission: ...

    async def complete(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        response: ConversationMessage,
        response_payload: dict[str, Any] | None = None,
    ) -> TurnRecord: ...

    async def stop(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        status: TurnStatus,
        error_code: str,
    ) -> TurnRecord: ...

    async def delete_session(self, *, user_id: UUID, session_id: UUID) -> None:
        """Delete every durable turn belonging to one user/session pair."""
