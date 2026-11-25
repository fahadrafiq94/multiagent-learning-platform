"""Atomic retry-safe turn admission/completion; model generation stays outside SQL."""

from __future__ import annotations

import hashlib
import re
from typing import Any
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb

from app.context import ConversationMessage, ConversationRole
from app.persistence.database import ApplicationDatabase
from app.persistence.records import checked, decode, insert, require_session, touch_session
from app.persistence.turn_models import TurnAdmission as TurnAdmission
from app.persistence.turn_models import TurnRecord as TurnRecord
from app.persistence.turn_models import TurnStatus as TurnStatus
from app.repositories.exceptions import RepositoryConflictError, RepositoryNotFoundError


class PostgresTurnStore:
    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def _find(self, connection, user_id, session_id, request_id, *, lock=False):
        cursor = await connection.execute(
            "SELECT * FROM application.turns WHERE user_id=%s AND session_id=%s AND request_id=%s"
            + (" FOR UPDATE" if lock else ""),
            (user_id, session_id, request_id),
        )
        return await cursor.fetchone()

    async def _locked(self, connection, user_id, session_id, request_id):
        await require_session(connection, user_id, session_id, lock=True)
        row = await self._find(connection, user_id, session_id, request_id, lock=True)
        if row is None:
            raise RepositoryNotFoundError(entity_name="ConversationTurn", entity_id=str(request_id))
        return row

    async def get(self, *, user_id: UUID, session_id: UUID, request_id: UUID) -> TurnRecord | None:
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id)
            row = await self._find(connection, user_id, session_id, request_id)
            return TurnRecord.model_validate(row) if row else None

    async def begin(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        content: str,
        trace_id: UUID | None = None,
        edit_message_id: UUID | None = None,
    ) -> TurnAdmission:
        turn_id = uuid5(session_id, f"{user_id}:{request_id}")
        message = ConversationMessage(
            message_id=uuid5(turn_id, "user"),
            user_id=user_id,
            session_id=session_id,
            role=ConversationRole.USER,
            content=content,
            trace_id=trace_id,
        )
        request_content = (
            f"edit:{edit_message_id}:{message.content}" if edit_message_id else message.content
        )
        request_hash = hashlib.sha256(request_content.encode()).hexdigest()
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id, lock=True)
            existing = await self._find(connection, user_id, session_id, request_id)
            if existing:
                if existing["request_hash"] != request_hash or (
                    existing["replaces_turn_id"] is None
                ) != (edit_message_id is None):
                    raise RepositoryConflictError(
                        "Request identifier was reused with different content."
                    )
                return TurnAdmission(turn=TurnRecord.model_validate(existing), created=False)
            cursor = await connection.execute(
                "SELECT 1 FROM application.turns WHERE user_id=%s AND session_id=%s AND "
                "status='processing'",
                (user_id, session_id),
            )
            if await cursor.fetchone():
                raise RepositoryConflictError("Another turn is processing in this conversation.")
            replaces_turn_id = None
            replaced_message_ids: tuple[UUID, ...] = ()
            if edit_message_id is not None:
                latest = await connection.execute(
                    "SELECT turn_id, message_id, role "
                    "FROM application.visible_conversation_messages "
                    "WHERE user_id=%s AND session_id=%s "
                    "ORDER BY created_at DESC, sequence_no DESC LIMIT 2",
                    (user_id, session_id),
                )
                messages = await latest.fetchall()
                if (
                    len(messages) != 2
                    or messages[0]["role"] != "assistant"
                    or messages[1]["role"] != "user"
                    or messages[1]["message_id"] != edit_message_id
                    or messages[0]["turn_id"] is None
                    or messages[0]["turn_id"] != messages[1]["turn_id"]
                ):
                    raise RepositoryConflictError(
                        "Only the latest completed exchange can be edited."
                    )
                replaces_turn_id = messages[0]["turn_id"]
                replaced_message_ids = (edit_message_id, messages[0]["message_id"])
            cursor = await connection.execute(
                "INSERT INTO "
                "application.turns(turn_id,user_id,session_id,request_id,"
                "status,request_hash,user_message_id,replaces_turn_id) "
                "VALUES (%s,%s,%s,%s,'processing',%s,%s,%s) RETURNING *",
                (
                    turn_id,
                    user_id,
                    session_id,
                    request_id,
                    request_hash,
                    message.message_id,
                    replaces_turn_id,
                ),
            )
            turn = TurnRecord.model_validate(await cursor.fetchone())
            await insert(connection, message, turn_id=turn_id)
            await touch_session(connection, user_id, session_id)
            return TurnAdmission(turn=turn, created=True, replaced_message_ids=replaced_message_ids)

    async def complete(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        response: ConversationMessage,
        response_payload: dict[str, Any] | None = None,
    ) -> TurnRecord:
        response = checked(response)
        if (
            response.user_id != user_id
            or response.session_id != session_id
            or response.role != ConversationRole.ASSISTANT
        ):
            raise RepositoryConflictError("Response does not match the turn owner/session/role.")
        async with self.database.connection() as connection:
            row = await self._locked(connection, user_id, session_id, request_id)
            if row["status"] == TurnStatus.COMPLETED:
                cursor = await connection.execute(
                    "SELECT * FROM application.conversation_messages WHERE user_id=%s AND "
                    "session_id=%s AND message_id=%s",
                    (user_id, session_id, row["assistant_message_id"]),
                )
                previous = decode(ConversationMessage, await cursor.fetchone())
                excluded = {"message_id", "created_at", "trace_id"}
                if previous.model_dump(exclude=excluded) != response.model_dump(exclude=excluded):
                    raise RepositoryConflictError(
                        "Completed turn already has a different response."
                    )
                return TurnRecord.model_validate(row)
            if row["status"] != TurnStatus.PROCESSING:
                raise RepositoryConflictError("Terminal turn cannot be completed again.")
            response = response.model_copy(
                update={"message_id": uuid5(row["turn_id"], "assistant")}
            )
            await insert(connection, response, turn_id=row["turn_id"])
            cursor = await connection.execute(
                "UPDATE application.turns SET "
                "status='completed',assistant_message_id=%s,response_payload=%s,"
                "updated_at=clock_timestamp() "
                "WHERE user_id=%s AND session_id=%s AND request_id=%s RETURNING *",
                (
                    response.message_id,
                    Jsonb(response_payload) if response_payload is not None else None,
                    user_id,
                    session_id,
                    request_id,
                ),
            )
            await touch_session(connection, user_id, session_id)
            return TurnRecord.model_validate(await cursor.fetchone())

    async def stop(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        status: TurnStatus,
        error_code: str,
    ) -> TurnRecord:
        if status not in {TurnStatus.FAILED, TurnStatus.INTERRUPTED}:
            raise ValueError("Stop requires failed or interrupted status.")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
            raise ValueError("Use a short error code, never raw exception text.")
        async with self.database.connection() as connection:
            row = await self._locked(connection, user_id, session_id, request_id)
            if row["status"] != TurnStatus.PROCESSING:
                if row["status"] == status and row["error_code"] == error_code:
                    return TurnRecord.model_validate(row)
                raise RepositoryConflictError("Terminal turn state cannot be overwritten.")
            cursor = await connection.execute(
                "UPDATE application.turns SET status=%s,error_code=%s,updated_at=clock_timestamp() "
                "WHERE user_id=%s AND session_id=%s AND request_id=%s RETURNING *",
                (status.value, error_code, user_id, session_id, request_id),
            )
            return TurnRecord.model_validate(await cursor.fetchone())

    async def delete_session(self, *, user_id: UUID, session_id: UUID) -> None:
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id, lock=True)
            processing = await connection.execute(
                "SELECT 1 FROM application.turns WHERE user_id=%s AND session_id=%s "
                "AND status='processing'",
                (user_id, session_id),
            )
            if await processing.fetchone():
                raise RepositoryConflictError("Another turn is processing in this conversation.")
            # The turn/message foreign keys are deferred to transaction commit.
            await connection.execute(
                "DELETE FROM application.conversation_messages WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            await connection.execute(
                "DELETE FROM application.turns WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            await connection.execute(
                "DELETE FROM application.session_contexts WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            await connection.execute(
                "DELETE FROM application.conversation_sessions WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
