"""Shared typed SQL serialization. All table identifiers come from this module."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from app.context import ConversationMessage, ConversationSession, SessionContext, UserContext
from app.identity import UserIdentity
from app.repositories.exceptions import RepositoryIntegrityError, RepositoryNotFoundError

Connection = AsyncConnection[dict[str, Any]]

FIELDS = {
    "identities": ("user_id", "student_id", "status", "created_at", "updated_at"),
    "user_contexts": ("user_id", "version", "created_at", "updated_at"),
    "conversation_sessions": (
        "session_id",
        "user_id",
        "title",
        "status",
        "created_at",
        "updated_at",
    ),
    "session_contexts": ("session_id", "user_id", "version", "updated_at"),
    "conversation_messages": ("message_id", "user_id", "session_id", "role", "created_at"),
}
MODELS: dict[type[BaseModel], str] = {
    UserIdentity: "identities",
    UserContext: "user_contexts",
    ConversationSession: "conversation_sessions",
    SessionContext: "session_contexts",
    ConversationMessage: "conversation_messages",
}


def checked[T: BaseModel](record: T) -> T:
    try:
        return type(record).model_validate(record.model_dump())
    except (ValidationError, ValueError):
        raise RepositoryIntegrityError("Application record is invalid.") from None


def decode[T: BaseModel](model: type[T], row: dict[str, Any] | None) -> T:
    try:
        if row is None:
            raise RepositoryIntegrityError("Stored application record is missing.")
        payload = dict(row["payload"])
        payload.update({name: row[name] for name in FIELDS[MODELS[model]]})
        return model.model_validate(payload)
    except (ValidationError, ValueError, TypeError, KeyError):
        raise RepositoryIntegrityError(f"Stored {model.__name__} record is invalid.") from None


async def insert[T: BaseModel](
    connection: Connection, record: T, *, turn_id: UUID | None = None
) -> T:
    record = checked(record)
    table = MODELS[type(record)]
    fields = [*FIELDS[table], "payload"]
    raw = record.model_dump(mode="json")
    # Native UUID/datetime values are preferable for indexed SQL columns.
    native = record.model_dump()
    values = [getattr(native[k], "value", native[k]) for k in FIELDS[table]] + [Jsonb(raw)]
    if isinstance(record, ConversationMessage):
        fields.append("turn_id")
        values.append(turn_id)
    cursor = await connection.execute(
        sql.SQL("INSERT INTO application.{} ({}) VALUES ({}) RETURNING *").format(
            sql.Identifier(table),
            sql.SQL(",").join(map(sql.Identifier, fields)),
            sql.SQL(",").join(sql.Placeholder() for _ in fields),
        ),
        values,
    )
    return decode(type(record), await cursor.fetchone())


async def require_user(connection: Connection, user_id: UUID) -> None:
    cursor = await connection.execute(
        "SELECT 1 FROM application.identities WHERE user_id=%s", (user_id,)
    )
    if await cursor.fetchone() is None:
        raise RepositoryNotFoundError(entity_name="UserIdentity", entity_id=str(user_id))


async def require_session(
    connection: Connection, user_id: UUID, session_id: UUID, *, lock: bool = False
):
    cursor = await connection.execute(
        "SELECT * FROM application.conversation_sessions WHERE user_id=%s AND session_id=%s"
        + (" FOR UPDATE" if lock else ""),
        (user_id, session_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise RepositoryNotFoundError(entity_name="ConversationSession", entity_id=str(session_id))
    return row


async def touch_session(connection: Connection, user_id: UUID, session_id: UUID) -> None:
    await connection.execute(
        "UPDATE application.conversation_sessions SET "
        "updated_at=GREATEST(updated_at,clock_timestamp()) "
        "WHERE user_id=%s AND session_id=%s",
        (user_id, session_id),
    )


def pagination(limit: int, offset: int = 0) -> None:
    if limit < 1 or offset < 0:
        raise ValueError("Pagination requires a positive limit and nonnegative offset.")
