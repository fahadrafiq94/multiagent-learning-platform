"""PostgreSQL implementations of the five existing application storage contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from app.context import ConversationMessage, ConversationSession, SessionContext, UserContext
from app.identity import IdentityProvider, UserIdentity
from app.persistence.database import ApplicationDatabase
from app.persistence.records import (
    Connection,
    checked,
    decode,
    insert,
    pagination,
    require_session,
    require_user,
    touch_session,
)
from app.persistence.sessions import PostgresSessionInitializer
from app.persistence.turns import PostgresTurnStore
from app.repositories.exceptions import RepositoryNotFoundError, VersionConflictError


class PostgresUserIdentityRepository:
    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def get_by_user_id(self, user_id: UUID) -> UserIdentity | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM application.identities WHERE user_id=%s", (user_id,)
            )
            row = await cursor.fetchone()
            return decode(UserIdentity, row) if row else None

    async def get_by_student_id(self, student_id: str) -> UserIdentity | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM application.identities WHERE student_id=%s", (student_id.strip(),)
            )
            row = await cursor.fetchone()
            return decode(UserIdentity, row) if row else None

    async def get_by_external_identity(
        self, *, provider: IdentityProvider, provider_instance: str, external_user_id: str
    ) -> UserIdentity | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT i.* FROM application.identities i JOIN "
                "application.external_identities e USING(user_id) "
                "WHERE e.provider=%s AND e.provider_instance=%s AND e.external_user_id=%s",
                (provider.value, provider_instance.strip(), external_user_id.strip()),
            )
            row = await cursor.fetchone()
            return decode(UserIdentity, row) if row else None

    async def _links(self, connection: Connection, identity: UserIdentity) -> None:
        for link in identity.external_identities:
            await connection.execute(
                "INSERT INTO application.external_identities VALUES (%s,%s,%s,%s)",
                (
                    link.provider.value,
                    link.provider_instance,
                    link.external_user_id,
                    identity.user_id,
                ),
            )

    async def create(self, identity: UserIdentity) -> UserIdentity:
        identity = checked(identity)
        async with self.database.connection() as connection:
            stored = await insert(connection, identity)
            await self._links(connection, identity)
            return stored

    async def update(self, identity: UserIdentity) -> UserIdentity:
        identity = checked(identity)
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM application.identities WHERE user_id=%s FOR UPDATE",
                (identity.user_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RepositoryNotFoundError(
                    entity_name="UserIdentity", entity_id=str(identity.user_id)
                )
            stored = identity.model_copy(
                update={"created_at": row["created_at"], "updated_at": datetime.now(UTC)}
            )
            cursor = await connection.execute(
                "UPDATE application.identities SET "
                "student_id=%s,status=%s,updated_at=%s,payload=%s "
                "WHERE user_id=%s RETURNING *",
                (
                    stored.student_id,
                    stored.status.value,
                    stored.updated_at,
                    Jsonb(stored.model_dump(mode="json")),
                    stored.user_id,
                ),
            )
            await connection.execute(
                "DELETE FROM application.external_identities WHERE user_id=%s", (stored.user_id,)
            )
            await self._links(connection, stored)
            return decode(UserIdentity, await cursor.fetchone())


class _ContextRepository[T: (UserContext, SessionContext)]:
    model: type[T]
    table: str

    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def _parent(
        self, connection: Connection, user_id: UUID, session_id: UUID | None = None
    ) -> None:
        if session_id is None:
            await require_user(connection, user_id)
        else:
            await require_session(connection, user_id, session_id)

    def _where(
        self, user_id: UUID, session_id: UUID | None = None
    ) -> tuple[sql.SQL, tuple[UUID, ...]]:
        if session_id is None:
            return sql.SQL("user_id=%s"), (user_id,)
        return sql.SQL("user_id=%s AND session_id=%s"), (user_id, session_id)

    async def _get(self, user_id: UUID, session_id: UUID | None = None) -> T | None:
        where, values = self._where(user_id, session_id)
        async with self.database.connection() as connection:
            await self._parent(connection, user_id, session_id)
            cursor = await connection.execute(
                sql.SQL("SELECT * FROM application.{} WHERE {}").format(
                    sql.Identifier(self.table), where
                ),
                values,
            )
            row = await cursor.fetchone()
            return decode(self.model, row) if row else None

    async def create(self, context: T) -> T:
        context = checked(context)
        async with self.database.connection() as connection:
            await self._parent(connection, context.user_id, getattr(context, "session_id", None))
            return await insert(connection, context)

    async def save(self, context: T, *, expected_version: int) -> T:
        context = checked(context)
        values: tuple[UUID, ...]
        where, values = self._where(context.user_id, getattr(context, "session_id", None))
        async with self.database.connection() as connection:
            await self._parent(connection, context.user_id, getattr(context, "session_id", None))
            cursor = await connection.execute(
                sql.SQL("SELECT * FROM application.{} WHERE {} FOR UPDATE").format(
                    sql.Identifier(self.table), where
                ),
                values,
            )
            row = await cursor.fetchone()
            identifier = str(getattr(context, "session_id", context.user_id))
            if row is None:
                raise RepositoryNotFoundError(entity_name=self.model.__name__, entity_id=identifier)
            if row["version"] != expected_version:
                raise VersionConflictError(
                    entity_name=self.model.__name__,
                    entity_id=identifier,
                    expected_version=expected_version,
                    actual_version=row["version"],
                )
            updates = {"version": row["version"] + 1, "updated_at": datetime.now(UTC)}
            if isinstance(context, UserContext):
                updates["created_at"] = row["created_at"]
            stored = context.model_copy(update=updates, deep=True)
            cursor = await connection.execute(
                sql.SQL(
                    "UPDATE application.{} SET version=%s,updated_at=%s,payload=%s WHERE "
                    "{} RETURNING *"
                ).format(sql.Identifier(self.table), where),
                (stored.version, stored.updated_at, Jsonb(stored.model_dump(mode="json")), *values),
            )
            return decode(self.model, await cursor.fetchone())


class PostgresUserContextRepository(_ContextRepository[UserContext]):
    model = UserContext
    table = "user_contexts"

    async def get(self, user_id: UUID) -> UserContext | None:
        return await self._get(user_id)


class PostgresSessionContextRepository(_ContextRepository[SessionContext]):
    model = SessionContext
    table = "session_contexts"

    async def get(self, *, user_id: UUID, session_id: UUID) -> SessionContext | None:
        return await self._get(user_id, session_id)


class PostgresConversationSessionRepository:
    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def create(self, session: ConversationSession) -> ConversationSession:
        session = checked(session)
        async with self.database.connection() as connection:
            await require_user(connection, session.user_id)
            return await insert(connection, session)

    async def get(self, *, user_id: UUID, session_id: UUID) -> ConversationSession | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM application.conversation_sessions WHERE user_id=%s AND "
                "session_id=%s",
                (user_id, session_id),
            )
            row = await cursor.fetchone()
            return decode(ConversationSession, row) if row else None

    async def list_for_user(
        self, user_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> list[ConversationSession]:
        pagination(limit, offset)
        async with self.database.connection() as connection:
            await require_user(connection, user_id)
            cursor = await connection.execute(
                "SELECT * FROM application.conversation_sessions WHERE user_id=%s "
                "ORDER BY updated_at DESC,session_id LIMIT %s OFFSET %s",
                (user_id, limit, offset),
            )
            return [decode(ConversationSession, row) for row in await cursor.fetchall()]

    async def update(self, session: ConversationSession) -> ConversationSession:
        session = checked(session)
        async with self.database.connection() as connection:
            await require_user(connection, session.user_id)
            current = await require_session(
                connection, session.user_id, session.session_id, lock=True
            )
            stored = session.model_copy(
                update={
                    "created_at": current["created_at"],
                    "updated_at": max(current["updated_at"], datetime.now(UTC)),
                }
            )
            cursor = await connection.execute(
                "UPDATE application.conversation_sessions SET "
                "title=%s,status=%s,updated_at=%s,payload=%s "
                "WHERE user_id=%s AND session_id=%s RETURNING *",
                (
                    stored.title,
                    stored.status.value,
                    stored.updated_at,
                    Jsonb(stored.model_dump(mode="json")),
                    stored.user_id,
                    stored.session_id,
                ),
            )
            return decode(ConversationSession, await cursor.fetchone())


class PostgresConversationRepository:
    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def append(self, message: ConversationMessage) -> ConversationMessage:
        message = checked(message)
        async with self.database.connection() as connection:
            await require_session(connection, message.user_id, message.session_id, lock=True)
            stored = await insert(connection, message)
            await touch_session(connection, message.user_id, message.session_id)
            return stored

    async def list_for_session(
        self, *, user_id: UUID, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[ConversationMessage]:
        pagination(limit, offset)
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id)
            cursor = await connection.execute(
                "SELECT * FROM application.conversation_messages WHERE user_id=%s AND "
                "session_id=%s "
                "ORDER BY created_at,sequence_no LIMIT %s OFFSET %s",
                (user_id, session_id, limit, offset),
            )
            return [decode(ConversationMessage, row) for row in await cursor.fetchall()]

    async def list_recent(
        self, *, user_id: UUID, session_id: UUID, limit: int = 10
    ) -> list[ConversationMessage]:
        pagination(limit)
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id)
            cursor = await connection.execute(
                "SELECT * FROM application.conversation_messages WHERE user_id=%s AND "
                "session_id=%s "
                "ORDER BY created_at DESC,sequence_no DESC LIMIT %s",
                (user_id, session_id, limit),
            )
            return [decode(ConversationMessage, row) for row in reversed(await cursor.fetchall())]

    async def count(self, *, user_id: UUID, session_id: UUID) -> int:
        async with self.database.connection() as connection:
            await require_session(connection, user_id, session_id)
            cursor = await connection.execute(
                "SELECT count(*) AS total FROM application.conversation_messages WHERE "
                "user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            row = await cursor.fetchone()
            return int(row["total"]) if row else 0


@dataclass(slots=True)
class PostgresRepositories:
    identities: PostgresUserIdentityRepository
    user_contexts: PostgresUserContextRepository
    sessions: PostgresConversationSessionRepository
    session_contexts: PostgresSessionContextRepository
    conversations: PostgresConversationRepository
    turns: PostgresTurnStore
    session_initializer: PostgresSessionInitializer


def create_postgres_repositories(database: ApplicationDatabase) -> PostgresRepositories:
    return PostgresRepositories(
        PostgresUserIdentityRepository(database),
        PostgresUserContextRepository(database),
        PostgresConversationSessionRepository(database),
        PostgresSessionContextRepository(database),
        PostgresConversationRepository(database),
        PostgresTurnStore(database),
        PostgresSessionInitializer(database),
    )
