"""Atomic durable conversation/session-context initialization."""

from app.context import ConversationSession, SessionContext
from app.persistence.database import ApplicationDatabase
from app.persistence.records import insert, require_user


class PostgresSessionInitializer:
    def __init__(self, database: ApplicationDatabase):
        self.database = database

    async def __call__(self, session: ConversationSession) -> ConversationSession:
        async with self.database.connection() as connection:
            await require_user(connection, session.user_id)
            stored = await insert(connection, session)
            await insert(
                connection,
                SessionContext(
                    user_id=session.user_id,
                    session_id=session.session_id,
                ),
            )
            return stored
