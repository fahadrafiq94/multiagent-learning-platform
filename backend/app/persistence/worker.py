"""One HTTP worker per database, with restart recovery and transaction fencing."""

from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from app.persistence.database import ApplicationDatabase
from app.repositories.exceptions import RepositoryConflictError, RepositoryUnavailableError


class ApplicationWorker:
    def __init__(self, database: ApplicationDatabase):
        self.database = database
        self.worker_id = uuid4()
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self.recovered_turns = 0

    async def start(self) -> None:
        """Acquire a session lock outside any transaction before recovering old work."""
        try:
            self._connection = await psycopg.AsyncConnection.connect(
                self.database.config.dsn.get_secret_value(),
                autocommit=True,
                row_factory=dict_row,
                connect_timeout=3,
                options="-c statement_timeout=5000",
            )
            cursor = await self._connection.execute(
                "SELECT pg_try_advisory_lock(562007) AS acquired, pg_backend_pid() AS pid"
            )
            result = await cursor.fetchone()
            if not result or not result["acquired"]:
                raise RepositoryConflictError("Another application worker is already running.")
            async with self.database.connection() as connection:
                await connection.execute(
                    "INSERT INTO application.runtime_owner(singleton,worker_id,backend_pid) "
                    "VALUES (true,%s,%s) ON CONFLICT(singleton) DO UPDATE SET "
                    "worker_id=excluded.worker_id,backend_pid=excluded.backend_pid,"
                    "started_at=clock_timestamp()",
                    (self.worker_id, result["pid"]),
                )
                cursor = await connection.execute(
                    "UPDATE application.turns SET status='interrupted',"
                    "error_code='worker_restarted',"
                    "updated_at=clock_timestamp() WHERE status='processing'"
                )
                self.recovered_turns = cursor.rowcount
            self.database.worker_id = self.worker_id
        except BaseException as exc:
            await self.close()
            if isinstance(exc, (psycopg.Error, OSError)):
                raise RepositoryUnavailableError("Application worker could not start.") from None
            raise

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
