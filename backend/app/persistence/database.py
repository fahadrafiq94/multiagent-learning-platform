from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.repositories.exceptions import (
    DuplicateRecordError,
    RepositoryConflictError,
    RepositoryIntegrityError,
    RepositoryPermissionError,
    RepositoryUnavailableError,
)


class ApplicationDatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dsn: SecretStr
    max_size: int = Field(default=4, ge=1, le=32)
    timeout_seconds: float = Field(default=5, gt=0)
    statement_timeout_ms: int = Field(default=5000, gt=0)


class ApplicationDatabase:
    """Lazy async pool. Each connection context is a short atomic transaction."""

    def __init__(self, config: ApplicationDatabaseConfig):
        self.config = config
        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]] | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self.worker_id: UUID | None = None

    async def start(self) -> None:
        async with self._lock:
            if self._closed:
                raise RepositoryUnavailableError("Application database is closed.")
            if self._pool is not None:
                return
            if type(asyncio.get_running_loop()).__name__ == "ProactorEventLoop":
                raise RepositoryUnavailableError("Psycopg on Windows requires a SelectorEventLoop.")
            pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]] | None = None
            try:
                pool = AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]](
                    self.config.dsn.get_secret_value(),
                    open=False,
                    min_size=1,
                    max_size=self.config.max_size,
                    timeout=self.config.timeout_seconds,
                    reconnect_timeout=self.config.timeout_seconds,
                    kwargs={
                        "row_factory": dict_row,
                        "connect_timeout": 3,
                        "options": "-c timezone=UTC "
                        f"-c statement_timeout={self.config.statement_timeout_ms}",
                    },
                )
                await pool.open(wait=True, timeout=self.config.timeout_seconds)
            except (psycopg.Error, PoolTimeout, OSError, ValueError):
                if pool is not None:
                    await pool.close()
                raise RepositoryUnavailableError(
                    "Application database could not be opened."
                ) from None
            except BaseException:
                if pool is not None:
                    await pool.close()
                raise
            self._pool = pool

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._pool is not None:
                await self._pool.close()
                self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection[dict[str, Any]]]:
        await self.start()
        try:
            if self._pool is None:
                raise RepositoryUnavailableError("Application database is closed.")
            async with self._pool.connection() as connection:
                if self.worker_id is not None:
                    cursor = await connection.execute(
                        "SELECT worker_id, EXISTS(SELECT 1 FROM pg_locks WHERE "
                        "locktype='advisory' AND classid=0 AND objid=562007 AND "
                        "objsubid=1 AND granted AND pid=owner.backend_pid) AS owns_lock "
                        "FROM application.runtime_owner AS owner WHERE singleton FOR SHARE"
                    )
                    owner = await cursor.fetchone()
                    if not owner or owner["worker_id"] != self.worker_id or not owner["owns_lock"]:
                        raise RepositoryUnavailableError(
                            "Application worker no longer owns runtime access."
                        )
                yield connection
        except psycopg.errors.UniqueViolation as exc:
            if exc.diag.constraint_name == "one_processing_turn":
                raise RepositoryConflictError(
                    "Another turn is processing in this conversation."
                ) from None
            raise DuplicateRecordError(
                entity_name="ApplicationRecord", key="unique identity"
            ) from None
        except psycopg.errors.InsufficientPrivilege:
            raise RepositoryPermissionError("Application database permission denied.") from None
        except psycopg.IntegrityError:
            raise RepositoryIntegrityError(
                "Application database rejected an invalid relationship."
            ) from None
        except (psycopg.Error, PoolTimeout, PoolClosed, OSError):
            raise RepositoryUnavailableError("Application database operation failed.") from None

    async def __aenter__(self) -> ApplicationDatabase:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
