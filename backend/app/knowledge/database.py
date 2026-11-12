from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgePermissionError,
    KnowledgeUnavailable,
)


class KnowledgeDatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dsn: SecretStr
    max_size: int = Field(default=4, ge=1, le=32)
    timeout_seconds: float = Field(default=5, gt=0)
    statement_timeout_ms: int = Field(default=2000, gt=0)


class KnowledgeDatabase:
    """Explicit, lazy lifecycle. Constructing this object performs no I/O."""

    def __init__(self, config: KnowledgeDatabaseConfig) -> None:
        self.config = config
        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        async with self._lock:
            if self._closed:
                raise KnowledgeUnavailable("Knowledge database has been closed.")
            if self._pool is not None:
                return
            if type(asyncio.get_running_loop()).__name__ == "ProactorEventLoop":
                raise KnowledgeUnavailable("Psycopg on Windows requires a SelectorEventLoop.")
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
                        "options": f"-c statement_timeout={self.config.statement_timeout_ms}",
                    },
                )
                await pool.open(wait=True, timeout=self.config.timeout_seconds)
            except (psycopg.Error, PoolTimeout, OSError, ValueError):
                if pool is not None:
                    await pool.close()
                raise KnowledgeUnavailable("Knowledge database could not be opened.") from None
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
            pool = self._pool
            if pool is None:
                raise KnowledgeUnavailable("Knowledge database is closed.")
            async with pool.connection() as connection:
                yield connection
        except psycopg.errors.UniqueViolation:
            raise KnowledgeConflict("Knowledge identity or fingerprint already exists.") from None
        except psycopg.errors.InsufficientPrivilege:
            raise KnowledgePermissionError("Knowledge database permission denied.") from None
        except psycopg.IntegrityError:
            raise KnowledgeIntegrityError(
                "Knowledge database constraint rejected the write."
            ) from None
        except (psycopg.Error, PoolTimeout, PoolClosed, OSError):
            raise KnowledgeUnavailable("Knowledge database operation failed.") from None

    async def __aenter__(self) -> KnowledgeDatabase:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
