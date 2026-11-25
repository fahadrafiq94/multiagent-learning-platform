from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from dotenv import dotenv_values
from psycopg import sql
from psycopg.rows import dict_row

from app.persistence.bootstrap import bootstrap
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.persistence.migrations import migrate
from app.repositories.in_memory import create_in_memory_repositories
from app.repositories.postgres import create_postgres_repositories
from scripts.application_credentials import local_application_dsn
from scripts.knowledge_infra import ROOT
from scripts.migrate_knowledge import local_dsn


def pytest_asyncio_loop_factories():
    return {"selector": asyncio.SelectorEventLoop}


@dataclass
class PostgresFixture:
    name: str
    database: ApplicationDatabase
    migrator: ApplicationDatabase

    @property
    def repositories(self):
        return create_postgres_repositories(self.database)


@pytest_asyncio.fixture
async def pg():
    if os.environ.get("FREDI_APPLICATION_REPOSITORY_INTEGRATION") != "1":
        pytest.skip("Opt-in: isolated real application PostgreSQL databases.")
    name = "fredi_app_test_" + uuid4().hex[:16]
    values = dotenv_values(ROOT / "backend/.env.knowledge")
    database = ApplicationDatabase(
        ApplicationDatabaseConfig(dsn=local_application_dsn(database=name))
    )
    migrator = ApplicationDatabase(
        ApplicationDatabaseConfig(dsn=local_application_dsn(role="migrator", database=name))
    )
    admin = await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(), autocommit=True
    )
    created = False
    try:
        await admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER fredi_app_migrator").format(sql.Identifier(name))
        )
        created = True
        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="admin", database=name).get_secret_value(), row_factory=dict_row
        ) as connection:
            await bootstrap(
                connection,
                {
                    "migrator": values.get("APPLICATION_MIGRATOR_PASSWORD") or "",
                    "runtime": values.get("APPLICATION_RUNTIME_PASSWORD") or "",
                },
            )
        assert await migrate(migrator) == [
            "001_application",
            "002_runtime",
            "003_session_deletion",
            "004_message_editing",
            "005_message_regeneration",
        ]
        yield PostgresFixture(name, database, migrator)
    finally:
        await database.close()
        await migrator.close()
        if created:
            await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        await admin.close()


@pytest.fixture(params=["memory", "postgres"])
def repositories(request):
    if request.param == "memory":
        return create_in_memory_repositories()
    return request.getfixturevalue("pg").repositories
