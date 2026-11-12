"""Provision local app roles/schema, preserving existing knowledge and passwords."""

import asyncio
import sys

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from app.persistence.bootstrap import bootstrap
from app.repositories.exceptions import RepositoryError
from scripts.application_credentials import ensure_credentials
from scripts.knowledge_infra import ROOT, InfrastructureError
from scripts.migrate_knowledge import local_dsn


async def execute() -> None:
    ensure_credentials()
    values = dotenv_values(ROOT / "backend/.env.knowledge")
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(), row_factory=dict_row
    ) as connection:
        await bootstrap(
            connection,
            {
                "migrator": values.get("APPLICATION_MIGRATOR_PASSWORD") or "",
                "runtime": values.get("APPLICATION_RUNTIME_PASSWORD") or "",
            },
        )


def main() -> int:
    try:
        asyncio.run(execute(), loop_factory=asyncio.SelectorEventLoop)
        print("Application roles/schema ready; existing secrets and knowledge preserved.")
        return 0
    except (RepositoryError, InfrastructureError, psycopg.Error, OSError, ValueError):
        print(
            "Application bootstrap failed; check privileged local access and bootstrap "
            "configuration.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
