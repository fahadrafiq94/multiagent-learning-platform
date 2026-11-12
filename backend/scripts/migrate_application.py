"""Apply app migrations using the dedicated migrator, not runtime/knowledge roles."""

import asyncio
import sys

from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.persistence.migrations import migrate
from app.repositories.exceptions import RepositoryError
from scripts.application_credentials import configured_dsn


async def execute() -> list[str]:
    async with ApplicationDatabase(
        ApplicationDatabaseConfig(dsn=configured_dsn(role="migrator"))
    ) as database:
        return await migrate(database)


def main() -> int:
    try:
        applied = asyncio.run(execute(), loop_factory=asyncio.SelectorEventLoop)
        print(
            "Applied: " + ", ".join(applied)
            if applied
            else "Application migrations already current."
        )
        return 0
    except (RepositoryError, OSError, ValueError):
        print(
            "Application migration failed; check migrator credentials, schema bootstrap "
            "and migration files.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
