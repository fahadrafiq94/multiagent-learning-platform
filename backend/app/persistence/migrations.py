from __future__ import annotations

import hashlib
from pathlib import Path

from app.persistence.database import ApplicationDatabase
from app.repositories.exceptions import RepositoryIntegrityError

SQL_ROOT = Path(__file__).with_name("sql")


async def migrate(database: ApplicationDatabase) -> list[str]:
    """Application-only migrations; no dependency on pgvector or knowledge dimensions."""
    paths = sorted(SQL_ROOT.glob("[0-9]*.sql"))
    if not paths:
        raise RepositoryIntegrityError("Application migration files are missing.")
    applied = []
    async with database.connection() as connection:
        await connection.execute("SELECT pg_advisory_xact_lock(562006)")
        cursor = await connection.execute(
            "SELECT pg_get_userbyid(nspowner) AS owner, current_user AS caller, "
            "has_schema_privilege(current_user, oid, 'CREATE') AS can_create "
            "FROM pg_namespace WHERE nspname='application'"
        )
        schema = await cursor.fetchone()
        if schema is None:
            raise RepositoryIntegrityError(
                "Application schema is missing; run application bootstrap."
            )
        if (
            schema["owner"] != "fredi_app_migrator"
            or not schema["can_create"]
            or schema["caller"] != "fredi_app_migrator"
        ):
            raise RepositoryIntegrityError(
                "Application migrations require the bootstrap migrator role."
            )
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS application.schema_migrations ("
            "version text PRIMARY KEY, checksum text NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cursor = await connection.execute(
            "SELECT version,checksum FROM application.schema_migrations"
        )
        recorded = {row["version"]: row["checksum"] for row in await cursor.fetchall()}
        if set(recorded) - {path.stem for path in paths}:
            raise RepositoryIntegrityError("Applied application migration files are missing.")
        for path in paths:
            content = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(content.encode()).hexdigest()
            if path.stem in recorded:
                if recorded[path.stem] != checksum:
                    raise RepositoryIntegrityError(
                        "Applied application migration checksum changed."
                    )
                continue
            await connection.execute(content)
            await connection.execute(
                "INSERT INTO application.schema_migrations(version,checksum) VALUES (%s,%s)",
                (path.stem, checksum),
            )
            applied.append(path.stem)
    return applied
