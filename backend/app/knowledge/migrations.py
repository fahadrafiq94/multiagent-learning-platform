from __future__ import annotations

import hashlib
from pathlib import Path

from app.knowledge.database import KnowledgeDatabase
from app.knowledge.errors import KnowledgeIntegrityError

SQL_ROOT = Path(__file__).with_name("sql")


async def migrate(database: KnowledgeDatabase, *, embedding_dimension: int) -> list[str]:
    """Transactional, serialized migrations; only bootstrap-owned knowledge schema is used."""
    if (
        isinstance(embedding_dimension, bool)
        or not isinstance(embedding_dimension, int)
        or not 1 <= embedding_dimension <= 16000
    ):
        raise ValueError("Embedding dimension must be an integer from 1 to 16000.")
    applied: list[str] = []
    paths = sorted(SQL_ROOT.glob("[0-9]*.sql"))
    if not paths:
        raise KnowledgeIntegrityError("Knowledge migration files are missing.")
    async with database.connection() as connection:
        await connection.execute("SELECT pg_advisory_xact_lock(562005)")
        cursor = await connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        )
        if await cursor.fetchone() is None:
            raise KnowledgeIntegrityError("Bootstrap must install the vector extension first.")
        cursor = await connection.execute(
            "SELECT pg_get_userbyid(nspowner) AS owner, "
            "has_schema_privilege(current_user, oid, 'USAGE') AS can_use, "
            "has_schema_privilege(current_user, oid, 'CREATE') AS can_create "
            "FROM pg_namespace WHERE nspname='knowledge'"
        )
        schema = await cursor.fetchone()
        if schema is None:
            raise KnowledgeIntegrityError(
                "Knowledge schema is missing; run the privileged knowledge bootstrap first."
            )
        if schema["owner"] != "fredi_knowledge_writer" or not (
            schema["can_use"] and schema["can_create"]
        ):
            raise KnowledgeIntegrityError(
                "Knowledge schema ownership or migration permissions are invalid; check bootstrap."
            )
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS knowledge.schema_migrations ("
            "version text PRIMARY KEY, checksum text NOT NULL, "
            "embedding_dimension integer NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cursor = await connection.execute("SELECT version FROM knowledge.schema_migrations")
        recorded = {row["version"] for row in await cursor.fetchall()}
        if recorded - {path.stem for path in paths}:
            raise KnowledgeIntegrityError(
                "Applied migration files are missing from this deployment."
            )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(source.encode()).hexdigest()
            cursor = await connection.execute(
                "SELECT checksum, embedding_dimension FROM knowledge.schema_migrations "
                "WHERE version=%s",
                (path.stem,),
            )
            row = await cursor.fetchone()
            if row:
                if row["checksum"] != checksum or row["embedding_dimension"] != embedding_dimension:
                    raise KnowledgeIntegrityError(
                        "Applied migration or embedding dimension changed."
                    )
                continue
            await connection.execute(source.replace("__DIMENSION__", str(embedding_dimension)))
            await connection.execute(
                "INSERT INTO knowledge.schema_migrations VALUES (%s,%s,%s,now())",
                (path.stem, checksum, embedding_dimension),
            )
            applied.append(path.stem)
    return applied
