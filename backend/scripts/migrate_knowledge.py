"""Apply knowledge migrations; no ingestion, activation, or student routing changes."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError, KnowledgeIntegrityError
from app.knowledge.migrations import migrate

ROOT = Path(__file__).resolve().parents[2]


def local_dsn(*, role: str = "writer", database: str = "fredi_knowledge") -> SecretStr:
    """Local Docker convenience only; never expose generated credentials in output."""
    if role not in {"admin", "writer", "reader"}:
        raise ValueError("Unknown database role.")
    values = dotenv_values(ROOT / "backend/.env.knowledge")
    key = "KNOWLEDGE_POSTGRES_PASSWORD" if role == "admin" else f"KNOWLEDGE_{role.upper()}_PASSWORD"
    password = values.get(key)
    if not password:
        raise KnowledgeError("Local knowledge credentials are missing.")
    return SecretStr(
        make_conninfo(
            host="127.0.0.1",
            port=values.get("KNOWLEDGE_POSTGRES_PORT") or "15432",
            user=f"fredi_knowledge_{role}",
            password=password,
            dbname=database,
        )
    )


async def execute(dimension: int) -> list[str]:
    configured = os.environ.get("KNOWLEDGE_INGEST_DATABASE_URL")
    dsn = SecretStr(configured) if configured else local_dsn()
    async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=dsn)) as database:
        return await migrate(database, embedding_dimension=dimension)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dimension", required=True, type=int)
    args = parser.parse_args()
    try:
        applied = asyncio.run(
            execute(args.embedding_dimension), loop_factory=asyncio.SelectorEventLoop
        )
        print(
            "Applied: " + ", ".join(applied) if applied else "Knowledge migrations already current."
        )
        return 0
    except KnowledgeIntegrityError as exc:
        print(f"Knowledge migration failed: {exc}", file=sys.stderr)
        return 1
    except (KnowledgeError, ValueError):
        print(
            "Knowledge migration failed; check configuration, schema version, and dimension.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
