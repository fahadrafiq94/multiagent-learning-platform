"""Import registered PDFs into staging. Output contains counts/IDs, never PDF text or secrets."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.extraction import PdfExtractor
from app.knowledge.importer import ImportOutcome, ImportRegistration, import_registered
from app.knowledge.postgres import PostgresKnowledgeRepository
from app.knowledge.schemas import KnowledgeWriteScope
from app.knowledge.source_store import FileSourceStore
from scripts.migrate_knowledge import local_dsn


def import_dsn() -> SecretStr:
    if configured := os.environ.get("KNOWLEDGE_INGEST_DATABASE_URL"):
        return SecretStr(configured)
    if host := os.environ.get("KNOWLEDGE_DB_HOST"):
        return SecretStr(
            make_conninfo(
                host=host,
                port="5432",
                dbname="fredi_knowledge",
                user="fredi_knowledge_writer",
                password=os.environ["KNOWLEDGE_WRITER_PASSWORD"],
            )
        )
    return local_dsn()


async def execute(args: argparse.Namespace) -> ImportOutcome:
    if args.registration.stat().st_size > 1024 * 1024:
        raise ValueError("Registration exceeds its size limit.")
    registration = ImportRegistration.model_validate_json(args.registration.read_bytes())
    # This command is an operator boundary, not a student endpoint. Domain/owner
    # metadata comes from the operator-controlled registration file.
    scope = KnowledgeWriteScope(
        access_domain=registration.collection.access_domain,
        user_id=registration.collection.owner_user_id,
        can_manage_knowledge=True,
    )
    async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=import_dsn())) as database:
        return await import_registered(
            registration,
            source_root=args.source_root,
            repository=PostgresKnowledgeRepository(database),
            source_store=FileSourceStore(args.source_store),
            extractor=PdfExtractor(),
            scope=scope,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
        print(result.model_dump_json())
        return 2 if result.failed_documents else 0
    except (KnowledgeError, ValueError, OSError, KeyError):
        print(
            "Import failed; check registration, input files and knowledge storage.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
