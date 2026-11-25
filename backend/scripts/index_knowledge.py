"""Index an authorized extracted revision; prints counts/IDs, never source text/vectors."""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.indexer import IndexConfig, index_revision
from app.knowledge.postgres import PostgresKnowledgeRepository
from app.knowledge.schemas import KnowledgeWriteScope
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.service import close_model_services
from scripts.import_knowledge import import_dsn


async def execute(args):
    if args.config.stat().st_size > 1024 * 1024:
        raise ValueError("Index configuration exceeds its size limit.")
    config = IndexConfig.model_validate_json(args.config.read_bytes())
    if config.embedding.provider != "ollama":
        raise ValueError("This sprint supports the configured Ollama embedding provider.")
    job_settings = settings.model_copy(update={"model_request_timeout_seconds": args.timeout})
    model_service = create_model_service(
        job_settings,
        role="embedding",
        legacy_model=config.embedding.model,
        base_url=args.model_url,
    )
    scope = KnowledgeWriteScope(
        access_domain=args.scope_domain,
        user_id=args.owner_user_id,
        can_manage_knowledge=True,
    )

    def progress(done, planned):
        if args.progress:
            print(
                json.dumps({"embedded_chunks": done, "planned_chunks": planned}),
                file=sys.stderr,
                flush=True,
            )

    async with (
        AsyncExitStack() as resources,
        KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=import_dsn())) as database,
    ):
        resources.push_async_callback(close_model_services, model_service)
        return await index_revision(
            args.source_revision,
            config=config,
            repository=PostgresKnowledgeRepository(database),
            model_service=model_service,
            scope=scope,
            progress=progress,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", type=UUID, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope-domain", required=True)
    parser.add_argument("--owner-user-id", type=UUID)
    parser.add_argument("--model-url")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive.")
        result = asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
        print(result.model_dump_json())
        return 0 if result.status == "ready" else 2
    except (KnowledgeError, ModelServiceError, ValueError, OSError, KeyError):
        print(
            "Indexing failed; check source revision, trusted scope, model profile and storage.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
