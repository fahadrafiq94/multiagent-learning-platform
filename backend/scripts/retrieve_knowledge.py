"""Read-only operator retrieval; student context integration belongs to later checkpoints."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import (
    MetadataRequest,
    RetrievalConfig,
    RetrievalRequest,
    RetrievalResult,
)
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import PostgresSearchRepository
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from scripts.migrate_knowledge import local_dsn


def retrieval_dsn() -> SecretStr:
    if configured := os.environ.get("KNOWLEDGE_RETRIEVAL_DATABASE_URL"):
        return SecretStr(configured)
    if host := os.environ.get("KNOWLEDGE_DB_HOST"):
        return SecretStr(
            make_conninfo(
                host=host,
                port="5432",
                dbname="fredi_knowledge",
                user="fredi_knowledge_reader",
                password=os.environ["KNOWLEDGE_READER_PASSWORD"],
            )
        )
    return local_dsn(role="reader")


async def execute(args) -> RetrievalResult:
    config = RetrievalConfig()
    if args.config is not None:
        if args.config.stat().st_size > 1024 * 1024:
            raise ValueError("Retrieval configuration exceeds its limit.")
        config = RetrievalConfig.model_validate_json(args.config.read_bytes())
    scope = AccessScope(access_domain=args.scope_domain, user_id=args.owner_user_id)
    async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=retrieval_dsn())) as database:
        repository = PostgresSearchRepository(database)
        if args.step_id is not None:
            return await KnowledgeRetrievalService(repository, config=config).lookup(
                MetadataRequest(
                    collection_id=args.collection,
                    revision_id=args.revision,
                    namespace=args.namespace,
                    step_id=args.step_id,
                    max_chars=args.max_chars,
                ),
                scope=scope,
            )
        pin = await repository.admit(args.collection, args.revision, scope)
        if pin is None:
            return RetrievalResult(
                status="unavailable", reason="revision_unavailable", collection_id=args.collection
            )
        profile = pin.revision.embedding
        job_settings = settings.model_copy(
            update={
                "model_provider": profile.provider,
                "ollama_base_url": args.model_url or settings.ollama_base_url,
                "ollama_embedding_model": profile.model,
                "ollama_chat_model": settings.ollama_chat_model or profile.model,
                "model_request_timeout_seconds": args.timeout,
            }
        )
        service = KnowledgeRetrievalService(repository, create_model_service(job_settings), config)
        return await service.search(
            RetrievalRequest(
                collection_id=args.collection,
                revision_id=args.revision,
                query=args.query,
                current_step_id=args.current_step,
                namespace=args.namespace,
                phase=args.phase,
                purpose=args.purpose,
                top_k=args.top_k,
                max_chars=args.max_chars,
            ),
            scope=scope,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=UUID, required=True)
    parser.add_argument(
        "--revision",
        type=UUID,
        help="Trusted explicit ready revision; otherwise use active revision.",
    )
    parser.add_argument("--scope-domain", required=True)
    parser.add_argument("--owner-user-id", type=UUID)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--query")
    operation.add_argument("--step-id")
    parser.add_argument("--namespace")
    parser.add_argument("--current-step")
    parser.add_argument("--phase")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--model-url")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--include-evidence", action="store_true", help="Include source quotes in operator output."
    )
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive.")
        result = asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
        payload = result.model_dump(mode="json")
        if not args.include_evidence:
            for item in payload["items"]:
                item.pop("content")
                item.pop("context_prefix")
        import json

        print(json.dumps(payload))
        return {"found": 0, "no_evidence": 2, "unavailable": 3, "error": 1}[result.status]
    except (KnowledgeError, ModelServiceError, ValueError, OSError, KeyError):
        print(
            "Retrieval failed; verify trusted scope, revision and read/model configuration.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
