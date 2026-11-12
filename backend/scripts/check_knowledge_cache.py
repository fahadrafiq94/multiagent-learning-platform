"""Read-only PostgreSQL/real Redis hydration smoke; prints counts, never source text."""

import argparse
import asyncio
import json
from dataclasses import asdict
from uuid import UUID

from dotenv import dotenv_values
from pydantic import SecretStr

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError, KnowledgeUnavailable
from app.knowledge.retrieval_schemas import MetadataRequest
from app.knowledge.runtime import create_knowledge_cache
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import PostgresSearchRepository
from scripts.knowledge_infra import ROOT
from scripts.retrieve_knowledge import retrieval_dsn


async def execute(args):
    local = dotenv_values(ROOT / "backend/.env.knowledge") if args.local else {}
    profile = settings.model_copy(
        update={
            "knowledge_cache_enabled": True,
            "knowledge_cache_redis_host": "127.0.0.1"
            if args.local
            else settings.knowledge_cache_redis_host,
            "knowledge_cache_redis_port": int(local.get("KNOWLEDGE_REDIS_PORT") or "16379")
            if args.local
            else settings.knowledge_cache_redis_port,
            "knowledge_cache_redis_password": SecretStr(local["KNOWLEDGE_REDIS_PASSWORD"])
            if args.local
            else settings.knowledge_cache_redis_password,
        }
    )
    dsn = retrieval_dsn()
    cache = create_knowledge_cache(profile, dsn)
    scope = AccessScope(access_domain=args.scope_domain, user_id=args.owner_user_id)
    try:
        async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=dsn)) as database:
            plain = PostgresSearchRepository(database)
            cached = PostgresSearchRepository(
                database, cache=cache, cache_ttl_seconds=profile.knowledge_cache_ttl_seconds
            )
            pin = await plain.admit(args.collection, args.revision, scope)
            if pin is None:
                raise KnowledgeUnavailable("Requested ready revision is unavailable.")
            async with database.connection() as connection:
                rows = await (
                    await connection.execute(
                        "SELECT chunk_id FROM knowledge.knowledge_chunks WHERE revision_id=%s "
                        "AND kind='evidence' ORDER BY chunk_id LIMIT 2",
                        (pin.revision.revision_id,),
                    )
                ).fetchall()
            ids = [row["chunk_id"] for row in rows]
            expected = await plain.evidence(pin, scope, ids)
            first = await cached.evidence(pin, scope, ids)
            before = asdict(cached._coalesced.stats)
            second = await cached.evidence(pin, scope, ids)
            if expected != first or expected != second:
                raise KnowledgeUnavailable("Cache smoke did not match authoritative evidence.")
            if args.step_id:
                request = MetadataRequest(
                    collection_id=args.collection,
                    step_id=args.step_id,
                    namespace=args.namespace,
                    fields=["name"],
                )
                expected_meta = await plain.metadata(pin, scope, request)
                if await cached.metadata(pin, scope, request) != expected_meta:
                    raise KnowledgeUnavailable("Cached metadata did not match PostgreSQL.")
                if await cached.metadata(pin, scope, request) != expected_meta:
                    raise KnowledgeUnavailable("Warm metadata did not match PostgreSQL.")
            after = asdict(cached._coalesced.stats)
            print(
                json.dumps(
                    {
                        "equivalent": True,
                        "chunks": len(expected),
                        "stats_after_first_chunks": before,
                        "stats_after_warm_and_metadata": after,
                        "warm_hits": after["hits"] - before["hits"],
                        "revision_id": str(pin.revision.revision_id),
                    }
                )
            )
            return 0 if after["hits"] > before["hits"] else 2
    finally:
        await cache.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=UUID, required=True)
    parser.add_argument("--scope-domain", required=True)
    parser.add_argument("--owner-user-id", type=UUID)
    parser.add_argument("--revision", type=UUID)
    parser.add_argument("--step-id")
    parser.add_argument("--namespace")
    parser.add_argument(
        "--local", action="store_true", help="Use existing protected native Redis credentials."
    )
    args = parser.parse_args()
    try:
        return asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
    except (KnowledgeError, ValueError, OSError, KeyError):
        print("Cache check failed; verify read scope, ready revision and protected Redis settings.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
