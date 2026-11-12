"""Measure real Redis hydration separately from model inference, using disposable keys."""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import dotenv_values
from pydantic import SecretStr

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.runtime import create_knowledge_cache
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import PostgresSearchRepository
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.migrate_knowledge import local_dsn
from scripts.run_student_simulation import save_json


class PrivateKeys:
    def __init__(self, cache):
        self.cache = cache
        self.prefix = "fredi:knw:v2:step6:" + uuid4().hex + ":"
        self.keys = set()

    def key(self, value):
        key = self.prefix + value.rsplit(":", 1)[-1]
        self.keys.add(key)
        return key

    async def get(self, key):
        return await self.cache.get(self.key(key))

    async def set(self, key, value, *, ttl_seconds):
        await self.cache.set(self.key(key), value, ttl_seconds=ttl_seconds)

    async def delete(self, key):
        await self.cache.delete(self.key(key))

    async def aclose(self):
        for key in self.keys:
            await self.cache.delete(key)
        remaining = await self.cache._client.exists(*self.keys) if self.keys else 0
        await self.cache.aclose()
        if remaining:
            raise ValueError("Private benchmark cache keys were not removed.")


async def execute(args):
    private_directory(args.output)
    source = json.loads((ROOT / "docs/knowledge/student-journey-step6-source-v1.json").read_bytes())
    values = dotenv_values(ROOT / "backend/.env.knowledge")
    dsn = local_dsn(role="reader")
    profile = settings.model_copy(
        update={
            "knowledge_cache_enabled": True,
            "knowledge_cache_redis_host": "127.0.0.1",
            "knowledge_cache_redis_port": int(values.get("KNOWLEDGE_REDIS_PORT") or "16379"),
            "knowledge_cache_redis_password": SecretStr(values["KNOWLEDGE_REDIS_PASSWORD"]),
        }
    )
    cache = PrivateKeys(create_knowledge_cache(profile, dsn))
    scope = AccessScope(access_domain=source["access_domain"], user_id=uuid4())
    try:
        async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=dsn)) as database:
            plain = PostgresSearchRepository(database)
            cached = PostgresSearchRepository(database, cache=cache)
            pin = await plain.admit(
                UUID(source["collection_id"]), UUID(source["revision_id"]), scope
            )
            if pin is None:
                raise ValueError("Expected published ready revision unavailable.")
            async with database.connection() as connection:
                rows = await (
                    await connection.execute(
                        "SELECT chunk_id FROM knowledge.knowledge_chunks WHERE revision_id=%s "
                        "AND kind='evidence' ORDER BY chunk_id LIMIT 4",
                        (pin.revision.revision_id,),
                    )
                ).fetchall()
            ids = [row["chunk_id"] for row in rows]
            if not ids:
                raise ValueError("Benchmark cannot pass with an empty corpus.")
            expected = await plain.evidence(pin, scope, ids)

            async def timed(repository):
                started = time.perf_counter()
                result = await repository.evidence(pin, scope, ids)
                duration = (time.perf_counter() - started) * 1000
                if result != expected:
                    raise ValueError("Cache response differs from authoritative PostgreSQL.")
                return duration

            disabled = [await timed(plain) for _ in range(10)]
            cold = await timed(cached)
            warm = [await timed(cached) for _ in range(10)]
            started = time.perf_counter()
            concurrent = await asyncio.gather(*(timed(cached) for _ in range(4)))
            batch_ms = (time.perf_counter() - started) * 1000
            denied = await plain.admit(
                UUID(source["collection_id"]),
                UUID(source["revision_id"]),
                AccessScope(access_domain="step6-denied", user_id=uuid4()),
            )
            result = {
                "scope": "Four-chunk hydration/ACL checks; excludes search, HTTP and inference.",
                "source": source,
                "equivalent": True,
                "chunks": len(expected),
                "cache_disabled_ms": disabled,
                "cache_cold_ms": cold,
                "cache_warm_ms": warm,
                "disabled_median_ms": statistics.median(disabled),
                "warm_median_ms": statistics.median(warm),
                "concurrency": 4,
                "concurrent_request_ms": concurrent,
                "concurrent_batch_ms": batch_ms,
                "stats": asdict(cached._coalesced.stats),
                "unauthorized_admission_denied": denied is None,
                "production_flags_changed": False,
            }
            if not result["unauthorized_admission_denied"] or not result["stats"]["hits"]:
                raise ValueError("Scoped real-cache acceptance failed.")
    finally:
        await cache.aclose()
    result["private_cache_keys_removed"] = True
    save_json(args.output / "results.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()
