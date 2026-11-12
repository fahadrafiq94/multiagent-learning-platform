"""Actual Redis + scoped read-role PostgreSQL; unique test keys, never FLUSHDB."""

import asyncio
import hashlib
import os
import subprocess
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from dotenv import dotenv_values
from redis.exceptions import RedisError

from app.knowledge.cache import RedisKnowledgeCache
from app.knowledge.database import KnowledgeDatabase
from app.knowledge.errors import KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import MetadataRequest, RetrievalRequest
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import PostgresSearchRepository
from scripts.knowledge_infra import ROOT
from tests.knowledge.test_retrieval import QueryModel, corpus, search_request

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("FREDI_KNOWLEDGE_REDIS_INTEGRATION") != "1",
        reason="Opt-in actual Redis and isolated PostgreSQL.",
    ),
]


class ScopedRedis:
    def __init__(self, cache):
        self.cache = cache
        self.prefix = "fredi:test:cp10:" + uuid4().hex + ":"
        self.keys = set()

    def native(self, key):
        result = self.prefix + key
        self.keys.add(result)
        return result

    async def get(self, key):
        return await self.cache.get(self.native(key))

    async def set(self, key, value, *, ttl_seconds):
        await self.cache.set(self.native(key), value, ttl_seconds=ttl_seconds)

    async def delete(self, key):
        await self.cache.delete(self.native(key))

    async def aclose(self):
        await self.cache.aclose()


@pytest_asyncio.fixture
async def redis_cache():
    env = dotenv_values(ROOT / "backend/.env.knowledge")
    backend = RedisKnowledgeCache(
        host="127.0.0.1",
        port=int(env.get("KNOWLEDGE_REDIS_PORT") or "16379"),
        password=env["KNOWLEDGE_REDIS_PASSWORD"],
        operation_timeout_seconds=0.5,
        signing_key=hashlib.sha256(uuid4().bytes).digest(),
    )
    cache = ScopedRedis(backend)
    try:
        assert await backend._client.ping()
        yield cache
    finally:
        if cache.keys:
            await backend._client.delete(*cache.keys)
        await cache.aclose()


async def repositories(pg, cache, **kwargs):
    collection, revision, _, scope = await corpus(pg, **kwargs)
    plain = PostgresSearchRepository(pg.reader_database)
    cached = PostgresSearchRepository(pg.reader_database, cache=cache)
    pin = await cached.admit(collection.collection_id, revision, scope)
    return collection, scope, pin, plain, cached


async def test_cached_search_and_metadata_equal_uncached_and_warm_calls_are_hits(
    postgres_fixture, redis_cache
):
    pg = postgres_fixture
    collection, scope, pin, plain, cached = await repositories(pg, redis_cache)
    query = search_request(
        collection, pin.revision.revision_id, current_step_id="FLOW-1", purpose="business reason"
    )
    expected = await KnowledgeRetrievalService(plain, QueryModel()).search(query, scope=scope)
    service = KnowledgeRetrievalService(cached, QueryModel())
    assert await service.search(query, scope=scope) == expected
    fills = cached._coalesced.stats.fills
    assert await service.search(query, scope=scope) == expected
    assert cached._coalesced.stats.hits and cached._coalesced.stats.fills == fills
    request = MetadataRequest(
        collection_id=collection.collection_id,
        revision_id=pin.revision.revision_id,
        step_id="FLOW-2",
        fields=["name", "prerequisites"],
    )
    expected_meta = await plain.metadata(pin, scope, request)
    assert expected_meta[0]
    assert await cached.metadata(pin, scope, request) == expected_meta
    fills = cached._coalesced.stats.fills
    assert await cached.metadata(pin, scope, request) == expected_meta
    assert cached._coalesced.stats.fills == fills


async def test_shared_hydration_reuses_across_students_but_personalized_search_is_separate(
    postgres_fixture, redis_cache
):
    collection, scope, pin, plain, cached = await repositories(postgres_fixture, redis_cache)
    a, b = (
        scope.model_copy(update={"user_id": uuid4()}),
        scope.model_copy(update={"user_id": uuid4()}),
    )
    request = search_request(collection, pin.revision.revision_id)
    candidates = await plain.candidates(pin, a, request, [0.2, 0.3, 0.4], [], 10)
    ids = [c.chunk_id for c in candidates]
    expected = await cached.evidence(pin, a, ids)
    fills = cached._coalesced.stats.fills
    assert await cached.evidence(pin, b, ids) == expected
    assert cached._coalesced.stats.fills == fills
    await cached.candidates(pin, a, request, [0.2, 0.3, 0.4], [], 10)
    await cached.candidates(pin, b, request, [0.2, 0.3, 0.4], [], 10)
    assert cached._coalesced.stats.fills == fills + 2


async def test_foreign_owner_domain_and_revoked_access_never_reuse_warm_evidence(
    postgres_fixture, redis_cache
):
    pg = postgres_fixture
    collection, scope, pin, plain, cached = await repositories(pg, redis_cache, owner=uuid4())
    request = search_request(collection, pin.revision.revision_id)
    ids = [c.chunk_id for c in await plain.candidates(pin, scope, request, [0.2, 0.3, 0.4], [], 10)]
    assert await cached.evidence(pin, scope, ids)
    for denied in (
        AccessScope(access_domain=scope.access_domain, user_id=uuid4()),
        AccessScope(access_domain="other", user_id=scope.user_id),
    ):
        with pytest.raises(KnowledgeNotFound):
            await cached.evidence(pin, denied, ids)
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
            "WHERE collection_id=%s",
            (collection.collection_id,),
        )
    with pytest.raises(KnowledgeNotFound):
        await cached.evidence(pin, scope, ids)


async def test_corrupt_bytes_and_valid_but_wrong_evidence_are_refilled(
    postgres_fixture, redis_cache, monkeypatch
):
    collection, scope, pin, plain, cached = await repositories(postgres_fixture, redis_cache)
    candidates = await plain.candidates(
        pin, scope, search_request(collection, pin.revision.revision_id), [0.2, 0.3, 0.4], [], 10
    )
    chunk_id = candidates[0].chunk_id
    expected = await cached.evidence(pin, scope, [chunk_id])
    key = cached._key(scope, pin, f"chunk:{chunk_id}")
    native = redis_cache.native(key)
    await redis_cache.cache._client.set(native, b'{"valid":"but unsigned"}')
    assert await cached.evidence(pin, scope, [chunk_id]) == expected
    wrong = expected[0].model_copy(update={"evidence_id": uuid4()})
    await redis_cache.set(key, wrong.model_dump_json().encode(), ttl_seconds=60)
    fetch = AsyncMock(wraps=cached._fetch_chunk)
    monkeypatch.setattr(cached, "_fetch_chunk", fetch)
    assert await cached.evidence(pin, scope, [chunk_id]) == expected
    assert fetch.await_count == 1
    await redis_cache.set(key, b"not-json", ttl_seconds=60)
    assert await cached.evidence(pin, scope, [chunk_id]) == expected


async def test_revision_version_and_query_context_change_keys(postgres_fixture, redis_cache):
    pg = postgres_fixture
    collection, scope, pin, _, cached = await repositories(pg, redis_cache)
    request = search_request(collection, pin.revision.revision_id)
    original = cached._key(scope, pin, "chunk:test")
    changed = pin.model_copy(
        update={"revision": pin.revision.model_copy(update={"version": pin.revision.version + 1})}
    )
    assert cached._key(scope, changed, "chunk:test") != original
    base = cached._search_digest(request, [], [0.2, 0.3, 0.4], 10)
    for updates in (
        {"query": "other"},
        {"purpose": "navigation"},
        {"namespace": "different"},
        {"phase": "different"},
        {"current_step_id": "FLOW-2"},
    ):
        assert (
            cached._search_digest(request.model_copy(update=updates), [], [0.2, 0.3, 0.4], 10)
            != base
        )
    # Ready snapshots are immutable via the repository. Inject a maintenance
    # version change in this scratch database to exercise the defensive cache key.
    data = await pg.writer.get_revision_data(pin.revision.revision_id, scope)
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_revisions SET version=version+1 WHERE revision_id=%s",
            (pin.revision.revision_id,),
        )
    with pytest.raises(KnowledgeIntegrityError):
        await cached.evidence(pin, scope, [data.chunks[0].chunk_id])
    current = await cached.admit(collection.collection_id, pin.revision.revision_id, scope)
    assert cached._key(scope, current, "chunk:test") != original


async def test_metadata_works_with_one_reader_connection(postgres_fixture, redis_cache):
    pg = postgres_fixture
    collection, scope, pin, plain, _ = await repositories(pg, redis_cache)
    single = KnowledgeDatabase(pg.reader_database.config.model_copy(update={"max_size": 1}))
    try:
        cached = PostgresSearchRepository(single, cache=redis_cache)
        request = MetadataRequest(
            collection_id=collection.collection_id, step_id="FLOW-2", fields=["name"]
        )
        expected = await plain.metadata(pin, scope, request)
        assert await asyncio.wait_for(cached.metadata(pin, scope, request), timeout=3) == expected
    finally:
        await single.close()


async def test_real_redis_expiry_and_client_restart_are_disposable(redis_cache):
    await redis_cache.set("ephemeral", b"value", ttl_seconds=1)
    assert await redis_cache.get("ephemeral") == b"value"
    assert 0 < await redis_cache.cache._client.pttl(redis_cache.native("ephemeral")) <= 1000
    await redis_cache.cache._client.connection_pool.disconnect()
    assert await redis_cache.get("ephemeral") == b"value"
    await asyncio.sleep(1.1)
    assert await redis_cache.get("ephemeral") is None


async def test_revocation_during_a_cache_hit_is_checked_again_before_return(
    postgres_fixture, redis_cache, monkeypatch
):
    pg = postgres_fixture
    collection, scope, pin, plain, cached = await repositories(pg, redis_cache)
    request = search_request(collection, pin.revision.revision_id)
    ids = [c.chunk_id for c in await plain.candidates(pin, scope, request, [0.2, 0.3, 0.4], [], 10)]
    await cached.evidence(pin, scope, ids)
    get = redis_cache.get
    revoked = False

    async def revoke(key):
        nonlocal revoked
        value = await get(key)
        if not revoked:
            revoked = True
            async with pg.writer_database.connection() as connection:
                await connection.execute(
                    "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
                    "WHERE collection_id=%s",
                    (collection.collection_id,),
                )
        return value

    monkeypatch.setattr(redis_cache, "get", revoke)
    with pytest.raises(KnowledgeNotFound):
        await cached.evidence(pin, scope, ids)


async def test_new_active_revision_does_not_use_previous_revision_entries(
    postgres_fixture, redis_cache
):
    pg = postgres_fixture
    collection, scope, pin, plain, cached = await repositories(pg, redis_cache)
    first = search_request(collection, pin.revision.revision_id)
    old = await KnowledgeRetrievalService(cached, QueryModel()).search(first, scope=scope)
    _, revision, _, _ = await corpus(
        pg, collection=collection, text="New independent revision about delivery verification."
    )
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
            "WHERE collection_id=%s",
            (revision, collection.collection_id),
        )
    query = RetrievalRequest(collection_id=collection.collection_id, query="delivery verification")
    result = await KnowledgeRetrievalService(cached, QueryModel()).search(query, scope=scope)
    expected = await KnowledgeRetrievalService(plain, QueryModel()).search(query, scope=scope)
    assert result == expected and result.revision_id == revision != old.revision_id


async def test_concurrent_real_redis_misses_have_one_hydration_fill(
    postgres_fixture, redis_cache, monkeypatch
):
    collection, scope, pin, plain, cached = await repositories(postgres_fixture, redis_cache)
    candidates = await plain.candidates(
        pin, scope, search_request(collection, pin.revision.revision_id), [0.2, 0.3, 0.4], [], 10
    )
    started, release = asyncio.Event(), asyncio.Event()
    fetch = cached._fetch_chunk

    async def blocked(*args):
        started.set()
        await release.wait()
        return await fetch(*args)

    spy = AsyncMock(side_effect=blocked)
    monkeypatch.setattr(cached, "_fetch_chunk", spy)
    tasks = [
        asyncio.create_task(cached.evidence(pin, scope, [candidates[0].chunk_id])) for _ in range(5)
    ]
    await started.wait()

    async def registered():
        while cached._coalesced.stats.waits < 4:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(registered(), timeout=5)
    finally:
        release.set()
    results = await asyncio.gather(*tasks)
    assert all(result == results[0] for result in results)
    assert spy.await_count == 1


async def test_isolated_redis_server_outage_and_restart_preserve_postgres_results(postgres_fixture):
    # Only this uniquely named disposable server is stopped. Never stop the
    # user's Redis, FLUSHDB, or mutate global Redis settings.
    name = "fredi-cp10-" + uuid4().hex[:16]
    password = uuid4().hex
    environment = {**os.environ, "FREDI_TEST_REDIS_PASSWORD": password}

    async def docker(*args):
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Disposable Redis container operation failed.")
        return result.stdout.strip()

    created = False
    cache = None

    async def connect():
        port = int(
            await docker(
                "inspect",
                "--format",
                '{{(index (index .NetworkSettings.Ports "6379/tcp") 0).HostPort}}',
                name,
            )
        )
        backend = RedisKnowledgeCache(
            host="127.0.0.1", port=port, password=password, operation_timeout_seconds=0.2
        )
        for _ in range(50):
            try:
                if await backend._client.ping():
                    return backend
            except (OSError, TimeoutError, RedisError):
                pass
            await asyncio.sleep(0.1)
        await backend.aclose()
        raise RuntimeError("Disposable Redis did not become ready.")

    try:
        await docker(
            "run",
            "--detach",
            "--pull",
            "never",
            "--name",
            name,
            "--publish",
            "127.0.0.1::6379",
            "--memory",
            "64m",
            "--cpus",
            "0.5",
            "--env",
            "FREDI_TEST_REDIS_PASSWORD",
            "redis:8.2.11-bookworm",
            "sh",
            "-ec",
            'exec redis-server --save "" --appendonly no '
            '--requirepass "$FREDI_TEST_REDIS_PASSWORD"',
        )
        created = True
        cache = ScopedRedis(await connect())
        collection, scope, pin, plain, cached = await repositories(postgres_fixture, cache)
        query = search_request(collection, pin.revision.revision_id)
        expected = await KnowledgeRetrievalService(plain, QueryModel()).search(query, scope=scope)
        service = KnowledgeRetrievalService(cached, QueryModel())
        assert await service.search(query, scope=scope) == expected
        await docker("stop", "--time", "1", name)
        assert await service.search(query, scope=scope) == expected
        await docker("start", name)
        await cache.cache.aclose()
        cache.cache = await connect()
        fills = cached._coalesced.stats.fills
        assert await service.search(query, scope=scope) == expected
        assert cached._coalesced.stats.fills > fills
    finally:
        if cache:
            await cache.aclose()
        if created:
            await docker("rm", "--force", name)
