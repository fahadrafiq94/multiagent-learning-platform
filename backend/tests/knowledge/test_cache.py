"""Deterministic cache behavior against a fake client; real-Redis checks are opt-in."""

from __future__ import annotations

import asyncio

import pytest

from app.knowledge.cache import (
    CoalescedCache,
    NoKnowledgeCache,
    RedisKnowledgeCache,
)


class FakeClient:
    """In-memory stand-in for redis.asyncio.Redis covering the used commands."""

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.fail = fail
        self.closed = False
        self.ttls: dict[str, int] = {}

    async def get(self, key: str):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: bytes, ex: int) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.ttls[key] = ex

    async def delete(self, key: str) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.store.pop(key, None)

    async def aclose(self) -> None:
        self.closed = True


def make_client(**kwargs) -> RedisKnowledgeCache:
    client = kwargs.pop("client", FakeClient())
    return RedisKnowledgeCache(
        host="127.0.0.1",
        port=6379,
        password=None,
        client=client,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_roundtrip_returns_exact_bytes_and_jittered_ttl_within_bounds():
    client = FakeClient()
    cache = make_client(client=client, ttl_jitter_seconds=300)
    await cache.set("fredi:knw:v1:a", b'{"x": 1}', ttl_seconds=3600)
    assert await cache.get("fredi:knw:v1:a") == b'{"x": 1}'
    assert 3600 <= client.ttls["fredi:knw:v1:a"] <= 3900


@pytest.mark.asyncio
async def test_missing_key_is_a_miss():
    cache = make_client()
    assert await cache.get("absent") is None


@pytest.mark.asyncio
async def test_outage_degrades_to_miss_and_skipped_store_never_raises():
    client = FakeClient(fail=True)
    cache = make_client(client=client)
    assert await cache.get("any") is None
    await cache.set("any", b"payload", ttl_seconds=60)
    await cache.delete("any")
    await cache.aclose()
    assert client.closed


@pytest.mark.asyncio
async def test_oversized_payload_is_not_stored_and_evicted_on_read():
    client = FakeClient()
    cache = make_client(client=client, max_payload_bytes=16)
    await cache.set("big", b"x" * 17, ttl_seconds=60)
    assert "big" not in client.store
    client.store["legacy"] = b"y" * 17
    assert await cache.get("legacy") is None
    assert "legacy" not in client.store


@pytest.mark.asyncio
async def test_ttl_contract_matches_no_knowledge_cache():
    for cache in (make_client(), NoKnowledgeCache()):
        with pytest.raises(ValueError):
            await cache.set("key", b"value", ttl_seconds=0)


def test_constructor_rejects_invalid_configuration():
    for bad in (
        {"ttl_jitter_seconds": -1},
        {"operation_timeout_seconds": 0},
        {"max_payload_bytes": 0},
    ):
        with pytest.raises(ValueError):
            make_client(**bad)


@pytest.mark.asyncio
async def test_no_knowledge_cache_is_a_silent_no_op():
    cache = NoKnowledgeCache()
    assert await cache.get("key") is None
    await cache.set("key", b"value", ttl_seconds=60)
    await cache.delete("key")
    await cache.aclose()
    assert await cache.get("key") is None


def make_coalesced(client: FakeClient, *, wait_seconds: float = 1.0) -> CoalescedCache:
    return CoalescedCache(make_client(client=client), ttl_seconds=60, wait_seconds=wait_seconds)


def encode_value(value):
    return value.encode()


def decode_value(blob):
    return blob.decode()


@pytest.mark.asyncio
async def test_cache_hit_skips_fetch_and_counts_stats():
    cache = make_coalesced(FakeClient())
    fetched = 0

    async def fetch():
        nonlocal fetched
        fetched += 1
        return "v1"

    first = await cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value)
    assert first == ("v1", False) and fetched == 1
    assert await cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value) == (
        "v1",
        True,
    )
    assert fetched == 1
    assert (cache.stats.hits, cache.stats.fills, cache.stats.waits) == (1, 1, 0)


@pytest.mark.asyncio
async def test_concurrent_misses_coalesce_into_one_fill():
    client = FakeClient()
    cache = make_coalesced(client)
    release = asyncio.Event()
    registered = asyncio.Event()
    fetched = 0

    async def fetch():
        nonlocal fetched
        fetched += 1
        registered.set()
        await release.wait()
        return "filled"

    first_task = asyncio.create_task(
        cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value)
    )
    await registered.wait()
    second_task = asyncio.create_task(
        cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value)
    )
    await asyncio.sleep(0)  # let the second call register as a waiter
    release.set()
    first, second = await asyncio.gather(first_task, second_task)
    assert (first, second) == (("filled", False), ("filled", True))
    assert fetched == 1 and await cache._cache.get("k") == b"filled"
    assert (cache.stats.fills, cache.stats.waits, cache.stats.misses) == (1, 1, 1)


@pytest.mark.asyncio
async def test_waiter_gives_up_after_bounded_wait_and_fetches_itself():
    cache = make_coalesced(FakeClient(), wait_seconds=0.05)
    release = asyncio.Event()
    registered = asyncio.Event()

    async def slow_fetch():
        registered.set()
        await release.wait()
        return "filler"

    async def own_fetch():
        return "own"

    first_task = asyncio.create_task(
        cache.get_or_fill("k", slow_fetch, encode=encode_value, decode=decode_value)
    )
    await registered.wait()
    # Runs inline: waits up to 0.05s for the (still-blocked) fill, then falls
    # back to fetching on its own instead of being held hostage.
    second = await cache.get_or_fill("k", own_fetch, encode=encode_value, decode=decode_value)
    assert second == ("own", False)
    assert cache.stats.waits == 1
    release.set()
    assert await first_task == ("filler", False)


@pytest.mark.asyncio
async def test_failed_fill_propagates_to_filler_and_wakes_waiters():
    client = FakeClient()
    cache = make_coalesced(client)
    release = asyncio.Event()
    registered = asyncio.Event()

    async def failing_fetch():
        registered.set()
        await release.wait()
        raise RuntimeError("db down")

    async def own_fetch():
        return "recovered"

    first_task = asyncio.create_task(
        cache.get_or_fill("k", failing_fetch, encode=encode_value, decode=decode_value)
    )
    await registered.wait()
    second_task = asyncio.create_task(
        cache.get_or_fill("k", own_fetch, encode=encode_value, decode=decode_value)
    )
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="db down"):
        await first_task
    # The waiter was woken by the failed fill, found nothing cached, and
    # fetched itself; the successful replacement (not the failure) is stored.
    assert await second_task == ("recovered", False)
    assert await cache._cache.get("k") == b"recovered"
    assert cache.stats.waits == 1


@pytest.mark.asyncio
async def test_corrupt_payload_is_deleted_and_refilled():
    client = FakeClient()
    client.store["k"] = b"not a valid payload"
    cache = make_coalesced(client)
    fetched = 0

    async def fetch():
        nonlocal fetched
        fetched += 1
        return "fresh"

    value, hit = await cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value)
    assert (value, hit) == ("fresh", False) and fetched == 1
    assert await cache._cache.get("k") == b"fresh"
    # A later read is a clean hit, proving the corrupt entry was evicted.
    assert await cache.get_or_fill("k", fetch, encode=encode_value, decode=decode_value) == (
        "fresh",
        True,
    )
    assert fetched == 1


@pytest.mark.asyncio
async def test_empty_list_payload_round_trips_as_a_hit():
    # An empty candidate list is a valid cached result, not a miss trigger.
    cache = make_coalesced(FakeClient())

    async def fetch():
        return []

    value, hit = await cache.get_or_fill(
        "k", fetch, encode=lambda items: b"[]", decode=lambda blob: []
    )
    assert (value, hit) == ([], False)
    assert await cache.get_or_fill("k", fetch, encode=lambda i: b"[]", decode=lambda b: []) == (
        [],
        True,
    )


def test_coalesced_cache_rejects_invalid_configuration():
    for bad in ({"ttl_seconds": 0}, {"wait_seconds": 0}):
        with pytest.raises(ValueError):
            CoalescedCache(NoKnowledgeCache(), **{"ttl_seconds": 60, **bad})


@pytest.mark.asyncio
async def test_tampered_or_cross_key_payload_is_never_trusted():
    client = FakeClient()
    cache = make_client(client=client)
    await cache.set("first", b"trusted", ttl_seconds=60)
    client.store["second"] = client.store["first"]
    assert await cache.get("second") is None
    client.store["first"] = client.store["first"][:-1] + b"X"
    assert await cache.get("first") is None


@pytest.mark.asyncio
async def test_hanging_redis_operation_has_total_timeout():
    class HangingClient(FakeClient):
        async def get(self, key):
            await asyncio.Event().wait()

    cache = make_client(client=HangingClient(), operation_timeout_seconds=0.01)
    assert await asyncio.wait_for(cache.get("key"), timeout=1) is None


@pytest.mark.asyncio
async def test_cancel_during_store_releases_fill_registry():
    entered = asyncio.Event()

    backend = FakeClient()

    async def blocked_set(key, value, ex):
        entered.set()
        await asyncio.Event().wait()

    backend.set = blocked_set
    cache = make_coalesced(backend)

    async def fetch():
        return "value"

    task = asyncio.create_task(
        cache.get_or_fill("key", fetch, encode=encode_value, decode=decode_value)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not cache._inflight
