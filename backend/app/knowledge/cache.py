"""Knowledge cache contract: PostgreSQL is authoritative, caching is disposable."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import redis.asyncio as aioredis


class KnowledgeCache(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def aclose(self) -> None: ...


class NoKnowledgeCache:
    """Correctness baseline; a disabled or failed cache must be indistinguishable from it."""

    async def get(self, key: str) -> bytes | None:
        return None

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive.")

    async def delete(self, key: str) -> None:
        pass

    async def aclose(self) -> None:
        pass


class RedisKnowledgeCache:
    """Best-effort Redis backend: every failure degrades to a miss or a skipped store.

    An outage or corrupt entry can only remove the speedup, never the data or the
    turn: PostgreSQL revalidation happens on the retrieval path regardless.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        password: str | None,
        ttl_jitter_seconds: int = 0,
        operation_timeout_seconds: float = 0.1,
        max_payload_bytes: int = 256 * 1024,
        client: aioredis.Redis | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        if ttl_jitter_seconds < 0:
            raise ValueError("TTL jitter must be nonnegative.")
        if operation_timeout_seconds <= 0:
            raise ValueError("Operation timeout must be positive.")
        if max_payload_bytes <= 0:
            raise ValueError("Payload limit must be positive.")
        self.ttl_jitter_seconds = ttl_jitter_seconds
        self.max_payload_bytes = max_payload_bytes
        self._timeout = operation_timeout_seconds
        # A runtime-specific application secret protects cache values independently
        # of Redis storage. Password fallback supports standalone operator clients;
        # anonymous clients use a process-local key and cannot reuse across restart.
        self._signing_key = signing_key or (
            hashlib.sha256(password.encode()).digest() if password else secrets.token_bytes(32)
        )
        self._client = client or aioredis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=False,
            socket_timeout=operation_timeout_seconds,
            socket_connect_timeout=operation_timeout_seconds,
        )

    async def get(self, key: str) -> bytes | None:
        try:
            async with asyncio.timeout(self._timeout):
                value: bytes | None = await self._client.get(key)
        except (aioredis.RedisError, OSError, TimeoutError):
            return None
        if value is None:
            return None
        if len(value) > self.max_payload_bytes + 32 or len(value) < 32:
            await self.delete(key)
            return None
        signature, payload = value[:32], value[32:]
        expected = hmac.digest(self._signing_key, key.encode() + b"\x00" + payload, "sha256")
        if not hmac.compare_digest(signature, expected):
            await self.delete(key)
            return None
        return payload

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive.")
        if len(value) > self.max_payload_bytes:
            return
        ttl = (
            ttl_seconds + secrets.randbelow(self.ttl_jitter_seconds + 1)
            if self.ttl_jitter_seconds
            else ttl_seconds
        )
        try:
            signature = hmac.digest(self._signing_key, key.encode() + b"\x00" + value, "sha256")
            async with asyncio.timeout(self._timeout):
                await self._client.set(key, signature + value, ex=ttl)
        except (aioredis.RedisError, OSError, TimeoutError):
            pass

    async def delete(self, key: str) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                await self._client.delete(key)
        except (aioredis.RedisError, OSError, TimeoutError):
            pass

    async def aclose(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                await self._client.aclose()
        except (aioredis.RedisError, OSError, TimeoutError):
            pass


_T = TypeVar("_T")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    fills: int = 0
    waits: int = 0


class CoalescedCache:
    """Per-process fill coalescing over a KnowledgeCache with bounded waiter waits.

    Safe with one HTTP worker per database: concurrent misses for the same key
    fetch once, then share. Waiters give up after `wait_seconds` and fetch
    themselves, so a slow or failed fill can never hold a request hostage.
    """

    def __init__(
        self,
        cache: KnowledgeCache,
        *,
        ttl_seconds: int,
        wait_seconds: float = 2.0,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive.")
        if wait_seconds <= 0:
            raise ValueError("Coalesce wait must be positive.")
        self._cache = cache
        self._ttl_seconds = ttl_seconds
        self._wait_seconds = wait_seconds
        self._inflight: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    async def get_or_fill(
        self,
        key: str,
        fetch: Callable[[], Awaitable[_T]],
        *,
        encode: Callable[[_T], bytes],
        decode: Callable[[bytes], _T],
    ) -> tuple[_T, bool]:
        """Return (value, hit). A corrupt payload is a miss, never an error."""
        if isinstance(self._cache, NoKnowledgeCache):
            self.stats.misses += 1
            uncached = await fetch()
            self.stats.fills += 1
            return uncached, False
        value: _T | None = await self._validated_get(key, decode)
        if value is not None:
            self.stats.hits += 1
            return value, True
        async with self._lock:
            event = self._inflight.get(key)
            if event is None:
                event = asyncio.Event()
                self._inflight[key] = event
                filler = True
            else:
                filler = False
        if not filler:
            self.stats.waits += 1
            try:
                await asyncio.wait_for(event.wait(), self._wait_seconds)
            except TimeoutError:
                pass
            value = await self._validated_get(key, decode)
            if value is not None:
                self.stats.hits += 1
                return value, True
        self.stats.misses += 1
        try:
            value = await fetch()
            self.stats.fills += 1
            try:
                await self._cache.set(key, encode(value), ttl_seconds=self._ttl_seconds)
            except (ValueError, TypeError, OSError, TimeoutError):
                # Serialization/cache availability cannot invalidate authoritative data.
                pass
            return value, False
        finally:
            if filler:
                self._finish_fill(key, event)

    def _finish_fill(self, key: str, event: asyncio.Event) -> None:
        # Single-threaded event loop: set before releasing the registry slot.
        # A later filler re-registers the key only after this method returns,
        # so popping the slot cannot drop a newer event.
        event.set()
        self._inflight.pop(key, None)

    async def _validated_get(self, key: str, decode: Callable[[bytes], _T]) -> _T | None:
        blob = await self._cache.get(key)
        if blob is None:
            return None
        try:
            return decode(blob)
        except Exception:
            await self._cache.delete(key)
            return None
