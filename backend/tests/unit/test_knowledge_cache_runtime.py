"""Optional cache construction and lifespan ownership remain separate from RAG grants."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.config.settings import Settings, settings
from app.knowledge.cache import NoKnowledgeCache
from app.knowledge.errors import KnowledgeUnavailable
from app.knowledge.runtime import create_knowledge_cache
from app.runtime import runtime_lifespan
from tests.knowledge.test_cache import FakeClient


def test_disabled_cache_does_not_require_redis_configuration():
    profile = Settings(_env_file=None, knowledge_cache_enabled=False)
    assert isinstance(create_knowledge_cache(profile, SecretStr("unused")), NoKnowledgeCache)
    with pytest.raises(KnowledgeUnavailable, match="Redis host"):
        create_knowledge_cache(
            profile.model_copy(
                update={"knowledge_cache_enabled": True, "knowledge_cache_redis_host": None}
            ),
            SecretStr("unused"),
        )


@pytest.mark.asyncio
async def test_reader_password_signing_is_shared_across_host_aliases_not_redis_storage(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("app.knowledge.cache.aioredis.Redis", lambda **kwargs: client)
    profile = Settings(
        _env_file=None, knowledge_cache_enabled=True, knowledge_cache_redis_host="redis"
    )
    a = create_knowledge_cache(profile, SecretStr("host=localhost password=reader-secret"))
    b = create_knowledge_cache(profile, SecretStr("host=postgres password=reader-secret"))
    other = create_knowledge_cache(profile, SecretStr("host=postgres password=different-reader"))
    await a.set("entry", b"valid", ttl_seconds=60)
    assert await b.get("entry") == b"valid"
    assert await other.get("entry") is None


@pytest.mark.asyncio
async def test_rag_lifespan_closes_cache_before_reader_database(monkeypatch):
    order = []
    cache = SimpleNamespace(aclose=AsyncMock(side_effect=lambda: order.append("cache")))
    database = SimpleNamespace(close=AsyncMock(side_effect=lambda: order.append("database")))
    monkeypatch.setattr(settings, "application_storage", "memory")
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "knowledge_intake_enabled", False)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: object())
    monkeypatch.setattr("app.runtime.build_orchestration_runtime", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.runtime.create_grounding_runtime", AsyncMock(return_value=(object(), database, cache))
    )
    async with runtime_lifespan(FastAPI()):
        pass
    assert order == ["cache", "database"]
