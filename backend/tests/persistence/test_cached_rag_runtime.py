"""Signed HTTP, both PostgreSQL stores and real optional Redis resources."""

import os
from uuid import uuid4

import pytest
from dotenv import dotenv_values
from pydantic import SecretStr

from app.config.settings import settings
from scripts.knowledge_infra import ROOT
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_cache_retrieval import redis_cache as redis_cache
from tests.knowledge.test_retrieval import corpus
from tests.persistence.test_grounded_runtime import body, config, create_student, running
from tests.persistence.test_http_runtime import headers

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("FREDI_KNOWLEDGE_REDIS_INTEGRATION") != "1",
        reason="Opt-in real Redis runtime.",
    ),
]


@pytest.mark.parametrize("outage", [False, True])
async def test_signed_grounded_chat_and_history_work_with_cache_or_auth_outage(
    pg,
    postgres_fixture,
    redis_cache,
    tmp_path,
    monkeypatch,
    outage,
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    local = dotenv_values(ROOT / "backend/.env.knowledge")
    monkeypatch.setattr(settings, "knowledge_cache_enabled", True)
    monkeypatch.setattr(settings, "knowledge_cache_redis_host", "127.0.0.1")
    monkeypatch.setattr(
        settings, "knowledge_cache_redis_port", int(local.get("KNOWLEDGE_REDIS_PORT") or "16379")
    )
    monkeypatch.setattr(
        settings,
        "knowledge_cache_redis_password",
        SecretStr("wrong" if outage else local["KNOWLEDGE_REDIS_PASSWORD"]),
    )
    keys = set()
    try:
        async with running(pg, kp, config(collection, revision, uid), monkeypatch, tmp_path) as (
            client,
            runtime,
            _,
            _,
        ):
            repository = runtime.grounding_service.planner.retrieval.repository
            cache = repository.cache
            original = cache._client.set

            async def record(key, *args, **kwargs):
                keys.add(key)
                return await original(key, *args, **kwargs)

            monkeypatch.setattr(cache._client, "set", record)
            user, session = await create_student(client, runtime, uid)
            for _ in range(2):
                response = await client.post(
                    "/orchestration/invoke", headers=headers(user), json=body(user, session)
                )
                assert response.status_code == 200 and response.json()["source_references"]
                assert response.json()["error"] is None
            history = await client.get(
                f"/conversations/{session.session_id}/messages", headers=headers(user)
            )
            assert len(history.json()["items"]) == 4
            if outage:
                assert repository._coalesced.stats.hits == 0
            else:
                assert repository._coalesced.stats.hits > 0
    finally:
        if keys:
            await redis_cache.cache._client.delete(*keys)
