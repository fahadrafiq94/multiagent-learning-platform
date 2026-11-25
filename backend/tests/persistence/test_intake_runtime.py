"""Lifespan-owned intake with actual signed identity and both PostgreSQL stores."""

from uuid import uuid4

import pytest

from app.config.settings import settings
from app.identity import UserIdentity
from app.knowledge.intake_schemas import IntakeConfig
from app.main import app
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_extraction import pdf_bytes
from tests.knowledge.test_indexer import Embeddings
from tests.knowledge.test_intake import setup
from tests.persistence.test_http_runtime import headers, running

pytestmark = pytest.mark.asyncio


async def test_enabled_intake_lifespan_and_restart(pg, postgres_fixture, monkeypatch, tmp_path):
    kp = postgres_fixture
    for field in ("embedding_provider", "embedding_base_url", "embedding_model"):
        monkeypatch.setattr(settings, field, None)
    collection, binding, _, _ = await setup(kp, tmp_path)
    path = tmp_path / "intake-config.json"
    path.write_text(IntakeConfig(bindings=[binding]).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(settings, "knowledge_intake_enabled", True)
    monkeypatch.setattr(settings, "knowledge_intake_config_file", path)
    monkeypatch.setattr(settings, "knowledge_ingest_database_url", kp.writer_database.config.dsn)
    monkeypatch.setattr(settings, "knowledge_source_root", tmp_path / "http-sources")
    monkeypatch.setattr(
        "app.knowledge.intake_runtime.create_model_service", lambda cfg, **options: Embeddings()
    )
    async with running(pg, monkeypatch) as (client, runtime, _):
        user = await runtime.repositories.identities.create(
            UserIdentity(user_id=binding.manager_user_ids[0], student_id=uuid4().hex)
        )
        url = f"/knowledge/collections/{collection.collection_id}"
        uploaded = await client.post(
            url + "/documents?title=New%20PDF",
            headers={**headers(user), "Content-Type": "application/pdf"},
            content=pdf_bytes("A generic course fact uploaded through HTTP."),
        )
        assert uploaded.status_code == 202, uploaded.text
        job = uploaded.json()
        # The upload request queues work, never waits for model inference.
        assert job["status"] == "queued"
        assert (
            await app.state.knowledge_intake.run_once(collection.collection_id)
        ).status == "succeeded"
        listing = await client.get(url + "/documents", headers=headers(user))
        assert listing.json()["documents"][0]["document_id"] == job["document_id"]
    assert app.state.knowledge_intake is None
    async with running(pg, monkeypatch) as (client, _, _):
        status = await client.get(url + "/jobs/" + job["job_id"], headers=headers(user))
        assert status.status_code == 200 and status.json()["status"] == "succeeded"
