"""Real backup/recovery, independent feature rollback and multi-student isolation."""

import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
import pytest

from app.config.settings import settings
from app.context import SessionContext, UserContext
from app.domain import GuidanceLevel
from app.knowledge.cache import NoKnowledgeCache
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.evaluation import EvaluationSuite
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from app.knowledge.integration import CourseBinding, RagRuntimeConfig
from app.knowledge.migrations import migrate
from app.knowledge.planner_schemas import PlannerConfig
from app.knowledge.schemas import EmbeddingProfile, KnowledgeCollection, KnowledgeWriteScope
from app.knowledge.source_store import FileSourceStore
from app.main import app
from app.persistence.turn_models import TurnStatus
from app.runtime import runtime_lifespan
from app.services.model_service import create_model_service
from scripts.evaluate_rag import ProfileModel
from scripts.knowledge_backup import backup, verified_manifest, verify_restore
from scripts.migrate_knowledge import local_dsn
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_retrieval import corpus
from tests.persistence.test_contracts import seed
from tests.persistence.test_grounded_runtime import body, config, create_student, running
from tests.persistence.test_http_runtime import SECRET, headers, student
from tests.persistence.test_http_runtime import running as plain_running

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(os.environ.get("FREDI_RAG_REAL_MODEL") != "1", reason="Opt-in real models")
async def test_real_router_planner_specialist_signed_http_and_durable_replay(
    pg, monkeypatch, tmp_path
):
    """No routing/teacher/planner doubles; only student storage is disposable."""
    suite, suite_hash = EvaluationSuite.read(
        Path(__file__).resolve().parents[3] / "docs/knowledge/rag-evaluation-v1.json"
    )
    uid = uuid4()
    model_name = os.environ.get("FREDI_RAG_MODEL", "qwen3.5:4b")
    endpoint = os.environ.get("FREDI_RAG_MODEL_URL", "http://127.0.0.1:11434")
    cfg = RagRuntimeConfig(
        planner=PlannerConfig(model=model_name, timeout_seconds=60),
        bindings=[
            CourseBinding(
                user_id=uid,
                collection_id=suite.collection_id,
                revision_id=suite.revision_id,
                access_domain=suite.access_domain,
            )
        ],
        planner_url=endpoint,
        embedding_url=endpoint,
    )
    cfg.prompt.reasoning_enabled = False
    cfg.prompt.reserved_output_tokens = 1024
    path = tmp_path / "trusted-real-rag.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    real = ProfileModel(
        create_model_service(
            settings.model_copy(
                update={
                    "ollama_base_url": endpoint,
                    "ollama_chat_model": model_name,
                    "model_request_timeout_seconds": 90,
                    "model_max_retries": 0,
                }
            )
        )
    )
    monkeypatch.setattr(settings, "application_storage", "postgres")
    monkeypatch.setattr(settings, "application_database_url", pg.database.config.dsn)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "knowledge_cache_enabled", False)
    monkeypatch.setattr(settings, "knowledge_intake_enabled", False)
    monkeypatch.setattr(settings, "rag_config_file", path)
    monkeypatch.setattr(settings, "rag_database_url", local_dsn(role="reader"))
    monkeypatch.setattr(settings, "ollama_base_url", endpoint)
    monkeypatch.setattr(settings, "ollama_chat_model", model_name)
    for name in (
        "orchestrator_model",
        "scenario_agent_model",
        "process_coach_agent_model",
        "ap_plus_navigator_agent_model",
    ):
        if hasattr(settings, name):
            monkeypatch.setattr(settings, name, None)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: real)
    async with runtime_lifespan(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            user, session = await create_student(client, app.state.runtime, uid)
            context = await app.state.runtime.user_context_service.get_or_create(user_id=uid)
            context.hint_state.guidance_level = GuidanceLevel.EXPLICIT
            context.hint_state.manually_overridden = True
            await app.state.runtime.user_context_service.save(
                user_id=uid, context=context, expected_version=context.version
            )
            payload = {
                **body(user, session),
                "user_message": "Explain why the business exercise needs a fiscal year at AP40-03. "
                "Use the supplied course evidence and do not change my progress.",
            }
            start = time.perf_counter()
            reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
            data = reply.json()
            history = await client.get(
                f"/conversations/{session.session_id}/messages", headers=headers(user)
            )
            before_replay = len(real.responses)
            replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
            report = dict(
                http_status=reply.status_code,
                suite_sha256=suite_hash,
                model=model_name,
                latency_seconds=round(time.perf_counter() - start, 2),
                rag=data.get("metadata", {}).get("rag"),
                citation_count=len(data.get("source_references", [])),
                real_route_and_teacher_calls=before_replay,
                replay_equal=replay.json() == data,
                replay_extra_model_calls=len(real.responses) - before_replay,
                history_count=len(history.json().get("items", [])),
                error=data.get("error"),
            )
            artifact = os.environ.get("FREDI_C11_REAL_REPORT")
            if artifact:
                Path(artifact).write_text(json.dumps(report, indent=2), encoding="utf-8")
            assert reply.status_code == 200 and data["error"] is None, report
            assert report["real_route_and_teacher_calls"] == 2, report
            assert report["citation_count"] > 0, report
            assert report["replay_equal"] and report["replay_extra_model_calls"] == 0, report
            assert report["history_count"] == 2, report


async def test_snapshot_restores_messages_context_queued_pdf_and_recovers_only_scratch(
    pg, tmp_path
):
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin", database=pg.name).get_secret_value()
    ) as connection:
        await connection.execute(
            "CREATE EXTENSION vector; "
            "CREATE SCHEMA knowledge AUTHORIZATION fredi_knowledge_writer; "
            "GRANT USAGE ON SCHEMA knowledge TO fredi_knowledge_reader; "
            "ALTER DEFAULT PRIVILEGES FOR ROLE fredi_knowledge_writer IN SCHEMA knowledge "
            "GRANT SELECT ON TABLES TO fredi_knowledge_reader"
        )
    user, _, session = await seed(pg.repositories)
    await pg.repositories.user_contexts.create(UserContext(user_id=user.user_id))
    await pg.repositories.session_contexts.create(
        SessionContext(
            user_id=user.user_id, session_id=session.session_id, current_step_id="AP40-05"
        )
    )
    scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=uuid4())
    await pg.repositories.turns.begin(**scope, content="Synthetic interrupted student question")
    async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=local_dsn(database=pg.name))) as db:
        await migrate(db, embedding_dimension=3)
        collection = KnowledgeCollection(name="Recovery fixture", access_domain="recovery-test")
        binding = IntakeBinding(
            collection_id=collection.collection_id,
            scope=KnowledgeWriteScope(
                access_domain=collection.access_domain, can_manage_knowledge=True
            ),
            manager_user_ids=[user.user_id],
            index=IndexConfig(
                embedding=EmbeddingProfile(
                    provider="test", model="fixed", revision="v1", dimension=3
                )
            ),
        )
        jobs = IntakeRepository(db)
        await jobs.knowledge.create_collection(collection, scope=binding.scope)
        sources = FileSourceStore(tmp_path / "source-store")
        service = IntakeService(jobs, sources, None, None, IntakeConfig(bindings=[binding]))
        queued = await service.submit(
            collection.collection_id, b"%PDF-1.4\nqueued bytes", title="Not yet processed"
        )
        output = tmp_path / "private-backup"
        result = await backup(
            output,
            database=pg.name,
            source_roots=[sources.root],
            volume_reader=lambda _: pytest.fail("Local bytes should suffice"),
        )
        assert result == dict(complete=True, source_count=1, table_count=17)
        assert verified_manifest(output)["sources"][0]["source_hash"] == queued.payload.source_hash
        restored = await verify_restore(output, recover=True)
        assert restored["verified"] and restored["recovered_turns"] == 1
        assert restored["scratch_database_removed"] is True
        assert (await pg.repositories.turns.get(**scope)).status == TurnStatus.PROCESSING
        assert (
            await pg.repositories.conversations.count(
                user_id=user.user_id, session_id=session.session_id
            )
            == 1
        )


async def test_disabling_redis_keeps_grounded_chat_and_history(
    pg, postgres_fixture, monkeypatch, tmp_path
):
    collection, revision, _, _ = await corpus(postgres_fixture)
    uid = uuid4()
    monkeypatch.setattr(settings, "knowledge_cache_enabled", False)
    monkeypatch.setattr(settings, "knowledge_cache_redis_host", None)
    async with running(
        pg, postgres_fixture, config(collection, revision, uid), monkeypatch, tmp_path
    ) as (client, runtime, _, _):
        repository = runtime.grounding_service.planner.retrieval.repository
        assert isinstance(repository.cache, NoKnowledgeCache)
        user, session = await create_student(client, runtime, uid)
        response = await client.post(
            "/orchestration/invoke", headers=headers(user), json=body(user, session)
        )
        assert response.status_code == 200 and response.json()["source_references"]
        history = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(user)
        )
        assert len(history.json()["items"]) == 2


async def test_disabling_rag_skips_broken_knowledge_and_redis_configuration(pg, monkeypatch):
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "knowledge_cache_enabled", True)
    monkeypatch.setattr(settings, "knowledge_cache_redis_host", None)
    monkeypatch.setattr(settings, "rag_config_file", None)
    monkeypatch.setattr(settings, "knowledge_intake_enabled", False)
    async with plain_running(pg, monkeypatch) as (client, runtime, _):
        assert runtime.grounding_service is None
        users = [await student(client, runtime) for _ in range(4)]
        responses = await asyncio.gather(
            *(
                client.post(
                    "/orchestration/invoke", headers=headers(user), json=body(user, session)
                )
                for user, session in users
            )
        )
        assert all(r.status_code == 200 and r.json()["error"] is None for r in responses)
        assert all(not r.json()["source_references"] for r in responses)
        for user, session in users:
            history = await client.get(
                f"/conversations/{session.session_id}/messages", headers=headers(user)
            )
            assert len(history.json()["items"]) == 2
            foreign, _ = next(pair for pair in users if pair[0].user_id != user.user_id)
            assert (
                await client.get(
                    f"/conversations/{session.session_id}/messages", headers=headers(foreign)
                )
            ).status_code == 404
