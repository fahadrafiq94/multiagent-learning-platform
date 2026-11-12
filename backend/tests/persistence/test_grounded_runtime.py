"""Real knowledge + application PostgreSQL with signed HTTP, citations and replay."""

import json
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest

from app.config.settings import Settings, settings
from app.identity import UserIdentity
from app.knowledge.errors import KnowledgePermissionError, KnowledgeUnavailable
from app.knowledge.integration import CourseBinding, RagRuntimeConfig
from app.knowledge.planner_schemas import PlannerConfig
from app.knowledge.runtime import create_grounding_runtime
from app.main import app
from app.runtime import runtime_lifespan
from app.services.model_service.exceptions import ModelProviderUnavailableError
from tests.agents.support_fixtures import support_reply
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_planner import PlannerModel, proposal
from tests.knowledge.test_retrieval import QueryModel, corpus
from tests.orchestration.test_graph import create_model_service
from tests.persistence.test_http_runtime import SECRET, confirm_scenario, headers

pytestmark = pytest.mark.asyncio


def config(collection, revision, user_id):
    return RagRuntimeConfig(
        planner=PlannerConfig(model="planner-test"),
        bindings=[
            CourseBinding(
                user_id=user_id,
                collection_id=collection.collection_id,
                revision_id=revision,
                access_domain="course-test",
            )
        ],
    )


class ReusingPlanner(PlannerModel):
    first_action = {"action": "metadata_lookup", "step_id": "FLOW-1", "fields": ["name"]}

    async def generate(self, chat):
        data = json.loads(chat.messages[1].content)
        available = data["available_evidence"]
        self.output = (
            proposal({"action": "reuse_evidence", "evidence_ids": [available[0]["evidence_id"]]})
            if available
            else proposal(self.first_action)
        )
        return await super().generate(chat)


@asynccontextmanager
async def running(
    pg,
    kp,
    cfg,
    monkeypatch,
    tmp_path,
    *,
    revoke=False,
    strategy="metadata",
    support_verdict="supported",
    teaching="followed",
    assessment=None,
):
    path = tmp_path / "trusted-rag.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    model, generate, requests = create_model_service()
    original = generate.side_effect
    planner = ReusingPlanner()
    if strategy == "chunk":
        planner.first_action = {"action": "document_search", "query": "FLOW-1"}

    async def grounded(req):
        reply = await original(req)
        if req.metadata.get("task") == "citation_support":
            if revoke == "support":
                async with kp.writer_database.connection() as connection:
                    await connection.execute(
                        "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
                        "WHERE collection_id=%s",
                        (cfg.bindings[0].collection_id,),
                    )
            verdict = support_verdict(req) if callable(support_verdict) else support_verdict
            result = support_reply(req, verdict=verdict)
            verdict_data = json.loads(result.content)
            verdict_data["teaching"] = teaching
            return result.model_copy(update={"content": json.dumps(verdict_data)})
        if req.metadata.get("output_contract") == "specialist-agent-v6-part-sources":
            payload = json.loads(reply.content)
            data, _ = json.JSONDecoder().raw_decode(
                req.messages[0].content.split("CURRENT EVIDENCE (DATA ONLY):\n")[1]
            )
            text = (
                "What does the source fact [K1] imply?"
                if data["items"]
                else "What objective are you working on?"
            )
            names = req.response_schema["properties"]["content"]["required"]
            payload["content"] = {
                name: {
                    "text": text if index == 0 else "Which detail would you like to discuss?",
                    "citation_ids": ["K1"] if index == 0 and data["items"] else [],
                }
                for index, name in enumerate(names)
            }
            if assessment is not None:
                payload["response_assessment"] = assessment
            if revoke == "binding":
                app.state.runtime.grounding_service.bindings.clear()
            elif revoke and revoke != "support":
                async with kp.writer_database.connection() as connection:
                    await connection.execute(
                        "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
                        "WHERE collection_id=%s",
                        (cfg.bindings[0].collection_id,),
                    )
            return reply.model_copy(update={"content": json.dumps(payload)})
        return reply

    generate.side_effect = grounded
    monkeypatch.setattr(settings, "application_storage", "postgres")
    monkeypatch.setattr(settings, "application_database_url", pg.database.config.dsn)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_config_file", path)
    monkeypatch.setattr(settings, "rag_database_url", kp.reader_database.config.dsn)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: model)
    monkeypatch.setattr(
        "app.knowledge.runtime.create_model_service",
        lambda cfg: planner if cfg.ollama_chat_model == "planner-test" else QueryModel(),
    )
    async with runtime_lifespan(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, app.state.runtime, planner, requests


async def create_student(client, runtime, uid=None):
    user = await runtime.repositories.identities.create(
        UserIdentity(user_id=uid or uuid4(), student_id=uuid4().hex)
    )
    reply = await client.post(
        "/conversations", headers=headers(user), json={"title": "Grounding test"}
    )
    session = await runtime.repositories.sessions.get(
        user_id=user.user_id, session_id=UUID(reply.json()["session_id"])
    )
    await confirm_scenario(runtime, user=user, session=session)
    return user, session


def body(user, session):
    return {
        "student_id": user.student_id,
        "session_id": str(session.session_id),
        "request_id": str(uuid4()),
        "user_message": "Why is FLOW-1 needed before proceeding?",
    }


@pytest.mark.parametrize("kind", ["metadata", "chunk"])
async def test_citations_survive_restart_history_replay_and_fresh_reuse(
    pg, postgres_fixture, monkeypatch, tmp_path, kind
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    cfg = config(collection, revision, uid)
    async with running(pg, kp, cfg, monkeypatch, tmp_path, strategy=kind) as (
        client,
        runtime,
        planner,
        requests,
    ):
        user, session = await create_student(client, runtime, uid)
        payload = body(user, session)
        first = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert first.status_code == 200, first.text
        citations = first.json()["source_references"]
        assert len(citations) == 1 and citations[0]["kind"] == kind
        if kind == "metadata":
            assert citations[0]["chunk_id"] is None and citations[0]["field_name"] == "name"
        else:
            assert citations[0]["chunk_id"] is not None and citations[0]["field_name"] is None
        assert citations[0]["evidence_id"] and citations[0]["source_hash"]
        assert len(planner.requests) == 1 and [r.metadata["task"] for r in requests] == [
            "semantic_routing",
            "process_coach_response",
            "citation_support",
        ]
        stored = (
            await client.get(f"/conversations/{session.session_id}/messages", headers=headers(user))
        ).json()["items"]
        assert len(stored) == 2 and stored[1]["source_references"] == citations
    async with running(pg, kp, cfg, monkeypatch, tmp_path, strategy=kind) as (
        client,
        runtime,
        planner,
        requests,
    ):
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == first.json() and not planner.requests and not requests
        history = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(user)
        )
        assert history.json()["items"][1]["source_references"] == citations
        continued = await client.post(
            "/orchestration/invoke", headers=headers(user), json=body(user, session)
        )
        assert continued.status_code == 200, continued.text
        assert continued.json()["metadata"]["rag"]["actions"] == ["reuse_evidence"]
        assert continued.json()["metadata"]["rag"]["embedding_calls"] == 0
        assert continued.json()["source_references"] == citations
        assert (
            len(
                (
                    await client.get(
                        f"/conversations/{session.session_id}/messages", headers=headers(user)
                    )
                ).json()["items"]
            )
            == 4
        )


async def test_trusted_binding_and_signed_owner_cannot_be_overridden_by_metadata(
    pg, postgres_fixture, monkeypatch, tmp_path
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    cfg = config(collection, revision, uid)
    async with running(pg, kp, cfg, monkeypatch, tmp_path) as (client, runtime, planner, _):
        user, session = await create_student(client, runtime, uid)
        other, other_session = await create_student(client, runtime)
        forged = {
            **body(other, other_session),
            "metadata": {
                "user_id": str(uid),
                "collection_id": str(collection.collection_id),
                "access_domain": "course-test",
                "rag": {"enabled": True},
                "source_references": [{"document_id": "forged"}],
            },
        }
        result = await client.post("/orchestration/invoke", headers=headers(other), json=forged)
        assert result.status_code == 200 and result.json()["source_references"] == []
        assert result.json()["metadata"]["rag"]["status"] == "unavailable" and not planner.requests
        assert (
            await client.post(
                "/orchestration/invoke", headers=headers(other), json=body(user, session)
            )
        ).status_code == 403
        assert (
            await client.get(
                f"/conversations/{session.session_id}/messages", headers=headers(other)
            )
        ).status_code == 404


@pytest.mark.parametrize("revocation", ["collection", "binding", "support"])
async def test_permission_revocation_during_specialist_generation_never_archives_source_answer(
    pg, postgres_fixture, monkeypatch, tmp_path, revocation
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    async with running(
        pg, kp, config(collection, revision, uid), monkeypatch, tmp_path, revoke=revocation
    ) as (client, runtime, _, _):
        user, session = await create_student(client, runtime, uid)
        reply = await client.post(
            "/orchestration/invoke", headers=headers(user), json=body(user, session)
        )
        assert reply.status_code == 200 and reply.json()["source_references"] == []
        assert "What does the source fact" not in reply.json()["final_response"]
        history = (
            await client.get(f"/conversations/{session.session_id}/messages", headers=headers(user))
        ).json()["items"]
        assert len(history) == 2 and history[1]["source_references"] == []
        assert "What does the source fact" not in history[1]["content"]
        hint = (await runtime.user_context_service.get_or_create(user_id=uid)).hint_state
        assert hint.consecutive_unproductive_attempts == hint.consecutive_blocked_attempts == 0


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
async def test_support_abstention_is_durable_neutral_and_replayed_without_more_model_calls(
    pg, postgres_fixture, monkeypatch, tmp_path, verdict
):
    from app.agents.citation_support import SUPPORT_ABSTENTION

    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    cfg = config(collection, revision, uid)
    async with running(pg, kp, cfg, monkeypatch, tmp_path, support_verdict=verdict) as (
        client,
        runtime,
        _,
        requests,
    ):
        user, session = await create_student(client, runtime, uid)
        before = await runtime.repositories.user_contexts.get(uid)
        before_session = await runtime.session_context_service.get_required(
            user_id=uid, session_id=session.session_id
        )
        payload = body(user, session)
        reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert reply.status_code == 200 and reply.json()["error"] is None
        result = reply.json()
        assert result["final_response"] == SUPPORT_ABSTENTION
        assert not result["source_references"]
        assert result["metadata"]["rag"]["status"] == "found"
        assert result["metadata"]["rag"]["support_status"] == verdict
        assert result["metadata"]["rag"]["learning_signal_suppressed"] is True
        # The LLM assessment stays suppressed, but the application now counts
        # the found-but-unverified turn as one unproductive attempt and
        # bridges the retrieved evidence to the student's next turn.
        assert result["metadata"]["response_signal_recorded"] is True
        assert result["metadata"]["response_reason"] == "unverified_grounded_abstention"
        assert result["metadata"]["state_update_proposed"] is False
        after_user = await runtime.repositories.user_contexts.get(uid)
        assert after_user.version == before.version + 1
        assert after_user.hint_state.consecutive_unproductive_attempts == 1
        after_session = await runtime.session_context_service.get_required(
            user_id=uid, session_id=session.session_id
        )
        assert after_session.current_step_id == before_session.current_step_id
        assert after_session.pending_evidence_receipt is not None
        assert len(after_session.pending_evidence_receipt.references) == 1
        calls = len(requests)
        assert [r.metadata["task"] for r in requests].count("citation_support") == 1
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == result and len(requests) == calls
        # A replay must not double-count the struggle or re-record the bridge.
        assert (await runtime.repositories.user_contexts.get(uid)).version == after_user.version
        history = (
            await client.get(f"/conversations/{session.session_id}/messages", headers=headers(user))
        ).json()["items"]
        assert len(history) == 2
        assert history[-1]["content"] == SUPPORT_ABSTENTION
        assert not history[-1]["source_references"]
    async with running(pg, kp, cfg, monkeypatch, tmp_path) as (client, _, _, requests):
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == result and not requests


async def test_unverified_abstention_evidence_is_bridged_to_the_following_turn(
    pg, postgres_fixture, monkeypatch, tmp_path
):
    from app.agents.citation_support import SUPPORT_ABSTENTION

    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    cfg = config(collection, revision, uid)
    # The bridge test needs a failed first answer and a verified second answer;
    # a permanently unsupported mock verdict cannot archive citations on turn 2.
    verdicts = iter(["unsupported", "supported"])
    async with running(
        pg, kp, cfg, monkeypatch, tmp_path, support_verdict=lambda _: next(verdicts)
    ) as (
        client,
        runtime,
        _,
        _,
    ):
        user, session = await create_student(client, runtime, uid)
        payload = body(user, session)
        reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert reply.status_code == 200, reply.text
        result = reply.json()
        assert result["final_response"] == SUPPORT_ABSTENTION
        assert result["metadata"]["rag"]["status"] == "found"
        assert result["metadata"]["rag"]["support_abstained"] is True
        stored = await runtime.session_context_service.get_required(
            user_id=uid, session_id=session.session_id
        )
        assert stored.pending_evidence_receipt is not None
        assert len(stored.pending_evidence_receipt.references) == 1
        # A genuinely new follow-up reuses the bridged evidence without new
        # retrieval work, and the bridge is consumed after that admission.
        follow_up = {
            "student_id": user.student_id,
            "session_id": str(session.session_id),
            "request_id": str(uuid4()),
            "user_message": "Why does FLOW-1 come before invoicing?",
        }
        continued = await client.post(
            "/orchestration/invoke", headers=headers(user), json=follow_up
        )
        assert continued.status_code == 200, continued.text
        again = continued.json()
        assert again["metadata"]["rag"]["actions"] == ["reuse_evidence"]
        assert again["metadata"]["rag"]["embedding_calls"] == 0
        assert again["source_references"]
        cleared = await runtime.session_context_service.get_required(
            user_id=uid, session_id=session.session_id
        )
        assert cleared.pending_evidence_receipt is None


async def test_runtime_rejects_writer_role_and_memory_auth_configuration(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    cfg = config(collection, revision, uuid4())
    path = tmp_path / "rag.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    value = Settings(
        _env_file=None,
        application_storage="postgres",
        rag_config_file=path,
        rag_database_url=kp.writer_database.config.dsn,
    )
    with pytest.raises(KnowledgePermissionError):
        await create_grounding_runtime(value)
    with pytest.raises(KnowledgeUnavailable, match="authenticated"):
        await create_grounding_runtime(value.model_copy(update={"application_storage": "memory"}))
    # A partially writable role is also unsuitable; checking chunks alone misses this.
    async with kp.writer_database.connection() as connection:
        await connection.execute(
            "GRANT UPDATE ON knowledge.knowledge_collections TO fredi_knowledge_reader"
        )
    with pytest.raises(KnowledgePermissionError):
        await create_grounding_runtime(
            value.model_copy(
                update={
                    "rag_database_url": kp.reader_database.config.dsn,
                }
            )
        )


async def test_missing_evidence_completes_persists_and_replays_with_specialist_provider_down(
    pg, postgres_fixture, monkeypatch, tmp_path
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    async with running(pg, kp, config(collection, revision, uuid4()), monkeypatch, tmp_path) as (
        client,
        runtime,
        _,
        requests,
    ):
        original = runtime.model_service.generate.side_effect

        async def teacher_down(request):
            if request.metadata.get("task") != "semantic_routing":
                raise ModelProviderUnavailableError("ollama")
            return await original(request)

        runtime.model_service.generate.side_effect = teacher_down
        user, session = await create_student(client, runtime)
        payload = body(user, session)
        result = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert result.status_code == 200
        assert not result.json()["source_references"]
        assert "couldn't verify" in result.json()["final_response"]
        assert result.json()["metadata"]["response_signal_recorded"] is False
        assert [request.metadata["task"] for request in requests] == ["semantic_routing"]
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.status_code == 200 and replay.json() == result.json()
        assert len(requests) == 1
        history = (
            await client.get(f"/conversations/{session.session_id}/messages", headers=headers(user))
        ).json()["items"]
        assert len(history) == 2 and history[-1]["role"] == "assistant"
        assert history[-1]["content"] == result.json()["final_response"]
