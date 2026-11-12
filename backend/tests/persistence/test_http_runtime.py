"""Real PostgreSQL + ASGI workflow, signed identity, replay and restart tests."""

import asyncio
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.identity.schemas import UserStatus
from app.main import app
from app.orchestration.schemas import OrchestrationRequest
from app.persistence.turn_models import TurnStatus
from app.runtime import runtime_lifespan
from tests.integration.test_orchestration_api import confirm_scenario, create_model_service

pytestmark = pytest.mark.asyncio
SECRET = SecretStr("http-pilot-test-secret-" * 3)


@asynccontextmanager
async def running(pg, monkeypatch):
    model, generate, _ = create_model_service(
        specialist_operations=[
            {
                "field": "current_step_id",
                "action": "set",
                "value": "8",
                "reason": "Student explicitly stated step 8.",
            },
        ]
    )
    monkeypatch.setattr(settings, "application_storage", "postgres")
    monkeypatch.setattr(settings, "application_database_url", pg.database.config.dsn)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: model)
    assert not app.dependency_overrides
    async with runtime_lifespan(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client, app.state.runtime, generate


def headers(user):
    return {"Authorization": "Bearer " + issue_token(user.user_id, SECRET)}


async def student(client, runtime):
    user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
    response = await client.post("/conversations", headers=headers(user), json={"title": "AP+"})
    assert response.status_code == 201, response.text
    session = await runtime.repositories.sessions.get(
        user_id=user.user_id,
        session_id=UUID(response.json()["session_id"]),
    )
    await confirm_scenario(runtime, user=user, session=session)
    return user, session


def body(user, session):
    return {
        "student_id": user.student_id,
        "session_id": str(session.session_id),
        "request_id": str(uuid4()),
        "user_message": "I am at step 8 in AP+ and cannot find the supplier field.",
    }


async def test_http_exchange_restart_replay_and_continue_preserve_context(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        user, session = await student(client, runtime)
        payload = body(user, session)
        first = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert first.status_code == 200, first.text
        assert first.json()["metadata"]["assistant_message_persisted"] is True
        assert generate.await_count == 2
        before = await runtime.repositories.user_contexts.get(user.user_id)
        retry = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert retry.json() == first.json() and generate.await_count == 2
        assert (
            await runtime.repositories.user_contexts.get(user.user_id)
        ).version == before.version
        history = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(user)
        )
        assert [m["role"] for m in history.json()["items"]] == ["user", "assistant"]
        assert history.json()["items"][1]["source_references"] == []
        assert history.json()["items"][0]["trace_id"] == history.json()["items"][1]["trace_id"]
        assert (await client.get("/ready")).json()["storage"] == "postgres"
    async with running(pg, monkeypatch) as (client, runtime, generate):
        reopened = await client.get(f"/conversations/{session.session_id}", headers=headers(user))
        assert reopened.status_code == 200
        assert reopened.json()["context"]["current_step_id"] == "8"
        assert (
            await runtime.repositories.user_contexts.get(user.user_id)
        ).scenario_card is not None
        retry = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert retry.json() == first.json() and generate.await_count == 0
        continued = await client.post(
            "/orchestration/invoke", headers=headers(user), json=body(user, session)
        )
        assert continued.status_code == 200
        assert generate.await_count == 2
        listing = await client.get("/conversations?limit=1&offset=0", headers=headers(user))
        assert listing.json()["items"][0]["session_id"] == str(session.session_id)
        page = await client.get(
            f"/conversations/{session.session_id}/messages?limit=1&offset=1", headers=headers(user)
        )
        assert page.json()["items"][0]["role"] == "assistant"
        status = await client.get(
            f"/conversations/{session.session_id}/turns/{payload['request_id']}",
            headers=headers(user),
        )
        assert status.json()["status"] == "completed"


async def test_http_identity_boundary_ownership_and_request_bounds(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        user, session = await student(client, runtime)
        other, _ = await student(client, runtime)
        assert (await client.get("/conversations")).status_code == 401
        blank_title = await client.post(
            "/conversations", headers=headers(user), json={"title": "   "}
        )
        assert blank_title.status_code == 422
        invalid = {"Authorization": "Bearer invalid"}
        assert (await client.get("/conversations", headers=invalid)).status_code == 401
        forged = await client.post(
            "/orchestration/invoke", headers=headers(other), json=body(user, session)
        )
        assert forged.status_code == 403
        foreign = await client.post(
            "/orchestration/invoke", headers=headers(other), json=body(other, session)
        )
        assert foreign.status_code == 404
        for path in [
            f"/conversations/{session.session_id}",
            f"/conversations/{session.session_id}/messages",
            f"/conversations/{session.session_id}/turns/{uuid4()}",
        ]:
            assert (await client.get(path, headers=headers(other))).status_code == 404
        for query in ["limit=0", "limit=101", "offset=-1"]:
            assert (
                await client.get("/conversations?" + query, headers=headers(user))
            ).status_code == 422
        missing = body(user, session)
        missing.pop("request_id")
        assert (
            await client.post("/orchestration/invoke", headers=headers(user), json=missing)
        ).status_code == 422
        oversized = {**body(user, session), "user_message": "x" * 4001}
        assert (
            await client.post("/orchestration/invoke", headers=headers(user), json=oversized)
        ).status_code == 422
        generate.assert_not_awaited()
        assert (
            await runtime.repositories.conversations.count(
                user_id=user.user_id, session_id=session.session_id
            )
            == 0
        )
        user.status = UserStatus.DISABLED
        await runtime.repositories.identities.update(user)
        assert (await client.get("/conversations", headers=headers(user))).status_code == 401


async def test_inflight_retry_and_different_request_cannot_duplicate_generation(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        user, session = await student(client, runtime)
        entered, release = asyncio.Event(), asyncio.Event()
        original = generate.side_effect

        async def slow(request):
            entered.set()
            await release.wait()
            return await original(request)

        generate.side_effect = slow
        payload = body(user, session)
        first = asyncio.create_task(
            client.post("/orchestration/invoke", headers=headers(user), json=payload)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            same = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
            other = await client.post(
                "/orchestration/invoke", headers=headers(user), json=body(user, session)
            )
            assert same.status_code == other.status_code == 409
            assert generate.await_count == 1
        finally:
            release.set()
            response = await first
        assert response.status_code == 200
        changed = {**payload, "user_message": "Different question"}
        assert (
            await client.post("/orchestration/invoke", headers=headers(user), json=changed)
        ).status_code == 409
        assert (
            await runtime.repositories.conversations.count(
                user_id=user.user_id, session_id=session.session_id
            )
            == 2
        )


@pytest.mark.parametrize("failure", ["exception", "graph_error", "oversized"])
async def test_failed_graph_or_invalid_response_is_explicit_and_never_reexecutes(
    pg, monkeypatch, failure
):
    async with running(pg, monkeypatch) as (client, runtime, _):
        user, session = await student(client, runtime)
        calls = 0

        class Graph:
            async def ainvoke(self, state):
                nonlocal calls
                calls += 1
                if failure == "exception":
                    raise RuntimeError("private-failure-marker")
                return {
                    **state,
                    "final_response": "x" * 4001 if failure == "oversized" else "Error",
                    "error": "private-marker" if failure == "graph_error" else None,
                }

        runtime.orchestration_service._graph = Graph()
        payload = body(user, session)
        result = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert result.status_code == 503 and "private" not in result.text
        assert (
            await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        ).status_code == 409
        assert calls == 1
        turn = await runtime.repositories.turns.get(
            user_id=user.user_id, session_id=session.session_id, request_id=payload["request_id"]
        )
        assert turn.status == TurnStatus.FAILED
        assert turn.error_code == (
            "invalid_response" if failure == "oversized" else "workflow_failed"
        )
        assert (
            await runtime.repositories.conversations.count(
                user_id=user.user_id, session_id=session.session_id
            )
            == 1
        )


async def test_cancellation_interrupts_turn_and_allows_deliberate_new_request(pg, monkeypatch):
    async with running(pg, monkeypatch) as (_, runtime, generate):
        user = await runtime.repositories.identities.create(UserIdentity(student_id="cancel-test"))
        session = await runtime.conversation_service.create_session(user_id=user.user_id)
        entered = asyncio.Event()

        class Graph:
            async def ainvoke(self, state):
                entered.set()
                await asyncio.Event().wait()

        runtime.orchestration_service._graph = Graph()
        request = OrchestrationRequest(**body(user, session))
        task = asyncio.create_task(runtime.orchestration_service.execute(request))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        turn = await runtime.repositories.turns.get(
            user_id=user.user_id, session_id=session.session_id, request_id=request.request_id
        )
        assert turn.status == TurnStatus.INTERRUPTED and turn.error_code == "request_cancelled"
        assert (
            await runtime.repositories.turns.begin(
                user_id=user.user_id,
                session_id=session.session_id,
                request_id=uuid4(),
                content="New attempt",
            )
        ).created
        generate.assert_not_awaited()


async def test_database_failure_is_503_without_memory_fallback(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        user, session = await student(client, runtime)
        await app.state.application_database.close()
        assert (await client.get("/ready")).status_code == 503
        assert (await client.get("/conversations", headers=headers(user))).status_code == 503
        result = await client.post(
            "/orchestration/invoke", headers=headers(user), json=body(user, session)
        )
        assert result.status_code == 503
        generate.assert_not_awaited()


async def test_session_initialization_failure_rolls_back_session_and_context(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, _):
        user = await runtime.repositories.identities.create(
            UserIdentity(student_id="atomic-session")
        )
        async with pg.migrator.connection() as connection:
            await connection.execute(
                "CREATE FUNCTION application.reject_context() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ "
                "BEGIN RAISE EXCEPTION 'private-marker'; END $$; "
                "CREATE TRIGGER reject_context BEFORE INSERT ON application.session_contexts "
                "FOR EACH ROW EXECUTE FUNCTION application.reject_context()"
            )
        response = await client.post("/conversations", headers=headers(user), json={})
        assert response.status_code == 503 and "private-marker" not in response.text
        assert await runtime.repositories.sessions.list_for_user(user.user_id) == []


async def test_completion_database_failure_cannot_report_false_success(pg, monkeypatch):
    async with running(pg, monkeypatch) as (client, runtime, _):
        user, session = await student(client, runtime)
        async with pg.migrator.connection() as connection:
            await connection.execute(
                "CREATE FUNCTION application.reject_complete() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ "
                "BEGIN IF NEW.status='completed' THEN RAISE EXCEPTION 'private-marker'; END IF; "
                "RETURN NEW; END $$; CREATE TRIGGER reject_complete BEFORE UPDATE "
                "ON application.turns FOR EACH ROW EXECUTE FUNCTION application.reject_complete()"
            )
        payload = body(user, session)
        response = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert response.status_code == 503 and "private-marker" not in response.text
        turn = await runtime.repositories.turns.get(
            user_id=user.user_id,
            session_id=session.session_id,
            request_id=UUID(payload["request_id"]),
        )
        assert turn.status == TurnStatus.FAILED and turn.error_code == "persistence_failed"
        assert (
            await runtime.repositories.conversations.count(
                user_id=user.user_id,
                session_id=session.session_id,
            )
            == 1
        )
