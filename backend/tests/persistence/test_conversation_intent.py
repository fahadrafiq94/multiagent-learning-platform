"""Signed HTTP, real PostgreSQL and actual knowledge admission with controlled LLM output."""

import json
from uuid import uuid4

import pytest

from app.context.schemas import SessionContextUpdate
from app.services.model_service.schemas import ModelChatResponse
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_retrieval import corpus
from tests.persistence.test_grounded_runtime import body, config, create_student, running
from tests.persistence.test_http_runtime import headers

pytestmark = pytest.mark.asyncio


async def test_context_replies_are_durable_isolated_and_mixed_requests_keep_real_grounding(
    pg, postgres_fixture, monkeypatch, tmp_path
):
    collection, revision, _, _ = await corpus(postgres_fixture)
    # Bind the authorized student after creation; no student-supplied access flags.
    uid = uuid4()
    cfg = config(collection, revision, uid)
    async with running(pg, postgres_fixture, cfg, monkeypatch, tmp_path) as (
        client,
        runtime,
        planner,
        requests,
    ):
        user, session = await create_student(client, runtime, uid)
        other, other_session = await create_student(client, runtime)
        await runtime.session_context_service.update(
            user_id=user.user_id,
            session_id=session.session_id,
            changes=SessionContextUpdate(current_step_id="FLOW-1"),
        )
        await runtime.session_context_service.update(
            user_id=other.user_id,
            session_id=other_session.session_id,
            changes=SessionContextUpdate(current_step_id="PRIVATE-OTHER"),
        )
        generate = runtime.model_service.generate
        original = generate.side_effect
        kind = "recall"

        async def dispatch(request):
            if request.metadata.get("task") != "semantic_routing":
                return await original(request)
            requests.append(request)
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "route": "process_coach" if kind == "mixed" else "fallback",
                        "reason": "Controlled semantic decision.",
                        "current_step": {
                            "action": "unchanged",
                            "step_id": None,
                            "source_quote": None,
                        },
                        "interaction": {
                            "kind": kind,
                            "target": "current_step_id" if kind == "recall" else None,
                            "source_quote": None,
                        },
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )

        generate.side_effect = dispatch
        before = await runtime.repositories.user_contexts.get(user.user_id)
        payload = {**body(user, session), "user_message": "Which step am I on?"}
        first = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert first.status_code == 200
        assert "FLOW-1" in first.json()["final_response"]
        assert "PRIVATE-OTHER" not in first.text
        assert first.json()["metadata"]["response_signal_recorded"] is False
        assert generate.await_count == 1 and planner.requests == []
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == first.json() and generate.await_count == 1
        kind = "acknowledgement"
        thanks = await client.post(
            "/orchestration/invoke",
            headers=headers(user),
            json={**body(user, session), "user_message": "Thank you, understood."},
        )
        assert thanks.status_code == 200 and "You're welcome" in thanks.text
        assert generate.await_count == 2 and planner.requests == []
        assert (
            await runtime.repositories.user_contexts.get(user.user_id)
        ).version == before.version
        kind = "mixed"
        mixed = await client.post(
            "/orchestration/invoke",
            headers=headers(user),
            json={**body(user, session), "user_message": "Thanks; which source record is FLOW-1?"},
        )
        assert mixed.status_code == 200
        assert len(planner.requests) == 1
        assert len(mixed.json()["source_references"]) == 1
        assert not mixed.json()["metadata"].get("context_reply_handled")
    async with running(pg, postgres_fixture, cfg, monkeypatch, tmp_path) as (
        client,
        _,
        _,
        requests,
    ):
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == first.json() and requests == []
        history = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(user)
        )
        assert len(history.json()["items"]) == 6
        assert history.json()["items"][1]["content"] == first.json()["final_response"]
        denied = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(other)
        )
        assert denied.status_code == 404
