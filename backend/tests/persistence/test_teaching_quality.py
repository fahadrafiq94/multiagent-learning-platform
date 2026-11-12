"""Real signed HTTP/SQL persistence; semantic decisions explicitly controlled."""

from uuid import uuid4

import pytest

from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_retrieval import corpus
from tests.persistence.test_grounded_runtime import body, config, create_student, running
from tests.persistence.test_http_runtime import headers


@pytest.mark.asyncio
@pytest.mark.parametrize("teaching", ["followed", "violated", "uncertain"])
async def test_pure_question_and_teaching_failure_are_neutral_durable_and_replayable(
    pg, postgres_fixture, monkeypatch, tmp_path, teaching
):
    kp = postgres_fixture
    collection, revision, _, _ = await corpus(kp)
    uid = uuid4()
    cfg = config(collection, revision, uid)
    async with running(
        pg,
        kp,
        cfg,
        monkeypatch,
        tmp_path,
        teaching=teaching,
        assessment={
            "assessability": "not_assessable",
            "student_claim": None,
            "understanding": "correct",
            "misconception_detected": True,
        },
    ) as (client, runtime, _, requests):
        user, session = await create_student(client, runtime, uid)
        other, _ = await create_student(client, runtime)
        before = (
            await runtime.user_context_service.get_required(user_id=uid)
        ).hint_state.model_dump()
        payload = body(user, session)
        reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert reply.status_code == 200 and not reply.json()["error"]
        result = reply.json()
        abstained = teaching != "followed"
        # An unverified grounded answer is now counted as one unproductive
        # attempt by the application; a verified answer moves no counters.
        assert result["metadata"]["response_signal_recorded"] is abstained
        if abstained:
            assert result["metadata"]["response_reason"] == "unverified_grounded_abstention"
        assert result["metadata"]["guidance_level_used"] == "minimal"
        assert result["metadata"]["guidance_level_next"] == "minimal"
        after = (await runtime.user_context_service.get_required(user_id=uid)).hint_state
        assert after.consecutive_unproductive_attempts == (1 if abstained else 0)
        if not abstained:
            assert after.model_dump() == before
        if teaching == "followed":
            assert result["metadata"]["response_understanding"] == "not_applicable"
            assert result["source_references"]
        else:
            assert not result["source_references"]
            assert result["metadata"]["rag"]["support_status"] == "passed"
            assert result["metadata"]["rag"]["teaching_status"] == teaching
            assert result["metadata"]["rag"]["learning_signal_suppressed"] is True
            assert result["metadata"]["state_update_proposed"] is False
        count = len(requests)
        assert [r.metadata["task"] for r in requests].count("citation_support") == 1
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == result and len(requests) == count
        history = await client.get(
            f"/conversations/{session.session_id}/messages", headers=headers(user)
        )
        assert len(history.json()["items"]) == 2
        assert history.json()["items"][-1]["content"] == result["final_response"]
        assert (
            await client.get(
                f"/conversations/{session.session_id}/messages", headers=headers(other)
            )
        ).status_code == 404
    async with running(pg, kp, cfg, monkeypatch, tmp_path) as (client, _, _, requests):
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == result and not requests
