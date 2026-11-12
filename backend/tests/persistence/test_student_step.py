"""Real PostgreSQL and signed HTTP: state provenance, replay, restart and onboarding."""

import json
from uuid import uuid4

import pytest

from app.services.model_service.schemas import ModelChatResponse
from tests.persistence.test_http_runtime import headers, running, student

pytestmark = pytest.mark.asyncio


def router(generate, step):
    original = generate.side_effect

    async def dispatch(request):
        if request.metadata.get("task") == "semantic_routing":
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "route": "scenario",
                        "reason": "Controlled signed-HTTP state test.",
                        "current_step": step,
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )
        return await original(request)

    generate.side_effect = dispatch


async def test_student_step_persists_with_source_id_replays_once_and_survives_restart(
    pg, monkeypatch
):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        user, session = await student(client, runtime)
        router(
            generate,
            {"action": "set", "step_id": "FLOW-B7", "source_quote": "I am now working on FLOW-B7."},
        )
        payload = {
            "student_id": user.student_id,
            "session_id": str(session.session_id),
            "request_id": str(uuid4()),
            "user_message": "I am now working on FLOW-B7.",
        }
        reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert reply.status_code == 200
        data = reply.json()
        assert data["metadata"]["student_step_status"] == "persisted"
        assert data["route"] == "scenario"
        detail = (
            await client.get(f"/conversations/{session.session_id}", headers=headers(user))
        ).json()
        assert detail["context"]["current_step_id"] == "FLOW-B7"
        version = detail["context"]["version"]
        history = (
            await client.get(f"/conversations/{session.session_id}/messages", headers=headers(user))
        ).json()["items"]
        assert data["metadata"]["student_step_source_message_id"] == history[0]["message_id"]
        assert len(history) == 2
        before = generate.await_count
        repeated = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert repeated.json() == data and generate.await_count == before
        assert (
            await runtime.session_context_service.get_required(
                user_id=user.user_id,
                session_id=session.session_id,
            )
        ).version == version

    async with running(pg, monkeypatch) as (client, runtime, generate):
        detail = (
            await client.get(f"/conversations/{session.session_id}", headers=headers(user))
        ).json()
        assert detail["context"]["current_step_id"] == "FLOW-B7"
        assert detail["context"]["version"] == version
        replay = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
        assert replay.json() == data and generate.await_count == 0
        router(generate, {"action": "unchanged", "step_id": None, "source_quote": None})
        hypothetical = {
            **payload,
            "request_id": str(uuid4()),
            "user_message": "For later, how would FLOW-C8 work? I am still on FLOW-B7.",
        }
        later = await client.post("/orchestration/invoke", headers=headers(user), json=hypothetical)
        assert later.status_code == 200
        assert later.json()["metadata"]["student_step_status"] == "unchanged"
        assert (
            await runtime.session_context_service.get_required(
                user_id=user.user_id,
                session_id=session.session_id,
            )
        ).current_step_id == "FLOW-B7"
        assert (
            await runtime.user_context_service.get_or_create(user_id=user.user_id)
        ).progress.steps == []


async def test_explicit_role_and_correction_are_saved_with_student_message_provenance(
    pg, monkeypatch
):
    async with running(pg, monkeypatch) as (client, runtime, generate):
        from app.identity import UserIdentity

        user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
        session = (await client.post("/conversations", headers=headers(user), json={})).json()
        roles = iter(["production planner", "inventory analyst"])

        async def extract(request):
            assert request.metadata["task"] == "scenario_draft_extraction"
            assert set(request.response_schema["required"]) == set(
                request.response_schema["properties"]
            )
            role = next(roles)
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "company_context": "UrbanCycle assembles bicycles."
                        if role == "production planner"
                        else None,
                        "learner_role": role,
                        "business_problem": None,
                        "business_goals": [],
                        "constraints": [],
                        "proposed_facts": [],
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )

        generate.side_effect = extract
        for role, message in [
            (
                "production planner",
                "I am a production planner at UrbanCycle. We assemble bicycles.",
            ),
            (
                "inventory analyst",
                "Correction: my role is inventory analyst; the company is unchanged.",
            ),
        ]:
            reply = await client.post(
                "/orchestration/invoke",
                headers=headers(user),
                json={
                    "student_id": user.student_id,
                    "session_id": session["session_id"],
                    "request_id": str(uuid4()),
                    "user_message": message,
                },
            )
            assert reply.status_code == 200
            assert reply.json()["metadata"]["scenario_role_status"] == "persisted"
            assert reply.json()["metadata"]["scenario_source_recorded"] is True
            detail = (
                await client.get(f"/conversations/{session['session_id']}", headers=headers(user))
            ).json()
            draft = detail["context"]["scenario_draft"]
            assert draft["learner_role"] == role
            assert draft["company_context"] == "UrbanCycle assembles bicycles."
            assert detail["context"]["scenario_setup_status"] == "collecting"
        history = (
            await client.get(
                f"/conversations/{session['session_id']}/messages", headers=headers(user)
            )
        ).json()["items"]
        assert draft["source_message_ids"] == [history[0]["message_id"], history[2]["message_id"]]
        assert (
            await runtime.user_context_service.get_or_create(user_id=user.user_id)
        ).scenario_card is None
