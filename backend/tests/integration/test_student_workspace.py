"""Student panel projection follows onboarding and hides internal context."""

import httpx
import pytest
from pydantic import SecretStr

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.main import app
from app.orchestration.factory import build_orchestration_runtime
from tests.integration.test_orchestration_api import create_model_service


@pytest.mark.asyncio
async def test_panel_uses_owner_scope_and_requires_confirmation(monkeypatch):
    secret = SecretStr("student-workspace-test-" * 3)
    model, _, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", secret)
    user = await runtime.repositories.identities.create(UserIdentity(student_id="panel-owner"))
    other = await runtime.repositories.identities.create(UserIdentity(student_id="panel-other"))
    headers = {"Authorization": "Bearer " + issue_token(user.user_id, secret)}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/conversations", headers=headers, json={})
        sid = created.json()["session_id"]
        path = f"/student-api/conversations/{sid}"
        assert (await client.get(path)).status_code == 401
        assert (
            await client.get(
                path, headers={"Authorization": "Bearer " + issue_token(other.user_id, secret)}
            )
        ).status_code == 404
        assert (await client.get(path, headers=headers)).json()["scenario"][
            "status"
        ] == "not_started"
        body = {
            "session_id": sid,
            "student_id": user.student_id,
            "user_message": "I plan bicycle production and want to improve delivery reliability.",
        }
        assert (
            await client.post("/orchestration/invoke", headers=headers, json=body)
        ).status_code == 200
        draft = await client.get(path, headers=headers)
        assert draft.status_code == 200
        assert draft.json()["scenario"]["status"] == "reviewing"
        assert draft.json()["scenario"]["learner_role"] == "Production Planner"
        assert (await runtime.scenario_card_service.get(user_id=user.user_id)) is None
        assert "pending_evidence_receipt" not in draft.text and "summary" not in draft.text
        assert "source_message_ids" not in draft.text
        assert draft.headers["cache-control"] == "no-store"
        body["user_message"] = "I confirm the scenario is correct."
        assert (
            await client.post("/orchestration/invoke", headers=headers, json=body)
        ).status_code == 200
        assert (await client.get(path, headers=headers)).json()["scenario"]["status"] == "confirmed"
        new = await client.post("/conversations", headers=headers, json={})
        panel = await client.get(
            f"/student-api/conversations/{new.json()['session_id']}", headers=headers
        )
        assert panel.json()["scenario"]["status"] == "confirmed"
        assert panel.json()["scenario"]["company_context"] == "UrbanCycle manufactures bicycles."


@pytest.mark.asyncio
async def test_rename_and_delete_are_owner_scoped(monkeypatch):
    secret = SecretStr("conversation-mutation-test-" * 3)
    model, _, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", secret)
    owner = await runtime.repositories.identities.create(UserIdentity(student_id="chat-owner"))
    other = await runtime.repositories.identities.create(UserIdentity(student_id="chat-other"))
    owner_headers = {"Authorization": "Bearer " + issue_token(owner.user_id, secret)}
    other_headers = {"Authorization": "Bearer " + issue_token(other.user_id, secret)}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/conversations", headers=owner_headers, json={})
        sid = created.json()["session_id"]
        assert (
            await client.patch(
                f"/conversations/{sid}", headers=other_headers, json={"title": "Stolen"}
            )
        ).status_code == 404
        renamed = await client.patch(
            f"/conversations/{sid}", headers=owner_headers, json={"title": "Renamed chat"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Renamed chat"
        assert (
            await client.delete(f"/conversations/{sid}", headers=other_headers)
        ).status_code == 404
        assert (
            await client.delete(f"/conversations/{sid}", headers=owner_headers)
        ).status_code == 204
        assert (await client.get(f"/conversations/{sid}", headers=owner_headers)).status_code == 404
