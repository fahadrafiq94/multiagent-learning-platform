"""Browser cookie identity and private conversation survive a real SQL runtime restart."""

from uuid import UUID, uuid4

import pytest

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.persistence.turn_models import TurnStatus
from tests.persistence.test_http_runtime import SECRET, running


@pytest.mark.asyncio
async def test_browser_identity_uses_durable_repository_after_restart(pg, monkeypatch):
    monkeypatch.setattr(settings, "student_browser_origin", "http://test")
    monkeypatch.setattr(settings, "student_browser_secure_cookie", False)
    monkeypatch.setattr(settings, "app_env", "development")
    async with running(pg, monkeypatch) as (client, runtime, _):
        user = await runtime.repositories.identities.create(
            UserIdentity(student_id="browser-pilot")
        )
        login = await client.post(
            "/auth/session",
            headers={"Origin": "http://test"},
            json={"credential": issue_token(user.user_id, SECRET)},
        )
        assert login.status_code == 200
        created = await client.post(
            "/conversations",
            json={"title": "Durable browser chat"},
            headers={"Origin": "http://test", "X-FREDI-CSRF": login.json()["csrf_token"]},
        )
        assert created.status_code == 201
        cookies = client.cookies
        identity = login.json()["user"]
    async with running(pg, monkeypatch) as (client, _, _):
        client.cookies.update(cookies)
        resumed = await client.get("/auth/session")
        assert resumed.status_code == 200 and resumed.json()["user"] == identity
        assert (await client.get("/me")).json() == identity
        conversations = await client.get("/conversations")
        assert conversations.json()["items"][0]["session_id"] == created.json()["session_id"]
        assert (
            await client.post(
                "/auth/logout",
                headers={"Origin": "http://test", "X-FREDI-CSRF": resumed.json()["csrf_token"]},
            )
        ).status_code == 204
        assert (await client.get("/me")).status_code == 401


@pytest.mark.asyncio
async def test_browser_onboarding_history_and_replay_survive_restart(pg, monkeypatch):
    monkeypatch.setattr(settings, "student_browser_origin", "http://test")
    monkeypatch.setattr(settings, "student_browser_secure_cookie", False)
    monkeypatch.setattr(settings, "app_env", "development")
    async with running(pg, monkeypatch) as (client, runtime, _):
        user = await runtime.repositories.identities.create(UserIdentity(student_id="workspace"))
        login = await client.post(
            "/auth/session",
            headers={"Origin": "http://test"},
            json={"credential": issue_token(user.user_id, SECRET)},
        )
        assert login.status_code == 200
        headers = {"Origin": "http://test", "X-FREDI-CSRF": login.json()["csrf_token"]}
        created = await client.post("/conversations", headers=headers, json={})
        assert created.status_code == 201
        sid = created.json()["session_id"]
        payload = {
            "session_id": sid,
            "student_id": user.student_id,
            "request_id": str(uuid4()),
            "user_message": "I plan bicycle production and want to improve delivery reliability.",
        }
        first = await client.post("/orchestration/invoke", headers=headers, json=payload)
        assert first.status_code == 200, first.text
        panel_path = f"/student-api/conversations/{sid}"
        assert (await client.get(panel_path)).json()["scenario"]["status"] == "reviewing"
        payload.update(request_id=str(uuid4()), user_message="I confirm the scenario is correct.")
        confirmed = await client.post("/orchestration/invoke", headers=headers, json=payload)
        assert confirmed.status_code == 200, confirmed.text
        assert (await client.get(panel_path)).json()["scenario"]["status"] == "confirmed"
        history_path = f"/conversations/{sid}/messages"
        history = (await client.get(history_path)).json()
        assert len(history["items"]) == 4
        cookies = client.cookies
    async with running(pg, monkeypatch) as (client, runtime, generate):
        client.cookies.update(cookies)
        assert (await client.get(panel_path)).json()["scenario"]["status"] == "confirmed"
        assert (await client.get(history_path)).json() == history
        replay = await client.post("/orchestration/invoke", headers=headers, json=payload)
        assert replay.status_code == 200 and replay.json() == confirmed.json()
        assert generate.await_count == 0
        assert (await client.get(history_path)).json() == history
        turn = await client.get(f"/conversations/{sid}/turns/{payload['request_id']}")
        assert turn.status_code == 200 and turn.json()["status"] == "completed"
        created = await client.post("/conversations", headers=headers, json={})
        assert created.status_code == 201
        inherited = await client.get(f"/student-api/conversations/{created.json()['session_id']}")
        assert inherited.json()["scenario"]["status"] == "confirmed"
        message_id = history["items"][0]["message_id"]
        edited = await client.patch(
            f"{history_path}/{message_id}",
            headers=headers,
            json={"content": "I plan production for several bicycle lines."},
        )
        assert edited.status_code == 200, edited.text
        updated = (await client.get(history_path)).json()["items"]
        assert updated[0]["content"] == "I plan production for several bicycle lines."
        assert updated[1:] == history["items"][1:]
        revision = {
            **payload,
            "request_id": str(uuid4()),
            "edit_message_id": message_id,
            "user_message": "How do I locate the supplier field in APplus?",
        }
        old_edit = await client.post("/orchestration/invoke", headers=headers, json=revision)
        assert old_edit.status_code == 409, old_edit.text
        revision["edit_message_id"] = history["items"][2]["message_id"]
        turns = runtime.repositories.turns
        assert turns is not None
        failed_scope = dict(user_id=user.user_id, session_id=UUID(sid), request_id=uuid4())
        await turns.begin(
            **failed_scope,
            content=revision["user_message"],
            edit_message_id=UUID(revision["edit_message_id"]),
        )
        assert (await client.get(history_path)).json()["items"] == updated
        await turns.stop(**failed_scope, status=TurnStatus.FAILED, error_code="workflow_failed")
        assert (await client.get(history_path)).json()["items"] == updated
        revised = await client.post("/orchestration/invoke", headers=headers, json=revision)
        assert revised.status_code == 200, revised.text
        calls = generate.await_count
        assert calls > 0
        regenerated = (await client.get(history_path)).json()["items"]
        assert regenerated[:2] == updated[:2]
        assert [m["content"] for m in regenerated[2:]] == [
            revision["user_message"],
            revised.json()["final_response"],
        ]
        assert regenerated[3]["message_id"] != history["items"][3]["message_id"]
        assert (
            await client.post("/orchestration/invoke", headers=headers, json=revision)
        ).json() == revised.json()
        assert generate.await_count == calls
        deleted = await client.delete(f"/conversations/{sid}", headers=headers)
        assert deleted.status_code == 204, deleted.text
        assert (await client.get(history_path)).status_code == 404
        assert (await client.get(panel_path)).status_code == 404
        assert (
            await client.get(f"/student-api/conversations/{created.json()['session_id']}")
        ).json()["scenario"]["status"] == "confirmed"
