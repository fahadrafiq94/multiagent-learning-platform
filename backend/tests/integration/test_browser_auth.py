"""Real auth boundary: cookie isolation, expiry, CSRF, logout and bearer compatibility."""

from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import Settings, settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.identity.schemas import UserStatus
from app.main import app
from app.orchestration.factory import build_orchestration_runtime
from tests.integration.test_orchestration_api import create_model_service

ORIGIN = "http://test"
SECRET = SecretStr("browser-auth-test-secret-" * 3)


@pytest.fixture
def runtime(monkeypatch):
    model, _, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr(settings, "student_browser_origin", ORIGIN)
    monkeypatch.setattr(settings, "student_browser_secure_cookie", False)
    monkeypatch.setattr(settings, "app_env", "development")
    return runtime


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN)


@pytest.mark.asyncio
async def test_cookie_sessions_isolate_students_and_protect_every_cookie_write(runtime):
    users = [
        await runtime.repositories.identities.create(UserIdentity(student_id=name))
        for name in ("student-one", "student-two")
    ]
    async with client() as one, client() as two:
        sessions = []
        for browser, user in zip((one, two), users, strict=True):
            token = issue_token(user.user_id, SECRET, lifetime_seconds=60)
            result = await browser.post(
                "/auth/session", headers={"Origin": ORIGIN}, json={"credential": token}
            )
            assert result.status_code == 200
            assert token not in result.text
            cookie = result.headers["set-cookie"]
            assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
            assert "Domain=" not in cookie
            assert 0 < int(cookie.split("Max-Age=", 1)[1].split(";", 1)[0]) <= 60
            assert result.headers["cache-control"] == "no-store"
            session = (await browser.get("/auth/session")).json()
            assert session == result.json()
            assert session["user"] == {"user_id": str(user.user_id), "student_id": user.student_id}
            assert "set-cookie" not in (await browser.get("/auth/session")).headers
            sessions.append(session)
        for headers in (
            {},
            {"Origin": ORIGIN},
            {"Origin": "http://evil", "X-FREDI-CSRF": sessions[0]["csrf_token"]},
        ):
            assert (await one.post("/conversations", headers=headers, json={})).status_code == 403
        safe = {"Origin": ORIGIN, "X-FREDI-CSRF": sessions[0]["csrf_token"]}
        created = await one.post("/conversations", headers=safe, json={"title": "Private"})
        assert created.status_code == 201
        identifier = UUID(created.json()["session_id"])
        assert (await two.get(f"/conversations/{identifier}")).status_code == 404
        assert (await two.get("/conversations")).json()["items"] == []
        assert (await one.post("/auth/logout", headers={"Origin": ORIGIN})).status_code == 403
        assert (await one.post("/auth/logout", headers=safe)).status_code == 204
        assert (await one.get("/me")).status_code == 401
        assert (await two.get("/me")).json()["student_id"] == "student-two"
        # Existing API clients retain bearer auth and do not need browser CSRF headers.
        assert (
            await one.post(
                "/conversations",
                json={},
                headers={"Authorization": "Bearer " + issue_token(users[0].user_id, SECRET)},
            )
        ).status_code == 201


@pytest.mark.asyncio
async def test_expired_and_disabled_browser_sessions_fail_closed(runtime, monkeypatch):
    user = await runtime.repositories.identities.create(UserIdentity(student_id="expiry"))
    token = issue_token(user.user_id, SECRET, lifetime_seconds=60)
    async with client() as browser:
        assert (
            await browser.post(
                "/auth/session", headers={"Origin": ORIGIN}, json={"credential": token}
            )
        ).status_code == 200
        monkeypatch.setattr("app.identity.pilot_tokens.time.time", lambda: 10**12)
        assert (await browser.get("/me")).status_code == 401
        assert (
            await browser.post(
                "/auth/session", headers={"Origin": ORIGIN}, json={"credential": token}
            )
        ).status_code == 401


@pytest.mark.asyncio
async def test_login_origin_payload_bounds_and_disabled_identity(runtime, monkeypatch):
    user = await runtime.repositories.identities.create(
        UserIdentity(student_id="disabled", status=UserStatus.DISABLED)
    )
    token = issue_token(user.user_id, SECRET)
    async with client() as browser:
        for origin in (None, "http://evil"):
            assert (
                await browser.post(
                    "/auth/session",
                    headers={"Origin": origin} if origin else {},
                    json={"credential": token},
                )
            ).status_code == 403
        for body in (
            {"credential": token},
            {"credential": token, "student_id": "forged"},
            {"credential": [token]},
        ):
            reply = await browser.post("/auth/session", headers={"Origin": ORIGIN}, json=body)
            assert reply.status_code == 401 and token not in reply.text
        assert (
            await browser.post(
                "/auth/session", headers={"Origin": ORIGIN}, content="credential=" + token
            )
        ).status_code == 415
        assert (
            await browser.post(
                "/auth/session",
                headers={"Origin": ORIGIN, "Content-Type": "application/json"},
                content="x" * 4097,
            )
        ).status_code == 413
        monkeypatch.setattr(settings, "app_env", "production")
        assert (
            await browser.post(
                "/auth/session", headers={"Origin": ORIGIN}, json={"credential": token}
            )
        ).status_code == 503


def test_browser_origin_is_a_trusted_normalized_origin():
    assert (
        Settings(
            _env_file=None, student_browser_origin="HTTPS://Example.COM:443/"
        ).student_browser_origin
        == "https://example.com"
    )
    for origin in (
        "https://user:password@example.com",
        "http://example.com/path",
        "http://example.com?x=1",
        "javascript:alert(1)",
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, student_browser_origin=origin)


@pytest.mark.asyncio
async def test_secure_cookie_and_frontend_asset_boundary(runtime, monkeypatch):
    monkeypatch.setattr(settings, "student_browser_secure_cookie", True)
    user = await runtime.repositories.identities.create(UserIdentity(student_id="secure"))
    async with client() as browser:
        reply = await browser.post(
            "/auth/session",
            headers={"Origin": ORIGIN},
            json={"credential": issue_token(user.user_id, SECRET)},
        )
        assert reply.status_code == 200
        cookie = reply.headers["set-cookie"]
        assert "__Host-fredi_pilot=" in cookie and "Secure" in cookie
        entry = await browser.get("/student")
        assert (
            entry.status_code == 200
            and "frame-ancestors 'none'" in entry.headers["content-security-policy"]
        )
        assert entry.headers["cache-control"] == "no-store"
        assert (await browser.get("/student/css/fredi.css")).status_code == 200
        for path in (
            "/student/.env",
            "/student/generated/openapi.json",
            "/student/package.json",
            "/student/tests/server.py",
        ):
            assert (await browser.get(path)).status_code == 404
