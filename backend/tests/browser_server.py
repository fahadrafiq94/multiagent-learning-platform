"""Isolated browser-test server. Never imported by the production application."""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from pydantic import SecretStr

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.main import app
from app.orchestration.factory import build_orchestration_runtime
from tests.integration.test_orchestration_api import create_model_service

SECRET = SecretStr(secrets.token_hex(32))
USERS: dict[str, UserIdentity] = {}


@asynccontextmanager
async def lifespan(_app):
    settings.app_env = "development"
    settings.application_pilot_auth_secret = SECRET
    settings.student_browser_origin = "http://127.0.0.1:8766"
    settings.student_browser_secure_cookie = False
    model, _, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    app.state.runtime = runtime
    for name in ("student-one", "student-two"):
        USERS[name] = await runtime.repositories.identities.create(UserIdentity(student_id=name))
    yield
    app.state.runtime = None


app.router.lifespan_context = lifespan


@app.get("/__test__/credentials")
def credentials():
    return {name: issue_token(user.user_id, SECRET) for name, user in USERS.items()}


@app.get("/__test__/short-credential")
def short_credential():
    return {"credential": issue_token(USERS["student-one"].user_id, SECRET, lifetime_seconds=3)}


app.mount(
    "/__prototype",
    StaticFiles(directory=Path(__file__).resolve().parents[2] / "frontend", html=True),
)
