"""Verified browser identity, without exposing provider links or trusting input IDs."""

from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.identity.schemas import UserStatus
from app.main import app
from app.orchestration.factory import build_orchestration_runtime
from tests.integration.test_orchestration_api import create_model_service


@pytest.mark.asyncio
async def test_current_user_is_verified_and_projects_only_public_identity(monkeypatch):
    secret = SecretStr("current-user-test-secret-" * 3)
    model, generate, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", secret)
    user = await runtime.repositories.identities.create(UserIdentity(student_id="student-one"))
    disabled = await runtime.repositories.identities.create(
        UserIdentity(student_id="disabled", status=UserStatus.DISABLED)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for token in (
            None,
            "invalid",
            issue_token(uuid4(), secret),
            issue_token(disabled.user_id, secret),
        ):
            headers = {"Authorization": "Bearer " + token} if token else {}
            assert (await client.get("/me", headers=headers)).status_code == 401
        response = await client.get(
            "/me?student_id=another-student",
            headers={"Authorization": "Bearer " + issue_token(user.user_id, secret)},
        )
        assert response.status_code == 200
        assert response.json() == {"user_id": str(user.user_id), "student_id": "student-one"}
        assert response.headers["Cache-Control"] == "no-store"
        generate.assert_not_called()
