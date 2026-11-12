"""Signed management HTTP admission, bounded PDF input and durable worker state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from app.config.settings import settings
from app.identity import UserIdentity
from app.main import app
from tests.knowledge.test_extraction import pdf_bytes
from tests.knowledge.test_intake import setup
from tests.persistence.test_http_runtime import SECRET, headers

pytestmark = pytest.mark.asyncio


async def test_signed_manager_upload_status_and_student_denial(
    postgres_fixture, tmp_path, monkeypatch
):
    collection, binding, _, service = await setup(postgres_fixture, tmp_path)
    manager = UserIdentity(user_id=binding.manager_user_ids[0], student_id="manager")
    student = UserIdentity(user_id=uuid4(), student_id="student")
    identities = AsyncMock()
    identities.get_by_user_id.side_effect = lambda uid: (
        manager if uid == manager.user_id else student
    )
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr(
        app.state,
        "runtime",
        SimpleNamespace(repositories=SimpleNamespace(identities=identities)),
        raising=False,
    )
    monkeypatch.setattr(app.state, "knowledge_intake", service, raising=False)
    url = f"/knowledge/collections/{collection.collection_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        content = pdf_bytes("Uploaded generic PDF course instructions.")
        assert (
            await client.post(url + "/documents?title=Manual", content=content)
        ).status_code == 401
        denied = await client.post(
            url + "/documents?title=Manual",
            content=content,
            headers={**headers(student), "Content-Type": "application/pdf"},
        )
        assert denied.status_code == 403
        wrong_type = await client.post(
            url + "/documents?title=Manual", content=content, headers=headers(manager)
        )
        assert wrong_type.status_code == 415
        too_large_limit = len(content) - 1
        with monkeypatch.context() as limited:
            limited.setattr("app.api.routes.knowledge.MAX_SOURCE_BYTES", too_large_limit)
            oversized = await client.post(
                url + "/documents?title=Manual",
                content=content,
                headers={**headers(manager), "Content-Type": "application/pdf"},
            )
            assert oversized.status_code == 413
        uploaded = await client.post(
            url + "/documents?title=Manual",
            content=content,
            headers={**headers(manager), "Content-Type": "application/pdf"},
        )
        assert uploaded.status_code == 202, uploaded.text
        job = uploaded.json()
        assert job["status"] == "queued" and "payload" not in job and "lease_token" not in job
        assert (await service.run_once(collection.collection_id)).status == "succeeded"
        status = await client.get(url + "/jobs/" + job["job_id"], headers=headers(manager))
        assert status.json()["status"] == "succeeded"
        denied_status = await client.get(url + "/jobs/" + job["job_id"], headers=headers(student))
        assert denied_status.status_code == 403
        retry = await client.post(
            url + "/jobs/" + job["job_id"] + "/retry", headers=headers(manager)
        )
        assert retry.status_code == 409
