"""Settings, actual runtime selection and resource ownership for Checkpoint 1."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.routes.model import router as model_router
from app.config.settings import Settings, settings
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from app.runtime import runtime_lifespan
from app.services.model_service.exceptions import ModelServiceError, UnsupportedModelProviderError
from app.services.model_service.factory import create_model_service
from app.services.model_service.ollama_provider import OllamaModelProvider
from app.services.model_service.service import ModelService
from app.services.model_service.vllm_provider import VLLMModelProvider
from tests.orchestration.test_graph import (
    confirm_scenario,
    create_user_and_session,
)
from tests.orchestration.test_graph import create_model_service as mock_model
from tests.services.model_service.test_model_service import FakeModelProvider
from tests.services.model_service.test_vllm_provider import provider


@pytest.mark.parametrize("env_name", ["PROCESS_COACH_MODEL", "PROCESS_COACH_AGENT_MODEL"])
def test_environment_selection_and_legacy_coach_alias(monkeypatch, env_name):
    monkeypatch.setenv("MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "http://inference:8000/v1")
    monkeypatch.setenv("VLLM_CHAT_MODEL", "served")
    monkeypatch.setenv("VLLM_API_KEY", "private-key")
    monkeypatch.setenv("VLLM_REASONING_CONTROL", "enable_thinking")
    monkeypatch.setenv(env_name, "coach-alias")
    cfg = Settings(_env_file=None)
    assert cfg.process_coach_agent_model == "coach-alias"
    assert isinstance(cfg.vllm_api_key, SecretStr) and "private-key" not in repr(cfg)
    service = create_model_service(cfg)
    assert isinstance(service._provider, VLLMModelProvider)
    assert service._provider.chat_model == "served" and service._provider._client is None


def test_ollama_remains_available_without_vllm_configuration():
    cfg = Settings(
        _env_file=None,
        model_provider="ollama",
        ollama_base_url="http://local",
        ollama_chat_model="chat",
        ollama_embedding_model="embedding",
    )
    assert isinstance(create_model_service(cfg)._provider, OllamaModelProvider)


@pytest.mark.parametrize(
    "values", [{}, {"vllm_base_url": "http://server"}, {"vllm_chat_model": "chat"}]
)
def test_explicit_vllm_never_falls_back_to_ollama_when_incomplete(values):
    cfg = Settings(_env_file=None, model_provider="vllm", **values)
    with pytest.raises(ModelServiceError) as error:
        create_model_service(cfg)
    assert error.value.code == "model_not_configured"


def test_unknown_or_automatic_provider_selection_is_not_silently_enabled():
    with pytest.raises(UnsupportedModelProviderError):
        create_model_service(Settings(_env_file=None, model_provider="auto"))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_build", [False, True])
async def test_asgi_lifespan_closes_provider_on_shutdown_and_partial_startup(
    monkeypatch, fail_build
):
    p, _ = provider()
    service = ModelService(p)
    monkeypatch.setattr(settings, "application_storage", "memory")
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "knowledge_intake_enabled", False)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: service)

    async def exercise():
        async with runtime_lifespan(FastAPI()):
            assert (await service.health()).status == "ok"

    if fail_build:

        def fail(**kwargs):
            raise RuntimeError("construction failed")

        monkeypatch.setattr("app.runtime.build_orchestration_runtime", fail)
        with pytest.raises(RuntimeError, match="construction failed"):
            await exercise()
    else:
        await exercise()
    assert p._closed
    if p._client:
        assert p._client.is_closed


@pytest.mark.asyncio
async def test_closing_legacy_stateless_provider_is_compatible():
    await ModelService(FakeModelProvider()).aclose()


@pytest.mark.asyncio
async def test_closeable_provider_is_delegated():
    p = FakeModelProvider()
    p.aclose = AsyncMock()
    await ModelService(p).aclose()
    p.aclose.assert_awaited_once()


def test_model_routes_reuse_lifespan_client_and_fail_without_startup(monkeypatch):
    p, calls = provider()
    service = ModelService(p)
    monkeypatch.setattr(settings, "application_storage", "memory")
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "knowledge_intake_enabled", False)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: service)
    app = FastAPI(lifespan=runtime_lifespan)
    app.include_router(model_router)
    with TestClient(app) as client:
        assert client.get("/model/health").json()["provider"] == "vllm"
        for _ in range(2):
            result = client.post(
                "/model/generate",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 32,
                },
            )
            assert result.status_code == 200 and result.json()["content"] == "Verified reply"
        assert not p._closed
        assert len([c for c in calls if c.url.path == "/v1/models"]) == 1
    assert p._closed and p._client.is_closed
    with TestClient(FastAPI()) as unused:
        # A router on an app without the managed runtime must fail cleanly.
        unused.app.include_router(model_router)
        assert unused.get("/model/health").status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,role,task",
    [
        ("Why do we need a purchase order?", "process_coach", "process_coach_response"),
        ("Which AP+ field do I use?", "ap_plus_navigator", "ap_plus_navigator_response"),
        ("Explain the company scenario.", "scenario", "scenario_agent_response"),
    ],
)
async def test_environment_model_overrides_reach_real_orchestration_requests(
    monkeypatch, message, role, task
):
    for name, value in {
        "orchestrator_model": "router-model",
        "scenario_agent_model": "scenario-model",
        "process_coach_agent_model": "coach-model",
        "ap_plus_navigator_model": "navigator-model",
    }.items():
        monkeypatch.setattr(settings, name, value)
    service, _, requests = mock_model()
    runtime = build_orchestration_runtime(model_service=service)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=user.student_id,
            user_message=message,
        )
    )
    assert response.route == role
    routed = [req for req in requests if req.metadata.get("task") == "semantic_routing"]
    assert routed and all(req.model == "router-model" for req in routed)
    selected = [req for req in requests if req.metadata.get("task") == task]
    assert (
        selected
        and selected[-1].model
        == {
            "scenario": "scenario-model",
            "process_coach": "coach-model",
            "ap_plus_navigator": "navigator-model",
        }[role]
    )
