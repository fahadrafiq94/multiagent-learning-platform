"""Independent trusted model routing, compatibility and credential boundaries."""

import pytest

from app.config.settings import Settings, settings
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from app.services.model_service.exceptions import ModelEmbeddingError, ModelServiceError
from app.services.model_service.factory import create_model_service
from app.services.model_service.profiles import resolve_profile
from app.services.model_service.schemas import (
    EmbeddingModelInfo,
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelHealthResponse,
    ModelMessage,
)
from app.services.model_service.service import ModelService, close_model_services
from tests.orchestration.test_graph import confirm_scenario, create_user_and_session
from tests.orchestration.test_graph import create_model_service as mock_model


def config(**values):
    return Settings(
        _env_file=None,
        **{
            "model_provider": "vllm",
            "vllm_base_url": "http://primary/v1",
            "vllm_chat_model": "chat",
            "vllm_api_key": "primary-secret",
            "vllm_reasoning_control": "enable_thinking",
            "vllm_reasoning_enabled": False,
            "ollama_base_url": "http://legacy:11434",
            "ollama_chat_model": "legacy-chat",
            "ollama_embedding_model": "emb:latest",
            **values,
        },
    )


@pytest.mark.parametrize(
    "field",
    [
        "orchestrator_provider",
        "summary_reasoning_enabled",
        "rag_planner_provider",
        "embedding_provider",
    ],
)
def test_blank_optional_environment_options_inherit(field):
    assert getattr(config(**{field: ""}), field) is None


def test_cross_endpoint_credentials_and_reasoning_are_not_inherited():
    cfg = config(process_coach_base_url="http://secondary/v1", process_coach_agent_model="small")
    role = resolve_profile(cfg, "process_coach")
    assert role.model == "small" and role.api_key is None
    assert role.reasoning_control == "unsupported" and role.reasoning_enabled is None
    same = resolve_profile(config(process_coach_base_url="http://primary"), "process_coach")
    assert same.api_key.get_secret_value() == "primary-secret"
    assert same.reasoning_enabled is False
    explicit = resolve_profile(
        config(process_coach_api_key="", process_coach_agent_model="small"), "process_coach"
    )
    assert explicit.api_key.get_secret_value() == ""
    assert "primary-secret" not in repr(role)


def test_explicit_role_provider_uses_its_own_provider_defaults():
    role = resolve_profile(config(summary_provider="ollama"), "summary")
    assert role.provider == "ollama" and role.model == "legacy-chat"
    assert role.base_url == "http://legacy:11434" and role.api_key is None


def test_legacy_planner_is_not_reinterpreted_as_vllm_when_primary_changes():
    selected = resolve_profile(
        config(), "rag_planner", legacy_model="old-planner", legacy_url="http://old:11434"
    )
    assert selected.provider == "ollama" and selected.model == "old-planner"
    assert selected.base_url == "http://old:11434"
    selected = resolve_profile(
        config(rag_planner_provider="vllm"),
        "rag_planner",
        legacy_model="old-planner",
        legacy_url="http://old:11434",
    )
    assert selected.provider == "vllm" and selected.model == "chat"
    assert selected.base_url == "http://primary/v1"


def test_environment_planner_and_cli_endpoint_precedence():
    cfg = config(
        rag_planner_provider="vllm",
        rag_planner_base_url="http://planner/v1",
        rag_planner_model="planner",
        rag_planner_api_key="planner-secret",
    )
    selected = resolve_profile(cfg, "rag_planner", legacy_model="old", legacy_url="http://old")
    assert selected.model == "planner" and selected.base_url == "http://planner/v1"
    # Explicit API credentials also must not follow a one-off CLI endpoint override.
    overridden = resolve_profile(cfg, "rag_planner", base_url="http://cli/v1")
    assert overridden.base_url == "http://cli/v1"
    assert overridden.api_key is None


@pytest.mark.parametrize(
    "url", ["http://user:secret@host", "http://host?key=secret", "file:///secret"]
)
def test_profile_urls_are_validated_without_leaking_secrets(url):
    with pytest.raises(ModelServiceError) as error:
        resolve_profile(config(embedding_base_url=url), "embedding")
    assert "secret" not in str(error.value.to_dict())


def test_embedding_model_and_provider_conflicts_fail_before_network():
    with pytest.raises(ModelServiceError):
        resolve_profile(config(embedding_model="different"), "embedding", legacy_model="emb:latest")
    with pytest.raises(ModelServiceError):
        resolve_profile(config(embedding_provider="vllm"), "embedding")
    profile = resolve_profile(
        config(embedding_model="emb", embedding_base_url="http://embedding:11434"),
        "embedding",
        legacy_model="emb:latest",
    )
    assert profile.provider == "ollama" and profile.base_url == "http://embedding:11434"


class MarkerProvider:
    def __init__(self, label, events, delegate, **kwargs):
        self.label, self.events, self.delegate, self.cfg = label, events, delegate, kwargs
        self.closed = 0

    async def generate(self, req):
        self.events.append(
            (
                self.cfg["base_url"],
                "generate",
                req.model or self.cfg["chat_model"],
                req.metadata.get("task"),
                self.cfg.get("api_key"),
            )
        )
        if self.delegate:
            response = await self.delegate.generate(req)
            return response.model_copy(
                update={"provider": self.label, "model": req.model or self.cfg["chat_model"]}
            )
        return ModelChatResponse(
            content="reply",
            model=req.model or self.cfg["chat_model"],
            provider=self.label,
            latency_ms=1,
        )

    async def stream(self, req):
        yield await self.generate(req)

    async def embed(self, req):
        self.events.append((self.cfg["base_url"], "embed", req.model, None, None))
        from app.services.model_service.schemas import ModelEmbeddingResponse

        return ModelEmbeddingResponse(
            embeddings=[[1.0, 0.0, 0.0] for _ in req.texts],
            model=req.model or self.cfg["embedding_model"],
            provider=self.label,
            latency_ms=1,
        )

    async def inspect_embedding_model(self, model):
        return EmbeddingModelInfo(
            provider=self.label, model=model, revision="a" * 64, dimension=3, context_length=512
        )

    async def health(self):
        return ModelHealthResponse(
            provider=self.label, status="ok", chat_model=self.cfg["chat_model"]
        )

    async def aclose(self):
        self.closed += 1


def markers(monkeypatch, delegate=None):
    created = []
    events = []

    def factory(label):
        def create(**kwargs):
            result = MarkerProvider(label, events, delegate, **kwargs)
            created.append(result)
            return result

        return create

    monkeypatch.setattr("app.services.model_service.factory.OllamaModelProvider", factory("ollama"))
    monkeypatch.setattr("app.services.model_service.factory.VLLMModelProvider", factory("vllm"))
    return created, events


@pytest.mark.asyncio
async def test_chat_embedding_and_roles_route_independently_and_close_once(monkeypatch):
    created, events = markers(monkeypatch)
    cfg = config(
        embedding_provider="ollama",
        embedding_base_url="http://embeddings:11434",
        embedding_model="emb",
        summary_provider="ollama",
        summary_base_url="http://summaries:11434",
        summary_model="small",
        process_coach_base_url="http://coach/v1",
        process_coach_agent_model="coach",
        scenario_agent_base_url="http://coach/v1",
        scenario_agent_model="coach",
    )
    model = create_model_service(cfg)
    req = ModelChatRequest(
        messages=[ModelMessage(role="user", content="Hello")],
        metadata={"role": "summary", "base_url": "http://untrusted"},
    )
    assert (await model.generate(req)).provider == "vllm"
    assert (await model.for_role("summary").generate(req)).provider == "ollama"
    await model.for_role("process_coach").generate(req)
    assert model.for_role("scenario_agent") is model.for_role("process_coach")
    embedded = await model.embed(ModelEmbeddingRequest(texts=["query"], model="emb:latest"))
    assert embedded.provider == "ollama"
    assert (await model.inspect_embedding_model("emb:latest")).dimension == 3
    assert [e[0] for e in events] == [
        "http://primary/v1",
        "http://summaries:11434",
        "http://coach/v1",
        "http://embeddings:11434",
    ]
    assert events[2][-1] is None
    with pytest.raises(ModelEmbeddingError):
        await model.embed(ModelEmbeddingRequest(texts=["x"], model="wrong"))
    with pytest.raises(ModelEmbeddingError):
        await model.inspect_embedding_model("wrong")
    assert len(events) == 4
    health = await model.health()
    assert health.status == "ok" and health.embedding_model == "emb"
    assert health.details["profiles"]["embedding"]["provider"] == "ollama"
    await model.aclose()
    await model.aclose()
    assert all(p.closed == 1 for p in created)


@pytest.mark.asyncio
async def test_same_primary_role_profile_is_shared_without_a_cycle(monkeypatch):
    created, _ = markers(monkeypatch)
    model = create_model_service(config(summary_base_url="http://primary"))
    assert model.for_role("summary") is model and len(created) == 1
    assert (await model.health()).status == "ok"
    await model.aclose()
    assert created[0].closed == 1


@pytest.mark.asyncio
async def test_actual_graph_selects_role_endpoint_not_only_model_name(monkeypatch):
    delegate, _, _ = mock_model()
    created, events = markers(monkeypatch, delegate)
    cfg = config(
        orchestrator_base_url="http://router/v1",
        orchestrator_model="router",
        process_coach_provider="ollama",
        process_coach_base_url="http://coach:11434",
        process_coach_agent_model="small",
    )
    for field in ("orchestrator_model", "process_coach_agent_model"):
        monkeypatch.setattr(settings, field, getattr(cfg, field))
    root = create_model_service(cfg)
    runtime = build_orchestration_runtime(model_service=root)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    result = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=user.student_id,
            user_message="Why do we need a purchase order?",
        )
    )
    assert result.route == "process_coach"
    assert any(e[:3] == ("http://router/v1", "generate", "router") for e in events)
    assert any(e[:3] == ("http://coach:11434", "generate", "small") for e in events)
    await root.aclose()
    assert all(p.closed == 1 for p in created)


@pytest.mark.asyncio
async def test_cleanup_closes_other_services_even_if_one_fails():
    closed = []

    class Resource:
        async def aclose(self):
            closed.append("good")

    class Broken:
        async def aclose(self):
            raise RuntimeError("close failed")

    resource = Resource()
    with pytest.raises(RuntimeError):
        await close_model_services(resource, resource, Broken())
    assert closed == ["good"]


@pytest.mark.asyncio
async def test_stateless_legacy_subclass_without_facade_constructor_closes_safely():
    class Legacy(ModelService):
        def __init__(self):
            self.requests = []

    model = Legacy()
    await close_model_services(model, model)
    await model.aclose()


@pytest.mark.asyncio
async def test_composed_health_has_one_total_deadline(monkeypatch):
    import asyncio

    monkeypatch.setattr("app.services.model_service.service.PROFILE_HEALTH_TIMEOUT_SECONDS", 0.001)
    created, _ = markers(monkeypatch)
    model = create_model_service(config(embedding_model="emb"))

    async def slow():
        await asyncio.sleep(1)
        return ModelHealthResponse(provider="ollama", status="ok")

    created[-1].health = slow
    result = await model.health()
    assert result.status == "unavailable" and result.details["reason"] == "profile_health_timeout"
    await model.aclose()
