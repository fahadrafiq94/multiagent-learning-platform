from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.api.routes import model as model_routes
from app.main import app
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
)


class FakeModelService:
    async def health(self) -> ModelHealthResponse:
        return ModelHealthResponse(
            provider="ollama",
            status="ok",
            latency_ms=1.0,
            chat_model="qwen3:4b",
            embedding_model="nomic-embed-text-v2-moe",
        )

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        return ModelChatResponse(
            content="fake response",
            model=request.model or "qwen3:4b",
            provider="ollama",
            latency_ms=1.0,
        )

    async def embed(
        self,
        request: ModelEmbeddingRequest,
    ) -> ModelEmbeddingResponse:
        return ModelEmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] for _ in request.texts],
            model=request.model or "nomic-embed-text-v2-moe",
            provider="ollama",
            latency_ms=1.0,
        )

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        model = request.model or "qwen3:4b"
        yield ModelStreamChunk(
            content="hello",
            done=False,
            model=model,
            provider="ollama",
        )
        yield ModelStreamChunk(
            content="",
            done=True,
            model=model,
            provider="ollama",
        )


def override_model_service() -> FakeModelService:
    return FakeModelService()


def setup_module() -> None:
    app.dependency_overrides[model_routes.get_model_service] = override_model_service


def teardown_module() -> None:
    app.dependency_overrides.clear()


client = TestClient(app)


def test_model_health_route() -> None:
    response = client.get("/model/health")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["status"] == "ok"


def test_model_generate_route() -> None:
    response = client.post(
        "/model/generate",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "model": "qwen3:4b",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "fake response"
    assert body["model"] == "qwen3:4b"


def test_model_embed_route() -> None:
    response = client.post(
        "/model/embed",
        json={
            "texts": ["hello", "world"],
            "model": "nomic-embed-text-v2-moe",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["embeddings"]) == 2
    assert body["model"] == "nomic-embed-text-v2-moe"


def test_model_stream_route() -> None:
    with client.stream(
        "POST",
        "/model/stream",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "model": "qwen3:4b",
        },
    ) as response:
        assert response.status_code == 200
        # Call read() to load the stream content before accessing response.text
        response.read()
        text = response.text

    assert '"content":"hello"' in text
    assert '"done":true' in text
