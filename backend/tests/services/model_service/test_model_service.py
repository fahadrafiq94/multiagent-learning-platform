from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelMessage,
    ModelStreamChunk,
)
from app.services.model_service.service import ModelService


class FakeModelProvider:
    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        return ModelChatResponse(
            content="fake response",
            model=request.model or "fake-model",
            provider="ollama",
            latency_ms=1.0,
        )

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        model = request.model or "fake-model"
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

    async def embed(
        self,
        request: ModelEmbeddingRequest,
    ) -> ModelEmbeddingResponse:
        return ModelEmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] for _ in request.texts],
            model=request.model or "fake-embedding-model",
            provider="ollama",
            latency_ms=1.0,
        )

    async def health(self) -> ModelHealthResponse:
        return ModelHealthResponse(
            provider="ollama",
            status="ok",
            latency_ms=1.0,
            chat_model="fake-model",
            embedding_model="fake-embedding-model",
        )


@pytest.mark.asyncio
async def test_model_service_generate_delegates_to_provider() -> None:
    service = ModelService(provider=FakeModelProvider())
    request = ModelChatRequest(
        messages=[ModelMessage(role="user", content="Hello")],
        model="qwen3:4b",
    )

    response = await service.generate(request)

    assert response.content == "fake response"
    assert response.model == "qwen3:4b"
    assert response.provider == "ollama"


@pytest.mark.asyncio
async def test_model_service_stream_delegates_to_provider() -> None:
    service = ModelService(provider=FakeModelProvider())
    request = ModelChatRequest(
        messages=[ModelMessage(role="user", content="Hello")],
        model="qwen3:4b",
    )

    chunks = [chunk async for chunk in service.stream(request)]

    assert len(chunks) == 2
    assert chunks[0].content == "hello"
    assert chunks[1].done is True


@pytest.mark.asyncio
async def test_model_service_embed_delegates_to_provider() -> None:
    service = ModelService(provider=FakeModelProvider())
    request = ModelEmbeddingRequest(
        texts=["one", "two"],
        model="nomic-embed-text-v2-moe",
    )

    response = await service.embed(request)

    assert len(response.embeddings) == 2
    assert response.model == "nomic-embed-text-v2-moe"


@pytest.mark.asyncio
async def test_model_service_health_delegates_to_provider() -> None:
    service = ModelService(provider=FakeModelProvider())

    response = await service.health()

    assert response.status == "ok"
    assert response.provider == "ollama"
