from __future__ import annotations

import pytest

from app.orchestration.graph import build_orchestration_graph
from app.services.model_service.exceptions import ModelGenerationError
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
)
from app.services.model_service.service import ModelService


class FakeModelService(ModelService):
    def __init__(self) -> None:
        # Override __init__ so no positional arguments are required
        pass

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        return ModelChatResponse(
            content=f"Model response to: {request.messages[-1].content}",
            model=request.model or "fake-model",
            provider="ollama",
            latency_ms=1.0,
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


class FailingModelService(FakeModelService):
    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        raise ModelGenerationError(
            "ollama",
            details={"reason": "test failure"},
        )


@pytest.mark.asyncio
async def test_graph_generates_model_response() -> None:
    graph = build_orchestration_graph(FakeModelService())

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "Hello orchestration",
            "metadata": {},
        }
    )

    assert result["error"] is None
    assert result["metadata"]["prepared"] is True
    assert result["metadata"]["model_called"] is True
    assert result["metadata"]["model_provider"] == "ollama"
    assert result["metadata"]["finalized"] is True
    assert result["final_response"] == "Model response to: Hello orchestration"


@pytest.mark.asyncio
async def test_graph_handles_empty_message_without_calling_model() -> None:
    graph = build_orchestration_graph(FakeModelService())

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "   ",
            "metadata": {},
        }
    )

    assert result["error"] == "User message cannot be empty."
    assert result["metadata"]["prepared"] is False
    assert result["metadata"].get("model_called") is None
    assert result["metadata"]["finalized"] is True
    assert result["final_response"] == (
        "I could not process the message because: User message cannot be empty."
    )


@pytest.mark.asyncio
async def test_graph_handles_model_service_error() -> None:
    graph = build_orchestration_graph(FailingModelService())

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "Hello",
            "metadata": {},
        }
    )

    assert result["error"] == "Model generation failed for provider: ollama"
    assert result["metadata"]["model_error"]["error"] == "model_generation_error"
    assert result["metadata"]["finalized"] is True
    assert result["final_response"] == (
        "I could not process the message because: Model generation failed for provider: ollama"
    )
