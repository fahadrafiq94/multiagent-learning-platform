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
        self.last_request: ModelChatRequest | None = None

    async def generate(
        self,
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        self.last_request = request

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


@pytest.mark.asyncio
async def test_graph_routes_to_scenario_path() -> None:
    model_service = FakeModelService()
    graph = build_orchestration_graph(model_service)

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("I want to define my company name and business problem."),
            "metadata": {},
        }
    )

    assert result["route"] == "scenario"
    assert result["metadata"]["selected_path"] == "scenario"
    assert result["metadata"]["routing_completed"] is True
    assert model_service.last_request is not None
    assert "Scenario placeholder path" in model_service.last_request.messages[0].content


@pytest.mark.asyncio
async def test_graph_routes_to_process_coach_path() -> None:
    model_service = FakeModelService()
    graph = build_orchestration_graph(model_service)

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Why is procurement important in this business process?"),
            "metadata": {},
        }
    )

    assert result["route"] == "process_coach"
    assert result["metadata"]["selected_path"] == "process_coach"
    assert model_service.last_request is not None
    assert "Process Coach placeholder path" in model_service.last_request.messages[0].content


@pytest.mark.asyncio
async def test_graph_routes_to_ap_plus_navigator_path() -> None:
    model_service = FakeModelService()
    graph = build_orchestration_graph(model_service)

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": ("Where can I find the purchase-order screen in AP+?"),
            "metadata": {},
        }
    )

    assert result["route"] == "ap_plus_navigator"
    assert result["metadata"]["selected_path"] == "ap_plus_navigator"
    assert model_service.last_request is not None
    assert "AP+ Navigator placeholder path" in model_service.last_request.messages[0].content


@pytest.mark.asyncio
async def test_graph_routes_to_fallback_path() -> None:
    model_service = FakeModelService()
    graph = build_orchestration_graph(model_service)

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "Hello there.",
            "metadata": {},
        }
    )

    assert result["route"] == "fallback"
    assert result["metadata"]["selected_path"] == "fallback"
