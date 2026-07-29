from __future__ import annotations

from collections.abc import AsyncIterator

from app.services.model_service.base import ModelProvider
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
)


class ModelService:
    """Application-facing model service facade.

    Agents and orchestration code should depend on this service instead of
    provider-specific implementations.
    """

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        return await self._provider.generate(request)

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        async for chunk in self._provider.stream(request):
            yield chunk

    async def embed(
        self,
        request: ModelEmbeddingRequest,
    ) -> ModelEmbeddingResponse:
        return await self._provider.embed(request)

    async def health(self) -> ModelHealthResponse:
        return await self._provider.health()
