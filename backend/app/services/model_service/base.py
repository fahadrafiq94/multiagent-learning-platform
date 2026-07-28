from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
)


class ModelProvider(Protocol):
    """Provider interface implemented by Ollama, vLLM, or future providers."""

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        """Generate a full non-streaming response."""
        ...

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Generate a streaming response."""
        ...

    async def embed(
        self,
        request: ModelEmbeddingRequest,
    ) -> ModelEmbeddingResponse:
        """Generate embeddings."""
        ...

    async def health(self) -> ModelHealthResponse:
        """Check provider health."""
        ...
