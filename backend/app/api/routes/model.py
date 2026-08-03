from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.services.model_service import ModelService, create_model_service
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
)

router = APIRouter(prefix="/model", tags=["model"])


def get_model_service() -> ModelService:
    """FastAPI dependency for the configured model service."""
    return create_model_service()


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.get("/health", response_model=ModelHealthResponse)
async def model_health(model_service: ModelServiceDep) -> ModelHealthResponse:
    """Check the configured model provider health."""
    return await model_service.health()


@router.post("/generate", response_model=ModelChatResponse)
async def generate(
    request: ModelChatRequest,
    model_service: ModelServiceDep,
) -> ModelChatResponse:
    """Generate a complete non-streaming model response."""
    try:
        return await model_service.generate(request)
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.to_dict(),
        ) from exc


@router.post("/embed", response_model=ModelEmbeddingResponse)
async def embed(
    request: ModelEmbeddingRequest,
    model_service: ModelServiceDep,
) -> ModelEmbeddingResponse:
    """Generate embeddings through the configured model provider."""
    try:
        return await model_service.embed(request)
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.to_dict(),
        ) from exc


@router.post("/stream")
async def stream(
    request: ModelChatRequest,
    model_service: ModelServiceDep,
) -> StreamingResponse:
    """Stream model response chunks as newline-delimited JSON."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in model_service.stream(request):
                yield _to_json_line(chunk)
        except ModelServiceError as exc:
            error_payload = {
                "error": exc.to_dict(),
                "done": True,
            }
            yield json.dumps(error_payload) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


def _to_json_line(chunk: ModelStreamChunk) -> str:
    return chunk.model_dump_json() + "\n"
