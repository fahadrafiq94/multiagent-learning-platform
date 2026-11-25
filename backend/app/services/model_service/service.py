from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.services.model_service.base import EmbeddingInspector, ModelProvider
from app.services.model_service.exceptions import ModelEmbeddingError, ModelServiceError
from app.services.model_service.profiles import canonical_model
from app.services.model_service.schemas import (
    EmbeddingModelInfo,
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
)

PROFILE_HEALTH_TIMEOUT_SECONDS = 10.0


@runtime_checkable
class _CloseableProvider(Protocol):
    async def aclose(self) -> None: ...


class ModelService:
    """Application-facing model service facade.

    Agents and orchestration code should depend on this service instead of
    provider-specific implementations.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        embedding_service: ModelService | None = None,
        role_services: dict[str, ModelService] | None = None,
        embedding_model: str | None = None,
        configured_model: str | None = None,
        default_reasoning_enabled: bool | None = None,
    ) -> None:
        self._provider = provider
        self._embedding_service = embedding_service
        self._role_services = role_services or {}
        self._embedding_model = embedding_model
        self.model_name = configured_model
        self._default_reasoning_enabled = default_reasoning_enabled
        self._closed = False

    @property
    def supports_role_profiles(self) -> bool:
        return bool(self._role_services)

    def for_role(self, role: str) -> ModelService:
        return self._role_services.get(role, self)

    def _request(self, request: ModelChatRequest) -> ModelChatRequest:
        if request.reasoning_enabled is None and self._default_reasoning_enabled is not None:
            return request.model_copy(update={"reasoning_enabled": self._default_reasoning_enabled})
        return request

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        return await self._provider.generate(self._request(request))

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        async for chunk in self._provider.stream(self._request(request)):
            yield chunk

    async def embed(
        self,
        request: ModelEmbeddingRequest,
    ) -> ModelEmbeddingResponse:
        if self._embedding_service is not None:
            return await self._embedding_service.embed(request)
        model = request.model or self._embedding_model
        self._check_embedding_model(model)
        return await self._provider.embed(request.model_copy(update={"model": model}))

    def _check_embedding_model(self, model: str | None) -> None:
        if (
            self._embedding_model
            and model
            and canonical_model(model) != canonical_model(self._embedding_model)
        ):
            raise ModelEmbeddingError(
                "ollama", details={"reason": "embedding_model_conflicts_with_profile"}
            )

    async def health(self) -> ModelHealthResponse:
        try:
            async with asyncio.timeout(PROFILE_HEALTH_TIMEOUT_SECONDS):
                return await self._health()
        except TimeoutError:
            return ModelHealthResponse(
                provider=getattr(self._provider, "provider_name", "ollama"),
                status="unavailable",
                chat_model=self.model_name,
                details={"reason": "profile_health_timeout"},
            )

    async def _health(self) -> ModelHealthResponse:
        health = await self._provider.health()
        if self._embedding_service is None and not self._role_services:
            return health
        profiles = {
            "chat": {
                "provider": health.provider,
                "model": health.chat_model,
                "status": health.status,
            }
        }
        ok = health.status == "ok"
        for role, service in {
            **self._role_services,
            **({"embedding": self._embedding_service} if self._embedding_service else {}),
        }.items():
            result = await service.health()
            status = result.status
            if role == "embedding" and service._embedding_model:
                try:
                    await service.inspect_embedding_model(service._embedding_model)
                except ModelServiceError:
                    status = "unavailable"
            profiles[role] = {
                "provider": result.provider,
                "model": service.model_name,
                "status": status,
            }
            ok = ok and status == "ok"
        return health.model_copy(
            update={
                "status": "ok" if ok else "unavailable",
                "embedding_model": self._embedding_service.model_name
                if self._embedding_service
                else health.embedding_model,
                "details": {**health.details, "profiles": profiles},
            }
        )

    async def aclose(self) -> None:
        """Close owned resources; older stateless providers need no cleanup."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        closers = []
        provider = getattr(self, "_provider", None)
        if isinstance(provider, _CloseableProvider):
            closers.append(provider.aclose())
        children = {
            id(child): child
            for child in [
                *getattr(self, "_role_services", {}).values(),
                getattr(self, "_embedding_service", None),
            ]
            if child is not None
        }
        closers.extend(child.aclose() for child in children.values())
        for result in await asyncio.gather(*closers, return_exceptions=True):
            if isinstance(result, BaseException):
                raise result

    async def inspect_embedding_model(self, model: str) -> EmbeddingModelInfo:
        if self._embedding_service is not None:
            return await self._embedding_service.inspect_embedding_model(model)
        self._check_embedding_model(model)
        if not isinstance(self._provider, EmbeddingInspector):
            raise ModelEmbeddingError(
                "unknown", details={"error": "Embedding inspection is unavailable."}
            )
        return await self._provider.inspect_embedding_model(model)


def service_for_role(service: ModelService, role: str) -> ModelService:
    """Optional capability preserves existing injected/recording service doubles."""
    if getattr(service, "supports_role_profiles", False) is True:
        return service.for_role(role)
    return service


async def close_model_services(*services: object) -> None:
    """Close all owned model resources even when one closer fails."""
    unique = {id(service): service for service in services if service is not None}
    closers = [getattr(service, "aclose", None) for service in unique.values()]
    for result in await asyncio.gather(
        *(close() for close in closers if callable(close)), return_exceptions=True
    ):
        if isinstance(result, BaseException):
            raise result
