from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

import httpx

from app.logging.logger import logger
from app.services.model_service.exceptions import (
    ModelEmbeddingError,
    ModelGenerationError,
    ModelProviderUnavailableError,
    ModelTimeoutError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelMessage,
    ModelStreamChunk,
    ModelUsage,
)


class OllamaModelProvider:
    """Ollama implementation of the model provider interface.

    This class is the only place in the backend that should know Ollama's
    HTTP API shape.
    """

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        default_temperature: float,
        default_max_tokens: int | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        """Generate a complete non-streaming chat response."""
        model = request.model or self.chat_model
        payload = self._build_chat_payload(request, model=model, stream=False)

        start = time.perf_counter()

        logger.info(
            "Model_generation_started",
            provider=self.provider_name,
            model=model,
            message_count=len(request.messages),
            stream=False,
        )

        try:
            response_data = await self._post_json_with_retries("/api/chat", payload)
        except ModelTimeoutError:
            raise
        except ModelProviderUnavailableError:
            raise
        except Exception as exc:
            raise ModelGenerationError(
                self.provider_name,
                details={"error": str(exc), "model": model},
            ) from exc

        latency_ms = self._latency_ms(start)
        content = self._extract_chat_content(response_data)

        logger.info(
            "Model_generation_completed",
            provider=self.provider_name,
            model=model,
            latency_ms=latency_ms,
            response_length=len(content),
        )

        return ModelChatResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            usage=self._extract_usage(response_data),
            raw=response_data,
        )

    async def stream(
        self,
        request: ModelChatRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Generate a streaming chat response."""
        model = request.model or self.chat_model
        payload = self._build_chat_payload(request, model=model, stream=True)

        logger.info(
            "Model_stream_started",
            provider=self.provider_name,
            model=model,
            message_count=len(request.messages),
            stream=True,
        )

        timeout = httpx.Timeout(self.timeout_seconds)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        chunk_data = self._parse_stream_line(line)
                        message = chunk_data.get("message") or {}
                        content = str(message.get("content") or "")
                        done = bool(chunk_data.get("done", False))

                        yield ModelStreamChunk(
                            content=content,
                            done=done,
                            model=model,
                            provider=self.provider_name,
                            raw=chunk_data,
                        )

                        if done:
                            break

        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                self.provider_name,
                details={"endpoint": "/api/chat", "model": model},
            ) from exc
        except httpx.ConnectError as exc:
            raise ModelProviderUnavailableError(
                self.provider_name,
                details={"endpoint": "/api/chat", "model": model},
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGenerationError(
                self.provider_name,
                details={
                    "endpoint": "/api/chat",
                    "model": model,
                    "status_code": exc.response.status_code,
                    "response_text": exc.response.text,
                },
            ) from exc
        except Exception as exc:
            raise ModelGenerationError(
                self.provider_name,
                details={"endpoint": "/api/chat", "model": model, "error": str(exc)},
            ) from exc

        logger.info(
            "Model_stream_completed",
            provider=self.provider_name,
            model=model,
        )

    async def embed(self, request: ModelEmbeddingRequest) -> ModelEmbeddingResponse:
        """Generate embeddings for one or more texts."""
        model = request.model or self.embedding_model

        payload: dict[str, Any] = {
            "model": model,
            "input": request.texts,
        }

        start = time.perf_counter()

        logger.info(
            "Model_embedding_started",
            provider=self.provider_name,
            model=model,
            text_count=len(request.texts),
        )

        try:
            response_data = await self._post_json_with_retries("/api/embed", payload)
        except ModelTimeoutError:
            raise
        except ModelProviderUnavailableError:
            raise
        except Exception as exc:
            raise ModelEmbeddingError(
                self.provider_name,
                details={"error": str(exc), "model": model},
            ) from exc

        latency_ms = self._latency_ms(start)
        embeddings = response_data.get("embeddings")

        if not isinstance(embeddings, list):
            raise ModelEmbeddingError(
                self.provider_name,
                details={
                    "error": "Ollama response did not contain embeddings list.",
                    "model": model,
                    "raw": response_data,
                },
            )

        logger.info(
            "Model_embedding_completed",
            provider=self.provider_name,
            model=model,
            latency_ms=latency_ms,
            embedding_count=len(embeddings),
        )

        return ModelEmbeddingResponse(
            embeddings=embeddings,
            model=model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw=response_data,
        )

    async def health(self) -> ModelHealthResponse:
        """Check whether Ollama is reachable."""
        start = time.perf_counter()

        try:
            response_data = await self._get_json_with_retries("/api/tags")
        except ModelTimeoutError:
            return ModelHealthResponse(
                provider=self.provider_name,
                status="unavailable",
                chat_model=self.chat_model,
                embedding_model=self.embedding_model,
                details={"reason": "timeout"},
            )
        except ModelProviderUnavailableError:
            return ModelHealthResponse(
                provider=self.provider_name,
                status="unavailable",
                chat_model=self.chat_model,
                embedding_model=self.embedding_model,
                details={"reason": "connection_error"},
            )
        except Exception as exc:
            return ModelHealthResponse(
                provider=self.provider_name,
                status="error",
                chat_model=self.chat_model,
                embedding_model=self.embedding_model,
                details={"reason": str(exc)},
            )

        return ModelHealthResponse(
            provider=self.provider_name,
            status="ok",
            latency_ms=self._latency_ms(start),
            chat_model=self.chat_model,
            embedding_model=self.embedding_model,
            details={
                "models": response_data.get("models", []),
            },
        )

    def _build_chat_payload(
        self,
        request: ModelChatRequest,
        *,
        model: str,
        stream: bool,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": (
                request.temperature if request.temperature is not None else self.default_temperature
            ),
        }

        max_tokens = request.max_tokens or self.default_max_tokens
        if max_tokens is not None:
            # Ollama uses num_predict for maximum generated tokens.
            options["num_predict"] = max_tokens

        return {
            "model": model,
            "messages": [self._message_to_ollama(message) for message in request.messages],
            "stream": stream,
            "options": options,
        }

    def _message_to_ollama(self, message: ModelMessage) -> dict[str, str]:
        return {
            "role": message.role,
            "content": message.content,
        }

    async def _post_json_with_retries(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            timeout = httpx.Timeout(self.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}{endpoint}", json=payload)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())

        return await self._run_with_retries(
            operation,
            endpoint=endpoint,
        )

    async def _get_json_with_retries(self, endpoint: str) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            timeout = httpx.Timeout(self.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}{endpoint}")
                response.raise_for_status()
                return cast(dict[str, Any], response.json())

        return await self._run_with_retries(
            operation,
            endpoint=endpoint,
        )

    async def _run_with_retries(
        self,
        operation: Callable[[], Awaitable[dict[str, Any]]],
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                res = await operation()
                return cast(dict[str, Any], res)
            except httpx.TimeoutException as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    raise ModelTimeoutError(
                        self.provider_name,
                        details={"endpoint": endpoint, "attempt": attempt + 1},
                    ) from exc
                await self._sleep_before_retry(attempt)
            except httpx.ConnectError as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    raise ModelProviderUnavailableError(
                        self.provider_name,
                        details={"endpoint": endpoint, "attempt": attempt + 1},
                    ) from exc
                await self._sleep_before_retry(attempt)
            except httpx.HTTPStatusError:
                # HTTP status errors are usually not fixed by retrying.
                raise

        raise ModelProviderUnavailableError(
            self.provider_name,
            details={
                "endpoint": endpoint,
                "error": str(last_exception) if last_exception else "unknown",
            },
        )

    async def _sleep_before_retry(self, attempt: int) -> None:
        backoff = self.retry_backoff_seconds * (attempt + 1)
        await asyncio.sleep(backoff)

    def _extract_chat_content(self, response_data: dict[str, Any]) -> str:
        message = response_data.get("message") or {}
        content = message.get("content")

        if not isinstance(content, str):
            raise ModelGenerationError(
                self.provider_name,
                details={
                    "error": "Ollama response did not contain message.content.",
                    "raw": response_data,
                },
            )

        return content

    def _extract_usage(self, response_data: dict[str, Any]) -> ModelUsage | None:
        prompt_tokens = response_data.get("prompt_eval_count")
        completion_tokens = response_data.get("eval_count")

        if prompt_tokens is None and completion_tokens is None:
            return None

        total_tokens: int | None = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return ModelUsage(
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens,
        )

    def _parse_stream_line(self, line: str) -> dict[str, Any]:
        try:
            data = json.loads(line)
            return cast(dict[str, Any], data)
        except Exception as exc:
            raise ModelGenerationError(
                self.provider_name,
                details={"error": "Failed to parse Ollama stream chunk.", "line": line},
            ) from exc

    def _latency_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
