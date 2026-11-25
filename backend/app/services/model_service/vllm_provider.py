"""OpenAI-compatible vLLM chat adapter; no implicit model or embedding fallback."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import httpx
from pydantic import SecretStr

from app.services.model_service.exceptions import (
    ModelEmbeddingError,
    ModelGenerationError,
    ModelProviderUnavailableError,
    ModelServiceError,
    ModelTimeoutError,
)
from app.services.model_service.schemas import (
    EmbeddingModelInfo,
    ModelChatRequest,
    ModelChatResponse,
    ModelEmbeddingRequest,
    ModelEmbeddingResponse,
    ModelHealthResponse,
    ModelStreamChunk,
    ModelUsage,
)


class VLLMModelProvider:
    provider_name: Literal["vllm"] = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        chat_model: str,
        api_key: SecretStr | None = None,
        reasoning_control: Literal["unsupported", "enable_thinking"] = "unsupported",
        default_reasoning_enabled: bool | None = None,
        timeout_seconds: float = 120,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        default_temperature: float = 0.2,
        default_max_tokens: int | None = None,
        default_context_tokens: int = 8192,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        try:
            url = httpx.URL(base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.host
                or url.username
                or url.password
                or url.query
                or url.fragment
                or not chat_model.strip()
                or timeout_seconds <= 0
                or max_retries < 0
                or retry_backoff_seconds < 0
                or default_context_tokens < 256
                or (default_max_tokens is not None and default_max_tokens <= 0)
            ):
                raise ValueError
            path = url.path.rstrip("/")
            if path.count("/v1") > 1:
                raise ValueError
            if not path.endswith("/v1"):
                path += "/v1"
            self.base_url = str(url.copy_with(path=path)).rstrip("/")
            self.server_url = self.base_url.removesuffix("/v1")
        except (ValueError, httpx.InvalidURL):
            raise ModelServiceError(
                "vLLM configuration is invalid.", code="model_not_configured", provider="vllm"
            ) from None
        self.chat_model = chat_model.strip()
        self.api_key = api_key
        self.reasoning_control = reasoning_control
        self.default_reasoning_enabled = default_reasoning_enabled
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens or 2048
        self.default_context_tokens = default_context_tokens
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._models: dict[str, int] | None = None
        self._models_lock = asyncio.Lock()
        self._closed = False

    def _http(self) -> httpx.AsyncClient:
        if self._closed:
            raise ModelProviderUnavailableError("vllm", details={"reason": "provider_closed"})
        if self._client is None:
            headers = {}
            if self.api_key is not None and self.api_key.get_secret_value():
                headers["Authorization"] = "Bearer " + self.api_key.get_secret_value()
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout_seconds,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        self._closed = True
        if self._client is not None:
            await self._client.aclose()

    @staticmethod
    def _invalid(reason: str) -> ModelGenerationError:
        return ModelGenerationError("vllm", details={"reason": reason})

    @staticmethod
    def _status(status: int) -> ModelServiceError:
        if status in {401, 403}:
            return ModelServiceError(
                "vLLM authentication failed.", code="model_authentication_error", provider="vllm"
            )
        if status == 429 or status >= 500:
            return ModelProviderUnavailableError("vllm", details={"status_code": status})
        return ModelGenerationError("vllm", details={"status_code": status})

    async def _json(
        self,
        method: str,
        url: str,
        payload: dict | None = None,
        *,
        retry: bool = True,
        status_only: bool = False,
    ) -> dict[str, Any]:
        retries = self.max_retries if retry else 0
        for attempt in range(retries + 1):
            try:
                response = await self._http().request(method, url, json=payload)
                if response.status_code >= 300:
                    if response.status_code in {429, 502, 503, 504} and attempt < retries:
                        await asyncio.sleep(min(5, self.retry_backoff_seconds * 2**attempt))
                        continue
                    raise self._status(response.status_code)
                if status_only:
                    return {}
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError
                return cast(dict[str, Any], data)
            except httpx.TimeoutException:
                # A read timeout can follow accepted work; never replay it automatically.
                raise ModelTimeoutError("vllm") from None
            except httpx.ConnectError:
                if attempt == retries:
                    raise ModelProviderUnavailableError("vllm") from None
                await asyncio.sleep(min(5, self.retry_backoff_seconds * 2**attempt))
            except httpx.TransportError:
                raise ModelProviderUnavailableError("vllm") from None
            except (ValueError, TypeError):
                raise self._invalid("invalid_provider_json") from None
        raise ModelProviderUnavailableError("vllm")

    async def _discover(self, *, retry: bool = True) -> dict[str, int]:
        async with self._models_lock:
            if self._models is None:
                data = await self._json("GET", self.base_url + "/models", retry=retry)
                try:
                    models = {}
                    for item in data["data"]:
                        name, limit = item["id"], item["max_model_len"]
                        if (
                            not isinstance(name, str)
                            or not name
                            or type(limit) is not int
                            or limit < 256
                        ):
                            raise ValueError
                        models[name] = limit
                    if not models or self.chat_model not in models:
                        raise ValueError
                    self._models = models
                except (KeyError, TypeError, ValueError):
                    raise self._invalid("configured_model_or_context_not_advertised") from None
            return self._models

    async def _prepare(self, request: ModelChatRequest, *, stream: bool) -> tuple[dict, int, int]:
        model = request.model or self.chat_model
        models = await self._discover()
        if model not in models:
            raise self._invalid("model_not_advertised")
        window = request.context_window_tokens or self.default_context_tokens
        output = request.max_tokens or self.default_max_tokens
        if window > models[model] or output >= window:
            raise self._invalid("invalid_context_budget")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature
            if request.temperature is not None
            else self.default_temperature,
            "max_completion_tokens": output,
            "stream": stream,
        }
        reasoning = self._reasoning(request)
        if reasoning is not None:
            if self.reasoning_control != "enable_thinking":
                raise self._invalid("reasoning_control_not_configured")
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning}
        if request.response_schema is not None:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", str(request.metadata.get("task", "response")))[
                :64
            ]
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name or "response", "schema": request.response_schema},
            }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        tokenizer = {
            "model": model,
            "messages": payload["messages"],
            "add_generation_prompt": True,
        }
        if "chat_template_kwargs" in payload:
            tokenizer["chat_template_kwargs"] = payload["chat_template_kwargs"]
        tokens = await self._json("POST", self.server_url + "/tokenize", tokenizer)
        count = tokens.get("count")
        if type(count) is not int or count < 0:
            raise self._invalid("invalid_token_count")
        if count + output > window:
            raise self._invalid("input_budget_exceeded")
        return payload, window, output

    def _usage(self, data: dict, window: int, output: int) -> ModelUsage:
        try:
            raw = data["usage"]
            prompt, completion = raw["prompt_tokens"], raw["completion_tokens"]
            if any(type(value) is not int or value < 0 for value in (prompt, completion)):
                raise ValueError
            if completion > output or prompt > window - output or prompt + completion > window:
                raise self._invalid("measured_budget_exceeded")
            return ModelUsage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
            )
        except (KeyError, TypeError, ValueError):
            raise self._invalid("invalid_usage") from None

    def _reasoning(self, request: ModelChatRequest) -> bool | None:
        return (
            request.reasoning_enabled
            if request.reasoning_enabled is not None
            else self.default_reasoning_enabled
        )

    def _check_message(self, message: dict, request: ModelChatRequest) -> None:
        if message.get("tool_calls") or message.get("function_call"):
            raise self._invalid("unexpected_tool_calls")
        if self._reasoning(request) is False and (
            message.get("reasoning") or message.get("reasoning_content")
        ):
            raise self._invalid("reasoning_control_not_honored")

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                payload, window, output = await self._prepare(request, stream=False)
                data = await self._json("POST", self.base_url + "/chat/completions", payload)
                try:
                    if data["model"] != payload["model"] or len(data["choices"]) != 1:
                        raise ValueError
                    choice = data["choices"][0]
                    if choice["finish_reason"] != "stop":
                        raise self._invalid("incomplete_or_unsupported_finish")
                    message = choice["message"]
                    self._check_message(message, request)
                    text = message["content"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    raise self._invalid("invalid_completion") from None
                return ModelChatResponse(
                    content=text,
                    model=payload["model"],
                    provider="vllm",
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    usage=self._usage(data, window, output),
                    raw=data,
                )
        except TimeoutError:
            raise ModelTimeoutError("vllm") from None

    async def stream(self, request: ModelChatRequest) -> AsyncIterator[ModelStreamChunk]:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                payload, window, output = await self._prepare(request, stream=True)
                async with self._http().stream(
                    "POST", self.base_url + "/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 300:
                        raise self._status(response.status_code)
                    finished, content_seen = False, False
                    usage = None
                    event: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            event.append(line[5:].lstrip())
                        if line or not event:
                            continue
                        encoded = "\n".join(event)
                        event = []
                        if encoded == "[DONE]":
                            if not finished or not content_seen or usage is None:
                                raise self._invalid("incomplete_stream")
                            yield ModelStreamChunk(
                                content="",
                                done=True,
                                model=payload["model"],
                                provider="vllm",
                                raw={"usage": usage.model_dump()},
                            )
                            return
                        data = json.loads(encoded)
                        if not isinstance(data, dict) or data.get("error"):
                            raise self._invalid("invalid_stream_event")
                        if data.get("model") != payload["model"]:
                            raise self._invalid("unexpected_stream_model")
                        if data.get("usage") is not None:
                            usage = self._usage(data, window, output)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        if len(choices) != 1:
                            raise self._invalid("invalid_stream_choices")
                        choice = choices[0]
                        delta = choice["delta"]
                        self._check_message(delta, request)
                        was_finished = finished
                        finish = choice.get("finish_reason")
                        if finish is not None:
                            if finish != "stop" or finished:
                                raise self._invalid("incomplete_or_unsupported_finish")
                            finished = True
                        text = delta.get("content")
                        if text is not None:
                            if not isinstance(text, str) or (was_finished and text):
                                raise self._invalid("invalid_stream_content")
                            if text:
                                content_seen = True
                                yield ModelStreamChunk(
                                    content=text,
                                    done=False,
                                    model=payload["model"],
                                    provider="vllm",
                                )
                    raise self._invalid("incomplete_stream")
        except (TimeoutError, httpx.TimeoutException):
            raise ModelTimeoutError("vllm") from None
        except httpx.TransportError:
            raise ModelProviderUnavailableError("vllm") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise self._invalid("invalid_stream_event") from None

    async def embed(self, request: ModelEmbeddingRequest) -> ModelEmbeddingResponse:
        raise ModelEmbeddingError(
            "vllm", details={"reason": "chat_endpoint_has_no_embedding_route"}
        )

    async def inspect_embedding_model(self, model: str) -> EmbeddingModelInfo:
        raise ModelEmbeddingError("vllm", details={"reason": "embedding_inspection_not_configured"})

    async def health(self) -> ModelHealthResponse:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(min(10, self.timeout_seconds)):
                await self._json("GET", self.server_url + "/health", retry=False, status_only=True)
                # Refresh rather than reporting a previously cached alias as currently ready.
                async with self._models_lock:
                    self._models = None
                await self._discover(retry=False)
            return ModelHealthResponse(
                provider="vllm",
                status="ok",
                chat_model=self.chat_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                details={"chat": "ready", "embeddings": "not_configured"},
            )
        except (ModelServiceError, TimeoutError):
            return ModelHealthResponse(
                provider="vllm",
                status="unavailable",
                chat_model=self.chat_model,
                details={"reason": "health_or_model_discovery_failed"},
            )
