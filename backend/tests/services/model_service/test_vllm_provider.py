"""Deterministic HTTP contracts; no live model or knowledge database required."""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from app.services.model_service.exceptions import (
    ModelEmbeddingError,
    ModelGenerationError,
    ModelProviderUnavailableError,
    ModelServiceError,
    ModelTimeoutError,
)
from app.services.model_service.schemas import ModelChatRequest, ModelEmbeddingRequest, ModelMessage
from app.services.model_service.vllm_provider import VLLMModelProvider


def request(**values):
    return ModelChatRequest(messages=[ModelMessage(role="user", content="Hello")], **values)


def completion(**values):
    return {
        "model": "chat",
        "choices": [{"finish_reason": "stop", "message": {"content": "Verified reply"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        **values,
    }


def provider(handler=None, **kwargs):
    calls = []

    async def route(req):
        calls.append(req)
        if handler:
            result = await handler(req)
            if result is not None:
                return result
        if req.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "chat", "max_model_len": 8192},
                        {"id": "other", "max_model_len": 4096},
                    ]
                },
            )
        if req.url.path == "/health":
            return httpx.Response(200)
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 12})
        return httpx.Response(200, json=completion())

    return VLLMModelProvider(
        base_url="http://server/v1",
        chat_model="chat",
        transport=httpx.MockTransport(route),
        retry_backoff_seconds=0,
        **kwargs,
    ), calls


@pytest.mark.asyncio
async def test_payload_uses_configured_model_schema_template_and_exact_budget():
    p, calls = provider(reasoning_control="enable_thinking", api_key=SecretStr("private-key"))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    before = json.dumps(schema)
    try:
        reply = await p.generate(
            request(
                response_schema=schema,
                reasoning_enabled=False,
                max_tokens=32,
                context_window_tokens=1024,
                temperature=0,
                metadata={"task": "a/structured task"},
            )
        )
        assert reply.provider == "vllm" and reply.model == "chat"
        assert reply.usage.total_tokens == 15
        tokenize, chat = [json.loads(c.content) for c in calls if c.method == "POST"]
        assert tokenize["chat_template_kwargs"] == {"enable_thinking": False}
        assert tokenize["add_generation_prompt"] is True
        assert chat["chat_template_kwargs"] == tokenize["chat_template_kwargs"]
        assert chat["response_format"]["json_schema"] == {
            "name": "a_structured_task",
            "schema": schema,
        }
        assert chat["max_completion_tokens"] == 32 and chat["temperature"] == 0
        assert not {"think", "num_ctx", "truncate_prompt_tokens", "options"}.intersection(chat)
        assert calls[-1].headers["Authorization"] == "Bearer private-key"
        assert json.dumps(schema) == before
        await p.generate(request())
        assert len([c for c in calls if c.url.path == "/v1/models"]) == 1
        assert "chat_template_kwargs" not in json.loads(calls[-1].content)
    finally:
        await p.aclose()
    with pytest.raises(ModelProviderUnavailableError):
        await p.generate(request())


@pytest.mark.asyncio
async def test_served_override_is_used_without_silently_using_default():
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=completion(model="other"))

    p, calls = provider(handler)
    try:
        result = await p.generate(request(model="other", context_window_tokens=4096))
        assert result.model == "other" and json.loads(calls[-1].content)["model"] == "other"
        with pytest.raises(ModelGenerationError, match="generation failed"):
            await p.generate(request(model="not-served"))
        assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == 1
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_configured_reasoning_default_is_used_and_request_override_wins():
    p, calls = provider(reasoning_control="enable_thinking", default_reasoning_enabled=False)
    try:
        await p.generate(request())
        assert json.loads(calls[-1].content)["chat_template_kwargs"] == {"enable_thinking": False}
        await p.generate(request(reasoning_enabled=True))
        assert json.loads(calls[-1].content)["chat_template_kwargs"] == {"enable_thinking": True}
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_server_cannot_ignore_disabled_reasoning_from_environment_default():
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json=completion(
                    choices=[
                        {
                            "finish_reason": "stop",
                            "message": {"content": "Answer", "reasoning": "hidden"},
                        }
                    ]
                ),
            )

    p, _ = provider(handler, reasoning_control="enable_thinking", default_reasoning_enabled=False)
    try:
        with pytest.raises(ModelGenerationError) as error:
            await p.generate(request())
        assert error.value.details["reason"] == "reasoning_control_not_honored"
    finally:
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        {"context_window_tokens": 9000},
        {"context_window_tokens": 256, "max_tokens": 256},
        {"model": "other"},
        {"reasoning_enabled": False},
    ],
)
async def test_invalid_alias_window_or_unsupported_control_never_generates(values):
    p, calls = provider()
    try:
        with pytest.raises(ModelGenerationError):
            await p.generate(request(**values))
        assert not any(c.url.path == "/v1/chat/completions" for c in calls)
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_exact_token_overflow_is_rejected_without_truncation_or_generation():
    async def handler(req):
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 250})

    p, calls = provider(handler)
    try:
        with pytest.raises(ModelGenerationError) as error:
            await p.generate(request(context_window_tokens=256, max_tokens=16))
        assert error.value.details["reason"] == "input_budget_exceeded"
        assert not any(c.url.path == "/v1/chat/completions" for c in calls)
    finally:
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"choices": []},
        {"model": "unexpected"},
        {"usage": None},
        {"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "a", "tool_calls": [{}]}}]},
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "a", "reasoning": "hidden"}}
            ]
        },
        {"usage": {"prompt_tokens": 12, "completion_tokens": 33}},
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 3}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 3}},
    ],
)
async def test_incomplete_malformed_unexpected_reasoning_or_over_budget_reply_is_rejected(bad):
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=completion(**bad))

    p, _ = provider(handler, reasoning_control="enable_thinking")
    try:
        with pytest.raises(ModelGenerationError):
            await p.generate(
                request(max_tokens=32, context_window_tokens=1024, reasoning_enabled=False)
            )
    finally:
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_cls",
    [
        (401, ModelServiceError),
        (403, ModelServiceError),
        (404, ModelGenerationError),
        (429, ModelProviderUnavailableError),
        (503, ModelProviderUnavailableError),
    ],
)
async def test_http_errors_are_sanitized_and_only_transient_statuses_retry(status, error_cls):
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(status, text="private-key student text internal-path")

    p, calls = provider(handler, max_retries=1)
    try:
        with pytest.raises(error_cls) as error:
            await p.generate(request())
        assert "private-key" not in json.dumps(error.value.to_dict())
        assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == (
            2 if status in {429, 503} else 1
        )
    finally:
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_cls", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_errors_are_sanitized_and_read_timeouts_are_not_replayed(error_cls):
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            raise error_cls("private transport details")

    p, calls = provider(handler, max_retries=1)
    try:
        with pytest.raises(ModelServiceError) as error:
            await p.generate(request())
        assert "private transport details" not in str(error.value.to_dict())
        assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == (
            1 if error_cls is httpx.ReadTimeout else 2
        )
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_cancelled_inference_is_not_retried_and_client_can_close():
    entered = asyncio.Event()

    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            entered.set()
            await asyncio.Event().wait()

    p, calls = provider(handler, max_retries=3)
    task = asyncio.create_task(p.generate(request()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == 1
    await p.aclose()


@pytest.mark.asyncio
async def test_wall_deadline_includes_discovery_and_tokenization():
    async def handler(req):
        await asyncio.sleep(0.1)

    p, _ = provider(handler, timeout_seconds=0.001)
    try:
        with pytest.raises(ModelTimeoutError):
            await p.generate(request())
    finally:
        await p.aclose()


def sse_event(**values):
    return {
        "model": "chat",
        "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        **values,
    }


@pytest.mark.asyncio
async def test_cancelled_stream_closes_wire_and_does_not_replay_partial_output():
    entered = asyncio.Event()

    class Wire(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield ("data: " + json.dumps(sse_event()) + "\n\n").encode()
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    wire = Wire()

    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(200, stream=wire)

    p, calls = provider(handler, max_retries=3)
    received = []

    async def consume():
        async for chunk in p.stream(request()):
            received.append(chunk)

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire.closed and len(received) == 1 and not received[0].done
    assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == 1
    await p.aclose()


def sse_response(events, done=True):
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if done:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, content=text, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_sse_deltas_usage_and_terminal_event_are_parsed_without_reasoning():
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            assert json.loads(req.content)["stream_options"] == {"include_usage": True}
            return sse_response(
                [
                    sse_event(),
                    sse_event(choices=[{"delta": {"content": "!"}, "finish_reason": "stop"}]),
                    sse_event(choices=[], usage={"prompt_tokens": 12, "completion_tokens": 3}),
                ]
            )

    p, _ = provider(handler)
    try:
        chunks = [c async for c in p.stream(request())]
        assert "".join(c.content for c in chunks) == "Hello!"
        assert [c.done for c in chunks] == [False, False, True]
        assert chunks[-1].raw["usage"]["completion_tokens"] == 3
    finally:
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "length",
        "missing_done",
        "missing_usage",
        "reasoning",
        "wrong_model",
        "provider_error",
        "bad_json",
    ],
)
async def test_sse_failure_never_yields_successful_done_or_retries(kind):
    async def handler(req):
        if req.url.path == "/v1/chat/completions":
            if kind == "bad_json":
                return httpx.Response(200, content="data: {bad}\n\n")
            events = [
                sse_event(),
                sse_event(
                    choices=[
                        {"delta": {}, "finish_reason": "length" if kind == "length" else "stop"}
                    ]
                ),
            ]
            if kind == "reasoning":
                events[0] = sse_event(
                    choices=[{"delta": {"reasoning": "hidden"}, "finish_reason": None}]
                )
            if kind == "wrong_model":
                events[0]["model"] = "wrong"
            if kind == "provider_error":
                events[0] = {"error": "private"}
            if kind != "missing_usage":
                events.append(
                    sse_event(choices=[], usage={"prompt_tokens": 12, "completion_tokens": 3})
                )
            return sse_response(events, done=kind != "missing_done")

    p, calls = provider(handler, reasoning_control="enable_thinking")
    chunks = []
    try:
        with pytest.raises(ModelGenerationError):
            async for chunk in p.stream(request(reasoning_enabled=False)):
                chunks.append(chunk)
        assert not any(c.done for c in chunks)
        assert len([c for c in calls if c.url.path == "/v1/chat/completions"]) == 1
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_health_accepts_empty_body_checks_alias_and_never_claims_embeddings():
    p, calls = provider()
    try:
        health = await p.health()
        assert health.status == "ok" and health.embedding_model is None
        assert health.details["embeddings"] == "not_configured"
        assert len(calls) == 2
        with pytest.raises(ModelEmbeddingError):
            await p.embed(ModelEmbeddingRequest(texts=["x"]))
        with pytest.raises(ModelEmbeddingError):
            await p.inspect_embedding_model("embedding")
        assert len(calls) == 2
    finally:
        await p.aclose()


@pytest.mark.asyncio
async def test_health_does_not_retry_and_ignores_stale_cached_aliases():
    failing = False

    async def handler(req):
        if failing and req.url.path == "/v1/models":
            return httpx.Response(503, text="private")

    p, calls = provider(handler, max_retries=3)
    try:
        assert (await p.health()).status == "ok"
        failing = True
        assert (await p.health()).status == "unavailable"
        assert len(calls) == 4
    finally:
        await p.aclose()


@pytest.mark.parametrize(
    "url",
    [
        "http://u:secret@server/v1",
        "http://server/v1?key=secret",
        "ftp://server",
        "http://server/v1/v1",
    ],
)
def test_invalid_url_never_leaks_credentials(url):
    with pytest.raises(ModelServiceError) as error:
        VLLMModelProvider(base_url=url, chat_model="chat")
    assert "secret" not in str(error.value.to_dict())
