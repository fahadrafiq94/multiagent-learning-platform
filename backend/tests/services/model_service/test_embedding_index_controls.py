from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.model_service.exceptions import ModelEmbeddingError
from app.services.model_service.ollama_provider import OllamaModelProvider
from app.services.model_service.schemas import ModelEmbeddingRequest

pytestmark = pytest.mark.asyncio


def provider():
    return OllamaModelProvider(
        base_url="http://test",
        chat_model="chat",
        embedding_model="embed",
        timeout_seconds=5,
        max_retries=0,
        retry_backoff_seconds=0,
        default_temperature=0.2,
        default_max_tokens=None,
    )


async def test_embedding_truncation_control_is_explicit_and_legacy_default_is_preserved():
    model = provider()
    model._post_json_with_retries = AsyncMock(return_value={"embeddings": [[1, 0, 0]]})
    await model.embed(ModelEmbeddingRequest(texts=["text"]))
    assert "truncate" not in model._post_json_with_retries.call_args.args[1]
    await model.embed(ModelEmbeddingRequest(texts=["text"], truncate=False))
    assert model._post_json_with_retries.call_args.args[1]["truncate"] is False


async def test_context_limit_signal_never_contains_source_text_or_raw_response():
    model = provider()
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://test/api/embed"),
        json={"error": "input length exceeds maximum context length: private-marker"},
    )
    model._post_json_with_retries = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "private-marker", request=response.request, response=response
        )
    )
    with pytest.raises(ModelEmbeddingError) as error:
        await model.embed(ModelEmbeddingRequest(texts=["private-marker"], truncate=False))
    assert error.value.details["input_too_long"] is True
    assert "private-marker" not in str(error.value.to_dict())


async def test_installed_model_inspection_resolves_digest_dimensions_and_context():
    model = provider()
    model._get_json_once = AsyncMock(
        return_value={"models": [{"name": "embed:latest", "digest": "digest"}]}
    )
    model._post_json_with_retries = AsyncMock(
        return_value={
            "capabilities": ["embedding"],
            "model_info": {
                "general.architecture": "bert",
                "bert.embedding_length": 3,
                "bert.context_length": 512,
            },
        }
    )
    info = await model.inspect_embedding_model("embed")
    assert info.revision == "digest" and info.dimension == 3 and info.context_length == 512
