"""Request-local reasoning control, preserving all existing chat defaults."""

from unittest.mock import AsyncMock

import pytest

from app.services.model_service.schemas import ModelChatRequest, ModelMessage
from tests.services.model_service.test_embedding_index_controls import provider


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, False, True])
async def test_reasoning_preference_is_explicit_and_default_payload_is_preserved(enabled):
    model = provider()
    model._post_json_with_retries = AsyncMock(return_value={"message": {"content": "Reply"}})
    await model.generate(
        ModelChatRequest(
            messages=[ModelMessage(role="user", content="Hello")],
            reasoning_enabled=enabled,
        )
    )
    payload = model._post_json_with_retries.call_args.args[1]
    if enabled is None:
        assert "think" not in payload
    else:
        assert payload["think"] is enabled
    assert payload["model"] == "chat" and payload["options"] == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_request_context_window_maps_to_actual_provider_window_without_global_changes():
    model = provider()
    model._post_json_with_retries = AsyncMock(return_value={"message": {"content": "Reply"}})
    await model.generate(
        ModelChatRequest(
            messages=[ModelMessage(role="user", content="Hello")],
            context_window_tokens=8192,
        )
    )
    assert model._post_json_with_retries.call_args.args[1]["options"]["num_ctx"] == 8192
    await model.generate(ModelChatRequest(messages=[ModelMessage(role="user", content="Hello")]))
    assert "num_ctx" not in model._post_json_with_retries.call_args.args[1]["options"]
