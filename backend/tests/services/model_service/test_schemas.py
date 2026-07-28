import pytest
from pydantic import ValidationError

from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelEmbeddingRequest,
    ModelMessage,
)


def test_model_message_accepts_valid_role_and_content() -> None:
    message = ModelMessage(role="user", content="Hello")

    assert message.role == "user"
    assert message.content == "Hello"


def test_model_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        ModelMessage(role="user", content="   ")


def test_chat_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        ModelChatRequest(messages=[])


def test_chat_request_accepts_messages() -> None:
    request = ModelChatRequest(
        messages=[
            ModelMessage(role="system", content="You are helpful."),
            ModelMessage(role="user", content="Explain procurement."),
        ],
        model="qwen3:4b",
        temperature=0.2,
    )

    assert request.model == "qwen3:4b"
    assert len(request.messages) == 2


def test_chat_request_rejects_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        ModelChatRequest(
            messages=[ModelMessage(role="user", content="Hello")],
            temperature=3.0,
        )


def test_embedding_request_accepts_texts() -> None:
    request = ModelEmbeddingRequest(
        texts=["first text", "second text"],
        model="nomic-embed-text-v2-moe",
    )

    assert request.model == "nomic-embed-text-v2-moe"
    assert len(request.texts) == 2


def test_embedding_request_rejects_blank_texts() -> None:
    with pytest.raises(ValidationError):
        ModelEmbeddingRequest(texts=["valid text", "   "])
