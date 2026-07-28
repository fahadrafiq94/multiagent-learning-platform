from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ModelRole = Literal["system", "user", "assistant", "tool"]
ModelProviderName = Literal["ollama", "vllm"]


class ModelMessage(BaseModel):
    """A single chat message passed to a model provider."""

    role: ModelRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must not be blank.")
        return value


class ModelUsage(BaseModel):
    """Token usage metadata when a provider returns it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ModelChatRequest(BaseModel):
    """Provider-independent chat/generation request."""

    messages: list[ModelMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, value: list[ModelMessage]) -> list[ModelMessage]:
        if not value:
            raise ValueError("At least one message is required.")
        return value


class ModelChatResponse(BaseModel):
    """Provider-independent chat/generation response."""

    content: str
    model: str
    provider: ModelProviderName
    latency_ms: float
    usage: ModelUsage | None = None
    raw: dict[str, Any] | None = None


class ModelStreamChunk(BaseModel):
    """A single streaming response chunk."""

    content: str = ""
    done: bool = False
    model: str
    provider: ModelProviderName
    raw: dict[str, Any] | None = None


class ModelEmbeddingRequest(BaseModel):
    """Provider-independent embedding request."""

    texts: list[str] = Field(min_length=1)
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("texts")
    @classmethod
    def texts_must_not_contain_blank_values(cls, value: list[str]) -> list[str]:
        for text in value:
            if not text.strip():
                raise ValueError("Embedding texts must not contain blank values.")
        return value


class ModelEmbeddingResponse(BaseModel):
    """Provider-independent embedding response."""

    embeddings: list[list[float]]
    model: str
    provider: ModelProviderName
    latency_ms: float
    raw: dict[str, Any] | None = None


class ModelHealthResponse(BaseModel):
    """Health status for a model provider."""

    provider: ModelProviderName
    status: Literal["ok", "unavailable", "error"]
    latency_ms: float | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
