from __future__ import annotations

from typing import Any


class ModelServiceError(Exception):
    """Base exception for all model service errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "model_service_error",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.provider = provider
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "provider": self.provider,
            "details": self.details,
        }


class UnsupportedModelProviderError(ModelServiceError):
    """Raised when the configured model provider is not supported."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Unsupported model provider: {provider}",
            code="unsupported_model_provider",
            provider=provider,
        )


class ModelProviderUnavailableError(ModelServiceError):
    """Raised when the provider cannot be reached."""

    def __init__(
        self,
        provider: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model provider is unavailable: {provider}",
            code="model_provider_unavailable",
            provider=provider,
            details=details,
        )


class ModelTimeoutError(ModelServiceError):
    """Raised when a provider request times out."""

    def __init__(
        self,
        provider: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model provider request timed out: {provider}",
            code="model_timeout",
            provider=provider,
            details=details,
        )


class ModelGenerationError(ModelServiceError):
    """Raised when chat/generation fails."""

    def __init__(
        self,
        provider: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model generation failed for provider: {provider}",
            code="model_generation_error",
            provider=provider,
            details=details,
        )


class ModelEmbeddingError(ModelServiceError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        provider: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model embedding failed for provider: {provider}",
            code="model_embedding_error",
            provider=provider,
            details=details,
        )
