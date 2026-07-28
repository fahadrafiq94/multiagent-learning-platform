from app.services.model_service.exceptions import (
    ModelProviderUnavailableError,
    UnsupportedModelProviderError,
)


def test_model_service_error_to_dict() -> None:
    error = ModelProviderUnavailableError(
        "ollama",
        details={"reason": "connection refused"},
    )

    assert error.to_dict() == {
        "error": "model_provider_unavailable",
        "message": "Model provider is unavailable: ollama",
        "provider": "ollama",
        "details": {"reason": "connection refused"},
    }


def test_unsupported_model_provider_error() -> None:
    error = UnsupportedModelProviderError("unknown")

    assert error.code == "unsupported_model_provider"
    assert error.provider == "unknown"
