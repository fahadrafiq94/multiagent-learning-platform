from __future__ import annotations

from app.config.settings import Settings, settings
from app.services.model_service.exceptions import UnsupportedModelProviderError
from app.services.model_service.ollama_provider import OllamaModelProvider
from app.services.model_service.service import ModelService


def create_model_service(app_settings: Settings = settings) -> ModelService:
    """Create the configured model service."""

    provider_name = app_settings.model_provider.lower().strip()

    if provider_name == "ollama":
        provider = OllamaModelProvider(
            base_url=app_settings.ollama_base_url,
            chat_model=app_settings.ollama_chat_model,
            embedding_model=app_settings.ollama_embedding_model,
            timeout_seconds=app_settings.model_request_timeout_seconds,
            max_retries=app_settings.model_max_retries,
            retry_backoff_seconds=app_settings.model_retry_backoff_seconds,
            default_temperature=app_settings.model_temperature,
            default_max_tokens=app_settings.model_max_tokens,
        )
        return ModelService(provider=provider)

    raise UnsupportedModelProviderError(provider_name)
