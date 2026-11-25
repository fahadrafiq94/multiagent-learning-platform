from __future__ import annotations

from app.config.settings import Settings, settings
from app.services.model_service.base import ModelProvider
from app.services.model_service.exceptions import ModelServiceError, UnsupportedModelProviderError
from app.services.model_service.ollama_provider import OllamaModelProvider
from app.services.model_service.profiles import (
    ROLE_MODELS,
    ModelProfile,
    has_role_override,
    resolve_profile,
)
from app.services.model_service.service import ModelService
from app.services.model_service.vllm_provider import VLLMModelProvider


def _build(
    app_settings: Settings,
    profile: ModelProfile,
    *,
    embedding: bool = False,
    bound_model: str | None = None,
) -> ModelService:
    provider: ModelProvider
    if profile.provider == "ollama":
        provider = OllamaModelProvider(
            base_url=profile.base_url,
            chat_model=profile.model,
            embedding_model=profile.model
            if embedding
            else (app_settings.ollama_embedding_model or profile.model),
            timeout_seconds=app_settings.model_request_timeout_seconds,
            max_retries=app_settings.model_max_retries,
            retry_backoff_seconds=app_settings.model_retry_backoff_seconds,
            default_temperature=app_settings.model_temperature,
            default_max_tokens=app_settings.model_max_tokens,
        )
    elif profile.provider == "vllm":
        provider = VLLMModelProvider(
            base_url=profile.base_url,
            chat_model=profile.model,
            api_key=profile.api_key,
            reasoning_control=profile.reasoning_control,
            default_reasoning_enabled=profile.reasoning_enabled,
            timeout_seconds=app_settings.model_request_timeout_seconds,
            max_retries=app_settings.model_max_retries,
            retry_backoff_seconds=app_settings.model_retry_backoff_seconds,
            default_temperature=app_settings.model_temperature,
            default_max_tokens=app_settings.model_max_tokens,
            default_context_tokens=app_settings.context_max_tokens,
        )
    else:
        raise UnsupportedModelProviderError(profile.provider)
    return ModelService(
        provider=provider,
        configured_model=profile.model,
        embedding_model=bound_model if embedding else None,
        default_reasoning_enabled=profile.reasoning_enabled,
    )


def create_model_service(
    app_settings: Settings = settings,
    *,
    role: str | None = None,
    legacy_model: str | None = None,
    legacy_url: str | None = None,
    base_url: str | None = None,
) -> ModelService:
    """Compose trusted profiles; explicit roles build independent, owned services."""
    if role is not None:
        profile = resolve_profile(
            app_settings, role, legacy_model=legacy_model, legacy_url=legacy_url, base_url=base_url
        )
        return _build(
            app_settings,
            profile,
            embedding=role == "embedding",
            bound_model=app_settings.embedding_model or legacy_model,
        )
    name = app_settings.model_provider.strip().lower()
    if name not in {"ollama", "vllm"}:
        raise UnsupportedModelProviderError(name)
    embeddings = any(
        getattr(app_settings, "embedding_" + key) for key in ("provider", "base_url", "model")
    )
    # Preserve the old Ollama completeness contract for legacy single-profile callers.
    if name == "ollama" and not embeddings and app_settings.ollama_embedding_model is None:
        raise ModelServiceError(
            "Ollama provider configuration is incomplete.",
            code="model_not_configured",
            provider="ollama",
        )
    primary = resolve_profile(app_settings)
    result = _build(app_settings, primary)
    if embeddings:
        result._embedding_service = create_model_service(app_settings, role="embedding")
    shared: dict[ModelProfile, ModelService] = {}
    for role in ROLE_MODELS:
        if not has_role_override(app_settings, role) and not (
            role == "rag_planner" and app_settings.rag_planner_model
        ):
            continue
        profile = resolve_profile(app_settings, role)
        if profile == primary:
            continue
        if profile not in shared:
            shared[profile] = _build(app_settings, profile)
        result._role_services[role] = shared[profile]
    return result
