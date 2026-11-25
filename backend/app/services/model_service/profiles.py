"""Resolve trusted model profiles without network access or agent-specific branching."""

from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import SecretStr

from app.config.settings import Settings
from app.services.model_service.exceptions import ModelServiceError

ROLE_MODELS = {
    "orchestrator": "orchestrator_model",
    "scenario_agent": "scenario_agent_model",
    "process_coach": "process_coach_agent_model",
    "ap_plus_navigator": "ap_plus_navigator_model",
    "summary": "summary_model",
    "rag_planner": "rag_planner_model",
}
ROLE_OPTIONS = ("provider", "base_url", "api_key", "reasoning_control", "reasoning_enabled")


@dataclass(frozen=True, repr=False)
class ModelProfile:
    provider: str
    base_url: str
    model: str
    api_key: SecretStr | None = None
    reasoning_control: Literal["unsupported", "enable_thinking"] = "unsupported"
    reasoning_enabled: bool | None = None


def config_error(reason: str) -> ModelServiceError:
    return ModelServiceError(
        "Model profile configuration is invalid.",
        code="model_not_configured",
        details={"reason": reason},
    )


def endpoint(url: str | None, provider: str) -> str:
    try:
        parsed = httpx.URL(url or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        path = parsed.path.rstrip("/")
        if provider == "vllm" and not path.endswith("/v1"):
            path += "/v1"
        return str(parsed.copy_with(path=path)).rstrip("/")
    except (ValueError, httpx.InvalidURL):
        raise config_error("invalid_endpoint") from None


def has_role_override(settings: Settings, role: str) -> bool:
    return any(getattr(settings, role + "_" + option) not in (None, "") for option in ROLE_OPTIONS)


def resolve_profile(
    settings: Settings,
    role: str | None = None,
    *,
    legacy_model: str | None = None,
    legacy_url: str | None = None,
    base_url: str | None = None,
) -> ModelProfile:
    primary = settings.model_provider.strip().lower()
    provider: str
    if role == "embedding":
        provider = settings.embedding_provider or "ollama"
        if provider != "ollama":
            raise config_error("embedding_provider_has_no_verified_inspection")
        if (
            settings.embedding_model
            and legacy_model
            and canonical_model(settings.embedding_model) != canonical_model(legacy_model)
        ):
            raise config_error("embedding_model_conflicts_with_index")
        model = settings.embedding_model or legacy_model or settings.ollama_embedding_model
        url = base_url or settings.embedding_base_url or legacy_url or settings.ollama_base_url
        if not model or not model.strip():
            raise config_error("embedding_model_missing")
        return ModelProfile(provider, endpoint(url, provider), model.strip())
    if role is not None and role not in ROLE_MODELS:
        raise config_error("unknown_role")
    if role is None:
        provider = primary
    elif (
        role == "rag_planner"
        and not has_role_override(settings, role)
        and not settings.rag_planner_model
    ):
        # Existing trusted JSON describes the old Ollama planner, not the newly
        # selected primary provider. Never reinterpret its alias as a vLLM alias.
        provider = "ollama"
    else:
        provider = getattr(settings, role + "_provider") or primary
    if provider not in {"ollama", "vllm"}:
        raise config_error("unsupported_provider")
    default_url = settings.vllm_base_url if provider == "vllm" else settings.ollama_base_url
    default_model = settings.vllm_chat_model if provider == "vllm" else settings.ollama_chat_model
    role_url = getattr(settings, role + "_base_url") if role else None
    role_model = getattr(settings, ROLE_MODELS[role]) if role else None
    url = (
        base_url
        or role_url
        or (legacy_url if role == "rag_planner" and provider == "ollama" else None)
        or default_url
    )
    model = (
        role_model
        or (legacy_model if role == "rag_planner" and provider == "ollama" else None)
        or default_model
    )
    if not model or not model.strip():
        raise config_error("chat_model_missing")
    normalized = endpoint(url, provider)
    same = (
        provider == primary and bool(default_url) and normalized == endpoint(default_url, provider)
    )
    key = getattr(settings, role + "_api_key") if role else None
    configured_url = (
        role_url
        or (legacy_url if role == "rag_planner" and provider == "ollama" else None)
        or default_url
    )
    if (
        key is not None
        and base_url
        and (not configured_url or normalized != endpoint(configured_url, provider))
    ):
        # CLI endpoint overrides do not carry credentials for the configured host.
        key = None
    if key is None and same and provider == "vllm":
        key = settings.vllm_api_key
    if provider == "ollama" and key is not None:
        raise config_error("ollama_api_key_not_supported")
    control = getattr(settings, role + "_reasoning_control") if role else None
    reasoning = getattr(settings, role + "_reasoning_enabled") if role else None
    if control is None:
        control = settings.vllm_reasoning_control if same and provider == "vllm" else "unsupported"
    if reasoning is None and same and provider == "vllm":
        reasoning = settings.vllm_reasoning_enabled
    return ModelProfile(provider, normalized, model.strip(), key, control, reasoning)


def canonical_model(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
