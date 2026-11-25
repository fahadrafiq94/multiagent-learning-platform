from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_LOG_LEVELS = frozenset(
    {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    },
)
_MODEL_ROLES = (
    "orchestrator",
    "scenario_agent",
    "process_coach",
    "ap_plus_navigator",
    "summary",
    "rag_planner",
)


class Settings(BaseSettings):
    app_name: str = "Multiagent Learning Platform"
    app_version: str = "0.5.0"

    app_env: str = "development"
    log_level: str = "INFO"

    # HTTP persistence must be explicitly configured; never fall back after DB failure.
    application_storage: Literal["memory", "postgres"] = "memory"
    application_database_url: SecretStr | None = None
    application_database_pool_size: int = Field(default=4, ge=1, le=32)
    application_pilot_auth_secret: SecretStr | None = None
    # Browser access is opt-in; APIs using existing bearer credentials remain available.
    student_browser_origin: str | None = None
    student_browser_secure_cookie: bool = True

    @field_validator("student_browser_origin")
    @classmethod
    def validate_browser_origin(cls, value: str | None) -> str | None:
        if not value:
            return None
        from urllib.parse import urlsplit

        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
            port = parsed.port
            host = parsed.hostname.lower()
            host = f"[{host}]" if ":" in host else host
            default_port = 443 if parsed.scheme == "https" else 80
            suffix = f":{port}" if port is not None and port != default_port else ""
            return f"{parsed.scheme}://{host}{suffix}"
        except ValueError:
            raise ValueError(
                "Student browser origin must be an HTTP(S) origin without a path."
            ) from None

    # Experimental student RAG is explicitly opt-in; course grants live in a
    # trusted server file, never client metadata or document text.
    rag_enabled: bool = False
    rag_config_file: Path | None = None
    rag_database_url: SecretStr | None = None
    rag_database_host: str | None = None
    rag_reader_password: SecretStr | None = None
    rag_database_pool_size: int = Field(default=4, ge=1, le=16)

    knowledge_intake_enabled: bool = False
    knowledge_intake_config_file: Path | None = None
    knowledge_ingest_database_url: SecretStr | None = None
    knowledge_db_host: str | None = None
    knowledge_writer_password: SecretStr | None = None
    knowledge_source_root: Path = Path(".knowledge-sources")

    # Checkpoint 10: disposable Redis acceleration for knowledge retrieval.
    # PostgreSQL remains authoritative; an outage only removes the speedup.
    knowledge_cache_enabled: bool = False
    knowledge_cache_redis_host: str | None = None
    knowledge_cache_redis_port: int = Field(default=6379, ge=1, le=65535)
    knowledge_cache_redis_password: SecretStr | None = None
    knowledge_cache_ttl_seconds: int = Field(default=3600, ge=30, le=86400)
    knowledge_cache_ttl_jitter_seconds: int = Field(default=300, ge=0, le=3600)
    knowledge_cache_operation_timeout_seconds: float = Field(default=0.1, gt=0.0, le=5.0)
    knowledge_cache_max_payload_bytes: int = Field(default=262144, ge=4096, le=1048576)

    @field_validator("application_pilot_auth_secret")
    @classmethod
    def validate_pilot_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("Pilot signing secret requires at least 32 characters.")
        return value

    # 4B.10: optional JSON-lines research export of request events.
    # None disables the file handler; events still reach the app log.
    observability_log_file: str | None = None
    observability_log_max_bytes: int = Field(default=10485760, ge=1024, le=104857600)
    observability_log_backup_count: int = Field(default=5, ge=1, le=100)

    # Model service
    model_provider: str = "ollama"

    # Ollama connection and model tags are deployment-specific and
    # must come from the environment; they are never hardcoded. The
    # module stays importable without them, and creating the model
    # service fails with a clear error (surfaced as a clean 503) when
    # they are missing.
    ollama_base_url: str | None = None

    ollama_chat_model: str | None = None

    ollama_embedding_model: str | None = None

    # Explicit remote chat selection; independent embeddings are configured below.
    vllm_base_url: str | None = None
    vllm_chat_model: str | None = None
    vllm_api_key: SecretStr | None = None
    vllm_reasoning_control: Literal["unsupported", "enable_thinking"] = "unsupported"
    vllm_reasoning_enabled: bool | None = None

    # Per-agent model overrides.
    #
    # None means:
    # "Use the selected provider's default chat model."
    orchestrator_model: str | None = None
    scenario_agent_model: str | None = None
    process_coach_agent_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PROCESS_COACH_MODEL", "PROCESS_COACH_AGENT_MODEL", "process_coach_agent_model"
        ),
    )
    ap_plus_navigator_model: str | None = None

    # Optional dedicated model for session-summary updates.
    # None means: "Use the selected provider's default chat model."
    summary_model: str | None = None

    # Optional independent role endpoints. Empty overrides inherit the primary
    # profile; credentials inherit only for the same normalized endpoint/provider.
    orchestrator_provider: Literal["ollama", "vllm"] | None = None
    orchestrator_base_url: str | None = None
    orchestrator_api_key: SecretStr | None = None
    orchestrator_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    orchestrator_reasoning_enabled: bool | None = None
    scenario_agent_provider: Literal["ollama", "vllm"] | None = None
    scenario_agent_base_url: str | None = None
    scenario_agent_api_key: SecretStr | None = None
    scenario_agent_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    scenario_agent_reasoning_enabled: bool | None = None
    process_coach_provider: Literal["ollama", "vllm"] | None = None
    process_coach_base_url: str | None = None
    process_coach_api_key: SecretStr | None = None
    process_coach_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    process_coach_reasoning_enabled: bool | None = None
    ap_plus_navigator_provider: Literal["ollama", "vllm"] | None = None
    ap_plus_navigator_base_url: str | None = None
    ap_plus_navigator_api_key: SecretStr | None = None
    ap_plus_navigator_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    ap_plus_navigator_reasoning_enabled: bool | None = None
    summary_provider: Literal["ollama", "vllm"] | None = None
    summary_base_url: str | None = None
    summary_api_key: SecretStr | None = None
    summary_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    summary_reasoning_enabled: bool | None = None

    rag_planner_provider: Literal["ollama", "vllm"] | None = None
    rag_planner_base_url: str | None = None
    rag_planner_model: str | None = None
    rag_planner_api_key: SecretStr | None = None
    rag_planner_reasoning_control: Literal["unsupported", "enable_thinking"] | None = None
    rag_planner_reasoning_enabled: bool | None = None
    embedding_provider: Literal["ollama", "vllm"] | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None

    @field_validator(
        "embedding_provider",
        *[
            f"{role}_{option}"
            for role in _MODEL_ROLES
            for option in ("provider", "reasoning_control", "reasoning_enabled")
        ],
        mode="before",
    )
    @classmethod
    def empty_role_option_inherits(cls, value):
        return None if value == "" else value

    # Model request behavior
    model_request_timeout_seconds: float = 120.0
    model_max_retries: int = 2
    model_retry_backoff_seconds: float = 0.5

    # Default generation behavior
    model_temperature: float = 0.2
    model_max_tokens: int | None = None

    # Context budgeting
    context_max_tokens: int = 8192

    context_reserved_output_tokens: int = 2048

    context_max_recent_messages: int = 20

    context_characters_per_token: float = 4.0

    # Structural overhead per rendered conversation message
    # (JSON fields: role, content, created_at, agent/route).
    context_message_overhead_tokens: int = 32

    # Budget reserved per request for the selected agent's own
    # system prompt, guidance policy, and state-update instructions,
    # which are never part of the measured context payload.
    context_reserved_agent_prompt_tokens: int = 1500

    # Session summary (Sprint 4B.8)
    # A summary update runs only after this many messages have fallen
    # out of the recent-history window since the last update.
    context_summary_min_new_messages: int = 4

    # Maximum items per structured list in the persisted summary.
    context_summary_max_items_per_list: int = 10

    # Maximum characters for the persisted summary overview.
    context_summary_max_overview_chars: int = 1200

    model_config = SettingsConfigDict(
        hide_input_in_errors=True,
        # Resolved relative to this file so the .env is found
        # regardless of the process working directory.
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(
        cls,
        value: str,
    ) -> str:
        normalized = value.upper().strip()

        if normalized not in _ALLOWED_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(_ALLOWED_LOG_LEVELS)}, got {value!r}."
            )

        return normalized


settings = Settings()
