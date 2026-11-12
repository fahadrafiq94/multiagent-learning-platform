from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
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

    # Per-agent model overrides.
    #
    # None means:
    # "Use ollama_chat_model as the default."
    orchestrator_model: str | None = None
    scenario_agent_model: str | None = None
    process_coach_agent_model: str | None = None
    ap_plus_navigator_model: str | None = None

    # Optional dedicated model for session-summary updates.
    # None means: "Use ollama_chat_model as the default."
    summary_model: str | None = None

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
