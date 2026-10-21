from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multiagent Learning Platform"
    app_version: str = "0.1.0"

    app_env: str = "development"
    log_level: str = "INFO"

    # Model service
    model_provider: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://ollama:11434"

    # Default chat model.
    #
    # Used whenever an agent does not provide its own model override.
    ollama_chat_model: str = "qwen3:4b"

    ollama_embedding_model: str = "nomic-embed-text-v2-moe"

    # Per-agent model overrides.
    #
    # None means:
    # "Use ollama_chat_model as the default."
    orchestrator_model: str | None = None
    scenario_agent_model: str | None = None
    process_coach_agent_model: str | None = None
    ap_plus_navigator_model: str | None = None

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

    context_message_overhead_tokens: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
