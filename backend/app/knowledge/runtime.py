"""Opt-in lifespan construction with a separate read-only knowledge pool."""

import hashlib
import secrets

from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr

from app.config.settings import Settings
from app.knowledge.cache import KnowledgeCache, NoKnowledgeCache, RedisKnowledgeCache
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgePermissionError, KnowledgeUnavailable
from app.knowledge.integration import RagRuntimeConfig, SpecialistGroundingService
from app.knowledge.planner import HybridRetrievalService
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.search_repository import PostgresSearchRepository
from app.services.model_service import create_model_service
from app.services.model_service.profiles import resolve_profile
from app.services.model_service.service import close_model_services

_PASSWORDLESS_CACHE_KEY = secrets.token_bytes(32)


def create_knowledge_cache(settings: Settings, dsn: SecretStr) -> KnowledgeCache:
    """Shared runtime/operator construction; never opens Redis during setup."""
    if settings.knowledge_cache_enabled:
        if (host := settings.knowledge_cache_redis_host) is None:
            raise KnowledgeUnavailable("Knowledge caching requires a Redis host.")
        # Reader credentials are independent of Redis storage permissions. Use
        # the password so native/Docker DSN host aliases share the same signature.
        secret = conninfo_to_dict(dsn.get_secret_value()).get("password")
        signing_key = (
            hashlib.sha256(b"fredi-knowledge-cache-v2\x00" + str(secret).encode()).digest()
            if secret
            else _PASSWORDLESS_CACHE_KEY
        )
        return RedisKnowledgeCache(
            host=host,
            port=settings.knowledge_cache_redis_port,
            password=(
                settings.knowledge_cache_redis_password.get_secret_value()
                if settings.knowledge_cache_redis_password
                else None
            ),
            ttl_jitter_seconds=settings.knowledge_cache_ttl_jitter_seconds,
            operation_timeout_seconds=settings.knowledge_cache_operation_timeout_seconds,
            max_payload_bytes=settings.knowledge_cache_max_payload_bytes,
            signing_key=signing_key,
        )
    return NoKnowledgeCache()


async def create_grounding_runtime(
    settings: Settings,
) -> tuple[SpecialistGroundingService, KnowledgeDatabase, KnowledgeCache]:
    if settings.application_storage != "postgres" or settings.rag_config_file is None:
        raise KnowledgeUnavailable(
            "RAG requires authenticated PostgreSQL runtime and trusted bindings."
        )
    try:
        config = RagRuntimeConfig.read(settings.rag_config_file)
    except (ValueError, OSError):
        raise KnowledgeUnavailable("RAG runtime configuration is invalid.") from None
    dsn = settings.rag_database_url
    if dsn is None and settings.rag_database_host and settings.rag_reader_password:
        dsn = SecretStr(
            make_conninfo(
                host=settings.rag_database_host,
                port="5432",
                dbname="fredi_knowledge",
                user="fredi_knowledge_reader",
                password=settings.rag_reader_password.get_secret_value(),
            )
        )
    if dsn is None:
        raise KnowledgeUnavailable("RAG requires a protected read-only database connection.")
    cache = create_knowledge_cache(settings, dsn)
    database = KnowledgeDatabase(
        KnowledgeDatabaseConfig(dsn=dsn, max_size=settings.rag_database_pool_size)
    )
    models = []
    try:
        async with database.connection() as connection:
            await connection.execute("SET TRANSACTION READ ONLY")
            row = await (
                await connection.execute(
                    "SELECT COALESCE(bool_or(has_table_privilege(current_user,r.oid, "
                    "'INSERT,UPDATE,DELETE,TRUNCATE')),false) "
                    "OR has_schema_privilege(current_user,'knowledge','CREATE') AS can_write "
                    "FROM pg_class r JOIN pg_namespace n ON n.oid=r.relnamespace "
                    "WHERE n.nspname='knowledge' AND r.relkind IN ('r','p','v')"
                )
            ).fetchone()
            if row is None or row["can_write"]:
                raise KnowledgePermissionError(
                    "RAG requires a role without knowledge write privileges."
                )
            await connection.execute("SELECT 1 FROM knowledge.knowledge_chunks LIMIT 0")
        planner_settings = settings.model_copy(
            update={
                "model_max_retries": 0,
                "model_request_timeout_seconds": config.planner.timeout_seconds,
            }
        )
        planner_url = str(config.planner_url) if config.planner_url else None
        embedding_url = str(config.embedding_url) if config.embedding_url else None
        selected = resolve_profile(
            planner_settings,
            "rag_planner",
            legacy_model=config.planner.model,
            legacy_url=planner_url,
        )
        # The planner request must name the model at the selected endpoint, not
        # an old Ollama alias left in the trusted JSON file.
        config = config.model_copy(
            update={"planner": config.planner.model_copy(update={"model": selected.model})}
        )
        embeddings = create_model_service(settings, role="embedding", legacy_url=embedding_url)
        models.append(embeddings)
        planner_model = create_model_service(
            planner_settings,
            role="rag_planner",
            legacy_model=config.planner.model,
            legacy_url=planner_url,
        )
        models.append(planner_model)
        retrieval = KnowledgeRetrievalService(
            PostgresSearchRepository(
                database,
                cache=cache,
                cache_ttl_seconds=settings.knowledge_cache_ttl_seconds,
            ),
            embeddings,
        )
        service = SpecialistGroundingService(
            HybridRetrievalService(retrieval, planner_model, config.planner),
            config,
        )
        return service, database, cache
    except BaseException:
        try:
            await close_model_services(*models)
        finally:
            await cache.aclose()
            await database.close()
        raise
