"""Explicit intake resources; independent of experimental student RAG enablement."""

from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.config.settings import Settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeUnavailable
from app.knowledge.extraction import PdfExtractor
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeConfig
from app.knowledge.source_store import FileSourceStore
from app.services.model_service import create_model_service
from app.services.model_service.profiles import resolve_profile


def create_intake_runtime(settings: Settings, *, require_auth: bool = True):
    if require_auth and (
        settings.application_storage != "postgres" or settings.application_pilot_auth_secret is None
    ):
        raise KnowledgeUnavailable("HTTP intake requires the authenticated PostgreSQL runtime.")
    dsn = settings.knowledge_ingest_database_url
    if dsn is None and settings.knowledge_db_host and settings.knowledge_writer_password:
        dsn = SecretStr(
            make_conninfo(
                host=settings.knowledge_db_host,
                port="5432",
                dbname="fredi_knowledge",
                user="fredi_knowledge_writer",
                password=settings.knowledge_writer_password.get_secret_value(),
            )
        )
    if settings.knowledge_intake_config_file is None or dsn is None:
        raise KnowledgeUnavailable("Intake requires trusted configuration and writer storage.")
    try:
        config = IntakeConfig.read(settings.knowledge_intake_config_file)
    except (OSError, ValueError):
        raise KnowledgeUnavailable("Intake configuration is invalid.") from None
    for binding in config.bindings:
        resolve_profile(settings, "embedding", legacy_model=binding.index.embedding.model)
    model = create_model_service(settings, role="embedding")
    database = KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=dsn))
    service = IntakeService(
        IntakeRepository(database),
        FileSourceStore(settings.knowledge_source_root),
        PdfExtractor(),
        model,
        config,
    )
    return service, database
