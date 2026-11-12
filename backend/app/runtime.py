"""Own HTTP runtime resources through ASGI lifespan, never through a global cache."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import settings
from app.knowledge.intake_runtime import create_intake_runtime
from app.knowledge.runtime import create_grounding_runtime
from app.orchestration.factory import build_orchestration_runtime
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.persistence.worker import ApplicationWorker
from app.repositories.exceptions import RepositoryUnavailableError
from app.repositories.postgres import create_postgres_repositories
from app.services.model_service import create_model_service


@asynccontextmanager
async def runtime_lifespan(app: FastAPI):
    database = None
    worker = None
    knowledge_database = None
    knowledge_cache = None
    intake_database = None
    try:
        repositories = None
        if settings.application_storage == "postgres":
            if (
                settings.application_database_url is None
                or settings.application_pilot_auth_secret is None
            ):
                raise RepositoryUnavailableError(
                    "PostgreSQL runtime requires database and pilot-auth configuration."
                )
            database = ApplicationDatabase(
                ApplicationDatabaseConfig(
                    dsn=settings.application_database_url,
                    max_size=settings.application_database_pool_size,
                )
            )
            worker = ApplicationWorker(database)
            await worker.start()
            repositories = create_postgres_repositories(database)
        grounding = None
        if settings.rag_enabled:
            grounding, knowledge_database, knowledge_cache = await create_grounding_runtime(
                settings
            )
        extra = {"grounding_service": grounding} if grounding is not None else {}
        app.state.runtime = build_orchestration_runtime(
            model_service=create_model_service(),
            repositories=repositories,
            **extra,
        )
        app.state.application_database = database
        app.state.application_worker = worker
        app.state.knowledge_database = knowledge_database
        app.state.knowledge_intake = None
        if settings.knowledge_intake_enabled:
            app.state.knowledge_intake, intake_database = create_intake_runtime(settings)
        yield
    finally:
        app.state.runtime = None
        app.state.knowledge_intake = None
        if intake_database is not None:
            await intake_database.close()
        if knowledge_cache is not None:
            await knowledge_cache.aclose()
        if knowledge_database is not None:
            await knowledge_database.close()
        if database is not None:
            await database.close()
        if worker is not None:
            await worker.close()
