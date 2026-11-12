from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.conversations import router as conversations_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.model import router as model_router
from app.api.routes.orchestration import router as orchestration_router
from app.config.settings import settings
from app.logging.config import configure_logging
from app.logging.logger import logger
from app.repositories.exceptions import (
    DuplicateRecordError,
    RepositoryConflictError,
    RepositoryError,
    RepositoryNotFoundError,
    VersionConflictError,
)
from app.runtime import runtime_lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    logger.info(
        "Application_started",
        version=settings.app_version,
    )

    async with runtime_lifespan(app):
        yield
    logger.info("Application_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    logger.info("Health_check_endpoint_called")
    return {"status": "ok"}


app.include_router(model_router)
app.include_router(orchestration_router)
app.include_router(conversations_router)
app.include_router(knowledge_router)


@app.exception_handler(RepositoryError)
async def repository_failure(request: Request, exc: RepositoryError):
    if isinstance(exc, RepositoryNotFoundError):
        return JSONResponse(
            status_code=404, content={"detail": "Conversation or record not found."}
        )
    if isinstance(exc, (RepositoryConflictError, VersionConflictError, DuplicateRecordError)):
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    "Request conflicts with current state; inspect turn status before retrying."
                )
            },
        )
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Application persistence or workflow is unavailable; "
                "inspect turn status before retrying."
            )
        },
    )


@app.get("/ready")
async def ready(request: Request):
    if getattr(request.app.state, "runtime", None) is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    database = getattr(request.app.state, "application_database", None)
    if database is not None:
        async with database.connection() as connection:
            await connection.execute("SELECT 1")
    return {"status": "ok", "storage": settings.application_storage}
