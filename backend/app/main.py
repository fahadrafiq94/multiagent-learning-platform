from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import settings
from app.logging.config import configure_logging
from app.logging.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    logger.info(
        "Application_started",
        version=settings.app_version,
    )

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
